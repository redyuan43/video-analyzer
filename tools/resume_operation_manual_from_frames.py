#!/usr/bin/env python3
"""Resume an operation-manual run from extracted frames and transcript.md."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analyzer.artifacts import write_orin_artifacts
from video_analyzer.audio_processor import AudioTranscript
from video_analyzer.cli import (
    analyze_frames_for_vl,
    create_client,
    create_operation_manual_text_client,
    read_page_context_metadata,
)
from video_analyzer.config import Config, get_model, resolve_temperature
from video_analyzer.frame import Frame
from video_analyzer.frame_manifest import MANIFEST_NAME, read_frame_manifest
from video_analyzer.frame_selection import (
    AUTO,
    FrameSelectionOptions,
    resolve_vl_context_gap_seconds,
    select_vl_frames,
)
from video_analyzer.manual import (
    embed_step_images,
    generate_operation_manual,
    prepare_frame_assets,
    read_context_file,
    review_operation_manual_markdown,
    write_frame_evidence_index,
)
from video_analyzer.ocr import run_ocr
from video_analyzer.local_model_runtime import local_model_runtime_session, local_model_stage
from video_analyzer.prompt import PromptLoader

LOGGER = logging.getLogger(__name__)
TIMESTAMP_RE = re.compile(
    r"^-\s+\[(?P<start>\d\d:\d\d:\d\d)\s+-\s+(?P<end>\d\d:\d\d:\d\d)\]\s+(?P<text>.*)$"
)


@dataclass(frozen=True)
class FrameLoadResult:
    frames: list[Frame]
    metadata: dict


def bypass_proxy_environment() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def seconds_from_timestamp(value: str) -> float:
    hours, minutes, seconds = [int(part) for part in value.split(":")]
    return float(hours * 3600 + minutes * 60 + seconds)


def read_transcript(path: Path) -> AudioTranscript:
    text = path.read_text(encoding="utf-8")
    language = ""
    segments = []
    full_text = []
    for line in text.splitlines():
        if line.startswith("- Language:"):
            language = line.split(":", 1)[1].strip()
            continue
        match = TIMESTAMP_RE.match(line)
        if not match:
            continue
        segment_text = match.group("text").strip()
        if not segment_text:
            continue
        segments.append(
            {
                "start": seconds_from_timestamp(match.group("start")),
                "end": seconds_from_timestamp(match.group("end")),
                "text": segment_text,
            }
        )
        full_text.append(segment_text)
    if not segments:
        full_text = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#") and not line.startswith("- Language:") and not line.startswith("- Segments:")
        ]
    return AudioTranscript(text="\n".join(full_text).strip(), segments=segments, language=language)


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(float(result.stdout.strip()), 0.0)


def load_frames(
    run_dir: Path,
    video_duration_seconds: float,
    source_analysis: Path | None,
    allow_estimated_timestamps: bool,
) -> FrameLoadResult:
    frame_paths = sorted(
        (run_dir / "frames").glob("frame_*.jpg"),
        key=lambda path: int(path.stem.split("_", 1)[1]),
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frames found under {run_dir / 'frames'}")

    frame_numbers = [int(path.stem.split("_", 1)[1]) for path in frame_paths]
    timestamp_sources = []
    if source_analysis:
        timestamp_sources.append(("analysis", source_analysis))
    timestamp_sources.extend(
        [
            ("manifest", run_dir / MANIFEST_NAME),
            ("analysis", run_dir / "analysis.json"),
        ]
    )
    timestamp_map = {}
    score_map = {}
    timestamp_metadata = {}
    for source_kind, timestamp_source in timestamp_sources:
        if source_kind == "manifest":
            candidate_timestamps, candidate_scores, candidate_metadata = read_frame_manifest(timestamp_source)
        else:
            candidate_timestamps, candidate_scores, candidate_metadata = _load_frame_timestamp_map(timestamp_source, frame_numbers)
        if set(frame_numbers).issubset(candidate_timestamps):
            timestamp_map = {number: candidate_timestamps[number] for number in frame_numbers}
            score_map = candidate_scores
            timestamp_metadata = candidate_metadata
            break
    if timestamp_map:
        return FrameLoadResult(
            frames=[
                Frame(
                    number=number,
                    path=path,
                    timestamp=timestamp_map[number],
                    score=score_map.get(number, 0.0),
                )
                for number, path in zip(frame_numbers, frame_paths)
            ],
            metadata={
                "backend": "resumed_existing_frames",
                "frame_count": len(frame_paths),
                "timestamp_source": timestamp_metadata,
                "estimated_timestamps": False,
            },
        )

    if not allow_estimated_timestamps:
        raise RuntimeError(
            "Frame timestamps were not found. Pass --source-analysis pointing at the original "
            "analysis.json, or pass --allow-estimated-frame-timestamps to use uniform estimates."
        )

    denominator = max(len(frame_paths) - 1, 1)
    return FrameLoadResult(
        frames=[
            Frame(
                number=number,
                path=path,
                timestamp=(video_duration_seconds * index / denominator),
                score=0.0,
            )
            for index, (number, path) in enumerate(zip(frame_numbers, frame_paths))
        ],
        metadata={
            "backend": "resumed_existing_frames",
            "frame_count": len(frame_paths),
            "timestamp_source": {
                "source": "uniform_estimate",
                "reason": "no frame timestamp metadata was available",
                "video_duration_seconds": video_duration_seconds,
            },
            "estimated_timestamps": True,
        },
    )


def _load_frame_timestamp_map(source_analysis: Path, required_frame_numbers: list[int]) -> tuple[dict[int, float], dict[int, float], dict]:
    if not source_analysis or not source_analysis.exists():
        return {}, {}, {"source": "missing", "path": str(source_analysis) if source_analysis else ""}

    payload = json.loads(source_analysis.read_text(encoding="utf-8"))
    candidates = [
        ("ocr_events", payload.get("ocr_events")),
        ("frame_selection.frames", (payload.get("metadata") or {}).get("frame_selection", {}).get("frames")),
        ("visual_events", payload.get("visual_events")),
        ("frame_analyses", payload.get("frame_analyses")),
    ]
    required = set(required_frame_numbers)
    for source_name, items in candidates:
        timestamp_map, score_map = _extract_timestamp_map(items)
        if required.issubset(timestamp_map):
            return (
                {number: timestamp_map[number] for number in required_frame_numbers},
                score_map,
                {
                    "source": source_name,
                    "path": str(source_analysis),
                },
            )
    return (
        {},
        {},
        {
            "source": "not_found",
            "path": str(source_analysis),
            "required_frame_count": len(required_frame_numbers),
        },
    )


def _extract_timestamp_map(items) -> tuple[dict[int, float], dict[int, float]]:
    timestamp_map: dict[int, float] = {}
    score_map: dict[int, float] = {}
    if not isinstance(items, list):
        return timestamp_map, score_map
    for item in items:
        if not isinstance(item, dict):
            continue
        frame_number = item.get("frame_number", item.get("number"))
        timestamp = item.get("timestamp")
        if frame_number is None or timestamp is None:
            continue
        number = int(frame_number)
        timestamp_map[number] = float(timestamp)
        if item.get("visual_change_score") is not None:
            score_map[number] = float(item["visual_change_score"])
        elif item.get("score") is not None:
            score_map[number] = float(item["score"])
    return timestamp_map, score_map


def configure_runtime(args: argparse.Namespace) -> Config:
    config = Config(args.config)
    config.config["task"] = "operation_manual"
    config.config["output_dir"] = str(args.run_dir)
    config.config["context_file"] = str(args.context_file)
    config.config["manual_language"] = args.manual_language
    config.config["keep_frames"] = True
    config.config["clients"]["default"] = "openai_api"
    config.config["clients"]["openai_api"]["api_key"] = "0"
    config.config["clients"]["openai_api"]["api_url"] = args.vision_base_url
    config.config["clients"]["openai_api"]["model"] = args.vision_model
    config.config.setdefault("operation_manual", {}).update(
        {
            "llm_base_url": args.llm_base_url,
            "vision_base_url": args.vision_base_url,
            "text_base_url": args.text_base_url,
            "vision_model": args.vision_model,
            "text_model": args.text_model,
            "text_temperature": args.text_temperature
            if args.text_temperature is not None
            else (1.0 if "deepseek.com" in args.text_base_url else 0.2),
            "deepseek_thinking": args.deepseek_thinking,
            "frame_no_think": True,
            "manual_no_think": True,
        }
    )
    config.config.setdefault("ocr", {}).update(
        {
            "provider": "auto",
            "base_url": args.ocr_base_url[0],
            "base_urls": args.ocr_base_url,
            "concurrency": args.ocr_concurrency,
            "cache": args.ocr_cache,
            "cache_dir": args.ocr_cache_dir,
            "timeout_seconds": args.ocr_timeout_seconds,
        }
    )
    config.config.setdefault("asr", {}).update({"provider": "vibevoice", "strategy": "balanced"})
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--config", default="config")
    parser.add_argument("--pipeline-mode", choices=["fast", "balanced", "deep"], default="balanced")
    parser.add_argument("--ocr-base-url", action="append", required=True)
    parser.add_argument("--ocr-concurrency", default="auto")
    parser.add_argument("--ocr-cache", choices=["on", "off", "refresh"], default="refresh")
    parser.add_argument("--ocr-cache-dir", default=".cache/video-analyzer/ocr")
    parser.add_argument("--ocr-timeout-seconds", type=float, default=30)
    parser.add_argument("--llm-base-url")
    parser.add_argument("--vision-base-url")
    parser.add_argument("--text-base-url")
    parser.add_argument("--vision-model", default="minicpm-v-4.5-v100")
    parser.add_argument("--text-model", default="hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive")
    parser.add_argument("--text-temperature", type=float)
    parser.add_argument("--deepseek-thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--manual-language", default="zh-CN")
    parser.add_argument("--source-analysis", type=Path, help="Original analysis.json to recover exact frame timestamps")
    parser.add_argument(
        "--allow-estimated-frame-timestamps",
        action="store_true",
        help="Allow uniform timestamp estimates when exact frame timestamps are unavailable",
    )
    parser.add_argument("--vl-concurrency", type=int, default=3)
    parser.add_argument("--vl-context-before", type=int, default=0)
    parser.add_argument("--vl-context-after", type=int, default=0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()
    services = (Config(args.config).get("endpoints") or {}).get("services", {})
    args.llm_base_url = args.llm_base_url or services.get("amd_fast_base_url")
    args.vision_base_url = args.vision_base_url or services.get("minicpm_v100_base_url") or args.llm_base_url
    args.text_base_url = args.text_base_url or services.get("amd_fast_base_url") or args.llm_base_url

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    bypass_proxy_environment()

    started = time.perf_counter()
    timings = {}
    config = configure_runtime(args)
    client = create_client(config)
    text_client = create_operation_manual_text_client(config, client)
    model = get_model(config)
    prompt_loader = PromptLoader(config.get("prompt_dir"), config.get("prompts", []))
    page_context = read_context_file(str(args.context_file))
    page_context_metadata = read_page_context_metadata(str(args.context_file), page_context)
    transcript = read_transcript(args.run_dir / "transcript.md")
    video_duration = probe_duration(args.video)
    frame_load = load_frames(
        args.run_dir,
        video_duration,
        source_analysis=args.source_analysis,
        allow_estimated_timestamps=args.allow_estimated_frame_timestamps,
    )
    frames = frame_load.frames

    with local_model_runtime_session(config.config, LOGGER, str(args.run_dir)):
        LOGGER.info("Resuming from %s frames and transcript.md; starting at OCR", len(frames))
        stage_started = time.perf_counter()
        ocr_config = config.get("ocr", {})
        with local_model_stage("ocr", config.config, LOGGER, str(args.run_dir)):
            ocr_events = run_ocr(
                frames=frames,
                provider=ocr_config.get("provider", "auto"),
                base_url=ocr_config.get("base_url", "auto"),
                model=ocr_config.get("model", "model"),
                prompt_mode=ocr_config.get("prompt_mode", "prompt_scene_spotting"),
                base_urls=ocr_config.get("base_urls"),
                ocr_concurrency=ocr_config.get("concurrency", "auto"),
                fallback_base_url=config.get("operation_manual", {}).get("llm_base_url"),
                fallback_model=config.get("operation_manual", {}).get("vision_model"),
                fallback_api_key="0",
                request_timeout_seconds=ocr_config.get("timeout_seconds", 120),
                max_tokens=ocr_config.get("max_tokens", 1024),
                max_image_long_side=ocr_config.get("max_image_long_side", 1280),
                retry_endpoints=bool(ocr_config.get("retry_endpoints", True)),
                cache_mode=ocr_config.get("cache", "on"),
                cache_dir=ocr_config.get("cache_dir", ".cache/video-analyzer/ocr"),
            )
        timings["ocr_seconds"] = round(time.perf_counter() - stage_started, 3)

        LOGGER.info("Selecting and analyzing VL frames")
        stage_started = time.perf_counter()
        options = FrameSelectionOptions(pipeline_mode=args.pipeline_mode)
        selected_frame_numbers, frame_decisions, frame_selection_metadata = select_vl_frames(
            frames=frames,
            ocr_events=ocr_events,
            transcript=transcript,
            video_duration_seconds=video_duration,
            options=options,
        )
        timings["frame_selection_seconds"] = round(time.perf_counter() - stage_started, 3)
        vl_started = time.perf_counter()
        analyzer = __import__("video_analyzer.analyzer", fromlist=["VideoAnalyzer"]).VideoAnalyzer(
            client,
            model,
            prompt_loader,
            config.get("clients", {}).get("temperature", 0.2),
            config.get("prompt", ""),
            frame_num_predict=config.get("response_length", {}).get("frame", 300),
            frame_no_think=True,
        )
        with local_model_stage("vl", config.config, LOGGER, str(args.run_dir)):
            frame_analyses = analyze_frames_for_vl(
                analyzer=analyzer,
                frames=frames,
                ocr_events=ocr_events,
                selected_frame_numbers=selected_frame_numbers,
                decisions=frame_decisions,
                concurrency=max(args.vl_concurrency, 1),
                context_before=max(args.vl_context_before, 0),
                context_after=max(args.vl_context_after, 0),
                context_max_gap=AUTO,
            )
        timings["vl_seconds"] = round(time.perf_counter() - vl_started, 3)

    LOGGER.info("Generating operation manual")
    stage_started = time.perf_counter()
    frame_assets = prepare_frame_assets(frames, args.run_dir)
    operation_manual = generate_operation_manual(
        client=text_client,
        text_model=config.get("operation_manual", {}).get("text_model"),
        frame_analyses=frame_analyses,
        frames=frames,
        transcript=transcript,
        asr_metadata={"strategy": "resume_from_transcript_md", "providers_run": ["vibevoice"]},
        ocr_events=ocr_events,
        page_context=page_context,
        language=args.manual_language,
        temperature=resolve_temperature(config.get("operation_manual", {}), config.get("clients", {}).get("temperature", 0.2)),
        frame_assets=frame_assets,
        no_think=True,
    )
    operation_manual["response"] = embed_step_images(operation_manual.get("response", ""), frames, frame_assets)
    operation_manual["quality_review"] = review_operation_manual_markdown(operation_manual.get("response", ""))
    operation_manual["quality_gate_passed"] = not any(
        issue.get("severity") == "error" for issue in operation_manual["quality_review"]
    )
    manual_filename = "operation_manual.md" if operation_manual["quality_gate_passed"] else "operation_manual.quality_failed.md"
    operation_manual["manual_path"] = str(args.run_dir / manual_filename)
    operation_manual["evidence_path"] = str(
        write_frame_evidence_index(frames, args.run_dir, ocr_events, frame_analyses, frame_assets)
    )
    timings["manual_generation_seconds"] = round(time.perf_counter() - stage_started, 3)
    timings["resume_total_seconds"] = round(time.perf_counter() - started, 3)

    ocr_endpoints = sorted(
        {
            event.provider.split(":", 1)[1]
            for event in ocr_events
            if event.provider.startswith("dots_mocr_vllm:")
        }
    )
    results = {
        "metadata": {
            "task": "operation_manual",
            "resume_from": str(args.run_dir),
            "resume_start": "ocr",
            "model": model,
            "vision_base_url": config.get("operation_manual", {}).get("vision_base_url"),
            "text_model": config.get("operation_manual", {}).get("text_model"),
            "text_base_url": config.get("operation_manual", {}).get("text_base_url"),
            "text_temperature": resolve_temperature(
                config.get("operation_manual", {}),
                config.get("clients", {}).get("temperature", 0.2),
            ),
            "context_file": str(args.context_file),
            "page_context": page_context_metadata,
            "frames_extracted": len(frames),
            "frames_processed": len(frame_analyses),
            "vl_frames_processed": len(selected_frame_numbers),
            "frame_selection": frame_selection_metadata,
            "frame_extraction": frame_load.metadata,
            "ocr": {
                "requested_endpoints": args.ocr_base_url,
                "effective_endpoints": ocr_endpoints,
                "effective_worker_count": len(ocr_endpoints),
                "concurrency": args.ocr_concurrency,
                "cache_mode": args.ocr_cache,
                "cache_dir": args.ocr_cache_dir,
                "cache_hits": sum(1 for event in ocr_events if event.cache_status == "hit"),
                "cache_misses": sum(1 for event in ocr_events if event.cache_status == "miss"),
                "cache_refreshes": sum(1 for event in ocr_events if event.cache_status == "refresh"),
                "cache_disabled": sum(1 for event in ocr_events if event.cache_status == "disabled"),
            },
            "asr_provider": "vibevoice",
            "asr_strategy": "balanced",
            "timings": timings,
            "audio_language": transcript.language,
            "transcription_successful": bool(transcript.text),
            "transcript_markdown": str(args.run_dir / "transcript.md"),
            "vl_context": {
                "before": args.vl_context_before,
                "after": args.vl_context_after,
                "max_gap_seconds": "auto",
                "resolved_max_gap_seconds": resolve_vl_context_gap_seconds(frames, AUTO),
            },
        },
        "transcript": {"text": transcript.text, "segments": transcript.segments},
        "asr": {"strategy": "resume_from_transcript_md", "providers_run": ["vibevoice"]},
        "ocr_events": [event.to_dict() for event in ocr_events],
        "visual_events": frame_analyses,
        "manual_steps": operation_manual,
        "uncertainties": [event.to_dict() for event in ocr_events if event.status != "ok"],
        "frame_analyses": frame_analyses,
        "video_description": None,
        "operation_manual": operation_manual,
    }
    orin_dir = write_orin_artifacts(args.run_dir, results, page_context)
    results["metadata"]["orin_dir"] = str(orin_dir)
    (args.run_dir / "analysis.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(operation_manual["manual_path"]).write_text(operation_manual.get("response", ""), encoding="utf-8")
    LOGGER.info("Analysis complete. Results saved to %s", args.run_dir / "analysis.json")
    return 0 if operation_manual["quality_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
