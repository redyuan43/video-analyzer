"""video-analyzer CLI 子流水线共享 helper：参数解析、checkpoint、签名与 client 构建。

由 video_analyzer.cli 导入使用；对外 API 保持不变。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

from .analysis_progress import write_analysis_progress
from .analyzer import VideoAnalyzer
from .artifacts import write_transcript_markdown
from .audio_processor import AudioTranscript
from .clients.generic_openai_api import GenericOpenAIAPIClient
from .clients.ollama import OllamaClient
from .config import Config, build_openai_extra_body, get_client, resolve_api_key
from .frame_selection import (
    FrameDecision,
    build_frame_context_window,
    make_skipped_visual_event,
    parse_auto_float,
    parse_auto_int,
)
from .frame_manifest import MANIFEST_NAME, read_frames_from_manifest
from .ocr import OCREvent
from .vl_checkpoint import analysis_signature, frame_sha256, write_vl_checkpoint

logger = logging.getLogger(__name__)
TRANSCRIPT_LINE_RE = re.compile(
    r"^-\s+\[(?P<start>\d\d:\d\d:\d\d)\s+-\s+(?P<end>\d\d:\d\d:\d\d)\]\s+(?P<text>.*)$"
)
DEFAULT_VL_SECONDS_PER_FRAME = 30.0
DEFAULT_VL_TARGET_SECONDS = 45 * 60


def media_has_video_stream(media_path: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=codec_type:stream_disposition=attached_pic",
                "-of",
                "json",
                str(media_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logger.warning("Could not probe video stream for %s; assuming video input: %s", media_path, exc)
        return True
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return bool(result.stdout.strip())
    for stream in payload.get("streams") or []:
        disposition = stream.get("disposition") or {}
        if stream.get("codec_type") == "video" and not disposition.get("attached_pic"):
            return True
    return False

def get_log_level(level_str: str) -> int:
    """Convert string log level to logging constant."""
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)


def seconds_from_timestamp(value: str) -> float:
    hours, minutes, seconds = [int(part) for part in value.split(":")]
    return float(hours * 3600 + minutes * 60 + seconds)


def read_transcript_markdown(path: Path) -> AudioTranscript:
    text = path.read_text(encoding="utf-8")
    language = ""
    segments = []
    full_text = []
    for line in text.splitlines():
        if line.startswith("- Language:"):
            language = line.split(":", 1)[1].strip()
            continue
        match = TRANSCRIPT_LINE_RE.match(line)
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
            if line.strip()
            and not line.startswith("#")
            and not line.startswith("- Language:")
            and not line.startswith("- Segments:")
        ]
    return AudioTranscript(text="\n".join(full_text).strip(), segments=segments, language=language)


def parse_jetson_frame_weights(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    weights: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid Jetson frame weight entry: {item}")
        host, weight = item.split("=", 1)
        weights[host.strip()] = max(float(weight), 0.1)
    return weights

def cleanup_files(output_dir: Path):
    """Clean up temporary files and directories."""
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.debug(f"Cleaned up frames directory: {frames_dir}")
            
        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.debug(f"Cleaned up audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def parse_auto_int_arg(value: str) -> int | str:
    try:
        parsed = parse_auto_int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed is None:
        raise argparse.ArgumentTypeError("value must be auto or a non-negative integer")
    return parsed


def parse_auto_float_arg(value: str) -> float | str:
    try:
        parsed = parse_auto_float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed is None:
        raise argparse.ArgumentTypeError("value must be auto or a non-negative number")
    return parsed


def append_evidence_boundary_section(
    markdown: str,
    frame_selection_metadata: dict,
    ocr_keyframe_metadata: dict,
    ocr_events: list,
) -> str:
    if not markdown or "## 证据边界与需复核" in markdown:
        return markdown
    notes: list[str] = []
    vl_policy = str(frame_selection_metadata.get("vl_frame_policy_resolved") or "").lower()
    vl_frame_count = frame_selection_metadata.get("vl_frames_processed")
    if vl_frame_count is None:
        vl_frame_count = frame_selection_metadata.get("vl_frames_count")
    if vl_frame_count is None:
        vl_frame_count = len(frame_selection_metadata.get("frames") or frame_selection_metadata.get("selected_vl_frames") or [])
    if vl_policy == "none" or int(vl_frame_count or 0) == 0:
        notes.append("本次未运行或未选中 VL 视觉理解帧，界面细节主要依赖 OCR、ASR 与截图证据，关键操作建议人工复核。")
    if ocr_keyframe_metadata.get("ocr_text_events_count") == 0:
        notes.append("OCR 没有形成稳定文本事件，涉及按钮文案、菜单项和页面状态的结论需结合截图复核。")
    failed_ocr = [
        event for event in ocr_events
        if getattr(event, "status", "ok") not in {"ok", "skipped"}
    ]
    if failed_ocr:
        notes.append(f"有 {len(failed_ocr)} 个 OCR 帧未成功解析，对应时间点的文字证据置信度较低。")
    if not notes:
        return markdown
    section = "\n\n## 证据边界与需复核\n\n" + "\n".join(f"- {note}" for note in notes)
    return markdown.rstrip() + section + "\n"


def recent_vl_seconds_per_frame(output_dir: Path, model: str, limit: int = 5) -> float:
    samples: list[float] = []
    try:
        runs_root = output_dir.parents[1]
        candidates = sorted(
            runs_root.glob("*/*/analysis.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        candidates = []
    for path in candidates:
        if path == output_dir / "analysis.json":
            continue
        try:
            metadata = (json.loads(path.read_text(encoding="utf-8")) or {}).get("metadata") or {}
            if str(metadata.get("model") or "") != str(model):
                continue
            timings = metadata.get("timings") or {}
            seconds = float(timings.get("vl_seconds") or 0.0)
            frames = int(metadata.get("vl_frames_processed") or 0)
        except Exception:
            continue
        if seconds > 0 and frames > 0:
            samples.append(seconds / frames)
        if len(samples) >= limit:
            break
    return float(median(samples)) if samples else DEFAULT_VL_SECONDS_PER_FRAME


def vl_signature_payload(
    analyzer: VideoAnalyzer,
    *,
    model: str,
    context_before: int,
    context_after: int,
    context_max_gap: float | str,
) -> dict[str, Any]:
    return {
        "model": model,
        "frame_prompt": analyzer.frame_prompt,
        "user_prompt": analyzer.user_prompt,
        "temperature": analyzer.temperature,
        "frame_num_predict": analyzer.frame_num_predict,
        "frame_no_think": analyzer.frame_no_think,
        "context_before": context_before,
        "context_after": context_after,
        "context_max_gap": context_max_gap,
    }


def ocr_signature_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": config.get("provider", "auto"),
        "base_url": config.get("base_url", "auto"),
        "base_urls": config.get("base_urls") or [],
        "model": config.get("model", "model"),
        "prompt_mode": config.get("prompt_mode", "prompt_scene_spotting"),
        "max_tokens": config.get("max_tokens", 1024),
        "max_image_long_side": config.get("max_image_long_side", 1280),
    }


def load_ocr_checkpoint(path: Path, frames, expected_signature: str) -> dict[int, OCREvent]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict) or payload.get("analysis_signature") != expected_signature:
        return {}
    frames_by_number = {int(frame.number): frame for frame in frames}
    loaded: dict[int, OCREvent] = {}
    for item in payload.get("frames") or []:
        if not isinstance(item, dict) or item.get("status") != "succeeded":
            continue
        frame_number = item.get("frame_number")
        frame = frames_by_number.get(int(frame_number)) if frame_number is not None else None
        if frame is None or item.get("frame_sha256") != frame_sha256(Path(frame.path)):
            continue
        event_payload = item.get("event")
        if isinstance(event_payload, dict):
            loaded[int(frame_number)] = OCREvent.from_dict(event_payload)
    return loaded


def write_ocr_checkpoint(
    path: Path,
    events_by_frame: dict[int, OCREvent],
    frames_by_number: dict[int, Any],
    signature: str,
    signature_payload: dict[str, Any],
) -> None:
    entries = []
    for frame_number, event in sorted(events_by_frame.items()):
        frame = frames_by_number.get(frame_number)
        if frame is None:
            continue
        entries.append(
            {
                "frame_number": frame_number,
                "frame_sha256": frame_sha256(Path(frame.path)),
                "status": "succeeded" if event.status == "ok" else "failed",
                "event": event.to_dict(),
            }
        )
    payload = {
        "version": 1,
        "analysis_signature": signature,
        "signature_payload": signature_payload,
        "frames": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def analyze_frames_for_vl(
    analyzer: VideoAnalyzer,
    frames,
    ocr_events,
    selected_frame_numbers: set[int],
    decisions: list[FrameDecision],
    concurrency: int,
    context_before: int,
    context_after: int,
    context_max_gap: float | str,
    checkpoint_path: Path | None = None,
    checkpoint_by_frame: dict[int, dict] | None = None,
    checkpoint_signature: str = "",
    checkpoint_signature_payload: dict[str, Any] | None = None,
    progress_callback=None,
):
    ocr_by_frame = {event.frame_number: event for event in ocr_events}
    context_ocr_texts = {event.frame_number: event.text for event in ocr_events if event.text}
    decisions_by_frame = {decision.frame_number: decision for decision in decisions}
    checkpoint_by_frame = dict(checkpoint_by_frame or {})
    frame_analyses = [checkpoint_by_frame.get(frame.number) for frame in frames]

    def save_checkpoint() -> None:
        if checkpoint_path is None or not checkpoint_signature:
            return
        write_vl_checkpoint(
            checkpoint_path,
            [item for item in frame_analyses if item is not None],
            signature=checkpoint_signature,
            signature_payload=checkpoint_signature_payload or {},
        )

    selected_total = len(selected_frame_numbers)
    reused = sum(
        1
        for frame in frames
        if frame.number in selected_frame_numbers and frame.number in checkpoint_by_frame
    )
    completed = reused
    succeeded = reused
    failed = 0
    analyzed_durations: list[float] = [
        float(item.get("duration_seconds"))
        for item in checkpoint_by_frame.values()
        if isinstance(item.get("duration_seconds"), (int, float))
        and float(item.get("duration_seconds")) > 0
    ]
    progress_started = time.perf_counter()

    def report_progress(current_frame_number: int | None = None) -> None:
        if progress_callback is None:
            return
        average_seconds = float(median(analyzed_durations)) if analyzed_durations else 0.0
        remaining = max(selected_total - completed, 0)
        progress_callback(
            {
                "total_selected": selected_total,
                "completed": completed,
                "succeeded": succeeded,
                "failed": failed,
                "reused": reused,
                "remaining": remaining,
                "current_frame_number": current_frame_number,
                "elapsed_seconds": round(time.perf_counter() - progress_started, 3),
                "average_frame_seconds": round(average_seconds, 3),
                "eta_seconds": round(remaining * average_seconds, 3) if average_seconds else None,
            }
        )

    def analyze_one(index_frame):
        index, frame = index_frame
        ocr_text = ocr_by_frame.get(frame.number).text if frame.number in ocr_by_frame else ""
        context_window = build_frame_context_window(
            frames=frames,
            current_frame=frame,
            before=context_before,
            after=context_after,
            max_gap_seconds=context_max_gap,
        )
        started = time.perf_counter()
        analysis = analyzer.analyze_frame(
            frame,
            ocr_text=ocr_text,
            context_window=context_window,
            context_ocr_texts=context_ocr_texts,
        )
        duration_seconds = round(time.perf_counter() - started, 3)
        response = str(analysis.get("response") or "")
        status = "failed" if response.startswith("Error analyzing frame ") else "succeeded"
        analysis.update(
            {
                "frame_number": int(frame.number),
                "timestamp": float(frame.timestamp),
                "frame_sha256": frame_sha256(Path(frame.path)),
                "status": status,
                "duration_seconds": duration_seconds,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                "analysis_signature": checkpoint_signature,
            }
        )
        return index, analysis

    selected = [(index, frame) for index, frame in enumerate(frames) if frame.number in selected_frame_numbers]
    skipped = [(index, frame) for index, frame in enumerate(frames) if frame.number not in selected_frame_numbers]
    for index, frame in skipped:
        if frame_analyses[index] is None:
            frame_analyses[index] = make_skipped_visual_event(frame, decisions_by_frame[frame.number])

    selected = [(index, frame) for index, frame in selected if frame_analyses[index] is None]
    report_progress()

    if not selected:
        save_checkpoint()
        return frame_analyses

    if concurrency <= 1:
        for index_frame in selected:
            index, analysis = analyze_one(index_frame)
            frame_analyses[index] = analysis
            completed += 1
            if analysis.get("status") == "succeeded":
                succeeded += 1
            else:
                failed += 1
            analyzed_durations.append(float(analysis.get("duration_seconds") or 0.0))
            save_checkpoint()
            report_progress(int(analysis["frame_number"]))
        return frame_analyses

    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as executor:
        futures = [executor.submit(analyze_one, item) for item in selected]
        for future in as_completed(futures):
            index, analysis = future.result()
            frame_analyses[index] = analysis
            completed += 1
            if analysis.get("status") == "succeeded":
                succeeded += 1
            else:
                failed += 1
            analyzed_durations.append(float(analysis.get("duration_seconds") or 0.0))
            save_checkpoint()
            report_progress(int(analysis["frame_number"]))
    return frame_analyses


def load_frame_analysis_checkpoint(path: Path | None) -> dict[int, dict]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    loaded = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        frame_number = item.get("frame_number", item.get("number"))
        if frame_number is None:
            continue
        loaded[int(frame_number)] = item
    return loaded


def write_frame_analysis_checkpoint(path: Path, analyses: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(analyses, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def scan_frames_count_from_metadata(metadata: dict) -> int | None:
    if not metadata:
        return None
    per_host = metadata.get("per_host") or []
    preview_total = sum(int(item.get("preview_frames") or 0) for item in per_host if isinstance(item, dict))
    if preview_total:
        return preview_total
    for key in ("scan_frames_count", "preview_frames", "raw_candidate_frames"):
        value = metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def frame_manifest_matches_request(metadata: dict, frame_extractor: str) -> bool:
    manifest_source = str(metadata.get("manifest_source") or "").strip().lower()
    requested = str(frame_extractor or "local").strip().lower()
    if requested == "jetson":
        return manifest_source == "jetson"
    if requested == "local":
        return manifest_source in {"local", "local_keyframes", "audio_only"}
    if requested == "local_gpu":
        return manifest_source == "local_gpu"
    return bool(manifest_source)


def reusable_frames_from_manifest(output_dir: Path, frame_extractor: str):
    frames, metadata = read_frames_from_manifest(output_dir / MANIFEST_NAME, output_dir)
    if not frames:
        return [], metadata
    if not frame_manifest_matches_request(metadata, frame_extractor):
        metadata["reuse_rejected_reason"] = (
            f"manifest source {metadata.get('manifest_source')!r} does not match requested extractor {frame_extractor!r}"
        )
        return [], metadata
    metadata = dict(metadata)
    metadata["backend"] = f"reused_{metadata.get('manifest_source') or 'frames_manifest'}"
    metadata["resumed_existing"] = True
    return frames, metadata

def create_client(config: Config):
    """Create the appropriate client based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = get_client(config)
    
    if client_type == "ollama":
        return OllamaClient(client_config["url"])
    elif client_type == "openai_api":
        return GenericOpenAIAPIClient(
            client_config["api_key"],
            client_config["api_url"],
            timeout_seconds=int(client_config.get("timeout_seconds", 600)),
            extra_body=build_openai_extra_body(config.get("clients", {}).get("openai_api", {}), client_config["api_url"]),
        )
    else:
        raise ValueError(f"Unknown client type: {client_type}")


def create_operation_manual_text_client(config: Config, fallback_client):
    """Create the text-generation client for operation manuals.

    Operation-manual runs can route visual frame analysis to one endpoint and
    final Markdown generation to another. Non-OpenAI clients keep the legacy
    single-client behavior.
    """
    if config.get("clients", {}).get("default") != "openai_api":
        return fallback_client

    manual_config = config.get("operation_manual", {})
    openai_config = config.get("clients", {}).get("openai_api", {})
    text_base_url = (
        manual_config.get("text_base_url")
        or manual_config.get("llm_base_url")
        or openai_config.get("api_url")
    )
    if not text_base_url:
        return fallback_client
    return GenericOpenAIAPIClient(
        resolve_api_key(
            openai_config.get("api_key"),
            manual_config.get("text_api_key_env") or openai_config.get("api_key_env"),
            text_base_url,
        ),
        text_base_url,
        timeout_seconds=int(manual_config.get("text_timeout_seconds") or openai_config.get("timeout_seconds", 600)),
        extra_body=build_openai_extra_body(manual_config, text_base_url),
    )


def create_operation_manual_fallback_client(
    config: Config,
) -> tuple[GenericOpenAIAPIClient | None, str, float | None]:
    manual_config = config.get("operation_manual", {})
    if not manual_config.get("text_fallback_enabled"):
        return None, "", None
    base_url = str(manual_config.get("text_fallback_base_url") or "").strip()
    model = str(manual_config.get("text_fallback_model") or "").strip()
    if not base_url or not model:
        return None, "", None
    fallback_settings = {
        "deepseek_thinking": manual_config.get(
            "text_fallback_deepseek_thinking"
        ),
        "reasoning_effort": manual_config.get(
            "text_fallback_reasoning_effort"
        ),
    }
    client = GenericOpenAIAPIClient(
        resolve_api_key(
            None,
            manual_config.get("text_fallback_api_key_env"),
            base_url,
        ),
        base_url,
        timeout_seconds=int(
            manual_config.get("text_fallback_text_timeout_seconds") or 900
        ),
        extra_body=build_openai_extra_body(fallback_settings, base_url),
    )
    temperature = manual_config.get("text_fallback_text_temperature")
    return client, model, float(temperature) if temperature is not None else None


def read_page_context_metadata(context_file: str, page_context: str) -> dict:
    metadata = {
        "context_file": context_file,
        "text_length": len(page_context or ""),
    }
    if not context_file:
        return metadata
    sidecar = Path(context_file).with_name("page_context.json")
    if not sidecar.exists():
        return metadata
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        metadata["diagnostics"] = [f"failed to read page_context.json: {exc}"]
        return metadata
    metadata.update(payload)
    metadata["text_length"] = len(page_context or "")
    return metadata


