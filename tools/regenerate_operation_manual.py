#!/usr/bin/env python3
"""Regenerate operation_manual.md from an existing operation-manual run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analyzer.audio_processor import AudioTranscript
from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature
from video_analyzer.frame import Frame
from video_analyzer.manual import (
    embed_step_images,
    generate_operation_manual,
    prepare_frame_assets,
    review_operation_manual_markdown,
    write_frame_evidence_index,
)
from video_analyzer.ocr import OCREvent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate operation_manual.md from analysis.json")
    parser.add_argument("run_dir", help="Existing operation-manual run directory")
    parser.add_argument("--config", default="config", help="Configuration directory")
    parser.add_argument("--profile", help="Runtime profile name")
    parser.add_argument("--llm-base-url", help="OpenAI-compatible LLM/VL base URL")
    parser.add_argument("--text-model", help="Text model for the final manual")
    parser.add_argument("--manual-language", default="zh-CN")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    analysis_path = run_dir / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing analysis.json: {analysis_path}")

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    config = Config(args.config)
    profile = config.get_runtime_profile(args.profile)
    manual_config = config.get("operation_manual", {})
    base_url = args.llm_base_url or profile.get("llm_base_url") or manual_config.get("llm_base_url")
    text_model = args.text_model or profile.get("text_model") or manual_config.get("text_model")
    if not base_url or not text_model:
        raise ValueError("LLM base URL and text model are required")

    transcript_payload = analysis.get("transcript") or {}
    transcript = None
    if transcript_payload.get("text"):
        transcript = AudioTranscript(
            text=transcript_payload.get("text") or "",
            segments=transcript_payload.get("segments") or [],
            language=(analysis.get("metadata") or {}).get("audio_language") or "unknown",
        )

    ocr_events = [
        OCREvent(
            frame_number=event.get("frame_number", index),
            timestamp=float(event.get("timestamp") or 0),
            provider=event.get("provider") or "",
            status=event.get("status") or "unknown",
            text=event.get("text") or "",
            items=event.get("items") or [],
            error=event.get("error"),
        )
        for index, event in enumerate(analysis.get("ocr_events") or [])
    ]

    visual_events = analysis.get("visual_events") or analysis.get("frame_analyses") or []
    frames = build_frames(run_dir, ocr_events, visual_events)
    frame_assets = prepare_frame_assets(frames, run_dir)

    client = GenericOpenAIAPIClient(
        resolve_api_key(
            api_key_env=profile.get("text_api_key_env") or profile.get("api_key_env"),
            api_url=base_url,
        ),
        base_url,
        timeout_seconds=args.timeout_seconds,
        extra_body=build_openai_extra_body(profile, base_url),
    )
    operation_manual = generate_operation_manual(
        client=client,
        text_model=text_model,
        frame_analyses=visual_events,
        frames=frames,
        transcript=transcript,
        asr_metadata=analysis.get("asr") or {},
        ocr_events=ocr_events,
        page_context=(analysis.get("metadata") or {}).get("page_description") or "",
        language=args.manual_language,
        temperature=resolve_temperature(profile, 0.0),
        frame_assets=frame_assets,
        no_think=True,
    )
    operation_manual["response"] = embed_step_images(
        operation_manual.get("response", ""),
        frames,
        frame_assets,
    )
    operation_manual["quality_review"] = review_operation_manual_markdown(operation_manual.get("response", ""))
    operation_manual["quality_gate_passed"] = not any(
        issue.get("severity") == "error" for issue in operation_manual["quality_review"]
    )
    manual_filename = "operation_manual.md" if operation_manual["quality_gate_passed"] else "operation_manual.quality_failed.md"
    manual_path = run_dir / manual_filename
    operation_manual["manual_path"] = str(manual_path)
    operation_manual["evidence_path"] = str(
        write_frame_evidence_index(
            frames=frames,
            output_dir=run_dir,
            ocr_events=ocr_events,
            frame_analyses=visual_events,
            frame_assets=frame_assets,
        )
    )

    manual_path.write_text(operation_manual.get("response", ""), encoding="utf-8")
    analysis["operation_manual"] = operation_manual
    analysis["manual_steps"] = operation_manual
    metadata = analysis.setdefault("metadata", {})
    metadata["text_model"] = text_model
    metadata["llm_base_url"] = base_url
    metadata["text_temperature"] = resolve_temperature(profile, 0.0)
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"manual: {manual_path}")
    print(f"quality_gate_passed: {operation_manual['quality_gate_passed']}")
    return 0 if operation_manual["quality_gate_passed"] else 2


def build_frames(run_dir: Path, ocr_events: list[OCREvent], visual_events: list[dict]) -> list[Frame]:
    frames: list[Frame] = []
    for index, _analysis in enumerate(visual_events):
        event = ocr_events[index] if index < len(ocr_events) else None
        frame_number = event.frame_number if event else index
        timestamp = event.timestamp if event else 0.0
        path = run_dir / "frames" / f"frame_{frame_number}.jpg"
        if not path.exists():
            path = run_dir / "manual_assets" / f"frame_{frame_number:03d}.jpg"
        frames.append(Frame(number=frame_number, path=path, timestamp=timestamp, score=0.0))
    return frames


if __name__ == "__main__":
    raise SystemExit(main())
