#!/usr/bin/env python3
"""Regenerate only the final operation manual from an existing analysis.json."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analyzer.audio_processor import AudioTranscript
from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.frame import Frame
from video_analyzer.manual import (
    DEFAULT_MAX_FRAME_EVIDENCE_CHARS,
    embed_step_images,
    generate_operation_manual,
    prepare_frame_assets,
    read_context_file,
    review_operation_manual_markdown,
    write_frame_evidence_index,
)
from video_analyzer.ocr import OCREvent

LOGGER = logging.getLogger(__name__)


def bypass_proxy_environment() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def load_frames(run_dir: Path, ocr_events: list[OCREvent]) -> list[Frame]:
    frame_paths = sorted(
        (run_dir / "frames").glob("frame_*.jpg"),
        key=lambda path: int(path.stem.split("_", 1)[1]),
    )
    timestamp_by_frame = {event.frame_number: event.timestamp for event in ocr_events}
    return [
        Frame(
            number=index,
            path=path,
            timestamp=timestamp_by_frame.get(index, 0.0),
            score=0.0,
        )
        for index, path in enumerate(frame_paths)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("--context-file", type=Path)
    parser.add_argument("--text-base-url", default="http://100.90.114.26:18081/v1")
    parser.add_argument("--text-model", default="hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive")
    parser.add_argument("--manual-language", default="zh-CN")
    parser.add_argument("--max-frame-evidence-chars", type=int, default=DEFAULT_MAX_FRAME_EVIDENCE_CHARS)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    bypass_proxy_environment()

    started = time.perf_counter()
    analysis_path = args.analysis_json.resolve()
    run_dir = analysis_path.parent
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    ocr_events = [OCREvent.from_dict(payload) for payload in analysis.get("ocr_events", [])]
    frames = load_frames(run_dir, ocr_events)
    frame_analyses = analysis.get("visual_events") or analysis.get("frame_analyses") or []
    transcript_payload = analysis.get("transcript") or {}
    transcript = AudioTranscript(
        text=str(transcript_payload.get("text") or ""),
        segments=list(transcript_payload.get("segments") or []),
        language=str(analysis.get("metadata", {}).get("audio_language") or ""),
    )
    context_file = args.context_file or Path(analysis.get("metadata", {}).get("context_file") or "")
    page_context = read_context_file(str(context_file)) if str(context_file) else ""

    LOGGER.info("Regenerating manual from %s visual events and %s OCR events", len(frame_analyses), len(ocr_events))
    client = GenericOpenAIAPIClient("0", args.text_base_url, timeout_seconds=args.timeout_seconds)
    frame_assets = prepare_frame_assets(frames, run_dir)
    operation_manual = generate_operation_manual(
        client=client,
        text_model=args.text_model,
        frame_analyses=frame_analyses,
        frames=frames,
        transcript=transcript,
        asr_metadata=analysis.get("asr") or {},
        ocr_events=ocr_events,
        page_context=page_context,
        language=args.manual_language,
        temperature=0.2,
        frame_assets=frame_assets,
        no_think=True,
        max_frame_evidence_chars=args.max_frame_evidence_chars,
    )
    operation_manual["response"] = embed_step_images(operation_manual.get("response", ""), frames, frame_assets)
    operation_manual["quality_review"] = review_operation_manual_markdown(operation_manual.get("response", ""))
    operation_manual["quality_gate_passed"] = not any(
        issue.get("severity") == "error" for issue in operation_manual["quality_review"]
    )
    manual_filename = "operation_manual.md" if operation_manual["quality_gate_passed"] else "operation_manual.quality_failed.md"
    operation_manual["manual_path"] = str(run_dir / manual_filename)
    operation_manual["evidence_path"] = str(
        write_frame_evidence_index(frames, run_dir, ocr_events, frame_analyses, frame_assets)
    )
    timings = analysis.setdefault("metadata", {}).setdefault("timings", {})
    timings["manual_regeneration_seconds"] = round(time.perf_counter() - started, 3)
    analysis["manual_steps"] = operation_manual
    analysis["operation_manual"] = operation_manual
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(operation_manual["manual_path"]).write_text(operation_manual.get("response", ""), encoding="utf-8")
    LOGGER.info("Manual regenerated at %s", operation_manual["manual_path"])
    return 0 if operation_manual["quality_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
