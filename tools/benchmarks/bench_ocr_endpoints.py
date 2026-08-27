#!/usr/bin/env python3
"""Benchmark DotsMOCR endpoints on a fixed set of extracted frames."""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analyzer.frame import Frame
from video_analyzer.ocr import DotsMOCRVLLMProvider, OCREvent

LOGGER = logging.getLogger(__name__)


def bypass_proxy_environment() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def parse_frame_numbers(value: str) -> list[int]:
    numbers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        numbers.append(int(item))
    if not numbers:
        raise argparse.ArgumentTypeError("at least one frame number is required")
    return numbers


def load_frames(frames_dir: Path, frame_numbers: list[int]) -> list[Frame]:
    frames = []
    for number in frame_numbers:
        path = frames_dir / f"frame_{number}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Missing frame: {path}")
        frames.append(Frame(number=number, path=path, timestamp=float(number), score=0.0))
    return frames


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event_to_row(frame: Frame, endpoint: str, started_at: str, elapsed: float, event: OCREvent) -> dict[str, Any]:
    return {
        "frame_number": frame.number,
        "frame_path": str(frame.path),
        "endpoint": endpoint,
        "started_at": started_at,
        "elapsed_seconds": round(elapsed, 3),
        "status": event.status,
        "provider": event.provider,
        "text_chars": len(event.text or ""),
        "items_count": len(event.items or []),
        "error": event.error,
    }


def run_one(provider: DotsMOCRVLLMProvider, frame: Frame) -> dict[str, Any]:
    endpoint = provider.selected_base_url or provider.base_url
    started_at = iso_now()
    started = time.perf_counter()
    event = provider.analyze_frame(frame)
    elapsed = time.perf_counter() - started
    row = event_to_row(frame, endpoint, started_at, elapsed, event)
    LOGGER.info(
        "frame=%s endpoint=%s status=%s elapsed=%.3fs chars=%s error=%s",
        frame.number,
        endpoint,
        event.status,
        elapsed,
        len(event.text or ""),
        (event.error or "")[:160],
    )
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_endpoint: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_endpoint.setdefault(row["endpoint"], []).append(row)

    summary = {}
    for endpoint, endpoint_rows in by_endpoint.items():
        elapsed = [float(row["elapsed_seconds"]) for row in endpoint_rows]
        ok_rows = [row for row in endpoint_rows if row["status"] == "ok"]
        summary[endpoint] = {
            "total": len(endpoint_rows),
            "ok": len(ok_rows),
            "error": len(endpoint_rows) - len(ok_rows),
            "ok_rate": round(len(ok_rows) / len(endpoint_rows), 3) if endpoint_rows else 0.0,
            "elapsed_p50_seconds": round(statistics.median(elapsed), 3) if elapsed else None,
            "elapsed_max_seconds": round(max(elapsed), 3) if elapsed else None,
            "failed_frames": [row["frame_number"] for row in endpoint_rows if row["status"] != "ok"],
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frame-numbers", type=parse_frame_numbers, required=True)
    parser.add_argument("--ocr-base-url", action="append", required=True)
    parser.add_argument("--model", default="model")
    parser.add_argument("--prompt-mode", default="prompt_scene_spotting")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-image-long-side", type=int, default=1280)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--probe-timeout-seconds", type=float, default=5)
    parser.add_argument("--warmup-timeout-seconds", type=float, default=120)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    bypass_proxy_environment()

    frames = load_frames(args.frames_dir, args.frame_numbers)
    endpoints = [endpoint.rstrip("/") for endpoint in args.ocr_base_url]
    providers = [
        DotsMOCRVLLMProvider(
            base_url=endpoint,
            model=args.model,
            prompt_mode=args.prompt_mode,
            timeout=int(args.timeout_seconds),
            max_tokens=args.max_tokens,
            max_image_long_side=args.max_image_long_side,
            probe_timeout_seconds=args.probe_timeout_seconds,
            warmup_timeout_seconds=args.warmup_timeout_seconds,
        )
        for endpoint in endpoints
    ]

    healthy = []
    for provider in providers:
        endpoint = provider.probe()
        LOGGER.info("probe endpoint=%s selected=%s", provider.base_url, endpoint)
        if endpoint:
            healthy.append(provider)
    if not healthy:
        raise RuntimeError("No DotsMOCR endpoint is reachable")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    max_workers = max(1, min(len(frames), len(healthy) * max(args.concurrency, 1)))

    def indexed_run(item: tuple[int, Frame]) -> dict[str, Any]:
        index, frame = item
        provider = healthy[index % len(healthy)]
        return run_one(provider, frame)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(indexed_run, item) for item in enumerate(frames)]
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: (int(row["frame_number"]), str(row["endpoint"])))
    payload = {
        "started_at": iso_now(),
        "frames_dir": str(args.frames_dir),
        "frame_numbers": args.frame_numbers,
        "endpoints": endpoints,
        "timeout_seconds": args.timeout_seconds,
        "prompt_mode": args.prompt_mode,
        "max_tokens": args.max_tokens,
        "max_image_long_side": args.max_image_long_side,
        "concurrency": args.concurrency,
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "summary": payload["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
