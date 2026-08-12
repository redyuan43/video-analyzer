#!/usr/bin/env python3
"""Benchmark CUDA EasyOCR on one or more Jetson NX2 images."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import easyocr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-storage-directory", type=Path, required=True)
    parser.add_argument("--languages", default="ch_sim,en")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw_value, *_ = line.replace(":", "").split()
        if key in {"MemAvailable", "SwapFree", "SwapTotal"}:
            values[key] = int(raw_value) * 1024
    return values


def serialize_items(result: list[tuple[Any, str, float]]) -> list[dict[str, Any]]:
    return [
        {
            "box": [[round(float(value), 2) for value in point] for point in box],
            "text": text,
            "confidence": round(float(confidence), 4),
        }
        for box, text, confidence in result
    ]


def main() -> int:
    args = parse_args()
    images = [path.resolve() for path in args.image]
    output = args.output.resolve()
    model_storage_directory = args.model_storage_directory.resolve()
    if missing := [str(path) for path in images if not path.is_file()]:
        raise SystemExit(f"missing images: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    model_storage_directory.mkdir(parents=True, exist_ok=True)
    before = meminfo()
    load_started = time.perf_counter()
    reader = easyocr.Reader(
        args.languages.split(","),
        gpu=args.gpu,
        model_storage_directory=str(model_storage_directory),
        verbose=False,
    )
    loaded_at = time.perf_counter()

    results = []
    for image in images:
        started = time.perf_counter()
        raw_items = reader.readtext(str(image), detail=1, paragraph=False)
        completed = time.perf_counter()
        items = serialize_items(raw_items)
        results.append(
            {
                "image": str(image),
                "bytes": image.stat().st_size,
                "inference_seconds": round(completed - started, 3),
                "items": items,
                "text": "\n".join(item["text"] for item in items),
            }
        )

    after = meminfo()
    total_inference = sum(item["inference_seconds"] for item in results)
    payload = {
        "model": {
            "engine": "easyocr",
            "languages": args.languages.split(","),
            "gpu": args.gpu,
            "model_storage_directory": str(model_storage_directory),
        },
        "timing": {
            "load_seconds": round(loaded_at - load_started, 3),
            "total_inference_seconds": round(total_inference, 3),
            "frames_per_second": round(len(results) / total_inference, 3) if total_inference else None,
            "seconds_per_frame": round(total_inference / len(results), 3) if results else None,
        },
        "memory": {
            "before": before,
            "after": after,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "swap_used_delta_bytes": before.get("SwapFree", 0) - after.get("SwapFree", 0),
        },
        "results": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
