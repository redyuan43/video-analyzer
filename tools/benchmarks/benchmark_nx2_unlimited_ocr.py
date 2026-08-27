#!/usr/bin/env python3
"""Benchmark a local Unlimited-OCR model on Jetson NX2."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--image-mode", choices=("gundam", "base"), default="gundam")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=128)
    return parser.parse_args()


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw_value, *_ = line.replace(":", "").split()
        if key in {"MemAvailable", "SwapFree", "SwapTotal"}:
            values[key] = int(raw_value) * 1024
    return values


def patch_dtype_scatter(model_dir: Path) -> bool:
    path = model_dir / "modeling_unlimitedocr.py"
    target = "inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)"
    replacement = (
        "inputs_embeds[idx].masked_scatter_("
        "images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch.to(inputs_embeds.dtype)"
        ")"
    )
    text = path.read_text(encoding="utf-8")
    if replacement in text:
        return False
    if target not in text:
        raise RuntimeError(f"Unlimited-OCR dtype patch target was not found in {path}")
    path.write_text(text.replace(target, replacement), encoding="utf-8")
    return True


def read_output_text(output_dir: Path, result: Any) -> str:
    if isinstance(result, str) and result.strip():
        return result.strip()
    if isinstance(result, dict):
        for key in ("text", "content", "result", "markdown"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    candidates = sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".json"}
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return ""


def main() -> int:
    args = parse_args()
    image = args.image.resolve()
    model_dir = args.model.resolve()
    output = args.output.resolve()
    if not image.is_file():
        raise SystemExit(f"missing image: {image}")
    if not model_dir.is_dir():
        raise SystemExit(f"missing model: {model_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    before = meminfo()
    load_started = time.perf_counter()
    patched = patch_dtype_scatter(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=model_dtype,
    ).eval().cuda()
    loaded_at = time.perf_counter()

    artifacts_dir = output.parent / f"{output.stem}-artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    inference_started = time.perf_counter()
    result = model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=str(image),
        output_path=str(artifacts_dir),
        base_size=1024,
        image_size=640 if args.image_mode == "gundam" else 1024,
        crop_mode=args.image_mode == "gundam",
        max_length=args.max_length,
        no_repeat_ngram_size=args.ngram_size,
        ngram_window=args.ngram_window,
        save_results=True,
    )
    completed_at = time.perf_counter()
    after = meminfo()
    text = read_output_text(artifacts_dir, result)

    payload = {
        "model": {
            "path": str(model_dir),
            "dtype": args.dtype,
            "image_mode": args.image_mode,
            "dtype_scatter_patch_applied": patched,
        },
        "image": {"path": str(image), "bytes": image.stat().st_size},
        "timing": {
            "load_seconds": round(loaded_at - load_started, 3),
            "inference_seconds": round(completed_at - inference_started, 3),
        },
        "memory": {
            "before": before,
            "after": after,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "swap_used_delta_bytes": before.get("SwapFree", 0) - after.get("SwapFree", 0),
        },
        "result": {
            "character_count": len(text),
            "text": text,
            "artifacts_dir": str(artifacts_dir),
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
