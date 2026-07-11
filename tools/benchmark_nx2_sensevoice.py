#!/usr/bin/env python3
"""Benchmark SenseVoice/FunASR on a Jetson NX2 audio sample."""

from __future__ import annotations

import argparse
import json
import re
import resource
import subprocess
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


MODEL_NAME = "iic/SenseVoiceSmall"
VAD_MODEL = "fsmn-vad"
PUNC_MODEL = "ct-punc-c"
SPECIAL_TOKEN = re.compile(r"<\|[^|>]+\|>")
MEANINGFUL_CHAR = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-transcript", type=Path)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--vad-model", default=VAD_MODEL)
    parser.add_argument("--punc-model", default=PUNC_MODEL)
    parser.add_argument("--hotword", default="")
    parser.add_argument("--batch-size-seconds", type=int, default=60)
    parser.add_argument("--merge-length-seconds", type=int, default=15)
    parser.add_argument(
        "--remote-code",
        type=Path,
        help="SenseVoice model.py used for native token timestamps.",
    )
    parser.add_argument(
        "--output-timestamp",
        action="store_true",
        help="Request native token-level timestamps from compatible SenseVoice remote code.",
    )
    return parser.parse_args()


def audio_duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as source:
            return source.getnframes() / source.getframerate()

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw_value, *_ = line.replace(":", "").split()
        if key in {"MemAvailable", "SwapFree", "SwapTotal"}:
            values[key] = int(raw_value) * 1024
    return values


def clean_text(value: str) -> str:
    value = SPECIAL_TOKEN.sub("", value)
    return rich_transcription_postprocess(value).replace("🎼", "").replace("😊", "").strip()


def comparable_text(value: str) -> str:
    return "".join(MEANINGFUL_CHAR.findall(value)).lower()


def reference_text(path: Path | None) -> str:
    if path is None:
        return ""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("text") or "")
    return path.read_text(encoding="utf-8")


def main() -> int:
    args = parse_args()
    audio = args.audio.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not audio.is_file():
        raise SystemExit(f"missing audio: {audio}")

    before = meminfo()
    load_started = time.perf_counter()
    model_kwargs: dict[str, Any] = {
        "model": args.model,
        "vad_model": args.vad_model,
        "vad_kwargs": {"max_single_segment_time": 30000},
        "punc_model": args.punc_model,
        "device": "cuda:0",
        "disable_update": True,
    }
    if args.remote_code:
        remote_code = args.remote_code.resolve()
        if not remote_code.is_file():
            raise SystemExit(f"missing remote code: {remote_code}")
        model_kwargs.update(
            trust_remote_code=True,
            remote_code=str(remote_code),
        )
    model = AutoModel(**model_kwargs)
    loaded_at = time.perf_counter()
    raw_result = model.generate(
        input=str(audio),
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=args.batch_size_seconds,
        merge_vad=True,
        merge_length_s=args.merge_length_seconds,
        hotword=args.hotword or None,
        output_timestamp=args.output_timestamp,
    )
    completed_at = time.perf_counter()
    after = meminfo()

    raw_text = "\n".join(str(item.get("text") or "") for item in raw_result)
    text = clean_text(raw_text)
    reference = reference_text(args.reference_transcript)
    comparable_result = comparable_text(text)
    comparable_reference = comparable_text(reference)
    similarity = (
        SequenceMatcher(None, comparable_result, comparable_reference).ratio()
        if comparable_reference
        else None
    )
    duration = audio_duration_seconds(audio)
    inference_seconds = completed_at - loaded_at
    payload: dict[str, Any] = {
        "model": {
            "asr": args.model,
            "vad": args.vad_model,
            "punc": args.punc_model,
            "hotword": args.hotword or None,
            "remote_code": str(args.remote_code.resolve()) if args.remote_code else None,
            "output_timestamp": args.output_timestamp,
        },
        "audio": {
            "path": str(audio),
            "duration_seconds": round(duration, 3),
        },
        "timing": {
            "load_seconds": round(loaded_at - load_started, 3),
            "inference_seconds": round(inference_seconds, 3),
            "rtf": round(inference_seconds / duration, 4),
            "realtime_multiplier": round(duration / inference_seconds, 3),
        },
        "memory": {
            "before": before,
            "after": after,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "swap_used_delta_bytes": (before.get("SwapFree", 0) - after.get("SwapFree", 0)),
        },
        "quality": {
            "reference_path": str(args.reference_transcript.resolve()) if args.reference_transcript else None,
            "reference_character_count": len(comparable_reference),
            "result_character_count": len(comparable_result),
            "reference_character_similarity": round(similarity, 4) if similarity is not None else None,
            "timestamp_token_count": sum(
                len(item.get("timestamp") or []) for item in raw_result
            ),
        },
        "text": text,
        "raw_result": raw_result,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
