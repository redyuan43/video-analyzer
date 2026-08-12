#!/usr/bin/env python3
"""Benchmark 3D-Speaker or pyannote community-1 on one audio recording."""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_MARKER = "__3DSPEAKER_JSON__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--provider", choices=("3dspeaker", "pyannote_community"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-speaker-count", type=int)
    parser.add_argument("--speaker-num", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--three-d-speaker-root", type=Path, default=Path("/tmp/3D-Speaker"))
    parser.add_argument(
        "--three-d-speaker-python",
        default="/home/ai/diarization-ab-venv/bin/python",
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    return parser.parse_args()


def normalize_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for turn in turns:
        try:
            start = float(turn.get("start", turn.get("start_sec", 0)))
            end = float(turn.get("end", turn.get("end_sec", 0)))
        except (TypeError, ValueError):
            continue
        speaker = str(turn.get("speaker") or turn.get("speaker_id") or "").strip()
        if speaker and end > start:
            normalized.append({"start": round(start, 3), "end": round(end, 3), "speaker": speaker})
    return sorted(normalized, key=lambda turn: (turn["start"], turn["end"], turn["speaker"]))


def run_3d_speaker(args: argparse.Namespace) -> list[dict[str, Any]]:
    command = [
        str(args.three_d_speaker_python),
        str(REPO_ROOT / "tools" / "run_3dspeaker_turns.py"),
        str(args.audio_path),
        "--project-root",
        str(args.three_d_speaker_root),
    ]
    if args.speaker_num:
        command.extend(["--speaker-num", str(args.speaker_num)])
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=7200)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "3D-Speaker failed")[-3000:])
    output = completed.stdout.strip()
    if JSON_MARKER in output:
        output = output.rsplit(JSON_MARKER, 1)[1].strip()
    payload = json.loads(output)
    return normalize_turns(payload.get("turns") if isinstance(payload, dict) else [])


def load_pyannote_pipeline(token: str):
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token() or ""
        except Exception:
            token = ""
    if not token:
        raise RuntimeError(
            "pyannote community-1 requires an HF token. "
            "Accept the model terms, then run 'hf auth login' or set HF_TOKEN."
        )
    # pyannote community-1 checkpoints contain trusted pyannote config objects.
    # PyTorch 2.6+ changed torch.load to weights_only=True by default, while
    # pyannote.audio 4.0.2 still relies on the previous behavior.
    import torch

    original_torch_load = torch.load
    if not getattr(torch.load, "_video_analyzer_pyannote_compat", False):
        @functools.wraps(original_torch_load)
        def load_trusted_pyannote_checkpoint(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        load_trusted_pyannote_checkpoint._video_analyzer_pyannote_compat = True
        torch.load = load_trusted_pyannote_checkpoint
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=token,
        )
    if pipeline is None:
        raise RuntimeError(
            "pyannote community-1 access is not enabled for this account. "
            "Open the model page while signed in and accept its user conditions."
        )
    return pipeline


def run_pyannote_community(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    pipeline = load_pyannote_pipeline(args.hf_token)
    device_used = "cpu"
    if args.device != "cpu":
        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
                device_used = "cuda"
            elif args.device == "cuda":
                raise RuntimeError("CUDA was requested but is unavailable")
        except Exception:
            if args.device == "cuda":
                raise
    options = {"num_speakers": args.speaker_num} if args.speaker_num else {}
    result = pipeline(str(args.audio_path), **options)
    diarization = getattr(result, "exclusive_speaker_diarization", result)
    turns = [
        {"start": segment.start, "end": segment.end, "speaker": speaker}
        for segment, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return normalize_turns(turns), device_used


def build_metrics(turns: list[dict[str, Any]], expected_speaker_count: int | None) -> dict[str, Any]:
    speaker_seconds: dict[str, float] = defaultdict(float)
    for turn in turns:
        speaker_seconds[turn["speaker"]] += turn["end"] - turn["start"]
    speakers = sorted(speaker_seconds)
    metrics: dict[str, Any] = {
        "turn_count": len(turns),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "speech_seconds": round(sum(speaker_seconds.values()), 3),
        "speaker_seconds": {speaker: round(seconds, 3) for speaker, seconds in speaker_seconds.items()},
    }
    if expected_speaker_count is not None:
        metrics["expected_speaker_count"] = expected_speaker_count
        metrics["speaker_count_match"] = len(speakers) == expected_speaker_count
    return metrics


def main() -> int:
    args = parse_args()
    args.audio_path = args.audio_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.audio_path.is_file():
        raise FileNotFoundError(f"audio does not exist: {args.audio_path}")
    started = time.perf_counter()
    if args.provider == "3dspeaker":
        turns = run_3d_speaker(args)
        device_used = "cuda"
    else:
        turns, device_used = run_pyannote_community(args)
    payload = {
        "provider": args.provider,
        "audio_path": str(args.audio_path),
        "device": device_used,
        "speaker_num_constraint": args.speaker_num,
        "metrics": build_metrics(turns, args.expected_speaker_count),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "turns": turns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["metrics"], "elapsed_seconds": payload["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
