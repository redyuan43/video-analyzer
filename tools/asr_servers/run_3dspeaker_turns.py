#!/usr/bin/env python3
"""Run native 3D-Speaker diarization and emit normalized speaker turns."""

from __future__ import annotations

import argparse
import contextlib
import json
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speaker-num", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    entrypoint = project_root / "speakerlab" / "bin" / "infer_diarization.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"3D-Speaker entrypoint not found: {entrypoint}")

    sys.path.insert(0, str(project_root))
    # ModelScope dependencies still access this NumPy 1.x compatibility alias.
    setattr(np, "NaN", np.nan)

    with tempfile.TemporaryDirectory(prefix="3dspeaker_turns_") as temp_dir:
        output_dir = Path(temp_dir) / "output"
        argv = [
            str(entrypoint),
            "--wav",
            str(args.audio_path.resolve()),
            "--out_dir",
            str(output_dir),
            "--out_type",
            "json",
            "--diable_progress_bar",
            "--nprocs",
            "1",
        ]
        if args.speaker_num:
            argv.extend(["--speaker_num", str(args.speaker_num)])

        original_argv = sys.argv
        try:
            sys.argv = argv
            # The upstream script prints model/download diagnostics to stdout.
            # Keep stdout machine-readable for the parent assignment process.
            with contextlib.redirect_stdout(sys.stderr):
                runpy.run_path(str(entrypoint), run_name="__main__")
        finally:
            sys.argv = original_argv

        outputs = sorted(output_dir.glob("*.json"))
        if not outputs:
            raise RuntimeError("3D-Speaker did not create a JSON diarization result")
        payload = json.loads(outputs[0].read_text(encoding="utf-8"))

    turns = []
    for value in payload.values():
        if not isinstance(value, dict):
            continue
        start = float(value.get("start", 0))
        end = float(value.get("stop", value.get("end", 0)))
        if end <= start:
            continue
        turns.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": str(value.get("speaker", "unknown")),
            }
        )
    turns.sort(key=lambda turn: (turn["start"], turn["end"]))
    print("__3DSPEAKER_JSON__" + json.dumps({"turns": turns}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
