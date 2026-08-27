#!/usr/bin/env python3
"""Run WeSpeaker diarization and emit normalized JSON turns."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model", default="chinese")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speaker-num", type=int)
    args = parser.parse_args()

    import wespeaker

    started = time.perf_counter()
    model = wespeaker.load_model(args.model)
    model.set_device(args.device)
    raw_turns = model.diarize(str(args.audio_path), "audio")
    turns = [
        {
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "speaker": f"speaker-{int(label) + 1}",
        }
        for _utt, start, end, label in raw_turns
        if float(end) > float(start)
    ]
    payload = {
        "provider": "wespeaker",
        "model": args.model,
        "device": args.device,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "turns": turns,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
