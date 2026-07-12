#!/usr/bin/env python3
"""Run the standalone 3D-Speaker backend and emit speaker turns as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "third_party" / "3d-speaker"))

    from diarization.backend_3dspeaker import ThirdPartyDiarizer
    from diarization.models import RuntimeProbe

    runtime = RuntimeProbe(
        accel="gpu" if args.device != "cpu" else "cpu",
        requested_device=args.device,
        resolved_device=args.device,
    )
    turns = ThirdPartyDiarizer(runtime, speaker_num=args.speaker_num).diarize(args.audio_path.resolve())
    print(
        json.dumps(
            {
                "turns": [
                    {
                        "speaker": turn.speaker_id,
                        "start": turn.start_sec,
                        "end": turn.end_sec,
                        "duration": turn.duration_sec,
                    }
                    for turn in turns
                ]
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
