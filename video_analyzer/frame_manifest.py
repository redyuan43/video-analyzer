from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .frame import Frame

MANIFEST_NAME = "frames_manifest.json"


def write_frame_manifest(frames: list[Frame], output_dir: Path, source: str) -> Path:
    payload = {
        "version": 1,
        "source": source,
        "frames": [
            {
                "frame_number": frame.number,
                "path": _relative_path(frame.path, output_dir),
                "timestamp": frame.timestamp,
                "score": frame.score,
            }
            for frame in frames
        ],
    }
    path = output_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_frame_manifest(path: Path) -> tuple[dict[int, float], dict[int, float], dict[str, Any]]:
    if not path.exists():
        return {}, {}, {"source": "missing", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamp_map: dict[int, float] = {}
    score_map: dict[int, float] = {}
    for item in payload.get("frames") or []:
        if not isinstance(item, dict):
            continue
        frame_number = item.get("frame_number", item.get("number"))
        timestamp = item.get("timestamp")
        if frame_number is None or timestamp is None:
            continue
        number = int(frame_number)
        timestamp_map[number] = float(timestamp)
        if item.get("score") is not None:
            score_map[number] = float(item["score"])
    return (
        timestamp_map,
        score_map,
        {
            "source": "frames_manifest",
            "path": str(path),
            "manifest_source": payload.get("source", ""),
        },
    )


def _relative_path(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)
