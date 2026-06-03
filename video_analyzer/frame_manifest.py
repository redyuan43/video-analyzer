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


def read_frames_from_manifest(path: Path, output_dir: Path) -> tuple[list[Frame], dict[str, Any]]:
    if not path.exists():
        return [], {"source": "missing", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames: list[Frame] = []
    missing_paths: list[str] = []
    for item in payload.get("frames") or []:
        if not isinstance(item, dict):
            continue
        frame_number = item.get("frame_number", item.get("number"))
        frame_path = item.get("path")
        timestamp = item.get("timestamp")
        if frame_number is None or frame_path is None or timestamp is None:
            continue
        path_value = Path(str(frame_path))
        if not path_value.is_absolute():
            path_value = output_dir / path_value
        if not path_value.is_file():
            missing_paths.append(str(path_value))
            continue
        frames.append(
            Frame(
                number=int(frame_number),
                path=path_value,
                timestamp=float(timestamp),
                score=float(item.get("score") or 0.0),
            )
        )
    return frames, {
        "source": "frames_manifest",
        "path": str(path),
        "manifest_source": payload.get("source", ""),
        "frame_count": len(frames),
        "missing_paths": missing_paths[:10],
        "missing_path_count": len(missing_paths),
    }


def _relative_path(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)
