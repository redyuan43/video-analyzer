from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_VERSION = 2
LEGACY_FRAME_RE = re.compile(r"^(?:Frame\s+|Error analyzing frame\s+)(\d+)\b", re.IGNORECASE)


def analysis_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frame_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vl_checkpoint(
    path: Path | None,
    frames: Iterable[Any],
    expected_signature: str,
    *,
    allow_legacy_ordered: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    metadata = {
        "version": None,
        "valid_successes": 0,
        "failed_entries": 0,
        "signature_match": False,
        "legacy_migrated": False,
    }
    if path is None or not path.is_file():
        return {}, metadata
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, metadata

    frame_list = list(frames)
    frame_by_number = {int(frame.number): frame for frame in frame_list}
    loaded: dict[int, dict[str, Any]] = {}

    if isinstance(payload, dict) and int(payload.get("version") or 0) == CHECKPOINT_VERSION:
        metadata["version"] = CHECKPOINT_VERSION
        metadata["signature_match"] = payload.get("analysis_signature") == expected_signature
        if not metadata["signature_match"]:
            return {}, metadata
        entries = payload.get("frames") or []
        if not isinstance(entries, list):
            return {}, metadata
        for item in entries:
            if not isinstance(item, dict):
                continue
            frame_number = item.get("frame_number")
            if frame_number is None:
                continue
            frame = frame_by_number.get(int(frame_number))
            if frame is None:
                continue
            status = str(item.get("status") or "")
            if status != "succeeded":
                metadata["failed_entries"] += 1
                continue
            if item.get("frame_sha256") != frame_sha256(Path(frame.path)):
                continue
            loaded[int(frame_number)] = item
        metadata["valid_successes"] = len(loaded)
        return loaded, metadata

    metadata["version"] = 1 if isinstance(payload, list) else None
    if not allow_legacy_ordered or not isinstance(payload, list):
        return {}, metadata

    for index, item in enumerate(payload):
        if index >= len(frame_list) or not isinstance(item, dict):
            break
        frame = frame_list[index]
        response = str(item.get("response") or "")
        match = LEGACY_FRAME_RE.match(response.strip())
        if match and int(match.group(1)) != int(frame.number):
            continue
        status = "failed" if response.startswith("Error analyzing frame ") else "succeeded"
        migrated = {
            **item,
            "frame_number": int(frame.number),
            "timestamp": float(frame.timestamp),
            "frame_sha256": frame_sha256(Path(frame.path)),
            "status": status,
            "analysis_signature": expected_signature,
        }
        if status == "succeeded":
            loaded[int(frame.number)] = migrated
        else:
            metadata["failed_entries"] += 1

    metadata["signature_match"] = True
    metadata["legacy_migrated"] = bool(loaded)
    metadata["valid_successes"] = len(loaded)
    return loaded, metadata


def write_vl_checkpoint(
    path: Path,
    analyses: Iterable[dict[str, Any]],
    *,
    signature: str,
    signature_payload: dict[str, Any],
) -> None:
    entries = [
        item
        for item in analyses
        if isinstance(item, dict)
        and item.get("frame_number") is not None
        and item.get("analysis_signature") == signature
        and item.get("status") in {"succeeded", "failed"}
    ]
    entries.sort(key=lambda item: int(item["frame_number"]))
    payload = {
        "version": CHECKPOINT_VERSION,
        "analysis_signature": signature,
        "signature_payload": signature_payload,
        "frames": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)
