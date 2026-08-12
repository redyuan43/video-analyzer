from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frame import Frame


@dataclass(frozen=True)
class FrameDedupAuditOptions:
    threshold_percent: float = 8.0
    window: int = 4
    signature_size: int = 16
    pixel_tolerance: int = 25
    max_gap_seconds: float = 50.0


def audit_frame_deduplication(
    frames: list[Frame],
    *,
    options: FrameDedupAuditOptions | None = None,
) -> dict[str, Any]:
    opts = options or FrameDedupAuditOptions()
    records: list[dict[str, Any]] = []
    if not frames:
        return _summary(opts, records, disabled_reason="")
    try:
        from PIL import Image
    except Exception:
        return _summary(opts, records, disabled_reason="Pillow is not available")

    ordered_frames = sorted(frames, key=lambda item: item.timestamp)
    frames_by_number = {int(frame.number): frame for frame in ordered_frames}
    recent_kept: list[tuple[int, list[tuple[int, int, int]]]] = []
    for frame in ordered_frames:
        try:
            signature = _signature(Image, frame.path, opts.signature_size)
        except Exception as exc:
            records.append(
                {
                    "frame_number": frame.number,
                    "timestamp": round(frame.timestamp, 3),
                    "path": _path_value(frame.path),
                    "baseline_action": "keep",
                    "similarity_action": "keep",
                    "treatment_action": "keep",
                    "decision_reason": "signature_error_fail_open",
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue
        distances = [
            {
                "frame_number": previous_number,
                "rgb_diff_percent": _pct_diff(signature, previous_signature, opts.pixel_tolerance),
            }
            for previous_number, previous_signature in recent_kept
        ]
        nearest = min(distances, key=lambda item: item["rgb_diff_percent"], default=None)
        should_drop = bool(nearest and nearest["rgb_diff_percent"] <= opts.threshold_percent)
        records.append(
            {
                "frame_number": frame.number,
                "timestamp": round(frame.timestamp, 3),
                "path": _path_value(frame.path),
                "baseline_action": "keep",
                "similarity_action": "drop" if should_drop else "keep",
                "treatment_action": "drop" if should_drop else "keep",
                "decision_reason": "near_duplicate" if should_drop else "visual_change",
                "nearest_kept_frame_number": nearest.get("frame_number") if nearest else None,
                "nearest_rgb_diff_percent": round(float(nearest["rgb_diff_percent"]), 4) if nearest else None,
                "status": "ok",
            }
        )
        if not should_drop:
            recent_kept.append((frame.number, signature))
            if len(recent_kept) > opts.window:
                recent_kept.pop(0)
    _repair_timeline_coverage(records, frames_by_number, opts.max_gap_seconds)
    return _summary(opts, records, disabled_reason="")


def write_frame_dedup_audit(
    frames: list[Frame],
    output_dir: Path,
    *,
    options: FrameDedupAuditOptions | None = None,
) -> tuple[Path, dict[str, Any]]:
    audit = audit_frame_deduplication(frames, options=options)
    path = output_dir / "frame_dedup_audit.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, audit


def select_audited_frames(frames: list[Frame], audit: dict[str, Any]) -> list[Frame]:
    """Return the active OCR input set, failing open on unusable audit data."""
    if not audit.get("enabled"):
        return list(frames)
    kept_numbers = {
        int(item["frame_number"])
        for item in audit.get("records") or []
        if item.get("treatment_action") == "keep" and item.get("frame_number") is not None
    }
    selected = [frame for frame in frames if int(frame.number) in kept_numbers]
    return selected or list(frames)


def _repair_timeline_coverage(
    records: list[dict[str, Any]],
    frames_by_number: dict[int, Frame],
    max_gap_seconds: float,
) -> None:
    if not records:
        return
    _promote_record(records[0], "timeline_boundary")
    _promote_record(records[-1], "timeline_boundary")
    if max_gap_seconds <= 0:
        return

    while True:
        kept_indexes = [
            index
            for index, item in enumerate(records)
            if item.get("treatment_action") == "keep"
        ]
        oversized_gap = next(
            (
                (left, right)
                for left, right in zip(kept_indexes, kept_indexes[1:])
                if float(records[right]["timestamp"]) - float(records[left]["timestamp"])
                > max_gap_seconds
            ),
            None,
        )
        if oversized_gap is None:
            return

        left, right = oversized_gap
        coverage_deadline = float(records[left]["timestamp"]) + max_gap_seconds
        candidates = [
            (index, item)
            for index, item in enumerate(records[left + 1 : right], start=left + 1)
            if item.get("treatment_action") == "drop"
            and float(item["timestamp"]) <= coverage_deadline
        ]
        if not candidates:
            candidates = [
                (index, item)
                for index, item in enumerate(records[left + 1 : right], start=left + 1)
                if item.get("treatment_action") == "drop"
            ]
        if not candidates:
            return
        selected_index, selected = max(
            candidates,
            key=lambda pair: (
                float(pair[1]["timestamp"]),
                float(frames_by_number.get(int(pair[1]["frame_number"])).score)
                if frames_by_number.get(int(pair[1]["frame_number"])) is not None
                else 0.0,
            ),
        )
        _promote_record(records[selected_index], "timeline_coverage")


def _promote_record(record: dict[str, Any], reason: str) -> None:
    if record.get("treatment_action") == "keep":
        return
    record["treatment_action"] = "keep"
    record["decision_reason"] = reason


def _signature(image_module: Any, path: Path, size: int) -> list[tuple[int, int, int]]:
    with image_module.open(path) as image:
        resized = image.convert("RGB").resize((size, size))
        getter = getattr(resized, "get_flattened_data", resized.getdata)
        return list(getter())


def _pct_diff(
    left: list[tuple[int, int, int]],
    right: list[tuple[int, int, int]],
    tolerance: int,
) -> float:
    changed = sum(
        max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])) > tolerance
        for a, b in zip(left, right)
    )
    return 100.0 * changed / max(len(left), 1)


def _summary(options: FrameDedupAuditOptions, records: list[dict[str, Any]], disabled_reason: str) -> dict[str, Any]:
    baseline_count = sum(1 for item in records if item.get("baseline_action") == "keep")
    similarity_keep = sum(1 for item in records if item.get("similarity_action") == "keep")
    treatment_keep = sum(1 for item in records if item.get("treatment_action") == "keep")
    treatment_drop = sum(1 for item in records if item.get("treatment_action") == "drop")
    coverage_promoted = sum(
        1
        for item in records
        if item.get("similarity_action") == "drop"
        and item.get("treatment_action") == "keep"
    )
    duplicate_ratio = treatment_drop / max(baseline_count, 1)
    kept_timestamps = [
        float(item["timestamp"])
        for item in records
        if item.get("treatment_action") == "keep"
    ]
    max_kept_gap = max(
        (right - left for left, right in zip(kept_timestamps, kept_timestamps[1:])),
        default=0.0,
    )
    return {
        "version": 2,
        "enabled": not disabled_reason,
        "disabled_reason": disabled_reason,
        "method": "sliding_window_rgb_diff_with_timeline_coverage",
        "options": {
            "threshold_percent": options.threshold_percent,
            "window": options.window,
            "signature_size": options.signature_size,
            "pixel_tolerance": options.pixel_tolerance,
            "max_gap_seconds": options.max_gap_seconds,
        },
        "summary": {
            "baseline_frame_count": baseline_count,
            "similarity_keep_count": similarity_keep,
            "coverage_promoted_count": coverage_promoted,
            "treatment_keep_count": treatment_keep,
            "treatment_drop_count": treatment_drop,
            "recommended_drop_ratio": round(duplicate_ratio, 4),
            "estimated_image_review_reduction_ratio": round(duplicate_ratio, 4),
            "max_kept_gap_seconds": round(max_kept_gap, 3),
        },
        "ab_test": {
            "name": "frame_dedup_sliding_window_rgb_diff",
            "baseline": "current selected candidate frames",
            "treatment": "drop frames whose RGB signature differs from the recent kept window by <= threshold",
            "primary_metric": "treatment_drop_count_without_reducing_timeline_coverage",
            "observed_delta": {
                "frame_count": treatment_keep - baseline_count,
                "image_review_reduction_ratio": round(duplicate_ratio, 4),
            },
        },
        "records": records,
    }


def _path_value(path: Path) -> str:
    return path.as_posix()
