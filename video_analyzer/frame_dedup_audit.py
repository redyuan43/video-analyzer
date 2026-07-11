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

    signatures: list[list[tuple[int, int, int]]] = []
    recent_kept: list[tuple[int, list[tuple[int, int, int]]]] = []
    for frame in sorted(frames, key=lambda item: item.timestamp):
        try:
            signature = _signature(Image, frame.path, opts.signature_size)
        except Exception as exc:
            records.append(
                {
                    "frame_number": frame.number,
                    "timestamp": round(frame.timestamp, 3),
                    "path": _path_value(frame.path),
                    "baseline_action": "keep",
                    "treatment_action": "keep",
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue
        signatures.append(signature)
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
                "treatment_action": "drop" if should_drop else "keep",
                "nearest_kept_frame_number": nearest.get("frame_number") if nearest else None,
                "nearest_rgb_diff_percent": round(float(nearest["rgb_diff_percent"]), 4) if nearest else None,
                "status": "ok",
            }
        )
        if not should_drop:
            recent_kept.append((frame.number, signature))
            if len(recent_kept) > opts.window:
                recent_kept.pop(0)
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
    treatment_keep = sum(1 for item in records if item.get("treatment_action") == "keep")
    treatment_drop = sum(1 for item in records if item.get("treatment_action") == "drop")
    duplicate_ratio = treatment_drop / max(baseline_count, 1)
    return {
        "version": 1,
        "enabled": not disabled_reason,
        "disabled_reason": disabled_reason,
        "method": "sliding_window_rgb_diff",
        "options": {
            "threshold_percent": options.threshold_percent,
            "window": options.window,
            "signature_size": options.signature_size,
            "pixel_tolerance": options.pixel_tolerance,
        },
        "summary": {
            "baseline_frame_count": baseline_count,
            "treatment_keep_count": treatment_keep,
            "treatment_drop_count": treatment_drop,
            "recommended_drop_ratio": round(duplicate_ratio, 4),
            "estimated_image_review_reduction_ratio": round(duplicate_ratio, 4),
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
