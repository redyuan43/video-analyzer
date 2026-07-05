from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy is a project dependency
    np = None


CANDIDATE_FRAME_STRATEGIES = {"auto", "legacy", "generic", "lecture", "operation"}


@dataclass(frozen=True)
class CandidateStrategyResult:
    selected: list[tuple[int, Any, float, float]]
    metadata: dict[str, Any]


def parse_candidate_frame_strategy(value: str | None) -> str:
    strategy = (value or "auto").strip().lower()
    if strategy not in CANDIDATE_FRAME_STRATEGIES:
        raise ValueError(f"Unknown candidate frame strategy: {value}")
    return strategy


def select_candidate_frames_with_strategy(
    candidates: list[tuple[int, Any, float, float]],
    candidate_budget: int,
    video_duration_seconds: float,
    pipeline_mode: str,
    strategy: str = "auto",
    transcript_segments: Iterable[dict[str, Any]] | None = None,
    legacy_selector: Callable[[list[tuple[int, Any, float, float]], int], list[tuple[int, Any, float, float]]] | None = None,
) -> CandidateStrategyResult:
    strategy = parse_candidate_frame_strategy(strategy)
    deduped = _dedupe_by_timestamp(sorted(candidates, key=lambda item: item[2]))
    if candidate_budget <= 0 or not deduped:
        return CandidateStrategyResult([], _metadata(strategy, "none", "empty", {}, {}, {}, {}))

    selector = legacy_selector or _density_budget
    legacy = selector(deduped, candidate_budget)
    profile = classify_video_profile(deduped, video_duration_seconds, transcript_segments)
    selected_strategy = _resolve_selected_strategy(strategy, profile)
    if selected_strategy == "legacy":
        return CandidateStrategyResult(
            legacy,
            _metadata(strategy, "legacy", "ok", profile, _metrics(deduped, video_duration_seconds), _metrics(legacy, video_duration_seconds), {}),
        )

    algorithm_trace: dict[str, Any] = {}
    strategy_selected = _run_paper_strategy(
        selected_strategy,
        deduped,
        candidate_budget,
        video_duration_seconds,
        pipeline_mode,
        transcript_segments,
        selector,
        algorithm_trace,
    )
    if pipeline_mode == "deep" and strategy == "auto":
        secondary_strategy = _secondary_strategy(selected_strategy)
        if secondary_strategy:
            algorithm_trace["secondary_strategy"] = secondary_strategy
            secondary_selected = _run_paper_strategy(
                secondary_strategy,
                deduped,
                max(1, candidate_budget // 2),
                video_duration_seconds,
                pipeline_mode,
                transcript_segments,
                selector,
                algorithm_trace,
            )
            strategy_selected = selector(_merge_unique(strategy_selected, secondary_selected), candidate_budget)
    guard = _quality_guard(
        selected_strategy,
        strategy_selected,
        legacy,
        candidate_budget,
        video_duration_seconds,
        transcript_segments,
    )
    final = strategy_selected
    status = _algorithm_status(selected_strategy, algorithm_trace)
    if not guard["passed"]:
        combined = _merge_unique(strategy_selected, legacy)
        final = selector(combined, min(candidate_budget, len(combined)))
        combined_guard = _quality_guard(
            selected_strategy,
            final,
            legacy,
            candidate_budget,
            video_duration_seconds,
            transcript_segments,
        )
        if combined_guard["passed"]:
            guard = {**combined_guard, "fallback": "paper_union_legacy"}
            status = "guarded_union"
        else:
            final = legacy
            guard = {**guard, "fallback": "legacy"}
            status = "guarded_legacy"

    return CandidateStrategyResult(
        final,
        _metadata(
            strategy,
            selected_strategy,
            status,
            profile,
            _metrics(deduped, video_duration_seconds),
            _metrics(final, video_duration_seconds),
            guard,
            algorithm_trace,
        ),
    )


def classify_video_profile(
    candidates: list[tuple[int, Any, float, float]],
    video_duration_seconds: float,
    transcript_segments: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not candidates:
        return {"profile": "operation", "confidence": 0.0, "reason": "empty"}
    visual = [_visual(item) for item in candidates]
    textness = [_textness(item) for item in candidates]
    duration = max(video_duration_seconds, candidates[-1][2] - candidates[0][2], 1.0)
    stable_text_ratio = sum(1 for item in candidates if _textness(item) >= 0.28 and _visual(item) <= 8.0) / len(candidates)
    high_motion_ratio = sum(1 for value in visual if value >= max(12.0, _percentile(visual, 70))) / len(candidates)
    text_mean = sum(textness) / len(textness)
    visual_mean = sum(visual) / len(visual)
    transcript_density = _transcript_density(transcript_segments, duration)

    if text_mean >= 0.30 and stable_text_ratio >= 0.45 and high_motion_ratio <= 0.35:
        confidence = min(0.95, 0.45 + stable_text_ratio * 0.45 + min(transcript_density, 1.0) * 0.10)
        return {
            "profile": "lecture",
            "confidence": round(confidence, 4),
            "reason": "stable_text_low_motion",
            "features": _profile_features(text_mean, visual_mean, stable_text_ratio, high_motion_ratio, transcript_density),
        }
    if text_mean >= 0.22 and stable_text_ratio >= 0.20:
        confidence = min(0.90, 0.38 + text_mean * 0.75 + high_motion_ratio * 0.15)
        return {
            "profile": "operation",
            "confidence": round(confidence, 4),
            "reason": "screen_text_with_state_changes",
            "features": _profile_features(text_mean, visual_mean, stable_text_ratio, high_motion_ratio, transcript_density),
        }
    confidence = min(0.90, 0.40 + high_motion_ratio * 0.35 + min(visual_mean / 30.0, 1.0) * 0.20)
    return {
        "profile": "generic",
        "confidence": round(confidence, 4),
        "reason": "visual_change_dominant",
        "features": _profile_features(text_mean, visual_mean, stable_text_ratio, high_motion_ratio, transcript_density),
    }


def _resolve_selected_strategy(requested_strategy: str, profile: dict[str, Any]) -> str:
    if requested_strategy != "auto":
        return requested_strategy
    if float(profile.get("confidence") or 0.0) < 0.65:
        return "operation"
    return str(profile.get("profile") or "operation")


def _secondary_strategy(primary: str) -> str | None:
    if primary == "generic":
        return "operation"
    if primary == "lecture":
        return "operation"
    if primary == "operation":
        return "lecture"
    return None


def _algorithm_status(strategy: str, algorithm_trace: dict[str, Any]) -> str:
    if strategy == "generic":
        lmske = algorithm_trace.get("lmske") or {}
        if lmske.get("embedding_source") != "clip_embedding" or lmske.get("shot_detector") != "transnetv2":
            return "degraded_backend_unavailable"
    if strategy == "operation":
        mskvs = algorithm_trace.get("mskvs") or {}
        if mskvs.get("clustering") != "sklearn_meanshift":
            return "degraded_backend_unavailable"
    return "ok"


def _run_paper_strategy(
    strategy: str,
    candidates: list[tuple[int, Any, float, float]],
    candidate_budget: int,
    video_duration_seconds: float,
    pipeline_mode: str,
    transcript_segments: Iterable[dict[str, Any]] | None,
    legacy_selector: Callable[[list[tuple[int, Any, float, float]], int], list[tuple[int, Any, float, float]]],
    algorithm_trace: dict[str, Any],
) -> list[tuple[int, Any, float, float]]:
    if strategy == "generic":
        return _select_lmske(candidates, candidate_budget, video_duration_seconds, legacy_selector, algorithm_trace)
    if strategy == "lecture":
        return _select_sspa(candidates, candidate_budget, video_duration_seconds, transcript_segments, legacy_selector, algorithm_trace)
    if strategy == "operation":
        return _select_mskvs_sspa(candidates, candidate_budget, video_duration_seconds, transcript_segments, legacy_selector, algorithm_trace)
    return legacy_selector(candidates, candidate_budget)


def _select_lmske(
    candidates: list[tuple[int, Any, float, float]],
    budget: int,
    video_duration_seconds: float,
    legacy_selector: Callable[[list[tuple[int, Any, float, float]], int], list[tuple[int, Any, float, float]]],
    algorithm_trace: dict[str, Any],
) -> list[tuple[int, Any, float, float]]:
    """LMSKE-shaped path: shot segmentation, embedding clustering, redundancy removal."""
    embedding_source = "clip_embedding" if any(_clip_embedding(item) for item in candidates) else "gffv_metadata_fallback"
    algorithm_trace["lmske"] = {
        "shot_detector": "transnetv2" if any(_payload_item(item).get("shot_boundary") for item in candidates) else "visual_score_fallback",
        "embedding_source": embedding_source,
        "clustering": "adaptive_center",
        "redundancy": "cosine",
    }
    shots = _shot_segments(candidates)
    selected: list[tuple[int, Any, float, float]] = []
    for shot in shots:
        if not shot:
            continue
        selected.append(_representative_by_feature_center(shot, video_duration_seconds))
    selected = _remove_feature_redundancy(selected, threshold=0.985)
    if len(selected) < budget:
        remainder = [item for item in candidates if item not in selected]
        selected = _merge_unique(selected, sorted(remainder, key=_lmske_rank, reverse=True))
    return legacy_selector(sorted(selected, key=lambda item: item[2]), min(budget, len(selected)))


def _select_sspa(
    candidates: list[tuple[int, Any, float, float]],
    budget: int,
    video_duration_seconds: float,
    transcript_segments: Iterable[dict[str, Any]] | None,
    legacy_selector: Callable[[list[tuple[int, Any, float, float]], int], list[tuple[int, Any, float, float]]],
    algorithm_trace: dict[str, Any],
) -> list[tuple[int, Any, float, float]]:
    """Spatio-temporal subtitle projection approximation over preview metadata."""
    algorithm_trace["sspa"] = {
        "curve_source": "subtitle_region_projection" if any(_payload_item(item).get("sspa_projection") is not None for item in candidates) else "textness_projection_fallback",
        "catastrophe_points": "projection_delta",
        "transcript_anchor": bool(transcript_segments),
    }
    stable_ranges = _stable_text_ranges(candidates)
    selected = [_middle_item(items) for items in stable_ranges if items]
    selected.extend(_catastrophe_neighbors(candidates))
    selected.extend(_transcript_anchor_items(candidates, transcript_segments))
    selected = _merge_unique(selected)
    if len(selected) < budget:
        fill = sorted(candidates, key=lambda item: (_textness(item), -_visual(item), item[3]), reverse=True)
        selected = _merge_unique(selected, fill)
    return legacy_selector(sorted(selected, key=lambda item: item[2]), min(budget, len(selected)))


def _select_mskvs_sspa(
    candidates: list[tuple[int, Any, float, float]],
    budget: int,
    video_duration_seconds: float,
    transcript_segments: Iterable[dict[str, Any]] | None,
    legacy_selector: Callable[[list[tuple[int, Any, float, float]], int], list[tuple[int, Any, float, float]]],
    algorithm_trace: dict[str, Any],
) -> list[tuple[int, Any, float, float]]:
    """MSKVS-shaped path: linear redundancy removal, mean-shift-like feature modes, SSPA text anchors."""
    sklearn_available = _has_sklearn_meanshift()
    algorithm_trace["mskvs"] = {
        "feature": "gffv_global_orientation",
        "redundancy": "linear_cosine",
        "clustering": "sklearn_meanshift" if sklearn_available else "internal_adaptive_meanshift",
        "rcr_metric": "redundancy_compression_ratio",
    }
    non_redundant = _linear_redundancy_filter(candidates, threshold=0.992)
    modes = _adaptive_mean_shift_modes(non_redundant, budget)
    text_anchors = _select_sspa(candidates, max(1, budget // 3), video_duration_seconds, transcript_segments, legacy_selector, algorithm_trace)
    selected = _merge_unique(modes, text_anchors)
    if len(selected) < budget:
        selected = _merge_unique(selected, sorted(non_redundant, key=_operation_rank, reverse=True))
    return legacy_selector(sorted(selected, key=lambda item: item[2]), min(budget, len(selected)))


def _shot_segments(candidates: list[tuple[int, Any, float, float]]) -> list[list[tuple[int, Any, float, float]]]:
    if any(_payload_item(item).get("shot_boundary") for item in candidates):
        shots: list[list[tuple[int, Any, float, float]]] = []
        current: list[tuple[int, Any, float, float]] = []
        for item in candidates:
            if current and _payload_item(item).get("shot_boundary"):
                shots.append(current)
                current = []
            current.append(item)
        if current:
            shots.append(current)
        return shots
    visual_values = [_visual(item) for item in candidates]
    threshold = max(12.0, _percentile(visual_values, 75) + _std(visual_values) * 0.25)
    shots: list[list[tuple[int, Any, float, float]]] = []
    current: list[tuple[int, Any, float, float]] = []
    for item in candidates:
        if current and _visual(item) >= threshold:
            shots.append(current)
            current = []
        current.append(item)
    if current:
        shots.append(current)
    return shots


def _representative_by_feature_center(
    items: list[tuple[int, Any, float, float]],
    video_duration_seconds: float,
) -> tuple[int, Any, float, float]:
    vectors = [_feature_vector(item, video_duration_seconds) for item in items]
    center = _mean_vector(vectors)
    return min(items, key=lambda item: _distance(_feature_vector(item, video_duration_seconds), center))


def _stable_text_ranges(candidates: list[tuple[int, Any, float, float]]) -> list[list[tuple[int, Any, float, float]]]:
    values = [_textness(item) for item in candidates]
    if not values:
        return []
    threshold = max(0.28, _percentile(values, 55))
    ranges: list[list[tuple[int, Any, float, float]]] = []
    current: list[tuple[int, Any, float, float]] = []
    for item in candidates:
        if _textness(item) >= threshold and _visual(item) <= 12.0:
            current.append(item)
            continue
        if current:
            ranges.append(current)
            current = []
    if current:
        ranges.append(current)
    return ranges


def _catastrophe_neighbors(candidates: list[tuple[int, Any, float, float]]) -> list[tuple[int, Any, float, float]]:
    textness = [_textness(item) for item in candidates]
    if len(textness) < 3:
        return []
    selected = []
    for idx in range(1, len(textness) - 1):
        before = textness[idx] - textness[idx - 1]
        after = textness[idx + 1] - textness[idx]
        if abs(before - after) >= 0.20:
            selected.append(candidates[idx])
    return selected


def _adaptive_mean_shift_modes(
    candidates: list[tuple[int, Any, float, float]],
    budget: int,
) -> list[tuple[int, Any, float, float]]:
    if not candidates:
        return []
    duration = max(candidates[-1][2] - candidates[0][2], 1.0)
    vectors = [_feature_vector(item, duration) for item in candidates]
    sklearn_modes = _sklearn_mean_shift_modes(candidates, vectors, budget, duration)
    if sklearn_modes:
        return sklearn_modes
    bandwidth = max(_median_pairwise_distance(vectors), 0.08)
    clusters: list[list[tuple[int, Any, float, float]]] = []
    centers: list[list[float]] = []
    for item, vector in zip(candidates, vectors):
        match = next((idx for idx, center in enumerate(centers) if _distance(vector, center) <= bandwidth), None)
        if match is None:
            centers.append(vector)
            clusters.append([item])
        else:
            clusters[match].append(item)
            centers[match] = _mean_vector([_feature_vector(row, duration) for row in clusters[match]])
    ranked_clusters = sorted(clusters, key=lambda rows: (len(rows), max(_operation_rank(row) for row in rows)), reverse=True)
    return [_representative_by_feature_center(rows, duration) for rows in ranked_clusters[:budget]]


def _linear_redundancy_filter(
    candidates: list[tuple[int, Any, float, float]],
    threshold: float,
) -> list[tuple[int, Any, float, float]]:
    kept = []
    last_vector = None
    duration = max(candidates[-1][2] - candidates[0][2], 1.0) if candidates else 1.0
    for item in candidates:
        vector = _feature_vector(item, duration)
        if last_vector is None or _cosine(vector, last_vector) < threshold:
            kept.append(item)
            last_vector = vector
    return kept


def _remove_feature_redundancy(
    candidates: list[tuple[int, Any, float, float]],
    threshold: float,
) -> list[tuple[int, Any, float, float]]:
    kept: list[tuple[int, Any, float, float]] = []
    duration = max(candidates[-1][2] - candidates[0][2], 1.0) if candidates else 1.0
    for item in sorted(candidates, key=lambda row: row[2]):
        vector = _feature_vector(item, duration)
        if all(_cosine(vector, _feature_vector(existing, duration)) < threshold for existing in kept):
            kept.append(item)
    return kept


def _quality_guard(
    strategy: str,
    selected: list[tuple[int, Any, float, float]],
    legacy: list[tuple[int, Any, float, float]],
    budget: int,
    video_duration_seconds: float,
    transcript_segments: Iterable[dict[str, Any]] | None,
) -> dict[str, Any]:
    selected_metrics = _metrics(selected, video_duration_seconds, transcript_segments)
    legacy_metrics = _metrics(legacy, video_duration_seconds, transcript_segments)
    reasons = []
    if len(selected) < min(budget, len(legacy)):
        reasons.append("candidate_count_below_legacy")
    if selected_metrics["coverage_60s_bucket_ratio"] + 0.05 < legacy_metrics["coverage_60s_bucket_ratio"]:
        reasons.append("coverage_below_legacy")
    if (
        strategy in {"lecture", "operation"}
        and legacy_metrics["stable_text_candidate_count"] > 0
        and selected_metrics["stable_text_candidate_count"] < max(1, int(legacy_metrics["stable_text_candidate_count"] * 0.85))
    ):
        reasons.append("stable_text_below_legacy")
    if selected_metrics["near_duplicate_gap_count"] > max(legacy_metrics["near_duplicate_gap_count"] + 2, int(len(selected) * 0.20)):
        reasons.append("too_many_near_duplicates")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "selected_metrics": selected_metrics,
        "legacy_metrics": legacy_metrics,
    }


def _metrics(
    candidates: list[tuple[int, Any, float, float]],
    video_duration_seconds: float,
    transcript_segments: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not candidates:
        return {
            "candidate_count": 0,
            "coverage_60s_bucket_ratio": 0.0,
            "near_duplicate_gap_count": 0,
            "stable_text_candidate_count": 0,
            "redundancy_ratio": 0.0,
            "transcript_anchor_coverage": 0.0,
        }
    timestamps = sorted(item[2] for item in candidates)
    duration = max(video_duration_seconds, timestamps[-1] - timestamps[0], 1.0)
    total_buckets = max(1, int(math.ceil(duration / 60.0)))
    covered = {int(timestamp // 60.0) for timestamp in timestamps}
    near_duplicates = sum(1 for previous, current in zip(timestamps, timestamps[1:]) if current - previous < 2.0)
    stable_text = sum(1 for item in candidates if _textness(item) >= 0.28 and _visual(item) <= 8.0)
    return {
        "candidate_count": len(candidates),
        "coverage_60s_bucket_ratio": round(min(len(covered) / total_buckets, 1.0), 4),
        "near_duplicate_gap_count": near_duplicates,
        "stable_text_candidate_count": stable_text,
        "redundancy_ratio": round(near_duplicates / max(len(candidates), 1), 4),
        "transcript_anchor_coverage": round(_transcript_anchor_coverage(timestamps, transcript_segments), 4),
    }


def _density_budget(candidates: list[tuple[int, Any, float, float]], max_frames: int) -> list[tuple[int, Any, float, float]]:
    if len(candidates) <= max_frames:
        return candidates
    selected_indexes = set()
    last_bucket = None
    for idx, (_, _, timestamp, _) in enumerate(candidates):
        bucket = int(timestamp // 20.0)
        if bucket != last_bucket:
            selected_indexes.add(idx)
            last_bucket = bucket
    remaining = max(max_frames - len(selected_indexes), 0)
    ranked = sorted((idx for idx in range(len(candidates)) if idx not in selected_indexes), key=lambda idx: candidates[idx][3], reverse=True)
    selected_indexes.update(ranked[:remaining])
    if len(selected_indexes) > max_frames:
        selected_indexes = set(sorted(selected_indexes)[:max_frames])
    return [candidates[idx] for idx in sorted(selected_indexes)]


def _metadata(
    requested_strategy: str,
    selected_strategy: str,
    status: str,
    profile: dict[str, Any],
    raw_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    guard: dict[str, Any],
    algorithm_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    algorithms = {
        "generic": "lmske_shot_embedding_clustering",
        "lecture": "sspa_spatio_temporal_subtitle_projection",
        "operation": "mskvs_gffv_adaptive_mean_shift_with_sspa",
        "legacy": "density_budget",
        "none": "none",
    }
    return {
        "requested_candidate_strategy": requested_strategy,
        "selected_candidate_strategy": selected_strategy,
        "paper_algorithm": algorithms.get(selected_strategy, selected_strategy),
        "paper_algorithm_status": status,
        "video_profile": profile.get("profile", selected_strategy),
        "video_profile_confidence": profile.get("confidence", 0.0),
        "video_profile_reason": profile.get("reason", ""),
        "strategy_observations": {
            "profile": profile,
            "raw_metrics": raw_metrics,
            "final_metrics": final_metrics,
            "paper_algorithm_trace": algorithm_trace or {},
        },
        "quality_guard": guard,
    }


def _feature_vector(item: tuple[int, Any, float, float], duration: float) -> list[float]:
    clip = _clip_embedding(item)
    if clip:
        return _normalize_vector(clip)
    histogram = _gffv(item)
    base = [
        min(_visual(item) / 40.0, 1.0),
        _textness(item),
        min(max(item[2] / max(duration, 1.0), 0.0), 1.0),
        min(max(item[3] / 80.0, 0.0), 1.0),
    ]
    return base + histogram


def _payload_item(item: tuple[int, Any, float, float]) -> dict[str, Any]:
    payload = item[1]
    if isinstance(payload, dict) and isinstance(payload.get("item"), dict):
        return payload["item"]
    if isinstance(payload, dict):
        return payload
    return {}


def _visual(item: tuple[int, Any, float, float]) -> float:
    payload = _payload_item(item)
    return float(payload.get("visual_score", item[3]) or 0.0)


def _textness(item: tuple[int, Any, float, float]) -> float:
    return float(_payload_item(item).get("textness_score", 0.0) or 0.0)


def _gffv(item: tuple[int, Any, float, float]) -> list[float]:
    values = _payload_item(item).get("gffv")
    if isinstance(values, list) and values:
        total = sum(float(value) for value in values) or 1.0
        return [float(value) / total for value in values[:8]]
    return [0.0] * 8


def _clip_embedding(item: tuple[int, Any, float, float]) -> list[float]:
    values = _payload_item(item).get("clip_embedding")
    if isinstance(values, list) and values:
        return [float(value) for value in values]
    return []


def _lmske_rank(item: tuple[int, Any, float, float]) -> float:
    return item[3] + _visual(item) * 0.6 + _textness(item) * 8.0


def _operation_rank(item: tuple[int, Any, float, float]) -> float:
    return item[3] + _textness(item) * 35.0 + _visual(item) * 0.4


def _middle_item(items: list[tuple[int, Any, float, float]]) -> tuple[int, Any, float, float]:
    return items[len(items) // 2]


def _merge_unique(*groups: Iterable[tuple[int, Any, float, float]]) -> list[tuple[int, Any, float, float]]:
    merged = {}
    for group in groups:
        for item in group:
            merged.setdefault(round(item[2], 3), item)
    return [merged[key] for key in sorted(merged)]


def _has_sklearn_meanshift() -> bool:
    try:
        from sklearn.cluster import MeanShift  # noqa: F401
        return True
    except Exception:
        return False


def _sklearn_mean_shift_modes(
    candidates: list[tuple[int, Any, float, float]],
    vectors: list[list[float]],
    budget: int,
    duration: float,
) -> list[tuple[int, Any, float, float]]:
    try:
        from sklearn.cluster import MeanShift, estimate_bandwidth
    except Exception:
        return []
    if len(candidates) < 2 or np is None:
        return []
    matrix = np.asarray(vectors, dtype=float)
    bandwidth = estimate_bandwidth(matrix, quantile=0.25, n_samples=min(len(vectors), 256))
    if not bandwidth or bandwidth <= 0:
        bandwidth = max(_median_pairwise_distance(vectors), 0.08)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            labels = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(matrix)
        except ValueError:
            return []
    clusters: dict[int, list[tuple[int, Any, float, float]]] = {}
    for item, label in zip(candidates, labels):
        clusters.setdefault(int(label), []).append(item)
    ranked = sorted(clusters.values(), key=lambda rows: (len(rows), max(_operation_rank(row) for row in rows)), reverse=True)
    return [_representative_by_feature_center(rows, duration) for rows in ranked[:budget]]


def _dedupe_by_timestamp(candidates: list[tuple[int, Any, float, float]], min_gap: float = 0.5) -> list[tuple[int, Any, float, float]]:
    deduped = []
    last_timestamp = -math.inf
    for item in candidates:
        if item[2] - last_timestamp < min_gap:
            continue
        deduped.append(item)
        last_timestamp = item[2]
    return deduped


def _transcript_anchor_items(
    candidates: list[tuple[int, Any, float, float]],
    transcript_segments: Iterable[dict[str, Any]] | None,
) -> list[tuple[int, Any, float, float]]:
    anchors = []
    for segment in transcript_segments or []:
        timestamp = _segment_midpoint(segment)
        if timestamp is None:
            continue
        anchors.append(min(candidates, key=lambda item: abs(item[2] - timestamp)))
    return _merge_unique(anchors)


def _segment_midpoint(segment: dict[str, Any]) -> float | None:
    start = segment.get("start", segment.get("start_time"))
    end = segment.get("end", segment.get("end_time"))
    try:
        if start is None and end is None:
            return None
        if end is None:
            return float(start)
        if start is None:
            return float(end)
        return (float(start) + float(end)) / 2.0
    except (TypeError, ValueError):
        return None


def _transcript_density(segments: Iterable[dict[str, Any]] | None, duration: float) -> float:
    values = [segment for segment in segments or [] if _segment_midpoint(segment) is not None]
    return len(values) / max(duration / 60.0, 1.0)


def _transcript_anchor_coverage(timestamps: list[float], transcript_segments: Iterable[dict[str, Any]] | None) -> float:
    anchors = [_segment_midpoint(segment) for segment in transcript_segments or []]
    anchors = [anchor for anchor in anchors if anchor is not None]
    if not anchors:
        return 0.0
    covered = sum(1 for anchor in anchors if min(abs(timestamp - anchor) for timestamp in timestamps) <= 30.0)
    return covered / len(anchors)


def _profile_features(text_mean: float, visual_mean: float, stable_text_ratio: float, high_motion_ratio: float, transcript_density: float) -> dict[str, float]:
    return {
        "textness_mean": round(text_mean, 4),
        "visual_score_mean": round(visual_mean, 4),
        "stable_text_ratio": round(stable_text_ratio, 4),
        "high_motion_ratio": round(high_motion_ratio, 4),
        "transcript_segments_per_minute": round(transcript_density, 4),
    }


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    if np is not None:
        return np.mean(np.asarray(vectors, dtype=float), axis=0).tolist()
    width = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def _normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        return values
    return [value / norm for value in values]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _median_pairwise_distance(vectors: list[list[float]]) -> float:
    distances = []
    for index, left in enumerate(vectors[:200]):
        for right in vectors[index + 1 : min(len(vectors), index + 16)]:
            distances.append(_distance(left, right))
    if not distances:
        return 0.2
    return _percentile(distances, 50)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if np is not None:
        return float(np.percentile(values, percentile))
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile / 100.0))
    return ordered[index]


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
