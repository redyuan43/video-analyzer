from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import ceil
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .audio_processor import AudioTranscript
from .frame import Frame
from .frame_selection import AUTO, parse_auto_float, parse_auto_int
from .ocr import OCREvent

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None
    np = None


OCR_KEYFRAME_STRATEGIES = {"auto", "scan-text", "legacy"}
MIN_LONG_VIDEO_SECONDS = 30 * 60
DEFAULT_TEXT_EVENT_SIMILARITY = 0.86


@dataclass(frozen=True)
class TextnessFeatures:
    textness_score: float
    sharpness_score: float
    edge_density: float
    component_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "textness_score": round(self.textness_score, 4),
            "sharpness_score": round(self.sharpness_score, 4),
            "edge_density": round(self.edge_density, 4),
            "component_score": round(self.component_score, 4),
        }


@dataclass(frozen=True)
class OCRKeyframeDecision:
    frame_number: int
    timestamp: float
    selected_for_ocr: bool
    selection_score: float
    reason: str
    skip_reason: str
    visual_change_score: float
    textness_score: float
    sharpness_score: float
    coverage_bucket: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "selected_for_ocr": self.selected_for_ocr,
            "selection_score": round(self.selection_score, 4),
            "reason": self.reason,
            "skip_reason": self.skip_reason,
            "visual_change_score": round(self.visual_change_score, 4),
            "textness_score": round(self.textness_score, 4),
            "sharpness_score": round(self.sharpness_score, 4),
            "coverage_bucket": self.coverage_bucket,
        }


def parse_ocr_keyframe_strategy(value: str | None) -> str:
    strategy = (value or AUTO).strip().lower()
    if strategy not in OCR_KEYFRAME_STRATEGIES:
        raise ValueError(f"Unknown OCR keyframe strategy: {value}")
    return strategy


def resolve_ocr_scan_sample_fps(value: str | float | int | None, pipeline_mode: str, video_duration_seconds: float) -> float:
    parsed = parse_auto_float(value)
    if isinstance(parsed, (float, int)):
        return max(float(parsed), 0.05)
    if video_duration_seconds >= MIN_LONG_VIDEO_SECONDS:
        return {"fast": 0.5, "balanced": 1.0, "deep": 2.0}.get(pipeline_mode, 1.0)
    return {"fast": 1.0, "balanced": 1.0, "deep": 2.0}.get(pipeline_mode, 1.0)


def resolve_ocr_keyframe_budget(
    video_duration_seconds: float,
    pipeline_mode: str,
    candidate_count: int,
    value: int | str = AUTO,
) -> int:
    if candidate_count <= 0:
        return 0
    parsed = parse_auto_int(value)
    if isinstance(parsed, int):
        return min(max(parsed, 1), candidate_count)

    minutes = max(video_duration_seconds, 1.0) / 60.0
    per_minute = {"fast": 2.0, "balanced": 4.0, "deep": 6.0}.get(pipeline_mode, 4.0)
    minimum = {"fast": 8, "balanced": 12, "deep": 18}.get(pipeline_mode, 12)
    ceiling = {"fast": 180, "balanced": 360, "deep": 720}.get(pipeline_mode, 360)
    budget = min(max(int(ceil(minutes * per_minute)), minimum), ceiling)
    return min(budget, candidate_count)


def resolve_ocr_keyframe_strategy(strategy: str, video_duration_seconds: float, candidate_count: int) -> str:
    strategy = parse_ocr_keyframe_strategy(strategy)
    if strategy != AUTO:
        return strategy
    if video_duration_seconds >= MIN_LONG_VIDEO_SECONDS or candidate_count > 24:
        return "scan-text"
    return "scan-text"


def image_textness(path: Path) -> TextnessFeatures:
    if cv2 is None or np is None:
        return TextnessFeatures(0.0, 0.0, 0.0, 0.0)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return TextnessFeatures(0.0, 0.0, 0.0, 0.0)

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > 640:
        scale = 640.0 / longest
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))

    sharpness = min(float(cv2.Laplacian(image, cv2.CV_64F).var()) / 1200.0, 1.0)
    edges = cv2.Canny(image, 80, 180)
    edge_density = min(float(np.count_nonzero(edges)) / float(edges.size) * 8.0, 1.0)

    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(connected, 8)

    plausible = 0
    area_total = float(connected.shape[0] * connected.shape[1])
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area <= 0:
            continue
        aspect = w / max(h, 1)
        area_ratio = area / area_total
        if 4 <= w <= connected.shape[1] * 0.95 and 4 <= h <= connected.shape[0] * 0.35 and 0.12 <= aspect <= 35 and area_ratio <= 0.20:
            plausible += 1

    component_score = min(plausible / 45.0, 1.0)
    textness = min((0.52 * component_score) + (0.30 * edge_density) + (0.18 * sharpness), 1.0)
    return TextnessFeatures(textness, sharpness, edge_density, component_score)


def select_ocr_keyframes(
    frames: List[Frame],
    transcript: Optional[AudioTranscript],
    video_duration_seconds: float,
    pipeline_mode: str,
    strategy: str = AUTO,
    budget: int | str = AUTO,
    scan_frames_count: int | None = None,
) -> tuple[List[Frame], list[OCRKeyframeDecision], dict[str, Any]]:
    resolved_strategy = resolve_ocr_keyframe_strategy(strategy, video_duration_seconds, len(frames))
    if not frames:
        return [], [], _metadata(resolved_strategy, strategy, budget, video_duration_seconds, 0, 0, 0, scan_frames_count, [])

    if resolved_strategy == "legacy":
        decisions = [
            OCRKeyframeDecision(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                selected_for_ocr=True,
                selection_score=1.0,
                reason="legacy_all_frames",
                skip_reason="",
                visual_change_score=frame.score,
                textness_score=0.0,
                sharpness_score=0.0,
                coverage_bucket=_coverage_bucket(frame.timestamp),
            )
            for frame in frames
        ]
        return frames, decisions, _metadata(resolved_strategy, strategy, budget, video_duration_seconds, len(frames), len(frames), len(frames), scan_frames_count, decisions)

    target = resolve_ocr_keyframe_budget(video_duration_seconds, pipeline_mode, len(frames), budget)
    scored = _score_frames(frames, transcript, video_duration_seconds)
    selected_numbers = _select_with_coverage(scored, target)
    decisions = [
        _decision(frame, score, frame.number in selected_numbers)
        for frame, score in scored
    ]
    selected = [frame for frame in frames if frame.number in selected_numbers]
    meta = _metadata(resolved_strategy, strategy, budget, video_duration_seconds, scan_frames_count or len(frames), len(frames), len(selected), scan_frames_count, decisions)
    return selected, decisions, meta


def build_ocr_text_events(ocr_events: Iterable[OCREvent], similarity_threshold: float = DEFAULT_TEXT_EVENT_SIMILARITY) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    last_event: dict[str, Any] | None = None
    for item in sorted(ocr_events, key=lambda event: (event.timestamp, event.frame_number)):
        normalized = normalize_ocr_text(item.text or "")
        if not normalized:
            continue
        if last_event and _similarity(last_event["normalized_text"], normalized) >= similarity_threshold:
            last_event["end_timestamp"] = item.timestamp
            last_event["frame_numbers"].append(item.frame_number)
            last_event["representative_frame_number"] = _choose_representative_frame(last_event, item)
            if len(normalized) > len(last_event["normalized_text"]):
                last_event["text"] = item.text
                last_event["normalized_text"] = normalized
            continue
        last_event = {
            "event_index": len(events),
            "start_timestamp": item.timestamp,
            "end_timestamp": item.timestamp,
            "frame_numbers": [item.frame_number],
            "representative_frame_number": item.frame_number,
            "text": item.text,
            "normalized_text": normalized,
            "status": item.status,
            "provider": item.provider,
        }
        events.append(last_event)
    for event in events:
        event["frame_count"] = len(event["frame_numbers"])
    return events


def normalize_ocr_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    normalized = re.sub(r"[|｜]+", " ", normalized)
    return normalized


def _score_frames(
    frames: List[Frame],
    transcript: Optional[AudioTranscript],
    video_duration_seconds: float,
) -> list[tuple[Frame, dict[str, Any]]]:
    asr_boundaries = _transcript_boundaries(transcript)
    scored = []
    for frame in frames:
        features = image_textness(frame.path)
        visual = min(max(frame.score, 0.0) / 30.0, 1.0)
        coverage = _coverage_bonus(frame.timestamp, video_duration_seconds)
        boundary = _boundary_bonus(frame.timestamp, asr_boundaries)
        score = (0.48 * features.textness_score) + (0.22 * visual) + (0.18 * coverage) + (0.12 * boundary)
        scored.append(
            (
                frame,
                {
                    "score": score,
                    "visual_change_score": frame.score,
                    "textness_score": features.textness_score,
                    "sharpness_score": features.sharpness_score,
                    "coverage_bucket": _coverage_bucket(frame.timestamp),
                },
            )
        )
    return scored


def _select_with_coverage(scored: list[tuple[Frame, dict[str, Any]]], budget: int) -> set[int]:
    if budget <= 0:
        return set()
    if len(scored) <= budget:
        return {frame.number for frame, _score in scored}

    selected_indexes: set[int] = {0, len(scored) - 1}
    bucket_first: dict[int, int] = {}
    for index, (_frame, score) in enumerate(scored):
        bucket_first.setdefault(int(score["coverage_bucket"]), index)
    selected_indexes.update(bucket_first.values())

    if len(selected_indexes) > budget:
        ranked_coverage = sorted(selected_indexes, key=lambda index: scored[index][1]["score"], reverse=True)
        selected_indexes = set(sorted(ranked_coverage[:budget]))
    remaining = max(budget - len(selected_indexes), 0)
    ranked = sorted(
        (index for index in range(len(scored)) if index not in selected_indexes),
        key=lambda index: scored[index][1]["score"],
        reverse=True,
    )
    selected_indexes.update(ranked[:remaining])
    return {scored[index][0].number for index in sorted(selected_indexes)[:budget]}


def _decision(frame: Frame, score: dict[str, Any], selected: bool) -> OCRKeyframeDecision:
    reason = "selected_by_textness_change_and_coverage" if selected else ""
    return OCRKeyframeDecision(
        frame_number=frame.number,
        timestamp=frame.timestamp,
        selected_for_ocr=selected,
        selection_score=float(score["score"]),
        reason=reason,
        skip_reason="" if selected else "below_ocr_keyframe_budget",
        visual_change_score=float(score["visual_change_score"]),
        textness_score=float(score["textness_score"]),
        sharpness_score=float(score["sharpness_score"]),
        coverage_bucket=int(score["coverage_bucket"]),
    )


def _metadata(
    resolved_strategy: str,
    requested_strategy: str,
    requested_budget: int | str,
    video_duration_seconds: float,
    scan_count: int,
    candidate_count: int,
    selected_count: int,
    scan_frames_count: int | None,
    decisions: Iterable[OCRKeyframeDecision],
) -> dict[str, Any]:
    return {
        "strategy": requested_strategy,
        "strategy_resolved": resolved_strategy,
        "video_duration_seconds": video_duration_seconds,
        "scan_frames_count": scan_frames_count if scan_frames_count is not None else scan_count,
        "ocr_candidate_frames_count": candidate_count,
        "ocr_frames_count": selected_count,
        "ocr_keyframe_budget": requested_budget,
        "frames": [decision.to_dict() for decision in decisions],
    }


def _coverage_bucket(timestamp: float, bucket_seconds: float = 60.0) -> int:
    return int(max(timestamp, 0.0) // bucket_seconds)


def _coverage_bonus(timestamp: float, video_duration_seconds: float) -> float:
    if video_duration_seconds <= 0:
        return 0.0
    position = timestamp / video_duration_seconds
    return 1.0 - min(abs(position - 0.5) * 2.0, 1.0)


def _transcript_boundaries(transcript: Optional[AudioTranscript]) -> list[float]:
    if not transcript:
        return []
    boundaries = []
    for segment in transcript.segments or []:
        for key in ("start", "end"):
            value = segment.get(key)
            if isinstance(value, (int, float)):
                boundaries.append(float(value))
    return sorted(boundaries)


def _boundary_bonus(timestamp: float, boundaries: list[float], window_seconds: float = 5.0) -> float:
    if not boundaries:
        return 0.0
    nearest = min(abs(timestamp - boundary) for boundary in boundaries)
    if nearest > window_seconds:
        return 0.0
    return 1.0 - (nearest / window_seconds)


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _choose_representative_frame(event: dict[str, Any], item: OCREvent) -> int:
    current_text = event.get("normalized_text") or ""
    if len(normalize_ocr_text(item.text or "")) > len(current_text):
        return item.frame_number
    return int(event.get("representative_frame_number") or item.frame_number)
