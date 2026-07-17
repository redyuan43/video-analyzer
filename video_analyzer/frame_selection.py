from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from statistics import median
from typing import Any, Iterable, List, Optional

from .audio_processor import AudioTranscript
from .frame import Frame
from .ocr import OCREvent

AUTO = "auto"
PIPELINE_MODES = {"fast", "balanced", "deep"}
VL_FRAME_POLICIES = {"auto", "all", "none"}
MIN_VL_CONTEXT_GAP_SECONDS = 8.0
MAX_VL_CONTEXT_GAP_SECONDS = 45.0
VL_CONTEXT_GAP_MULTIPLIER = 3.0
DEFAULT_VL_TARGET_SECONDS = 45 * 60
DEFAULT_VL_SECONDS_PER_FRAME = 30.0
VL_TIME_SAFETY_FACTOR = 1.10


@dataclass(frozen=True)
class FrameSelectionOptions:
    pipeline_mode: str = "balanced"
    candidate_frames: int | str = AUTO
    min_vl_frames: int | str = AUTO
    max_vl_frames: int | str = AUTO
    vl_frame_policy: str = AUTO
    explicit_max_frames: Optional[int] = None
    vl_target_seconds: float = DEFAULT_VL_TARGET_SECONDS
    vl_seconds_per_frame: float = DEFAULT_VL_SECONDS_PER_FRAME


@dataclass(frozen=True)
class FrameDecision:
    frame_number: int
    timestamp: float
    selected_for_vl: bool
    selection_score: float
    reason: str
    skip_reason: str
    ocr_status: str
    ocr_chars: int
    ocr_summary: str
    visual_change_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "selected_for_vl": self.selected_for_vl,
            "selection_score": round(self.selection_score, 4),
            "reason": self.reason,
            "skip_reason": self.skip_reason,
            "ocr_status": self.ocr_status,
            "ocr_chars": self.ocr_chars,
            "ocr_summary": self.ocr_summary,
            "visual_change_score": round(self.visual_change_score, 4),
        }


@dataclass(frozen=True)
class FrameContextItem:
    frame: Frame
    role: str
    gap_seconds: float


def parse_auto_float(value: str | float | int | None) -> float | str | None:
    if value is None or isinstance(value, (float, int)):
        return value
    if str(value).strip().lower() == AUTO:
        return AUTO
    parsed = float(value)
    if parsed < 0:
        raise ValueError("value must be auto or a non-negative number")
    return parsed


def parse_auto_int(value: str | int | None) -> int | str | None:
    if value is None or isinstance(value, int):
        return value
    if str(value).strip().lower() == AUTO:
        return AUTO
    parsed = int(value)
    if parsed < 0:
        raise ValueError("value must be auto or a non-negative integer")
    return parsed


def resolve_vl_context_gap_seconds(frames: List[Frame], value: float | str = AUTO) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    timestamps = sorted(frame.timestamp for frame in frames)
    gaps = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:])
        if later > earlier
    ]
    if not gaps:
        return MIN_VL_CONTEXT_GAP_SECONDS
    return min(
        MAX_VL_CONTEXT_GAP_SECONDS,
        max(MIN_VL_CONTEXT_GAP_SECONDS, median(gaps) * VL_CONTEXT_GAP_MULTIPLIER),
    )


def build_frame_context_window(
    frames: List[Frame],
    current_frame: Frame,
    before: int = 3,
    after: int = 2,
    max_gap_seconds: float | str = AUTO,
) -> list[FrameContextItem]:
    if not frames:
        return []
    ordered = sorted(frames, key=lambda frame: frame.timestamp)
    index_by_number = {frame.number: index for index, frame in enumerate(ordered)}
    current_index = index_by_number.get(current_frame.number)
    if current_index is None:
        return [FrameContextItem(current_frame, "current", 0.0)]

    gap_limit = resolve_vl_context_gap_seconds(ordered, max_gap_seconds)
    selected_indexes = {current_index}

    previous_index = current_index
    for index in range(current_index - 1, -1, -1):
        if len([item for item in selected_indexes if item < current_index]) >= max(before, 0):
            break
        if ordered[previous_index].timestamp - ordered[index].timestamp > gap_limit:
            break
        selected_indexes.add(index)
        previous_index = index

    previous_index = current_index
    for index in range(current_index + 1, len(ordered)):
        if len([item for item in selected_indexes if item > current_index]) >= max(after, 0):
            break
        if ordered[index].timestamp - ordered[previous_index].timestamp > gap_limit:
            break
        selected_indexes.add(index)
        previous_index = index

    context = []
    for index in sorted(selected_indexes):
        frame = ordered[index]
        role = "current"
        if index < current_index:
            role = "previous"
        elif index > current_index:
            role = "next"
        context.append(
            FrameContextItem(
                frame=frame,
                role=role,
                gap_seconds=abs(frame.timestamp - current_frame.timestamp),
            )
        )
    return context


def resolve_candidate_frame_budget(
    video_duration_seconds: float,
    pipeline_mode: str,
    candidate_frames: int | str = AUTO,
    explicit_max_frames: Optional[int] = None,
) -> int:
    if isinstance(candidate_frames, int):
        budget = max(candidate_frames, 1)
    else:
        minutes = max(video_duration_seconds, 1.0) / 60.0
        per_minute = {"fast": 4.0, "balanced": 6.0, "deep": 8.0}.get(pipeline_mode, 6.0)
        minimum = {"fast": 8, "balanced": 12, "deep": 16}.get(pipeline_mode, 12)
        ceiling = {"fast": 240, "balanced": 720, "deep": 1200}.get(pipeline_mode, 720)
        budget = min(max(int(ceil(minutes * per_minute)), minimum), ceiling)
    if explicit_max_frames is not None:
        budget = min(budget, max(explicit_max_frames, 1))
    return budget


def select_vl_frames(
    frames: List[Frame],
    ocr_events: List[OCREvent],
    transcript: Optional[AudioTranscript],
    video_duration_seconds: float,
    options: FrameSelectionOptions,
) -> tuple[set[int], list[FrameDecision], dict[str, Any]]:
    if not frames:
        return set(), [], _metadata(options, video_duration_seconds, 0, 0, 0, [])

    policy = options.vl_frame_policy
    if policy == AUTO:
        policy = "none" if options.pipeline_mode == "fast" else "auto"

    scored = _score_frames(frames, ocr_events, transcript, video_duration_seconds)
    if policy == "none":
        selected_numbers: set[int] = set()
        budget_details = {
            "vl_quality_budget": 0,
            "vl_time_capacity": 0,
            "vl_budget_resolved": 0,
            "vl_projected_seconds": 0.0,
        }
    elif policy == "all":
        selected_numbers = {frame.number for frame, _ in scored}
        seconds_per_frame = max(float(options.vl_seconds_per_frame or DEFAULT_VL_SECONDS_PER_FRAME), 0.1)
        target_seconds = max(float(options.vl_target_seconds or DEFAULT_VL_TARGET_SECONDS), 0.0)
        time_capacity = max(1, floor(target_seconds / (seconds_per_frame * VL_TIME_SAFETY_FACTOR)))
        budget_details = {
            "vl_quality_budget": len(selected_numbers),
            "vl_time_capacity": min(time_capacity, len(selected_numbers)),
            "vl_budget_resolved": len(selected_numbers),
            "vl_projected_seconds": round(len(selected_numbers) * seconds_per_frame, 3),
        }
    else:
        budget_details = resolve_vl_frame_budget_details(
            frames=frames,
            ocr_events=ocr_events,
            transcript=transcript,
            video_duration_seconds=video_duration_seconds,
            options=options,
        )
        selected_numbers = _select_with_coverage(scored, int(budget_details["vl_budget_resolved"]))

    decisions = [
        _build_decision(frame, score, frame.number in selected_numbers, policy)
        for frame, score in scored
    ]
    meta = _metadata(
        options=options,
        video_duration_seconds=video_duration_seconds,
        candidate_count=len(frames),
        ocr_count=len(ocr_events),
        vl_count=len(selected_numbers),
        decisions=decisions,
    )
    meta["vl_frame_policy_resolved"] = policy
    meta.update(budget_details)
    meta["vl_time_target_seconds"] = round(max(options.vl_target_seconds, 0.0), 3)
    meta["vl_seconds_per_frame_estimate"] = round(max(options.vl_seconds_per_frame, 0.0), 3)
    meta["vl_time_target_bypassed"] = policy == "all"
    return selected_numbers, decisions, meta


def resolve_vl_frame_budget(
    frames: List[Frame],
    ocr_events: List[OCREvent],
    transcript: Optional[AudioTranscript],
    video_duration_seconds: float,
    options: FrameSelectionOptions,
) -> int:
    return int(
        resolve_vl_frame_budget_details(
            frames=frames,
            ocr_events=ocr_events,
            transcript=transcript,
            video_duration_seconds=video_duration_seconds,
            options=options,
        )["vl_budget_resolved"]
    )


def resolve_vl_frame_budget_details(
    frames: List[Frame],
    ocr_events: List[OCREvent],
    transcript: Optional[AudioTranscript],
    video_duration_seconds: float,
    options: FrameSelectionOptions,
) -> dict[str, float | int]:
    candidate_count = len(frames)
    if candidate_count == 0:
        return {
            "vl_quality_budget": 0,
            "vl_time_capacity": 0,
            "vl_budget_resolved": 0,
            "vl_projected_seconds": 0.0,
        }

    min_budget = _resolve_auto_vl_min(
        video_duration_seconds,
        candidate_count,
        options.min_vl_frames,
        options.pipeline_mode,
    )
    max_budget = _resolve_auto_vl_max(
        video_duration_seconds,
        candidate_count,
        options.max_vl_frames,
        options.pipeline_mode,
    )
    if min_budget > max_budget:
        min_budget = max_budget

    density = _combined_density(ocr_events, transcript, video_duration_seconds)
    ratio = 0.14 + (0.32 * density)
    if options.pipeline_mode == "deep":
        ratio *= 1.25
    quality_budget = min(max(int(ceil(candidate_count * ratio)), min_budget), max_budget, candidate_count)

    seconds_per_frame = max(float(options.vl_seconds_per_frame or DEFAULT_VL_SECONDS_PER_FRAME), 0.1)
    target_seconds = max(float(options.vl_target_seconds or DEFAULT_VL_TARGET_SECONDS), 0.0)
    time_capacity = max(1, floor(target_seconds / (seconds_per_frame * VL_TIME_SAFETY_FACTOR)))
    resolved = min(quality_budget, time_capacity, candidate_count)
    resolved = max(min(resolved, candidate_count), min(min_budget, candidate_count))
    if options.explicit_max_frames is not None:
        resolved = min(resolved, max(int(options.explicit_max_frames), 0))
    return {
        "vl_quality_budget": quality_budget,
        "vl_time_capacity": min(time_capacity, candidate_count),
        "vl_budget_resolved": resolved,
        "vl_projected_seconds": round(resolved * seconds_per_frame, 3),
    }


def make_skipped_visual_event(frame: Frame, decision: FrameDecision) -> dict[str, Any]:
    return {
        "frame_number": frame.number,
        "timestamp": frame.timestamp,
        "status": "skipped",
        "skip_reason": decision.skip_reason,
        "ocr_summary": decision.ocr_summary,
        "selection_score": round(decision.selection_score, 4),
        "response": (
            "VL analysis skipped. "
            f"OCR summary: {decision.ocr_summary or 'no OCR text'}"
        ),
    }


def _score_frames(
    frames: List[Frame],
    ocr_events: List[OCREvent],
    transcript: Optional[AudioTranscript],
    video_duration_seconds: float,
) -> list[tuple[Frame, dict[str, Any]]]:
    ocr_by_frame = {event.frame_number: event for event in ocr_events}
    chapter_density = _asr_density(transcript, video_duration_seconds)
    scored = []
    for frame in frames:
        ocr = ocr_by_frame.get(frame.number)
        ocr_text = ocr.text if ocr and ocr.text else ""
        ocr_chars = len(ocr_text.strip())
        ocr_density = min(ocr_chars / 500.0, 1.0)
        visual_density = min(max(frame.score, 0.0) / 30.0, 1.0)
        coverage_bonus = _coverage_bonus(frame, video_duration_seconds)
        score = (0.45 * ocr_density) + (0.30 * visual_density) + (0.15 * chapter_density) + (0.10 * coverage_bonus)
        scored.append(
            (
                frame,
                {
                    "score": score,
                    "ocr_status": ocr.status if ocr else "not_run",
                    "ocr_chars": ocr_chars,
                    "ocr_summary": _summarize_ocr(ocr_text),
                    "visual_change_score": frame.score,
                },
            )
        )
    return scored


def _select_with_coverage(scored: list[tuple[Frame, dict[str, Any]]], budget: int) -> set[int]:
    if budget <= 0:
        return set()
    if len(scored) <= budget:
        return {frame.number for frame, _ in scored}

    selected_indexes = {0, len(scored) - 1}
    coverage_slots = min(budget, max(2, int(ceil(budget * 0.60))))
    for slot in range(coverage_slots):
        start = floor(slot * len(scored) / coverage_slots)
        end = max(start + 1, floor((slot + 1) * len(scored) / coverage_slots))
        window = range(start, min(end, len(scored)))
        selected_indexes.add(max(window, key=lambda idx: scored[idx][1]["score"]))

    remaining = max(budget - len(selected_indexes), 0)
    ranked = sorted(
        (idx for idx in range(len(scored)) if idx not in selected_indexes),
        key=lambda idx: scored[idx][1]["score"],
        reverse=True,
    )
    selected_indexes.update(ranked[:remaining])
    return {scored[idx][0].number for idx in sorted(selected_indexes)[:budget]}


def _build_decision(frame: Frame, score: dict[str, Any], selected: bool, policy: str) -> FrameDecision:
    reason = "selected_by_policy_all" if policy == "all" else "selected_by_score_and_coverage"
    skip_reason = "" if selected else "pipeline_fast" if policy == "none" else "below_vl_budget"
    return FrameDecision(
        frame_number=frame.number,
        timestamp=frame.timestamp,
        selected_for_vl=selected,
        selection_score=float(score["score"]),
        reason=reason if selected else "",
        skip_reason=skip_reason,
        ocr_status=str(score["ocr_status"]),
        ocr_chars=int(score["ocr_chars"]),
        ocr_summary=str(score["ocr_summary"]),
        visual_change_score=float(score["visual_change_score"]),
    )


def _metadata(
    options: FrameSelectionOptions,
    video_duration_seconds: float,
    candidate_count: int,
    ocr_count: int,
    vl_count: int,
    decisions: Iterable[FrameDecision],
) -> dict[str, Any]:
    return {
        "pipeline_mode": options.pipeline_mode,
        "video_duration_seconds": video_duration_seconds,
        "candidate_frames": options.candidate_frames,
        "candidate_frames_count": candidate_count,
        "ocr_frames_count": ocr_count,
        "vl_frames_count": vl_count,
        "min_vl_frames": options.min_vl_frames,
        "max_vl_frames": options.max_vl_frames,
        "vl_frame_policy": options.vl_frame_policy,
        "explicit_max_frames": options.explicit_max_frames,
        "frames": [decision.to_dict() for decision in decisions],
    }


def _resolve_auto_vl_min(
    video_duration_seconds: float,
    candidate_count: int,
    value: int | str,
    pipeline_mode: str = "balanced",
) -> int:
    if isinstance(value, int):
        return min(value, candidate_count)
    minutes = video_duration_seconds / 60.0
    if minutes <= 5:
        budget = 4
    elif minutes <= 30:
        budget = 8
    else:
        budget = 12
    if pipeline_mode == "deep":
        budget = int(ceil(budget * 1.5))
    return min(budget, candidate_count)


def _resolve_auto_vl_max(
    video_duration_seconds: float,
    candidate_count: int,
    value: int | str,
    pipeline_mode: str = "balanced",
) -> int:
    if isinstance(value, int):
        return min(max(value, 0), candidate_count)
    minutes = video_duration_seconds / 60.0
    if minutes <= 5:
        budget = 18
    elif minutes <= 30:
        budget = 72
    else:
        budget = min(72 + int(ceil((minutes - 30) * 3)), 300)
    if pipeline_mode == "deep":
        budget = min(int(ceil(budget * 1.5)), 300)
    return min(budget, candidate_count)


def _combined_density(
    ocr_events: List[OCREvent],
    transcript: Optional[AudioTranscript],
    video_duration_seconds: float,
) -> float:
    ocr_density = 0.0
    if ocr_events:
        avg_chars = sum(len((event.text or "").strip()) for event in ocr_events) / len(ocr_events)
        ocr_density = min(avg_chars / 350.0, 1.0)
    return min((0.65 * ocr_density) + (0.35 * _asr_density(transcript, video_duration_seconds)), 1.0)


def _asr_density(transcript: Optional[AudioTranscript], video_duration_seconds: float) -> float:
    if not transcript or not transcript.segments or video_duration_seconds <= 0:
        return 0.0
    segments_per_minute = len(transcript.segments) / max(video_duration_seconds / 60.0, 1.0)
    return min(segments_per_minute / 8.0, 1.0)


def _coverage_bonus(frame: Frame, video_duration_seconds: float) -> float:
    if video_duration_seconds <= 0:
        return 0.0
    position = frame.timestamp / video_duration_seconds
    return 1.0 - min(abs(position - 0.5) * 2.0, 1.0)


def _summarize_ocr(text: str, max_chars: int = 160) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."
