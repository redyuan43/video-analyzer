from pathlib import Path

from video_analyzer.audio_processor import AudioTranscript
from video_analyzer.speaker_diarization import (
    SpeakerEstimate,
    apply_speaker_merge_map,
    build_speaker_merge_map,
    choose_target_speaker_count,
    select_dense_windows,
)


def test_choose_target_prefers_consensus_count():
    estimates = [
        SpeakerEstimate("ecapa", 2, confidence=0.3),
        SpeakerEstimate("3dspeaker", 2, confidence=0.7),
        SpeakerEstimate("resemblyzer", 3, confidence=0.4),
    ]

    assert choose_target_speaker_count(4, estimates) == 2


def test_choose_target_keeps_current_on_tie_with_current_count():
    estimates = [
        SpeakerEstimate("ecapa", 2, confidence=0.3),
        SpeakerEstimate("3dspeaker", 3, confidence=0.7),
    ]

    assert choose_target_speaker_count(3, estimates) == 3


def test_choose_target_keeps_current_when_estimators_disagree():
    estimates = [
        SpeakerEstimate("ecapa", 3, confidence=0.3),
        SpeakerEstimate("3dspeaker", 2, confidence=0.7),
    ]

    assert choose_target_speaker_count(4, estimates) == 4


def test_build_merge_map_uses_vibevoice_audit_edges():
    transcript = AudioTranscript(
        text="",
        language="zh",
        segments=[
            {"Start": 0, "End": 1, "Speaker": "Speaker A", "Content": "甲"},
            {"Start": 1, "End": 2, "Speaker": "Speaker B", "Content": "乙"},
            {"Start": 2, "End": 3, "Speaker": "Speaker C", "Content": "甲继续"},
            {"Start": 3, "End": 4, "Speaker": "Speaker D", "Content": "乙继续"},
        ],
        metadata={
            "audit_chunks": {
                "speaker_assignments": [
                    {"chunk_index": 0, "local_speaker": "A", "global_speaker": "Speaker A"},
                    {"chunk_index": 0, "local_speaker": "B", "global_speaker": "Speaker B"},
                    {"chunk_index": 1, "local_speaker": "A", "global_speaker": "Speaker C"},
                    {"chunk_index": 1, "local_speaker": "B", "global_speaker": "Speaker D"},
                ],
                "global_speaker_edges": [
                    {"left": "0:A", "right": "1:A", "score": 0.93},
                    {"left": "0:B", "right": "1:B", "score": 0.91},
                ],
            }
        },
    )

    merge_map, notes = build_speaker_merge_map(transcript, target_speaker_count=2, min_score=0.78)

    assert merge_map
    assert len(set(merge_map.values())) <= 2
    assert any("merged" in note for note in notes)


def test_apply_speaker_merge_map_updates_both_speaker_fields():
    transcript = AudioTranscript(
        text="hello",
        language="zh",
        segments=[
            {"Speaker": "Speaker A", "speaker_id": "Speaker A", "Content": "a"},
            {"Speaker": "Speaker C", "speaker_id": "Speaker C", "Content": "c"},
        ],
        metadata={},
    )

    refined = apply_speaker_merge_map(transcript, {"Speaker C": "Speaker A"})

    assert refined.segments[1]["Speaker"] == "Speaker A"
    assert refined.segments[1]["speaker_id"] == "Speaker A"
    assert refined.segments[1]["speaker_merge_source"] == "Speaker C"


def test_select_dense_windows_prefers_multi_speaker_dense_region():
    transcript = AudioTranscript(
        text="",
        language="zh",
        segments=[
            {"Start": 0, "End": 5, "Speaker": "Speaker A", "Content": "开场"},
            {"Start": 120, "End": 130, "Speaker": "Speaker A", "Content": "很多内容" * 10},
            {"Start": 132, "End": 140, "Speaker": "Speaker B", "Content": "回应" * 10},
            {"Start": 142, "End": 150, "Speaker": "Speaker C", "Content": "第三人" * 10},
        ],
        metadata={},
    )

    windows = select_dense_windows(transcript, duration=300, sample_seconds=90, max_windows=1)

    assert windows == [120.0]
