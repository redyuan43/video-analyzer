from pathlib import Path
from unittest.mock import patch

import requests

from video_analyzer.audio_processor import AudioTranscript
from video_analyzer.speaker_diarization import (
    SpeakerEstimate,
    apply_speaker_merge_map,
    assign_speakers_by_overlap,
    build_speaker_merge_map,
    choose_target_speaker_count,
    process_transcript_speakers,
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


def test_assign_speakers_by_overlap_labels_unlabeled_asr_segments():
    transcript = AudioTranscript(
        text="甲乙",
        language="zh",
        segments=[
            {"start": 0.0, "end": 2.0, "text": "甲"},
            {"start": 2.0, "end": 4.0, "text": "乙"},
            {"start": 4.0, "end": 5.0, "text": "无匹配"},
        ],
        metadata={},
    )

    assigned, report = assign_speakers_by_overlap(
        transcript,
        [
            {"start": 0.0, "end": 2.1, "speaker": "spk_01"},
            {"start": 2.1, "end": 4.0, "speaker": "spk_02"},
        ],
    )

    assert assigned.segments[0]["speaker"] == "说话人 1"
    assert assigned.segments[1]["speaker"] == "说话人 2"
    assert "speaker" not in assigned.segments[2]
    assert report["assigned_segment_count"] == 2
    assert report["unassigned_segment_count"] == 1
    assert report["final_speaker_count"] == 2


def test_3dspeaker_assignment_uses_native_helper_inside_branch_actor(tmp_path):
    from video_analyzer import speaker_diarization

    external_python = tmp_path / "python"
    external_python.touch()
    project_root = tmp_path / "3D-Speaker"
    project_root.mkdir()
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    with patch.object(speaker_diarization.subprocess, "run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = '__3DSPEAKER_JSON__{"turns":[]}'
        run.return_value.stderr = ""
        speaker_diarization.run_3dspeaker_assignment(
            audio_path,
            {
                "external_python": str(external_python),
                "diarization_project_root": str(project_root),
            },
        )

    command = run.call_args.args[0]
    assert command[1].endswith("run_3dspeaker_turns.py")


def test_remote_3dspeaker_streams_audio_with_token(tmp_path, monkeypatch):
    from video_analyzer import speaker_diarization

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "turns": [
                    {
                        "speaker": "spk_01",
                        "start": 0.0,
                        "end": 1.0,
                        "duration": 1.0,
                    }
                ],
                "turn_count": 1,
                "detected_speakers": ["spk_01"],
                "detected_speaker_count": 1,
                "node": "nano1",
                "device": "cuda",
                "elapsed_seconds": 0.5,
            }

    class Session:
        trust_env = True

        def post(self, endpoint, **kwargs):
            assert endpoint == "http://nano1:5021/api/diarization/turns"
            assert kwargs["headers"]["X-Audio-Diarization-Token"] == "secret"
            assert kwargs["data"].read() == b"audio"
            return Response()

        def close(self):
            return None

    monkeypatch.setenv("NANO_DIARIZATION_TOKEN", "secret")
    monkeypatch.setattr(requests, "Session", Session)
    turns, report = speaker_diarization.run_remote_3dspeaker_assignment(
        audio_path,
        {
            "endpoint": "http://nano1:5021/api/diarization/turns",
            "token_env": "NANO_DIARIZATION_TOKEN",
        },
    )

    assert len(turns) == 1
    assert report["node"] == "nano1"
    assert report["device"] == "cuda"
    assert report["route"] == "nano_gpu"


def test_remote_3dspeaker_falls_back_to_ai_local(tmp_path):
    from video_analyzer import speaker_diarization

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    with (
        patch.object(
            speaker_diarization,
            "run_remote_3dspeaker_assignment",
            return_value=([], {"error": "nano busy"}),
        ),
        patch.object(
            speaker_diarization,
            "run_3dspeaker_assignment",
            return_value=(
                [{"speaker": "spk_01", "start": 0.0, "end": 1.0}],
                {"backend": "3dspeaker"},
            ),
        ),
    ):
        turns, report = speaker_diarization.run_diarization_assignment(
            audio_path,
            {
                "backend": "remote_3dspeaker_http",
                "fallback_backend": "3dspeaker",
            },
        )

    assert len(turns) == 1
    assert report["route"] == "ai_local_fallback"
    assert report["fallback_reason"] == "nano busy"


def test_prepared_assignment_is_reused_after_parallel_asr():
    transcript = AudioTranscript(
        text="hello",
        language="en",
        segments=[{"start": 0.0, "end": 1.0, "text": "hello"}],
        metadata={},
    )
    prepared = (
        [{"start": 0.0, "end": 1.0, "speaker": "spk_01"}],
        {
            "enabled": True,
            "mode": "assignment",
            "backend": "wespeaker",
            "original_speaker_count": 0,
            "original_speakers": [],
            "notes": [],
        },
    )

    with patch(
        "video_analyzer.speaker_diarization.run_diarization_assignment"
    ) as backend:
        assigned, report = process_transcript_speakers(
            Path("/tmp/audio.wav"),
            transcript,
            {"enabled": True, "backend": "wespeaker"},
            prepared_assignment=prepared,
        )

    backend.assert_not_called()
    assert assigned.segments[0]["speaker"] == "说话人 1"
    assert report["final_speaker_count"] == 1


def test_prepared_assignment_overrides_existing_vibevoice_speakers():
    transcript = AudioTranscript(
        text="hello",
        language="en",
        segments=[
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "Speaker A",
                "text": "hello",
            }
        ],
        metadata={},
    )
    prepared = (
        [{"start": 0.0, "end": 1.0, "speaker": "spk_02"}],
        {"enabled": True, "mode": "assignment", "backend": "3dspeaker", "notes": []},
    )

    assigned, report = process_transcript_speakers(
        Path("/tmp/audio.wav"),
        transcript,
        {"enabled": True, "backend": "3dspeaker"},
        prepared_assignment=prepared,
    )

    assert assigned.segments[0]["speaker"] == "说话人 2"
    assert report["final_speaker_count"] == 1


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
