"""Hybrid speaker diarization refinement for audio transcripts."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .audio_processor import AudioTranscript

logger = logging.getLogger(__name__)

DEFAULT_EXTERNAL_PYTHON = "/home/ai/diarization-ab-venv/bin/python"
DEFAULT_3D_SPEAKER_ROOT = "/tmp/3D-Speaker"
DEFAULT_SAMPLE_SECONDS = 90.0
DEFAULT_MAX_WINDOWS = 2
DEFAULT_TIMEOUT_SECONDS = 240.0
DEFAULT_MERGE_MIN_SCORE = 0.78
DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class SpeakerEstimate:
    source: str
    speaker_count: int | None
    confidence: float = 0.0
    elapsed_seconds: float = 0.0
    detail: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "speaker_count": self.speaker_count,
            "confidence": self.confidence,
            "elapsed_seconds": self.elapsed_seconds,
            "detail": self.detail,
            "skipped": self.skipped,
        }


def process_transcript_speakers(
    audio_path: Path,
    transcript: AudioTranscript,
    config: dict[str, Any] | None = None,
) -> tuple[AudioTranscript, dict[str, Any]]:
    """Assign missing speaker labels, or refine labels supplied by the ASR."""
    config = config or {}
    if not _truthy(config.get("enabled"), default=True):
        return transcript, {"enabled": False, "reason": "disabled"}

    current_speakers = _speaker_ids(transcript.segments or [])
    if current_speakers:
        return refine_transcript_speakers(audio_path, transcript, config)

    backend = str(config.get("backend") or "3dspeaker")
    report: dict[str, Any] = {
        "enabled": True,
        "mode": "assignment",
        "backend": backend,
        "original_speaker_count": 0,
        "original_speakers": [],
        "notes": [],
    }
    if not _truthy(config.get("assignment_enabled"), default=True):
        report["notes"].append("speaker assignment disabled")
        transcript.metadata = _with_hybrid_metadata(transcript.metadata, report)
        return transcript, report

    turns, assignment_report = run_diarization_assignment(audio_path, config)
    report.update(assignment_report)
    if not turns:
        report["notes"].append("no diarization turns produced")
        transcript.metadata = _with_hybrid_metadata(transcript.metadata, report)
        return transcript, report

    assigned, assignment_stats = assign_speakers_by_overlap(transcript, turns)
    report.update(assignment_stats)
    assigned.metadata = _with_hybrid_metadata(assigned.metadata, report)
    return assigned, report


def run_diarization_assignment(
    audio_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = config or {}
    backend = str(config.get("backend") or "3dspeaker")
    if backend == "3dspeaker":
        return run_3dspeaker_assignment(audio_path, config)
    if backend == "pyannote_community":
        return run_pyannote_assignment(audio_path, config)
    if backend == "wespeaker":
        return run_wespeaker_assignment(audio_path, config)
    return [], {"backend": backend, "error": f"unknown diarization backend: {backend}"}


def _run_assignment_command(
    command: list[str],
    backend: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", str(env.get("DIARIZATION_GPU_ID") or "0"))
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return [], {
            "backend": backend,
            "error": f"{backend} assignment timed out",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    report = {
        "backend": backend,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if completed.returncode != 0:
        report["error"] = (completed.stderr or completed.stdout or f"{backend} failed")[-3000:]
        return [], report
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        report["error"] = f"invalid {backend} output: {completed.stdout[-1500:]}"
        return [], report
    turns = [
        {
            "start": float(turn.get("start", turn.get("start_sec", 0))),
            "end": float(turn.get("end", turn.get("end_sec", 0))),
            "speaker": str(turn.get("speaker") or turn.get("speaker_id") or ""),
        }
        for turn in (payload.get("turns") or [])
        if isinstance(turn, dict)
    ]
    turns = [turn for turn in turns if turn["speaker"] and turn["end"] > turn["start"]]
    speakers = sorted({turn["speaker"] for turn in turns})
    report.update(
        {
            "turn_count": len(turns),
            "detected_speakers": speakers,
            "detected_speaker_count": len(speakers),
        }
    )
    return turns, report


def run_pyannote_assignment(
    audio_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = config or {}
    python = str(config.get("external_python") or "/home/ai/pyannote-community-venv/bin/python")
    helper = Path(__file__).resolve().parents[1] / "tools" / "run_diarization_benchmark.py"
    with tempfile.TemporaryDirectory(prefix="pyannote_assignment_") as temp:
        output = Path(temp) / "result.json"
        command = [
            python,
            str(helper),
            str(audio_path),
            "--provider",
            "pyannote_community",
            "--output",
            str(output),
            "--device",
            str(config.get("device") or "cpu"),
        ]
        speaker_num = config.get("speaker_num")
        if speaker_num not in (None, "", 0, "0"):
            command.extend(["--speaker-num", str(int(speaker_num))])
        turns, report = _run_assignment_command(
            command,
            "pyannote_community",
            float(config.get("assignment_timeout_seconds") or DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS),
        )
        if not turns and output.is_file():
            payload = json.loads(output.read_text(encoding="utf-8"))
            turns = list(payload.get("turns") or [])
            speakers = sorted(
                {
                    str(turn.get("speaker") or turn.get("speaker_id") or "")
                    for turn in turns
                    if isinstance(turn, dict)
                    and (turn.get("speaker") or turn.get("speaker_id"))
                }
            )
            report.update(
                {
                    "turn_count": len(turns),
                    "detected_speakers": speakers,
                    "detected_speaker_count": len(speakers),
                }
            )
        return turns, report


def run_wespeaker_assignment(
    audio_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = config or {}
    python = str(config.get("external_python") or DEFAULT_EXTERNAL_PYTHON)
    if python == "/home/ai/wespeaker-venv/bin/python" and not Path(python).exists():
        python = DEFAULT_EXTERNAL_PYTHON
    helper = Path(__file__).resolve().parents[1] / "tools" / "run_wespeaker_diarization.py"
    command = [
        python,
        str(helper),
        str(audio_path),
        "--model",
        str(config.get("model_id") or "chinese"),
        "--device",
        str(config.get("device") or "cuda"),
    ]
    speaker_num = config.get("speaker_num")
    if speaker_num not in (None, "", 0, "0"):
        command.extend(["--speaker-num", str(int(speaker_num))])
    return _run_assignment_command(
        command,
        "wespeaker",
        float(config.get("assignment_timeout_seconds") or DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS),
    )


def run_3dspeaker_assignment(
    audio_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = config or {}
    external_python = Path(
        str(
            config.get("external_python")
            or os.environ.get("VIDEO_ANALYZER_DIARIZATION_PYTHON")
            or DEFAULT_EXTERNAL_PYTHON
        )
    )
    project_root_value = (
        config.get("diarization_project_root")
        or os.environ.get("VIDEO_ANALYZER_DIARIZATION_ROOT")
    )
    report: dict[str, Any] = {
        "external_python": str(external_python),
        "diarization_project_root": str(project_root_value or ""),
    }
    if not external_python.is_file():
        report["error"] = f"missing external python: {external_python}"
        return [], report
    if not project_root_value:
        report["error"] = "missing diarization_project_root"
        return [], report

    project_root = Path(str(project_root_value)).expanduser().resolve()
    helper_path = Path(__file__).resolve().parents[1] / "tools" / "run_3dspeaker_turns.py"
    if not project_root.is_dir():
        report["error"] = f"missing diarization project root: {project_root}"
        return [], report
    if not helper_path.is_file():
        report["error"] = f"missing diarization helper: {helper_path}"
        return [], report

    command = [
        str(external_python),
        str(helper_path),
        str(audio_path),
        "--project-root",
        str(project_root),
        "--device",
        str(config.get("assignment_device") or "cuda"),
    ]
    speaker_num = config.get("speaker_num")
    if speaker_num not in (None, "", 0, "0"):
        command.extend(["--speaker-num", str(int(speaker_num))])

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(config.get("assignment_timeout_seconds") or DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS),
        )
    except subprocess.TimeoutExpired:
        report["error"] = "3D-Speaker assignment timed out"
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return [], report

    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    if completed.returncode != 0:
        report["error"] = (completed.stderr or completed.stdout or "3D-Speaker assignment failed")[-2000:]
        return [], report
    try:
        output = completed.stdout.strip()
        marker = "__3DSPEAKER_JSON__"
        if marker in output:
            output = output.rsplit(marker, 1)[1].strip()
        payload = json.loads(output)
    except json.JSONDecodeError:
        report["error"] = f"invalid 3D-Speaker output: {completed.stdout[-1000:]}"
        return [], report
    turns = payload.get("turns") if isinstance(payload, dict) else None
    if not isinstance(turns, list):
        report["error"] = "3D-Speaker output is missing turns"
        return [], report
    report["turn_count"] = len(turns)
    report["detected_speakers"] = sorted(
        {str(turn.get("speaker")) for turn in turns if isinstance(turn, dict) and turn.get("speaker")}
    )
    report["detected_speaker_count"] = len(report["detected_speakers"])
    return [turn for turn in turns if isinstance(turn, dict)], report


def assign_speakers_by_overlap(
    transcript: AudioTranscript,
    turns: list[dict[str, Any]],
) -> tuple[AudioTranscript, dict[str, Any]]:
    normalized_turns = []
    for turn in turns:
        start = _float_value(turn.get("start", turn.get("start_sec")))
        end = _float_value(turn.get("end", turn.get("end_sec")))
        speaker = str(turn.get("speaker") or turn.get("speaker_id") or "").strip()
        if speaker and end > start:
            normalized_turns.append((start, end, speaker))

    assigned_count = 0
    unassigned_count = 0
    segments: list[dict[str, Any]] = []
    for segment in transcript.segments or []:
        updated = dict(segment)
        start = _float_value(first_present_value(segment, ("start", "start_time", "Start")))
        end = _float_value(first_present_value(segment, ("end", "end_time", "End")))
        if end <= start:
            end = start + 0.001
        overlaps: dict[str, float] = defaultdict(float)
        for turn_start, turn_end, speaker in normalized_turns:
            overlap = min(end, turn_end) - max(start, turn_start)
            if overlap > 0:
                overlaps[speaker] += overlap
        if overlaps:
            speaker = _display_speaker_label(max(overlaps.items(), key=lambda item: (item[1], item[0]))[0])
            updated["speaker"] = speaker
            updated["Speaker"] = speaker
            updated["speaker_id"] = speaker
            assigned_count += 1
        else:
            unassigned_count += 1
        segments.append(updated)

    return (
        AudioTranscript(
            text=transcript.text,
            segments=segments,
            language=transcript.language,
            metadata=dict(transcript.metadata or {}),
        ),
        {
            "assigned_segment_count": assigned_count,
            "unassigned_segment_count": unassigned_count,
            "final_speaker_count": len(_speaker_ids(segments)),
            "final_speakers": _speaker_ids(segments),
        },
    )


def refine_transcript_speakers(
    audio_path: Path,
    transcript: AudioTranscript,
    config: dict[str, Any] | None = None,
) -> tuple[AudioTranscript, dict[str, Any]]:
    """Refine over-split speaker labels without requiring user-supplied speaker count.

    The VibeVoice/WavLM transcript remains the primary transcript. External
    diarization backends only estimate likely speaker count. If WavLM produces
    more speaker IDs than the external consensus, this function merges speaker
    IDs using VibeVoice audit similarity edges.
    """
    config = config or {}
    if not _truthy(config.get("enabled"), default=True):
        report = {"enabled": False, "reason": "disabled"}
        return transcript, report

    current_speakers = _speaker_ids(transcript.segments or [])
    report: dict[str, Any] = {
        "enabled": True,
        "original_speaker_count": len(current_speakers),
        "original_speakers": current_speakers,
        "estimates": [],
        "target_speaker_count": len(current_speakers),
        "merge_map": {},
        "notes": [],
    }
    if len(current_speakers) <= 1:
        report["notes"].append("single speaker or no speaker labels; refinement skipped")
        transcript.metadata = _with_hybrid_metadata(transcript.metadata, report)
        return transcript, report

    estimates = collect_speaker_estimates(audio_path, transcript, config)
    report["estimates"] = [estimate.to_dict() for estimate in estimates]
    target = choose_target_speaker_count(len(current_speakers), estimates)
    report["target_speaker_count"] = target
    if target >= len(current_speakers):
        report["notes"].append("current speaker count is compatible with external estimates")
        transcript.metadata = _with_hybrid_metadata(transcript.metadata, report)
        return transcript, report

    strong_consensus = _strong_consensus(target, estimates)
    merge_map, merge_notes = build_speaker_merge_map(
        transcript=transcript,
        target_speaker_count=target,
        min_score=float(config.get("merge_min_score") or DEFAULT_MERGE_MIN_SCORE),
        allow_best_effort=strong_consensus,
    )
    if not merge_map and strong_consensus and _truthy(config.get("enable_ecapa_merge"), default=True):
        merge_map, ecapa_notes = build_ecapa_speaker_merge_map(audio_path, transcript, target, config)
        merge_notes.extend(ecapa_notes)
    report["merge_map"] = merge_map
    report["notes"].extend(merge_notes)
    if not merge_map:
        transcript.metadata = _with_hybrid_metadata(transcript.metadata, report)
        return transcript, report

    refined = apply_speaker_merge_map(transcript, merge_map)
    final_speakers = _speaker_ids(refined.segments or [])
    report["final_speaker_count"] = len(final_speakers)
    report["final_speakers"] = final_speakers
    refined.metadata = _with_hybrid_metadata(refined.metadata, report)
    return refined, report


def collect_speaker_estimates(
    audio_path: Path,
    transcript: AudioTranscript,
    config: dict[str, Any] | None = None,
) -> list[SpeakerEstimate]:
    config = config or {}
    duration = _wav_duration(audio_path)
    windows = select_dense_windows(
        transcript,
        duration=duration,
        sample_seconds=float(config.get("sample_seconds") or DEFAULT_SAMPLE_SECONDS),
        max_windows=int(config.get("max_windows") or DEFAULT_MAX_WINDOWS),
    )
    if not windows:
        return [SpeakerEstimate("hybrid", None, detail="no dense transcript windows available", skipped=True)]

    estimates: list[SpeakerEstimate] = []
    external_python = str(
        config.get("external_python")
        or os.environ.get("VIDEO_ANALYZER_DIARIZATION_PYTHON")
        or DEFAULT_EXTERNAL_PYTHON
    )
    timeout = float(config.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    if not Path(external_python).exists():
        return [SpeakerEstimate("external", None, detail=f"missing external python: {external_python}", skipped=True)]

    with tempfile.TemporaryDirectory(prefix="speaker_refine_") as temp_dir:
        temp_root = Path(temp_dir)
        clips = []
        for index, start in enumerate(windows):
            clip = temp_root / f"window_{index:02d}.wav"
            if _extract_clip(audio_path, clip, start, float(config.get("sample_seconds") or DEFAULT_SAMPLE_SECONDS)):
                clips.append((start, clip))

        if _truthy(config.get("enable_ecapa"), default=True):
            estimates.extend(_run_ecapa_estimates(external_python, audio_path, transcript, clips, timeout))
        if _truthy(config.get("enable_3dspeaker"), default=True):
            root = str(config.get("three_d_speaker_root") or os.environ.get("THREED_SPEAKER_ROOT") or DEFAULT_3D_SPEAKER_ROOT)
            estimates.extend(_run_3dspeaker_estimates(external_python, root, clips, timeout))

    return estimates


def select_dense_windows(
    transcript: AudioTranscript,
    duration: float,
    sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
    max_windows: int = DEFAULT_MAX_WINDOWS,
) -> list[float]:
    segments = [_normalized_segment(segment) for segment in transcript.segments or []]
    segments = [segment for segment in segments if segment["text"] and not _is_non_speech(segment["text"])]
    if not segments:
        return [0.0] if duration else []

    candidates = {max(0.0, float(int(segment["start"] // 30) * 30)) for segment in segments}
    scored: list[tuple[float, float]] = []
    for start in candidates:
        end = start + sample_seconds
        inside = [segment for segment in segments if segment["end"] > start and segment["start"] < end]
        if not inside:
            continue
        speakers = {segment["speaker"] for segment in inside if segment["speaker"] and segment["speaker"].lower() != "unknown"}
        text_len = sum(len(segment["text"]) for segment in inside)
        score = len(inside) * 3 + len(speakers) * 10 + min(text_len / 120, 20)
        scored.append((score, start))
    scored.sort(reverse=True)

    windows: list[float] = []
    for _score, start in scored:
        if duration:
            start = min(start, max(0.0, duration - sample_seconds))
        if all(abs(start - existing) >= sample_seconds * 0.8 for existing in windows):
            windows.append(round(start, 3))
        if len(windows) >= max_windows:
            break
    return windows


def choose_target_speaker_count(current_count: int, estimates: list[SpeakerEstimate]) -> int:
    counts = [
        int(estimate.speaker_count)
        for estimate in estimates
        if estimate.speaker_count is not None and 1 <= int(estimate.speaker_count) <= 12 and not estimate.skipped
    ]
    if not counts:
        return current_count
    tally = Counter(counts)
    top_count, top_votes = tally.most_common(1)[0]
    if top_votes >= 2:
        return top_count
    if current_count in tally:
        return current_count
    return current_count


def build_speaker_merge_map(
    transcript: AudioTranscript,
    target_speaker_count: int,
    min_score: float = DEFAULT_MERGE_MIN_SCORE,
    allow_best_effort: bool = False,
) -> tuple[dict[str, str], list[str]]:
    speakers = _speaker_ids(transcript.segments or [])
    if target_speaker_count >= len(speakers):
        return {}, ["merge skipped: target is not smaller than current speaker count"]

    parent = {speaker: speaker for speaker in speakers}
    pair_scores = _speaker_similarity_edges(transcript.metadata or {})
    if not pair_scores:
        return {}, ["merge skipped: no VibeVoice speaker similarity edges available"]

    notes: list[str] = []
    ordered_pairs = sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)
    for (left, right), score in ordered_pairs:
        if _group_count(parent) <= target_speaker_count:
            break
        if score < min_score and not allow_best_effort:
            continue
        if score < 0.55:
            continue
        if _find(parent, left) == _find(parent, right):
            continue
        winner = _larger_speaker(transcript.segments or [], _find(parent, left), _find(parent, right), parent)
        loser = _find(parent, right if winner == _find(parent, left) else left)
        parent[loser] = winner
        notes.append(f"merged {loser} into {winner} by audit edge score {score:.3f}")

    if _group_count(parent) > target_speaker_count:
        notes.append("merge stopped before target: not enough reliable similarity edges")

    merge_map = {speaker: _find(parent, speaker) for speaker in speakers if _find(parent, speaker) != speaker}
    return merge_map, notes


def apply_speaker_merge_map(transcript: AudioTranscript, merge_map: dict[str, str]) -> AudioTranscript:
    segments: list[dict[str, Any]] = []
    for segment in transcript.segments or []:
        updated = dict(segment)
        speaker = _segment_speaker(updated)
        if speaker in merge_map:
            target = merge_map[speaker]
            if "Speaker" in updated:
                updated["Speaker"] = target
            if "speaker_id" in updated:
                updated["speaker_id"] = target
            updated["speaker_merge_source"] = speaker
        segments.append(updated)

    metadata = dict(transcript.metadata or {})
    return AudioTranscript(
        text=transcript.text,
        segments=segments,
        language=transcript.language,
        metadata=metadata,
    )


def build_ecapa_speaker_merge_map(
    audio_path: Path,
    transcript: AudioTranscript,
    target_speaker_count: int,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Merge speaker IDs by clustering per-speaker ECAPA centroids.

    This fallback is only intended for strong external consensus cases where
    VibeVoice audit edges are unavailable. It clusters existing speaker labels,
    not individual utterances, so it is conservative and deterministic.
    """
    config = config or {}
    speakers = _speaker_ids(transcript.segments or [])
    if target_speaker_count >= len(speakers):
        return {}, ["ecapa merge skipped: target is not smaller than current speaker count"]
    external_python = str(
        config.get("external_python")
        or os.environ.get("VIDEO_ANALYZER_DIARIZATION_PYTHON")
        or DEFAULT_EXTERNAL_PYTHON
    )
    if not Path(external_python).exists():
        return {}, [f"ecapa merge skipped: missing external python: {external_python}"]

    script = r'''
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

audio_path = Path(sys.argv[1])
segments_path = Path(sys.argv[2])
target = int(sys.argv[3])
started = time.time()
sample_rate, audio = wavfile.read(str(audio_path))
audio = audio.astype(np.float32)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if np.max(np.abs(audio)) > 2:
    audio = audio / 32768.0
segments = json.loads(segments_path.read_text(encoding="utf-8"))
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/tmp/video_analyzer_speaker_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},
)
vectors = defaultdict(list)
counts = Counter()
for segment in segments:
    speaker = segment["speaker"]
    start = max(0, int(float(segment["start"]) * sample_rate))
    end = min(len(audio), int(float(segment["end"]) * sample_rate))
    piece = audio[start:end]
    if len(piece) < int(sample_rate * 0.45):
        continue
    counts[speaker] += 1
    with torch.no_grad():
        vector = classifier.encode_batch(torch.tensor(piece).unsqueeze(0)).squeeze().detach().cpu().numpy().astype(np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    vectors[speaker].append(vector)
speakers = sorted(vectors)
if len(speakers) <= target:
    print(json.dumps({"merge_map": {}, "elapsed_seconds": time.time() - started}))
    raise SystemExit(0)
centroids = []
for speaker in speakers:
    centroid = np.mean(np.vstack(vectors[speaker]), axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    centroids.append(centroid)
labels = AgglomerativeClustering(n_clusters=target, metric="cosine", linkage="average").fit_predict(np.vstack(centroids))
groups = defaultdict(list)
for speaker, label in zip(speakers, labels):
    groups[int(label)].append(speaker)
merge_map = {}
for members in groups.values():
    representative = sorted(members, key=lambda item: (-counts[item], item))[0]
    for speaker in members:
        if speaker != representative:
            merge_map[speaker] = representative
print(json.dumps({"merge_map": merge_map, "elapsed_seconds": time.time() - started, "speakers": speakers}))
'''
    with tempfile.TemporaryDirectory(prefix="ecapa_speaker_merge_") as temp_dir:
        temp_root = Path(temp_dir)
        script_path = temp_root / "merge.py"
        segments_path = temp_root / "segments.json"
        script_path.write_text(script, encoding="utf-8")
        segments_path.write_text(json.dumps(_speaker_segments(transcript), ensure_ascii=False), encoding="utf-8")
        estimate = _run_json_payload(
            [
                external_python,
                str(script_path),
                str(audio_path),
                str(segments_path),
                str(target_speaker_count),
            ],
            timeout=float(config.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS),
        )
    if not estimate:
        return {}, ["ecapa merge skipped: external merge failed"]
    merge_map = {str(key): str(value) for key, value in (estimate.get("merge_map") or {}).items()}
    if not merge_map:
        return {}, ["ecapa merge produced no merge map"]
    return merge_map, [f"ecapa merged over-split speakers to target {target_speaker_count}"]


def _run_ecapa_estimates(
    python_path: str,
    audio_path: Path,
    transcript: AudioTranscript,
    clips: list[tuple[float, Path]],
    timeout: float,
) -> list[SpeakerEstimate]:
    script = r'''
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from speechbrain.inference.speaker import EncoderClassifier
from scipy.io import wavfile

clip_path = Path(sys.argv[1])
segments_path = Path(sys.argv[2])
started = time.time()
sample_rate, audio = wavfile.read(str(clip_path))
audio = audio.astype(np.float32)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if np.max(np.abs(audio)) > 2:
    audio = audio / 32768.0
segments = json.loads(segments_path.read_text(encoding="utf-8"))
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/tmp/video_analyzer_speaker_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},
)
embeddings = []
for segment in segments:
    start = max(0, int(float(segment["start"]) * sample_rate))
    end = min(len(audio), int(float(segment["end"]) * sample_rate))
    piece = audio[start:end]
    if len(piece) < int(sample_rate * 0.45):
        continue
    with torch.no_grad():
        vector = classifier.encode_batch(torch.tensor(piece).unsqueeze(0)).squeeze().detach().cpu().numpy().astype(np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    embeddings.append(vector)
if len(embeddings) <= 1:
    print(json.dumps({"speaker_count": len(embeddings), "confidence": 0.0, "elapsed_seconds": time.time() - started}))
    raise SystemExit(0)
matrix = np.vstack(embeddings)
scores = []
for k in range(2, min(6, len(embeddings) - 1) + 1):
    labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(matrix)
    if len(set(labels)) < 2:
        continue
    raw = float(silhouette_score(matrix, labels, metric="cosine"))
    scores.append((raw - 0.035 * (k - 2), raw, k))
if not scores:
    speaker_count = 1
    confidence = 0.0
else:
    scores.sort(reverse=True)
    _adjusted, confidence, speaker_count = scores[0]
print(json.dumps({"speaker_count": speaker_count, "confidence": confidence, "elapsed_seconds": time.time() - started}))
'''
    estimates: list[SpeakerEstimate] = []
    for start, clip in clips:
        with tempfile.TemporaryDirectory(prefix="ecapa_segments_") as temp_dir:
            segments_path = Path(temp_dir) / "segments.json"
            segments_path.write_text(
                json.dumps(_segments_for_window(transcript, start, _clip_duration(clip)), ensure_ascii=False),
                encoding="utf-8",
            )
            script_path = Path(temp_dir) / "ecapa_estimate.py"
            script_path.write_text(script, encoding="utf-8")
            estimates.append(_run_json_estimate([python_path, str(script_path), str(clip), str(segments_path)], "ecapa", timeout))
    return estimates


def _run_3dspeaker_estimates(
    python_path: str,
    root: str,
    clips: list[tuple[float, Path]],
    timeout: float,
) -> list[SpeakerEstimate]:
    root_path = Path(root)
    script_path = root_path / "speakerlab" / "bin" / "infer_diarization.py"
    if not script_path.is_file():
        return [SpeakerEstimate("3dspeaker", None, detail=f"missing 3D-Speaker script: {script_path}", skipped=True)]

    estimates: list[SpeakerEstimate] = []
    for _start, clip in clips:
        with tempfile.TemporaryDirectory(prefix="threed_speaker_") as temp_dir:
            out_dir = Path(temp_dir) / "out"
            wrapper = Path(temp_dir) / "run_3dspeaker.py"
            wrapper.write_text(
                "\n".join(
                    [
                        "import runpy, sys",
                        "import numpy as np",
                        "setattr(np, 'NaN', np.nan)",
                        f"sys.argv = [{str(script_path)!r}, '--wav', {str(clip)!r}, '--out_dir', {str(out_dir)!r}, '--out_type', 'json', '--diable_progress_bar', '--nprocs', '1']",
                        "runpy.run_path(sys.argv[0], run_name='__main__')",
                    ]
                ),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = str(root_path)
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    [python_path, str(wrapper)],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                estimates.append(SpeakerEstimate("3dspeaker", None, detail="timeout", skipped=True))
                continue
            elapsed = round(time.perf_counter() - started, 3)
            if completed.returncode != 0:
                estimates.append(SpeakerEstimate("3dspeaker", None, elapsed_seconds=elapsed, detail=completed.stdout[-800:], skipped=True))
                continue
            files = list(out_dir.glob("*.json"))
            if not files:
                estimates.append(SpeakerEstimate("3dspeaker", None, elapsed_seconds=elapsed, detail="no json output", skipped=True))
                continue
            data = json.loads(files[0].read_text(encoding="utf-8"))
            speakers = {str(item.get("speaker")) for item in data.values() if isinstance(item, dict)}
            estimates.append(SpeakerEstimate("3dspeaker", len(speakers), confidence=0.75, elapsed_seconds=elapsed))
    return estimates


def _run_json_estimate(command: list[str], source: str, timeout: float) -> SpeakerEstimate:
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return SpeakerEstimate(source, None, detail="timeout", skipped=True)
    elapsed = round(time.perf_counter() - started, 3)
    if completed.returncode != 0:
        return SpeakerEstimate(source, None, elapsed_seconds=elapsed, detail=completed.stdout[-800:], skipped=True)
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        count = payload.get("speaker_count")
        return SpeakerEstimate(
            source,
            int(count) if count is not None else None,
            confidence=float(payload.get("confidence") or 0.0),
            elapsed_seconds=float(payload.get("elapsed_seconds") or elapsed),
        )
    except Exception as exc:
        return SpeakerEstimate(source, None, elapsed_seconds=elapsed, detail=f"invalid output: {exc}; {completed.stdout[-500:]}", skipped=True)


def _run_json_payload(command: list[str], timeout: float) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0:
        logger.warning("external speaker merge failed: %s", completed.stdout[-800:])
        return None
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        logger.warning("external speaker merge returned invalid output: %s", completed.stdout[-800:])
        return None


def _speaker_similarity_edges(metadata: dict[str, Any]) -> dict[tuple[str, str], float]:
    audit = _find_audit_chunks(metadata)
    if not isinstance(audit, dict):
        return {}
    assignments = audit.get("speaker_assignments") or []
    node_to_global: dict[str, str] = {}
    for item in assignments:
        if not isinstance(item, dict):
            continue
        chunk = item.get("chunk_index")
        local = item.get("local_speaker")
        global_speaker = item.get("global_speaker")
        if chunk is None or local is None or not global_speaker:
            continue
        node_to_global[f"{chunk}:{local}"] = str(global_speaker)

    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for edge in audit.get("global_speaker_edges") or []:
        if not isinstance(edge, dict):
            continue
        left = node_to_global.get(str(edge.get("left")))
        right = node_to_global.get(str(edge.get("right")))
        if not left or not right or left == right:
            continue
        try:
            score = float(edge.get("score"))
        except (TypeError, ValueError):
            continue
        key = tuple(sorted((left, right)))
        scores[key].append(score)
    return {key: max(values) for key, values in scores.items()}


def _find_audit_chunks(metadata: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(metadata.get("audit_chunks"), dict):
        return metadata["audit_chunks"]
    deep = metadata.get("deep_transcript_metadata")
    if isinstance(deep, dict) and isinstance(deep.get("audit_chunks"), dict):
        return deep["audit_chunks"]
    return None


def _segments_for_window(transcript: AudioTranscript, start: float, duration: float) -> list[dict[str, Any]]:
    end = start + duration
    selected = []
    for segment in transcript.segments or []:
        normalized = _normalized_segment(segment)
        if normalized["end"] <= start or normalized["start"] >= end:
            continue
        if not normalized["text"] or _is_non_speech(normalized["text"]):
            continue
        selected.append(
            {
                "start": max(normalized["start"], start) - start,
                "end": min(normalized["end"], end) - start,
                "text": normalized["text"],
            }
        )
    return selected


def _speaker_segments(transcript: AudioTranscript) -> list[dict[str, Any]]:
    selected = []
    for segment in transcript.segments or []:
        normalized = _normalized_segment(segment)
        if not normalized["speaker"] or normalized["speaker"].lower() == "unknown":
            continue
        if not normalized["text"] or _is_non_speech(normalized["text"]):
            continue
        selected.append(
            {
                "speaker": normalized["speaker"],
                "start": normalized["start"],
                "end": normalized["end"],
                "text": normalized["text"],
            }
        )
    return selected


def _normalized_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": _float_first(segment, ("Start", "start_time", "start")),
        "end": _float_first(segment, ("End", "end_time", "end")),
        "speaker": _segment_speaker(segment),
        "text": str(segment.get("Content") or segment.get("text") or "").strip(),
    }


def _segment_speaker(segment: dict[str, Any]) -> str:
    for key in ("Speaker", "speaker_id", "speaker", "speakerId"):
        value = segment.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _speaker_ids(segments: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            speaker
            for speaker in (_segment_speaker(segment) for segment in segments)
            if speaker and speaker.lower() not in {"unknown", "none", "null"}
        }
    )


def _float_first(segment: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        try:
            value = segment.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _is_non_speech(text: str) -> bool:
    return text.strip().lower() in {"[noise]", "[music]", "[silence]", "[applause]", "[speech]", "[human sounds]"}


def _extract_clip(audio_path: Path, output_path: Path, start: float, duration: float) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(audio_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return output_path.is_file()
    except Exception:
        return False


def _wav_duration(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else 0.0
    except Exception:
        return 0.0


def _clip_duration(path: Path) -> float:
    return _wav_duration(path) or DEFAULT_SAMPLE_SECONDS


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def first_present_value(values: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in values and values[key] not in (None, ""):
            return values[key]
    return None


def _display_speaker_label(value: str) -> str:
    match = re.fullmatch(r"spk[_ -]?0*(\d+)", value.strip(), flags=re.IGNORECASE)
    if match:
        return f"说话人 {int(match.group(1))}"
    return value


def _strong_consensus(target: int, estimates: list[SpeakerEstimate]) -> bool:
    return sum(1 for estimate in estimates if estimate.speaker_count == target and not estimate.skipped) >= 2


def _with_hybrid_metadata(metadata: dict[str, Any] | None, report: dict[str, Any]) -> dict[str, Any]:
    updated = dict(metadata or {})
    updated["hybrid_speaker_diarization"] = report
    return updated


def _group_count(parent: dict[str, str]) -> int:
    return len({_find(parent, speaker) for speaker in parent})


def _find(parent: dict[str, str], value: str) -> str:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _larger_speaker(segments: list[dict[str, Any]], left: str, right: str, parent: dict[str, str]) -> str:
    counts = Counter(_find(parent, _segment_speaker(segment)) for segment in segments if _segment_speaker(segment) in parent)
    if counts[left] != counts[right]:
        return left if counts[left] > counts[right] else right
    return min(left, right)
