#!/usr/bin/env python3
"""Expose FireRed ASR plus local 3D-Speaker diarization as one HTTP provider."""

from __future__ import annotations

import os
import hashlib
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from flask import Flask, jsonify, request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from video_analyzer.audio_processor import AudioTranscript
from video_analyzer.speaker_diarization import (
    assign_speakers_by_overlap,
    run_3dspeaker_assignment,
)


EDGE_URL = os.environ.get(
    "FIRERED_BENCHMARK_URL",
    "http://edge.taild500c8.ts.net:8031/v1/transcribe",
).rstrip("/")
EDGE_TOKEN = os.environ.get("FIRERED_BENCHMARK_TOKEN", "").strip()
PIPELINE_TOKEN = os.environ.get("VIDEO_ANALYZER_AUDIO_PIPELINE_TOKEN", "").strip()
THREED_ROOT = os.environ.get("FIRERED_3DSPEAKER_ROOT", "/tmp/3D-Speaker").strip()
THREED_PYTHON = os.environ.get(
    "FIRERED_3DSPEAKER_PYTHON",
    "/home/ai/diarization-ab-venv/bin/python",
).strip()
REQUEST_TIMEOUT_SECONDS = max(
    60,
    int(os.environ.get("FIRERED_3DSPEAKER_REQUEST_TIMEOUT_SECONDS", "7200")),
)

FIRERED_CONCURRENCY = max(1, int(os.environ.get("FIRERED_CONCURRENCY", "1")))
THREED_SPEAKER_CONCURRENCY = max(
    1, int(os.environ.get("THREED_SPEAKER_CONCURRENCY", "1"))
)
CACHE_DIR = Path(
    os.environ.get(
        "FIRERED_3DSPEAKER_CACHE_DIR",
        str(Path.home() / ".cache" / "video-analyzer" / "firered-3dspeaker"),
    )
).expanduser()
FIRERED_SEMAPHORE = threading.Semaphore(FIRERED_CONCURRENCY)
THREED_SPEAKER_SEMAPHORE = threading.Semaphore(THREED_SPEAKER_CONCURRENCY)
ALIGNMENT_SEMAPHORE = threading.Semaphore(
    max(FIRERED_CONCURRENCY, THREED_SPEAKER_CONCURRENCY)
)
app = Flask("firered_3dspeaker")
_SINGLE_FLIGHT_GUARD = threading.Lock()
_SINGLE_FLIGHTS: dict[tuple[str, str, str], tuple[threading.Lock, int]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(config: dict) -> str:
    raw = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def firered_fingerprint() -> str:
    return _fingerprint({"schema": 1, "url": EDGE_URL, "model": "firered_asr2_aed"})


def diarization_fingerprint() -> str:
    return _fingerprint({
        "schema": 1, "root": THREED_ROOT, "python": THREED_PYTHON,
        "device": "cuda",
    })


def alignment_fingerprint() -> str:
    return _fingerprint({"schema": 1, "algorithm": "assign_speakers_by_overlap"})


def alignment_input_fingerprint(
    firered_artifact: dict,
    diarization_artifact: dict,
    algorithm_fingerprint: str,
) -> str:
    return _fingerprint({
        "schema": 1,
        "firered_artifact_sha256": _fingerprint(firered_artifact),
        "diarization_artifact_sha256": _fingerprint(diarization_artifact),
        "alignment_fingerprint": algorithm_fingerprint,
    })


def pipeline_fingerprint() -> str:
    config = {
        "schema": 2,
        "firered": firered_fingerprint(),
        "diarization": diarization_fingerprint(),
        "alignment": alignment_fingerprint(),
    }
    return _fingerprint(config)


def cache_path(audio_sha256: str, fingerprint: str, stage: str) -> Path:
    return CACHE_DIR / audio_sha256 / fingerprint / f"{stage}.json"


def read_stage_cache(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_stage_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class SingleFlight:
    def __init__(self, key: tuple[str, str, str]):
        self.key = key
        self.lock: threading.Lock | None = None

    def __enter__(self) -> None:
        with _SINGLE_FLIGHT_GUARD:
            lock, users = _SINGLE_FLIGHTS.get(self.key, (threading.Lock(), 0))
            _SINGLE_FLIGHTS[self.key] = (lock, users + 1)
            self.lock = lock
        lock.acquire()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        assert self.lock is not None
        self.lock.release()
        with _SINGLE_FLIGHT_GUARD:
            lock, users = _SINGLE_FLIGHTS[self.key]
            if users == 1:
                del _SINGLE_FLIGHTS[self.key]
            else:
                _SINGLE_FLIGHTS[self.key] = (lock, users - 1)


def run_cached_stage(
    stage: str,
    audio_sha256: str,
    fingerprint: str,
    semaphore: threading.Semaphore,
    operation,
    validator,
) -> tuple[dict, dict]:
    started = time.perf_counter()
    path = cache_path(audio_sha256, fingerprint, stage)
    cached = read_stage_cache(path)
    if cached is not None and validator(cached):
        return cached, {
            "cache_hit": True,
            "queue_seconds": 0.0,
            "run_seconds": 0.0,
            "total_seconds": round(time.perf_counter() - started, 3),
        }
    queue_started = time.perf_counter()
    with SingleFlight((stage, audio_sha256, fingerprint)):
        # Another same-key request may have populated the cache while this one waited.
        cached = read_stage_cache(path)
        if cached is not None and validator(cached):
            return cached, {
                "cache_hit": True,
                "queue_seconds": round(time.perf_counter() - queue_started, 3),
                "run_seconds": 0.0,
                "total_seconds": round(time.perf_counter() - started, 3),
            }
        with semaphore:
            queue_seconds = time.perf_counter() - queue_started
            run_started = time.perf_counter()
            payload = operation()
            write_stage_cache(path, payload)
            run_seconds = time.perf_counter() - run_started
    return payload, {
        "cache_hit": False,
        "queue_seconds": round(queue_seconds, 3),
        "run_seconds": round(run_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
    }


def require_token() -> None:
    token = request.headers.get("X-Audio-Pipeline-Token", "").strip()
    if PIPELINE_TOKEN and token != PIPELINE_TOKEN:
        raise PermissionError("audio pipeline token is invalid")


def transcribe_with_firered(audio_path: Path) -> dict:
    headers = {"X-ASR-Benchmark-Token": EDGE_TOKEN} if EDGE_TOKEN else {}
    session = requests.Session()
    session.trust_env = False
    try:
        with audio_path.open("rb") as audio:
            response = session.post(
                EDGE_URL,
                headers=headers,
                data={"model_id": "firered_asr2_aed"},
                files={"audio": (audio_path.name, audio, "audio/wav")},
                timeout=(30, REQUEST_TIMEOUT_SECONDS),
            )
    finally:
        session.close()
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not str(payload.get("text") or "").strip():
        raise RuntimeError("FireRed benchmark endpoint returned no transcript")
    return payload


def run_pipeline(audio_path: Path) -> dict:
    started = time.perf_counter()
    audio_sha256 = sha256_file(audio_path)
    fingerprints = {
        "firered": firered_fingerprint(),
        "diarization": diarization_fingerprint(),
        "alignment": alignment_fingerprint(),
    }
    fingerprint = pipeline_fingerprint()

    def diarize() -> dict:
        turns, report = run_3dspeaker_assignment(
            audio_path,
            {
                "external_python": THREED_PYTHON,
                "diarization_project_root": THREED_ROOT,
                "assignment_device": "cuda",
                "assignment_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            },
        )
        return {"turns": turns, "report": report}

    with ThreadPoolExecutor(max_workers=2) as executor:
        firered_future = executor.submit(
            run_cached_stage,
            "firered",
            audio_sha256,
            fingerprints["firered"],
            FIRERED_SEMAPHORE,
            lambda: transcribe_with_firered(audio_path),
            lambda payload: bool(str(payload.get("text") or "").strip())
            and isinstance(payload.get("segments") or [], list),
        )
        diarization_future = executor.submit(
            run_cached_stage,
            "3dspeaker",
            audio_sha256,
            fingerprints["diarization"],
            THREED_SPEAKER_SEMAPHORE,
            diarize,
            lambda payload: isinstance(payload.get("turns"), list)
            and bool(payload.get("turns"))
            and isinstance(payload.get("report") or {}, dict),
        )
        firered, firered_timing = firered_future.result()
        diarization_payload, diarization_timing = diarization_future.result()

    turns = list(diarization_payload.get("turns") or [])
    diarization = dict(diarization_payload.get("report") or {})
    if not turns:
        detail = diarization.get("error") or "3D-Speaker produced no turns"
        raise RuntimeError(detail)
    alignment_input = alignment_input_fingerprint(
        firered,
        diarization_payload,
        fingerprints["alignment"],
    )

    def align() -> dict:
        transcript = AudioTranscript(
            text=str(firered.get("text") or ""),
            segments=list(firered.get("segments") or []),
            language=str(firered.get("language") or "Chinese"),
            metadata={},
        )
        assigned, assignment = assign_speakers_by_overlap(transcript, turns)
        return {
            "text": assigned.text,
            "segments": assigned.segments,
            "assignment": assignment,
        }

    alignment_payload, alignment_timing = run_cached_stage(
        "alignment",
        alignment_input,
        fingerprints["alignment"],
        ALIGNMENT_SEMAPHORE,
        align,
        lambda payload: bool(str(payload.get("text") or "").strip())
        and isinstance(payload.get("segments"), list)
        and isinstance(payload.get("assignment"), dict),
    )
    assigned = AudioTranscript(
        text=str(alignment_payload.get("text") or ""),
        segments=list(alignment_payload.get("segments") or []),
        language=str(firered.get("language") or "Chinese"),
        metadata={},
    )
    assignment = dict(alignment_payload.get("assignment") or {})
    diarization.update(assignment)
    speaker_count = int(diarization.get("final_speaker_count") or 0)
    return {
        "success": True,
        "text": assigned.text,
        "segments": assigned.segments,
        "transcript_raw": {
            "text": str(firered.get("text") or ""),
            "segments": list(firered.get("segments") or []),
            "language": str(firered.get("language") or "Chinese"),
            "metadata": {
                key: value
                for key, value in firered.items()
                if key not in {"text", "segments", "language"}
            },
        },
        "language": assigned.language,
        "speaker_count": speaker_count,
        "provider": "firered_3dspeaker",
        "speaker_diarization_applied": True,
        "speaker_diarization": diarization,
        "diarization": diarization,
        "metadata": {
            "provider": "firered_3dspeaker",
            "speaker_diarization_applied": True,
            "speaker_diarization": diarization,
            "firered": {
                key: value
                for key, value in firered.items()
                if key not in {"text", "segments"}
            },
            "audio_sha256": audio_sha256,
            "pipeline_fingerprint": fingerprint,
            "config_fingerprint": fingerprint,
            "firered_fingerprint": fingerprints["firered"],
            "diarization_fingerprint": fingerprints["diarization"],
            "alignment_fingerprint": fingerprints["alignment"],
            "stages": {
                "firered": firered_timing,
                "3dspeaker": diarization_timing,
                "alignment": alignment_timing,
            },
        },
        "audio_sha256": audio_sha256,
        "pipeline_fingerprint": fingerprint,
        "config_fingerprint": fingerprint,
        "firered_fingerprint": fingerprints["firered"],
        "diarization_fingerprint": fingerprints["diarization"],
        "alignment_fingerprint": fingerprints["alignment"],
        "cache_hit": bool(
            firered_timing["cache_hit"]
            and diarization_timing["cache_hit"]
            and alignment_timing["cache_hit"]
        ),
        "stage_timings": {
            "firered": firered_timing,
            "3dspeaker": diarization_timing,
            "alignment": alignment_timing,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


@app.get("/api/health")
def health():
    try:
        require_token()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 401
    return jsonify({
        "status": "ok",
        "edge_url": EDGE_URL,
        "three_d_speaker_root": THREED_ROOT,
        "three_d_speaker_ready": Path(THREED_ROOT).is_dir(),
        "three_d_speaker_python_ready": Path(THREED_PYTHON).is_file(),
        "firered_concurrency": FIRERED_CONCURRENCY,
        "three_d_speaker_concurrency": THREED_SPEAKER_CONCURRENCY,
        "pipeline_fingerprint": pipeline_fingerprint(),
    })


@app.post("/api/firered-3dspeaker/transcribe")
def transcribe():
    try:
        require_token()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 401
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"error": "audio file is required"}), 400
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(prefix="firered_3dspeaker_", suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
        audio.save(handle)
    try:
        return jsonify(run_pipeline(path))
    except requests.RequestException as exc:
        return jsonify({"error": f"FireRed endpoint unavailable: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/firered/transcribe")
def transcribe_firered_only():
    try:
        require_token()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 401
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"error": "audio file is required"}), 400
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(prefix="firered_asr2_", suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
        audio.save(handle)
    try:
        audio_sha256 = sha256_file(path)
        payload, timing = run_cached_stage(
            "firered",
            audio_sha256,
            firered_fingerprint(),
            FIRERED_SEMAPHORE,
            lambda: transcribe_with_firered(path),
            lambda item: bool(str(item.get("text") or "").strip())
            and isinstance(item.get("segments") or [], list),
        )
        return jsonify(
            {
                "success": True,
                "provider": "firered_asr2",
                "text": str(payload.get("text") or ""),
                "segments": list(payload.get("segments") or []),
                "language": str(payload.get("language") or "Chinese"),
                "metadata": {
                    key: value
                    for key, value in payload.items()
                    if key not in {"text", "segments", "language"}
                },
                "audio_sha256": audio_sha256,
                "stage_timing": timing,
            }
        )
    except requests.RequestException as exc:
        return jsonify({"error": f"FireRed endpoint unavailable: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FIRERED_3DSPEAKER_HOST", "127.0.0.1"),
        port=int(os.environ.get("FIRERED_3DSPEAKER_PORT", "8013")),
        threaded=True,
    )
