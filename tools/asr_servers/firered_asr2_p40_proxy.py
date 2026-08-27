#!/usr/bin/env python3
"""Multi-worker proxy for local FireRedASR2-AED workers."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, request

from tools.asr_servers.asr_ray_workers import (
    audio_duration,
    dispatch_asr_chunks,
    materialize_fixed_chunks,
    materialize_segment_chunks,
    merge_asr_results,
    normalize_audio,
    request_choice,
    request_float,
)


app = Flask("firered_asr2_p40")
WORKER_PORTS = [
    int(value)
    for value in os.environ.get("FIRERED_ASR2_WORKER_PORTS", "18400").split(",")
    if value.strip()
]
CHUNK_SECONDS = float(os.environ.get("FIRERED_ASR2_CHUNK_SECONDS", "30"))
CHUNK_OVERLAP_SECONDS = float(os.environ.get("FIRERED_ASR2_CHUNK_OVERLAP_SECONDS", "3"))
SINGLE_PASS_SECONDS = float(os.environ.get("FIRERED_ASR2_SINGLE_PASS_SECONDS", "35"))
REQUEST_TIMEOUT = float(os.environ.get("FIRERED_ASR2_REQUEST_TIMEOUT", "1800"))
SEGMENTATION_MODE = os.environ.get("FIRERED_ASR2_SEGMENTATION_MODE", "vad").strip().lower()
VAD_MODEL_DIR = os.environ.get(
    "FIRERED_VAD_MODEL",
    "/home/ai/models/firered/FireRedVAD/VAD",
)
VAD_MAX_SEGMENT_SECONDS = float(os.environ.get("FIRERED_VAD_MAX_SEGMENT_SECONDS", "50"))
VAD_HARD_LIMIT_SECONDS = float(os.environ.get("FIRERED_VAD_HARD_LIMIT_SECONDS", "55"))
VAD_SPEECH_THRESHOLD = float(os.environ.get("FIRERED_VAD_SPEECH_THRESHOLD", "0.4"))
VAD_MODEL = None
VAD_LOCK = threading.Lock()


def worker_url(port: int, path: str = "/api/asr/transcribe") -> str:
    return f"http://127.0.0.1:{port}{path}"


def load_vad():
    global VAD_MODEL
    with VAD_LOCK:
        if VAD_MODEL is None:
            from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

            model_dir = Path(VAD_MODEL_DIR)
            if not model_dir.is_dir():
                raise FileNotFoundError(f"FireRedVAD model directory missing: {model_dir}")
            VAD_MODEL = FireRedVad.from_pretrained(
                str(model_dir),
                FireRedVadConfig(
                    use_gpu=False,
                    speech_threshold=VAD_SPEECH_THRESHOLD,
                    max_speech_frame=max(1, int(VAD_HARD_LIMIT_SECONDS * 100)),
                ),
            )
    return VAD_MODEL


def detect_speech_segments(path: Path) -> list[tuple[float, float]]:
    result, _probabilities = load_vad().detect(str(path))
    return [
        (float(start), float(end))
        for start, end in (result.get("timestamps") or [])
        if float(end) > float(start)
    ]


@app.get("/api/health")
def health():
    import requests

    workers = []
    for port in WORKER_PORTS:
        try:
            response = requests.get(worker_url(port, "/api/health"), timeout=2)
            payload = response.json() if response.ok else {}
            workers.append({"port": port, "ready": response.ok, **payload})
        except Exception as exc:
            workers.append({"port": port, "ready": False, "error": str(exc)})
    return jsonify(
        {
            "status": "ok",
            "ready": all(item["ready"] for item in workers),
            "model": "FireRedTeam/FireRedASR2-AED",
            "worker_count": len(workers),
            "workers": workers,
            "dispatch_mode": "ray",
            "segmentation_mode": SEGMENTATION_MODE,
            "vad_model": VAD_MODEL_DIR,
            "vad_model_ready": Path(VAD_MODEL_DIR).is_dir(),
        }
    )


@app.post("/api/asr/transcribe")
def transcribe():
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"success": False, "error": "audio file is required"}), 400
    form = {key: str(value) for key, value in request.form.items()}
    try:
        segmentation_mode = request_choice(
            form,
            "segmentation_mode",
            SEGMENTATION_MODE,
            {"fixed", "vad"},
        )
        chunk_seconds = request_float(
            form,
            "chunk_duration_sec",
            CHUNK_SECONDS,
            minimum=1.0,
        )
        overlap_seconds = request_float(
            form,
            "chunk_overlap_sec",
            CHUNK_OVERLAP_SECONDS,
            minimum=0.0,
        )
        single_pass_seconds = request_float(
            form,
            "single_pass_max_duration_sec",
            SINGLE_PASS_SECONDS,
            minimum=1.0,
        )
        vad_max_segment_seconds = request_float(
            form,
            "vad_max_segment_sec",
            VAD_MAX_SEGMENT_SECONDS,
            minimum=1.0,
        )
        if overlap_seconds >= chunk_seconds:
            raise ValueError("chunk_overlap_sec must be smaller than chunk_duration_sec")
        if vad_max_segment_seconds > VAD_HARD_LIMIT_SECONDS:
            raise ValueError(
                f"vad_max_segment_sec must not exceed {VAD_HARD_LIMIT_SECONDS:g}"
            )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="firered_asr2_pool_") as temp:
        source = Path(temp) / f"audio{suffix}"
        audio.save(source)
        duration = audio_duration(source)
        fallback_used = False
        fallback_reason = ""
        vad_segments: list[tuple[float, float]] = []
        if segmentation_mode == "vad":
            normalized = normalize_audio(source, Path(temp) / "normalized.wav")
            try:
                vad_segments = detect_speech_segments(normalized)
            except Exception as exc:
                fallback_used = True
                fallback_reason = str(exc)
                segmentation_mode = "fixed"
            else:
                if not vad_segments:
                    return jsonify(
                        {
                            "success": False,
                            "error": "no_speech_detected",
                            "provider": "firered_asr2",
                            "dispatch_mode": "ray",
                            "segmentation_mode": "vad",
                            "audio_duration_seconds": round(duration, 3),
                            "worker_count": len(WORKER_PORTS),
                            "chunk_count": 0,
                            "vad_segments": [],
                        }
                    ), 422
                chunks = materialize_segment_chunks(
                    normalized,
                    Path(temp),
                    vad_segments,
                    hard_limit_seconds=min(
                        vad_max_segment_seconds,
                        VAD_HARD_LIMIT_SECONDS,
                    ),
                )
        if segmentation_mode == "fixed":
            if duration <= single_pass_seconds:
                fixed_chunk_seconds = max(duration, 1.0)
                fixed_overlap_seconds = 0.0
            else:
                fixed_chunk_seconds = chunk_seconds
                fixed_overlap_seconds = overlap_seconds
            chunks = materialize_fixed_chunks(
                source,
                Path(temp),
                duration,
                chunk_seconds=fixed_chunk_seconds,
                overlap_seconds=fixed_overlap_seconds,
            )
        results = dispatch_asr_chunks(
            [worker_url(port) for port in WORKER_PORTS],
            chunks,
            form,
            request_timeout=REQUEST_TIMEOUT,
        )
        return jsonify(
            merge_asr_results(
                "firered_asr2",
                results,
                segmentation_mode=segmentation_mode,
                audio_duration_seconds=duration,
                worker_count=len(WORKER_PORTS),
                segmentation_metadata={
                    "chunk_duration_sec": chunk_seconds,
                    "chunk_overlap_sec": overlap_seconds,
                    "single_pass_max_duration_sec": single_pass_seconds,
                    "vad_max_segment_sec": vad_max_segment_seconds,
                    "vad_segments": [
                        [round(start, 3), round(end, 3)]
                        for start, end in vad_segments
                    ],
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                },
            )
        )


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FIRERED_ASR2_PROXY_HOST", "0.0.0.0"),
        port=int(os.environ.get("FIRERED_ASR2_PROXY_PORT", "18014")),
        threaded=True,
    )
