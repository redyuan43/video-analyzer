#!/usr/bin/env python3
"""Single-GPU FireRedASR2-AED HTTP worker."""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, request


app = Flask("firered_asr2_worker")
MODEL = None
MODEL_LOCK = threading.Lock()
MODEL_DIR = os.environ.get("FIRERED_ASR2_MODEL", "/home/ai/models/firered/FireRedASR2-AED")


def load_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

            MODEL = FireRedAsr2.from_pretrained(
                "aed",
                MODEL_DIR,
                FireRedAsr2Config(
                    use_gpu=True,
                    use_half=True,
                    beam_size=3,
                    nbest=1,
                    return_timestamp=True,
                ),
            )
    return MODEL


def normalize_audio(source: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(target),
        ],
        check=True,
        timeout=120,
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "ready": True,
            "model_loaded": MODEL is not None,
            "model": "FireRedTeam/FireRedASR2-AED",
        }
    )


@app.post("/api/asr/transcribe")
def transcribe():
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"success": False, "error": "audio file is required"}), 400
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="firered_asr2_") as temp:
        source = Path(temp) / f"input{suffix}"
        target = Path(temp) / "audio.wav"
        audio.save(source)
        normalize_audio(source, target)
        result = load_model().transcribe(["audio"], [str(target)])[0]
    text = str(result.get("text") or "")
    duration = float(result.get("dur_s") or 0)
    words = [
        {"text": str(item[0]), "start": float(item[1]), "end": float(item[2])}
        for item in (result.get("timestamp") or [])
        if isinstance(item, (list, tuple)) and len(item) >= 3
    ]
    return jsonify(
        {
            "success": True,
            "provider": "firered_asr2",
            "text": text,
            "segments": [
                {
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                    "confidence": result.get("confidence"),
                }
            ] if text else [],
            "words": words,
            "language": "zh-CN",
            "rtf": result.get("rtf"),
        }
    )


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FIRERED_ASR2_WORKER_HOST", "127.0.0.1"),
        port=int(os.environ.get("FIRERED_ASR2_WORKER_PORT", "18400")),
        threaded=True,
    )
