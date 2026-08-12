#!/usr/bin/env python3
"""Multi-worker proxy for local FireRedASR2-AED workers."""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, jsonify, request

from tools.qwen3_asr_p40_proxy import (
    audio_duration,
    dedupe_join,
    offset_segments,
    post_audio,
    split_audio,
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


def worker_url(port: int, path: str = "/api/asr/transcribe") -> str:
    return f"http://127.0.0.1:{port}{path}"


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
        }
    )


@app.post("/api/asr/transcribe")
def transcribe():
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"success": False, "error": "audio file is required"}), 400
    form = {key: str(value) for key, value in request.form.items()}
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="firered_asr2_pool_") as temp:
        source = Path(temp) / f"audio{suffix}"
        audio.save(source)
        duration = audio_duration(source)
        if len(WORKER_PORTS) == 1 or duration <= SINGLE_PASS_SECONDS:
            payload = post_audio(worker_url(WORKER_PORTS[0]), source, form)
            payload["worker_count"] = 1
            return jsonify(payload)
        chunks = split_audio(
            source,
            Path(temp),
            duration,
            chunk_seconds=CHUNK_SECONDS,
            overlap_seconds=CHUNK_OVERLAP_SECONDS,
        )
        results = []
        with ThreadPoolExecutor(max_workers=len(WORKER_PORTS)) as executor:
            futures = {
                executor.submit(
                    post_audio,
                    worker_url(WORKER_PORTS[index % len(WORKER_PORTS)]),
                    chunk,
                    form,
                ): (index, start, length)
                for index, (chunk, start, length) in enumerate(chunks)
            }
            for future in as_completed(futures):
                index, start, length = futures[future]
                results.append((index, start, length, future.result()))
        text = ""
        segments = []
        words = []
        for _index, start, length, payload in sorted(results):
            text = dedupe_join(text, str(payload.get("text") or ""))
            segments.extend(offset_segments(payload, start, length))
            for source_word in payload.get("words") or []:
                word = dict(source_word)
                word["start"] = round(float(word.get("start") or 0) + start, 3)
                word["end"] = round(float(word.get("end") or 0) + start, 3)
                words.append(word)
        return jsonify(
            {
                "success": True,
                "provider": "firered_asr2",
                "text": text,
                "segments": segments,
                "words": words,
                "language": "zh-CN",
                "worker_count": len(WORKER_PORTS),
                "chunk_count": len(chunks),
            }
        )


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FIRERED_ASR2_PROXY_HOST", "0.0.0.0"),
        port=int(os.environ.get("FIRERED_ASR2_PROXY_PORT", "18014")),
        threaded=True,
    )
