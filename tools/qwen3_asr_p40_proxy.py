#!/usr/bin/env python3
"""Lazy multi-worker HTTP proxy for local Qwen3-ASR on P40 GPUs."""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from flask import Flask, jsonify, request


app = Flask("qwen3_asr_p40")
ROOT = Path(__file__).resolve().parents[1]
CAPSWRITER_ROOT = Path(os.environ.get("QWEN3_ASR_ROOT", "/home/ai/CapsWriter-Offline-Windows-64bit-main"))
WORKER_PYTHON = os.environ.get("QWEN3_ASR_PYTHON", "/home/ai/vllm-p40-nightly-test/bin/python")
MODEL_PATH = os.environ.get(
    "QWEN3_ASR_MODEL",
    str(CAPSWRITER_ROOT / "models" / "Qwen3-ASR-1.7B"),
)
WORKER_SPECS = os.environ.get("QWEN3_ASR_WORKERS", "0:18300").split(",")
CHUNK_SECONDS = float(os.environ.get("QWEN3_ASR_CHUNK_SECONDS", "120"))
CHUNK_OVERLAP_SECONDS = float(os.environ.get("QWEN3_ASR_CHUNK_OVERLAP_SECONDS", "10"))
SINGLE_PASS_SECONDS = float(os.environ.get("QWEN3_ASR_SINGLE_PASS_SECONDS", "150"))
STARTUP_TIMEOUT = float(os.environ.get("QWEN3_ASR_STARTUP_TIMEOUT", "60"))
REQUEST_TIMEOUT = float(os.environ.get("QWEN3_ASR_REQUEST_TIMEOUT", "1800"))
LOG_DIR = Path(os.environ.get("QWEN3_ASR_LOG_DIR", str(ROOT / "tmp" / "qwen3-asr-p40" / "logs")))
PROCESSES: list[subprocess.Popen] = []
LOG_HANDLES = []
START_LOCK = threading.Lock()


def parsed_workers() -> list[tuple[int, int]]:
    workers = []
    for raw in WORKER_SPECS:
        gpu_text, port_text = raw.strip().split(":", 1)
        gpu, port = int(gpu_text), int(port_text)
        if gpu == 3:
            raise ValueError("GPU 3 is reserved for the Foundation-Sec security model")
        workers.append((gpu, port))
    if not workers:
        raise ValueError("at least one Qwen3-ASR worker is required")
    return workers


def worker_url(port: int, path: str = "/api/asr/transcribe") -> str:
    return f"http://127.0.0.1:{port}{path}"


def worker_ready(port: int) -> bool:
    try:
        response = requests.get(worker_url(port, "/api/health"), timeout=2, proxies={"http": None, "https": None})
        return response.ok
    except requests.RequestException:
        return False


def ensure_workers() -> None:
    with START_LOCK:
        workers = parsed_workers()
        if all(worker_ready(port) for _gpu, port in workers):
            return
        stop_workers()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        for gpu, port in workers:
            log_handle = (LOG_DIR / f"worker-gpu{gpu}-port{port}.log").open("ab")
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(CAPSWRITER_ROOT),
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "QWEN_CUDA_DEVICE": "0",
                    "ASR_DEVICE": "cuda:0",
                    "ASR_REQUIRE_ACCEL": "1",
                    "QWEN_ASR_MODEL": MODEL_PATH,
                    "HTTP_API_PORT": str(port),
                    "ENABLE_TTS": "0",
                    "TTS_ENABLED": "0",
                    "TRANSLATE_ENABLED": "0",
                    "CAPS_TRAINING_DATA_ENABLED": "0",
                    "NO_PROXY": "127.0.0.1,localhost",
                    "no_proxy": "127.0.0.1,localhost",
                }
            )
            process = subprocess.Popen(
                [
                    WORKER_PYTHON,
                    str(CAPSWRITER_ROOT / "http_api_server.py"),
                    "--qwen3-asr-worker",
                ],
                cwd=CAPSWRITER_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            PROCESSES.append(process)
            LOG_HANDLES.append(log_handle)
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if all(worker_ready(port) for _gpu, port in workers):
                return
            if any(process.poll() is not None for process in PROCESSES):
                break
            time.sleep(1)
        raise RuntimeError("Qwen3-ASR worker HTTP services did not become ready")


def stop_workers() -> None:
    while PROCESSES:
        process = PROCESSES.pop()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    while LOG_HANDLES:
        LOG_HANDLES.pop().close()


atexit.register(stop_workers)


def audio_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(completed.stdout.strip())


def split_audio(
    path: Path,
    directory: Path,
    duration: float,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
) -> list[tuple[Path, float, float]]:
    step = max(1.0, chunk_seconds - overlap_seconds)
    chunks = []
    start = 0.0
    index = 0
    while start < duration:
        length = min(chunk_seconds, duration - start)
        output = directory / f"chunk-{index:04d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{length:.3f}",
                "-i",
                str(path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(output),
            ],
            check=True,
            timeout=120,
        )
        chunks.append((output, start, length))
        index += 1
        start += step
    return chunks


def post_audio(url: str, path: Path, form: dict[str, str]) -> dict:
    session = requests.Session()
    session.trust_env = False
    try:
        with path.open("rb") as audio:
            response = session.post(
                url,
                files={"audio": (path.name, audio, "audio/wav")},
                data=form,
                timeout=(30, REQUEST_TIMEOUT),
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("success") is False or payload.get("error"):
            raise RuntimeError(str(payload.get("error") or payload))
        return payload
    finally:
        session.close()


def dedupe_join(left: str, right: str) -> str:
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    limit = min(120, len(left), len(right))
    for size in range(limit, 3, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return f"{left}\n{right}"


def offset_segments(payload: dict, start: float, duration: float) -> list[dict]:
    segments = []
    for source in payload.get("segments") or []:
        if not isinstance(source, dict):
            continue
        item = dict(source)
        if item.get("start") is not None:
            item["start"] = round(float(item["start"]) + start, 3)
        if item.get("end") is not None:
            item["end"] = round(float(item["end"]) + start, 3)
        segments.append(item)
    if not segments and str(payload.get("text") or "").strip():
        segments.append(
            {
                "start": round(start, 3),
                "end": round(start + duration, 3),
                "text": str(payload.get("text") or "").strip(),
            }
        )
    return segments


@app.get("/api/health")
def health():
    workers = [
        {"gpu": gpu, "port": port, "http_ready": worker_ready(port)}
        for gpu, port in parsed_workers()
    ]
    return jsonify(
        {
            "status": "ok",
            "ready": all(worker["http_ready"] for worker in workers),
            "model": MODEL_PATH,
            "worker_count": len(workers),
            "workers": workers,
        }
    )


@app.post("/api/asr/transcribe")
def transcribe():
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"success": False, "error": "audio file is required"}), 400
    ensure_workers()
    form = {key: str(value) for key, value in request.form.items()}
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="qwen3_asr_") as temp:
        source = Path(temp) / f"audio{suffix}"
        audio.save(source)
        duration = audio_duration(source)
        workers = parsed_workers()
        if len(workers) == 1 or duration <= SINGLE_PASS_SECONDS:
            payload = post_audio(worker_url(workers[0][1]), source, form)
            payload.update({"provider": "qwen3_asr", "worker_count": 1})
            return jsonify(payload)
        chunks = split_audio(source, Path(temp), duration)
        results = []
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = {
                executor.submit(
                    post_audio,
                    worker_url(workers[index % len(workers)][1]),
                    chunk_path,
                    form,
                ): (index, start, length)
                for index, (chunk_path, start, length) in enumerate(chunks)
            }
            for future in as_completed(futures):
                index, start, length = futures[future]
                results.append((index, start, length, future.result()))
        text = ""
        segments = []
        for _index, start, length, payload in sorted(results):
            text = dedupe_join(text, str(payload.get("text") or ""))
            segments.extend(offset_segments(payload, start, length))
        return jsonify(
            {
                "success": True,
                "provider": "qwen3_asr",
                "text": text,
                "segments": segments,
                "language": "zh-CN",
                "worker_count": len(workers),
                "chunk_count": len(chunks),
                "audio_duration_seconds": round(duration, 3),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("QWEN3_ASR_PROXY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN3_ASR_PROXY_PORT", "18013")))
    args = parser.parse_args()
    ensure_workers()
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
