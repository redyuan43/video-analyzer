#!/usr/bin/env python3
"""Lazy multi-worker HTTP proxy for local Qwen3-ASR on P40 GPUs."""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.asr_servers.asr_ray_workers import (
    audio_duration,
    dispatch_asr_chunks,
    materialize_fixed_chunks,
    merge_asr_results,
    request_float,
)

app = Flask("qwen3_asr_p40")
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
            raise ValueError("GPU 3 is reserved and must not run Qwen3-ASR")
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
    try:
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
        if overlap_seconds >= chunk_seconds:
            raise ValueError("chunk_overlap_sec must be smaller than chunk_duration_sec")
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="qwen3_asr_") as temp:
        source = Path(temp) / f"audio{suffix}"
        audio.save(source)
        duration = audio_duration(source)
        workers = parsed_workers()
        if duration <= single_pass_seconds:
            chunks = materialize_fixed_chunks(
                source,
                Path(temp),
                duration,
                chunk_seconds=max(duration, 1.0),
                overlap_seconds=0,
            )
        else:
            chunks = materialize_fixed_chunks(
                source,
                Path(temp),
                duration,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
            )
        results = dispatch_asr_chunks(
            [worker_url(port) for _gpu, port in workers],
            chunks,
            form,
            request_timeout=REQUEST_TIMEOUT,
        )
        return jsonify(
            merge_asr_results(
                "qwen3_asr",
                results,
                segmentation_mode="fixed",
                audio_duration_seconds=duration,
                worker_count=len(workers),
                segmentation_metadata={
                    "chunk_duration_sec": chunk_seconds,
                    "chunk_overlap_sec": overlap_seconds,
                    "single_pass_max_duration_sec": single_pass_seconds,
                },
            )
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
