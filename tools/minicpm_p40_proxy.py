#!/usr/bin/env python3
"""Lazy OpenAI-compatible proxy for local MiniCPM-V llama.cpp workers."""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class WorkerSpec:
    gpu: int
    port: int
    log_path: Path


class LlamaWorkerPool:
    def __init__(
        self,
        *,
        server_bin: Path,
        model_path: Path,
        mmproj_path: Path,
        model_alias: str,
        host: str,
        workers: list[WorkerSpec],
        ctx_size: int,
        parallel: int,
        startup_timeout: int,
        stop_timeout: int,
        backend_timeout: int,
        extra_args: list[str],
    ) -> None:
        self.server_bin = server_bin
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.model_alias = model_alias
        self.host = host
        self.workers = workers
        self.ctx_size = ctx_size
        self.parallel = parallel
        self.startup_timeout = startup_timeout
        self.stop_timeout = stop_timeout
        self.backend_timeout = backend_timeout
        self.extra_args = extra_args
        self._processes: dict[int, subprocess.Popen] = {}
        self._log_files = {}
        self._last_error: dict[int, str] = {}
        self._next_worker = 0
        self._lock = threading.RLock()

    def shutdown(self) -> None:
        self.stop_all()

    def choose_worker(self) -> WorkerSpec:
        with self._lock:
            self.ensure_ready()
            ready_workers = [worker for worker in self.workers if self._is_http_ready(worker)]
            if not ready_workers:
                errors = "; ".join(
                    f"gpu{worker.gpu}:{self._last_error.get(worker.port, 'not ready')}"
                    for worker in self.workers
                )
                raise RuntimeError(f"no MiniCPM worker is ready ({errors})")
            worker = ready_workers[self._next_worker % len(ready_workers)]
            self._next_worker += 1
            return worker

    def ensure_ready(self) -> None:
        pending_workers: list[WorkerSpec] = []
        for worker in self.workers:
            if self._is_http_ready(worker):
                continue
            proc = self._processes.get(worker.port)
            if proc is None or proc.poll() is not None:
                self._start_worker(worker)
            pending_workers.append(worker)
        self._wait_for_ready(pending_workers)

    def stop_all(self) -> None:
        with self._lock:
            processes = list(self._processes.items())
            self._processes.clear()
        for _, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + self.stop_timeout
        for _, proc in processes:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for handle in list(self._log_files.values()):
            try:
                handle.close()
            except OSError:
                pass
        self._log_files.clear()

    def health(self) -> dict:
        with self._lock:
            workers = []
            for worker in self.workers:
                proc = self._processes.get(worker.port)
                workers.append(
                    {
                        "gpu": worker.gpu,
                        "port": worker.port,
                        "pid": proc.pid if proc and proc.poll() is None else None,
                        "http_ready": self._is_http_ready(worker),
                        "log": str(worker.log_path),
                        "last_error": self._last_error.get(worker.port),
                    }
                )
        return {
            "ready": all(item["http_ready"] for item in workers),
            "model": self.model_alias,
            "workers": workers,
        }

    def _start_worker(self, worker: WorkerSpec) -> None:
        worker.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = worker.log_path.open("ab")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker.gpu)
        env["NO_PROXY"] = env.get("NO_PROXY", "127.0.0.1,localhost")
        env["no_proxy"] = env.get("no_proxy", "127.0.0.1,localhost")
        command = [
            str(self.server_bin),
            "--host",
            self.host,
            "--port",
            str(worker.port),
            "--model",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--alias",
            self.model_alias,
            "--ctx-size",
            str(self.ctx_size),
            "--parallel",
            str(self.parallel),
            "--n-gpu-layers",
            "999",
            "--split-mode",
            "none",
            "--main-gpu",
            "0",
            *self.extra_args,
        ]
        logging.info("starting MiniCPM worker gpu=%s port=%s", worker.gpu, worker.port)
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        self._processes[worker.port] = proc
        old_log = self._log_files.pop(worker.port, None)
        if old_log:
            old_log.close()
        self._log_files[worker.port] = log_file

    def _wait_for_ready(self, workers: Iterable[WorkerSpec]) -> None:
        pending = list(workers)
        if not pending:
            return
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            pending = [worker for worker in pending if not self._is_http_ready(worker)]
            if not pending:
                return
            for worker in pending:
                proc = self._processes.get(worker.port)
                if proc and proc.poll() is not None:
                    self._last_error[worker.port] = f"exited with code {proc.returncode}"
            time.sleep(1)
        for worker in pending:
            self._last_error[worker.port] = f"not ready after {self.startup_timeout}s"

    def _is_http_ready(self, worker: WorkerSpec) -> bool:
        try:
            conn = http.client.HTTPConnection(self.host, worker.port, timeout=2)
            conn.request("GET", "/health")
            response = conn.getresponse()
            response.read()
            conn.close()
            return 200 <= response.status < 300
        except OSError:
            return False


def make_handler(pool: LlamaWorkerPool):
    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path in {"/health", "/api/health"}:
                self._send_json(pool.health())
                return
            if self.path == "/v1/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": pool.model_alias,
                                "object": "model",
                                "owned_by": "video-analyzer",
                            }
                        ],
                    }
                )
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if not self.path.startswith("/v1/"):
                self.send_error(404)
                return
            try:
                worker = pool.choose_worker()
                self._proxy_to_worker(worker)
            except Exception as exc:
                logging.exception("MiniCPM proxy request failed")
                self._send_json({"error": str(exc)}, status=503)

        def _proxy_to_worker(self, worker: WorkerSpec) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            split = urlsplit(self.path)
            path = split.path
            if split.query:
                path = f"{path}?{split.query}"
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
            }
            conn = http.client.HTTPConnection(pool.host, worker.port, timeout=pool.backend_timeout)
            try:
                conn.request(self.command, path, body=body, headers=headers)
                response = conn.getresponse()
                payload = response.read()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in HOP_BY_HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("X-MiniCPM-Worker", f"gpu{worker.gpu}:{worker.port}")
                self.end_headers()
                self.wfile.write(payload)
            finally:
                conn.close()

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: object) -> None:
            logging.info("%s - %s", self.address_string(), fmt % args)

    return ProxyHandler


def parse_workers(value: str, log_dir: Path) -> list[WorkerSpec]:
    workers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        gpu_text, port_text = item.split(":", 1)
        gpu = int(gpu_text)
        port = int(port_text)
        workers.append(WorkerSpec(gpu=gpu, port=port, log_path=log_dir / f"worker-gpu{gpu}-port{port}.log"))
    if not workers:
        raise ValueError("at least one worker is required")
    return workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("MINICPM_PROXY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MINICPM_PROXY_PORT", "18082")))
    parser.add_argument("--backend-host", default=os.getenv("MINICPM_BACKEND_HOST", "127.0.0.1"))
    parser.add_argument("--workers", default=os.getenv("MINICPM_WORKERS", "0:18182,1:18183,2:18184,3:18185,4:18186"))
    parser.add_argument("--log-dir", type=Path, default=Path(os.getenv("MINICPM_LOG_DIR", "tmp/minicpm-p40/logs")))
    parser.add_argument("--server-bin", type=Path, default=Path(os.getenv("MINICPM_SERVER_BIN", "/home/ai/mtp-q8/llama.cpp-mtp/build/bin/llama-server")))
    parser.add_argument("--model-path", type=Path, default=Path(os.getenv("MINICPM_MODEL_PATH", "/home/ai/.lmstudio/models/openbmb/MiniCPM-V-4_5-gguf/ggml-model-Q4_K_M.gguf")))
    parser.add_argument("--mmproj-path", type=Path, default=Path(os.getenv("MINICPM_MMPROJ_PATH", "/home/ai/.lmstudio/models/openbmb/MiniCPM-V-4_5-gguf/mmproj-model-f16.gguf")))
    parser.add_argument("--model-alias", default=os.getenv("MINICPM_MODEL_ALIAS", "minicpm-v-4.5-v100"))
    parser.add_argument("--ctx-size", type=int, default=int(os.getenv("MINICPM_CTX_SIZE", "8192")))
    parser.add_argument("--parallel", type=int, default=int(os.getenv("MINICPM_PARALLEL", "1")))
    parser.add_argument("--startup-timeout", type=int, default=int(os.getenv("MINICPM_STARTUP_TIMEOUT", "300")))
    parser.add_argument("--stop-timeout", type=int, default=int(os.getenv("MINICPM_STOP_TIMEOUT", "30")))
    parser.add_argument("--backend-timeout", type=int, default=int(os.getenv("MINICPM_BACKEND_TIMEOUT", "600")))
    parser.add_argument("--llama-extra-arg", action="append", default=[])
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    for path in (args.server_bin, args.model_path, args.mmproj_path):
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> int:
    args = parse_args()
    validate_paths(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log_dir / "proxy.log"),
            logging.StreamHandler(),
        ],
    )
    pool = LlamaWorkerPool(
        server_bin=args.server_bin,
        model_path=args.model_path,
        mmproj_path=args.mmproj_path,
        model_alias=args.model_alias,
        host=args.backend_host,
        workers=parse_workers(args.workers, args.log_dir),
        ctx_size=args.ctx_size,
        parallel=args.parallel,
        startup_timeout=args.startup_timeout,
        stop_timeout=args.stop_timeout,
        backend_timeout=args.backend_timeout,
        extra_args=args.llama_extra_arg,
    )

    def shutdown(signum: int, _frame: object) -> None:
        logging.info("received signal %s; stopping MiniCPM workers", signum)
        pool.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(pool))
    logging.info("MiniCPM proxy listening on %s:%s", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        pool.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
