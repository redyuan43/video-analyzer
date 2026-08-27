#!/usr/bin/env python3
"""On-demand OpenAI-compatible pool for the local text GGUF model.

The BONSAI_LOCAL_* names are retained as a compatibility interface for the
existing service and workflow configuration.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from flask import Flask, Response, jsonify, request


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_DIR = ROOT / "tmp" / "bonsai-local-pool"
RUNTIME_DIR = Path(os.environ.get("BONSAI_LOCAL_RUNTIME_DIR", DEFAULT_RUNTIME_DIR))
PID_PATH = RUNTIME_DIR / "pool.pid"
STATE_PATH = RUNTIME_DIR / "state.json"
LOG_DIR = RUNTIME_DIR / "logs"
CONFIG_PATH = Path(os.environ.get("BONSAI_LOCAL_CONFIG", RUNTIME_DIR / "config.json"))


def _load_runtime_config() -> dict[str, Any]:
    use_runtime_config = (
        os.environ.get("BONSAI_LOCAL_USE_RUNTIME_CONFIG") == "1"
        or (len(sys.argv) > 1 and sys.argv[1] == "serve")
    )
    if not use_runtime_config:
        return {}
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Local text pool config must be a JSON object: {CONFIG_PATH}")
    return payload


RUNTIME_CONFIG = _load_runtime_config()


def _setting(name: str, default: str) -> str:
    return str(os.environ.get(name, RUNTIME_CONFIG.get(name, default)))


HOST = _setting("BONSAI_LOCAL_HOST", "127.0.0.1")
PORT = int(_setting("BONSAI_LOCAL_PORT", "18103"))
BACKEND_BASE_PORT = int(_setting("BONSAI_LOCAL_BACKEND_BASE_PORT", "18110"))
GPU_IDS = tuple(
    item.strip()
    for item in _setting("BONSAI_LOCAL_GPU_IDS", "3,0,1,2,4,5").split(",")
    if item.strip()
)
WORKER_COUNT = int(_setting("BONSAI_LOCAL_WORKER_COUNT", str(len(GPU_IDS))))
MODEL_PATH = Path(
    _setting(
        "BONSAI_LOCAL_MODEL",
        "/home/ai/model-sources/unsloth-Qwen3.8-27B-Q2_K_XL/"
        "Qwen3.8-27B-UD-Q2_K_XL.gguf",
    )
)
MMPROJ_VALUE = _setting("BONSAI_LOCAL_MMPROJ", "").strip()
MMPROJ_PATH = Path(MMPROJ_VALUE) if MMPROJ_VALUE else None
LLAMA_SERVER = Path(
    _setting(
        "BONSAI_LOCAL_LLAMA_SERVER",
        "/home/ai/llama.cpp-github/build-cuda-qwen38-sm61-sm70/bin/llama-server",
    )
)
MODEL_ALIAS = _setting("BONSAI_LOCAL_MODEL_ALIAS", "Qwen/Qwen3.8-27B-Q2-MTP4")
CONTEXT_SIZE = int(_setting("BONSAI_LOCAL_CONTEXT_SIZE", "65536"))
CACHE_TYPE_K = _setting("BONSAI_LOCAL_CACHE_TYPE_K", "q8_0")
CACHE_TYPE_V = _setting("BONSAI_LOCAL_CACHE_TYPE_V", "q8_0")
SPEC_TYPE = _setting("BONSAI_LOCAL_SPEC_TYPE", "draft-mtp")
SPEC_DRAFT_N_MAX = int(_setting("BONSAI_LOCAL_SPEC_DRAFT_N_MAX", "4"))
DEFAULT_ENABLE_THINKING = (
    _setting("BONSAI_LOCAL_ENABLE_THINKING", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
EXTRA_ARGS = tuple(shlex.split(_setting("BONSAI_LOCAL_EXTRA_ARGS", "")))
REQUEST_TIMEOUT = float(_setting("BONSAI_LOCAL_REQUEST_TIMEOUT", "1800"))
ACQUIRE_TIMEOUT = float(_setting("BONSAI_LOCAL_ACQUIRE_TIMEOUT", "900"))
STARTUP_TIMEOUT = float(_setting("BONSAI_LOCAL_STARTUP_TIMEOUT", "900"))


@dataclass(frozen=True)
class Worker:
    gpu_id: str
    gpu_uuid: str
    name: str
    port: int

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def gpu_inventory() -> dict[str, tuple[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    inventory: dict[str, tuple[str, str]] = {}
    for line in result.stdout.splitlines():
        index, gpu_uuid, name = (part.strip() for part in line.split(",", 2))
        inventory[index] = (gpu_uuid, name)
    return inventory


def selected_gpu_ids() -> tuple[str, ...]:
    if not 1 <= WORKER_COUNT <= len(GPU_IDS):
        raise RuntimeError(
            "BONSAI_LOCAL_WORKER_COUNT must be between 1 and the configured GPU count"
        )
    return GPU_IDS[:WORKER_COUNT]


def configured_workers() -> list[Worker]:
    inventory = gpu_inventory()
    workers: list[Worker] = []
    for offset, gpu_id in enumerate(selected_gpu_ids()):
        if gpu_id not in inventory:
            raise RuntimeError(f"Configured BONSAI GPU {gpu_id} is not present")
        gpu_uuid, name = inventory[gpu_id]
        if "P40" not in name and "V100" not in name:
            raise RuntimeError(
                f"Configured local text GPU {gpu_id} is not a Tesla P40 or V100: {name}"
            )
        workers.append(Worker(gpu_id, gpu_uuid, name, BACKEND_BASE_PORT + offset))
    if not workers:
        raise RuntimeError("At least one BONSAI_LOCAL_GPU_IDS entry is required")
    return workers


def worker_command(worker: Worker) -> list[str]:
    command = [
        str(LLAMA_SERVER),
        "--host",
        "127.0.0.1",
        "--port",
        str(worker.port),
        "--device",
        "CUDA0",
        "--model",
        str(MODEL_PATH),
        "--alias",
        MODEL_ALIAS,
        "--ctx-size",
        str(CONTEXT_SIZE),
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--fit",
        "off",
        "--parallel",
        "1",
        "--batch-size",
        "1024",
        "--ubatch-size",
        "1024",
        "--cache-type-k",
        CACHE_TYPE_K,
        "--cache-type-v",
        CACHE_TYPE_V,
        "--flash-attn",
        "on",
        "--kv-offload",
        "--jinja",
        "--no-webui",
    ]
    if SPEC_TYPE != "none":
        command.extend(
            [
                "--spec-type",
                SPEC_TYPE,
                "--spec-draft-n-max",
                str(SPEC_DRAFT_N_MAX),
                "--spec-draft-ngl",
                "all",
                "--spec-draft-type-k",
                CACHE_TYPE_K,
                "--spec-draft-type-v",
                CACHE_TYPE_V,
            ]
        )
    if MMPROJ_PATH is not None:
        command.extend(
            [
                "--mmproj",
                str(MMPROJ_PATH),
                "--mmproj-offload",
                "--image-min-tokens",
                "1024",
            ]
        )
    command.extend(EXTRA_ARGS)
    return command


def worker_env(worker: Worker) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": worker.gpu_uuid,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "BONSAI_LOCAL_POOL_WORKER": worker.gpu_id,
        }
    )
    return env


def worker_ready(worker: Worker) -> bool:
    try:
        response = requests.get(f"{worker.url}/health", timeout=2, proxies={"http": None, "https": None})
        return response.ok
    except requests.RequestException:
        return False


def pool_health() -> dict[str, Any] | None:
    try:
        response = requests.get(f"http://{HOST}:{PORT}/api/health", timeout=2, proxies={"http": None, "https": None})
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


def read_pid() -> int | None:
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_state(workers: list[Worker]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": HOST,
        "port": PORT,
        "model": MODEL_ALIAS,
        "context_size": CONTEXT_SIZE,
        "cache": {"k": CACHE_TYPE_K, "v": CACHE_TYPE_V},
        "speculative": {"type": SPEC_TYPE, "draft_n_max": SPEC_DRAFT_N_MAX},
        "workers": [asdict(worker) for worker in workers],
        "started_at": time.time(),
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_state() -> None:
    for path in (PID_PATH, STATE_PATH):
        path.unlink(missing_ok=True)


def port_pids(port: int) -> list[int]:
    result = subprocess.run(
        ["fuser", "-n", "tcp", str(port)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = f"{result.stdout}\n{result.stderr}"
    return [int(token) for token in output.split() if token.isdigit()]


def command_line(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def stop_process_group(pid: int) -> None:
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not process_alive(pid):
            return
        time.sleep(0.2)
    if process_alive(pid):
        os.killpg(pgid, signal.SIGKILL)


def stop_foreign_bonsai_listener() -> None:
    for pid in port_pids(PORT):
        cmdline = command_line(pid)
        if "llama-server" not in cmdline or str(MODEL_PATH) not in cmdline:
            raise RuntimeError(f"Port {PORT} is occupied by unrelated process pid={pid}: {cmdline}")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process_alive(pid):
            time.sleep(0.2)
        if process_alive(pid):
            os.kill(pid, signal.SIGKILL)


def _spawn_pool(gpu_ids: tuple[str, ...]) -> subprocess.Popen[bytes]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BONSAI_LOCAL_GPU_IDS"] = ",".join(gpu_ids)
    env["BONSAI_LOCAL_WORKER_COUNT"] = str(len(gpu_ids))
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "serve"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=(LOG_DIR / "pool.log").open("ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_for_pool(process: subprocess.Popen[bytes]) -> dict[str, Any] | None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        health = pool_health()
        if health and health.get("ready_workers") == health.get("worker_count"):
            return health
        if process.poll() is not None:
            break
        time.sleep(1)
    return None


def start_pool() -> int:
    health = pool_health()
    if health:
        print(json.dumps(health, ensure_ascii=False))
        return 0
    pid = read_pid()
    if process_alive(pid):
        raise RuntimeError(f"BONSAI pool pid={pid} is running but health check failed")
    clear_state()
    stop_foreign_bonsai_listener()
    gpu_ids = selected_gpu_ids()
    process = _spawn_pool(gpu_ids)
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    health = _wait_for_pool(process)
    if health:
        print(json.dumps(health, ensure_ascii=False))
        return 0
    stop_process_group(process.pid)
    clear_state()
    raise RuntimeError(f"BONSAI pool did not become ready; see {LOG_DIR / 'pool.log'}")


def stop_pool() -> int:
    pid = read_pid()
    if process_alive(pid):
        stop_process_group(pid)
    else:
        for candidate in port_pids(PORT):
            cmdline = command_line(candidate)
            if "bonsai_local_pool.py serve" in cmdline:
                stop_process_group(candidate)
    clear_state()
    print("BONSAI local pool stopped")
    return 0


class PoolServer:
    def __init__(self, workers: list[Worker]) -> None:
        self.workers = workers
        self.worker_priority = {
            worker.port: priority for priority, worker in enumerate(workers)
        }
        self.available: queue.PriorityQueue[tuple[int, Worker]] = (
            queue.PriorityQueue()
        )
        for worker in workers:
            self._release(worker)
        self.app = Flask("bonsai_local_pool")
        self._register_routes()

    def _release(self, worker: Worker) -> None:
        self.available.put((self.worker_priority[worker.port], worker))

    def _available_gpu_ids(self) -> list[str]:
        with self.available.mutex:
            return [
                worker.gpu_id
                for _, worker in sorted(self.available.queue)
            ]

    def _register_routes(self) -> None:
        @self.app.get("/api/health")
        @self.app.get("/health")
        def health() -> Response:
            ready = sum(worker_ready(worker) for worker in self.workers)
            return jsonify(
                {
                    "ok": ready == len(self.workers),
                    "model": MODEL_ALIAS,
                    "worker_count": len(self.workers),
                    "ready_workers": ready,
                    "available_workers": self.available.qsize(),
                    "available_gpu_ids": self._available_gpu_ids(),
                    "context_size": CONTEXT_SIZE,
                    "cache": {"k": CACHE_TYPE_K, "v": CACHE_TYPE_V},
                    "speculative": {
                        "type": SPEC_TYPE,
                        "draft_n_max": SPEC_DRAFT_N_MAX,
                    },
                    "workers": [
                        {
                            "gpu_id": worker.gpu_id,
                            "name": worker.name,
                            "port": worker.port,
                            "ready": worker_ready(worker),
                        }
                        for worker in self.workers
                    ],
                }
            )

        @self.app.get("/v1/models")
        def models() -> Response:
            return jsonify({"object": "list", "data": [{"id": MODEL_ALIAS, "object": "model"}]})

        @self.app.route("/v1/<path:path>", methods=["POST"])
        def forward(path: str) -> Response:
            acquire_timeout = ACQUIRE_TIMEOUT
            requested_timeout = request.headers.get("X-Bonsai-Acquire-Timeout")
            if requested_timeout is not None:
                try:
                    acquire_timeout = min(
                        ACQUIRE_TIMEOUT,
                        max(0.0, float(requested_timeout)),
                    )
                except (TypeError, ValueError):
                    return jsonify(
                        {
                            "error": {
                                "message": "invalid X-Bonsai-Acquire-Timeout",
                                "type": "invalid_request_error",
                            }
                        }
                    ), 400
            try:
                _, worker = self.available.get(timeout=acquire_timeout)
            except queue.Empty:
                return jsonify({"error": {"message": "BONSAI worker pool is busy", "type": "server_error"}}), 503
            try:
                body = request.get_json(silent=True)
                if isinstance(body, dict) and path == "chat/completions":
                    body.setdefault("model", MODEL_ALIAS)
                    template_kwargs = body.setdefault("chat_template_kwargs", {})
                    if isinstance(template_kwargs, dict):
                        template_kwargs.setdefault(
                            "enable_thinking",
                            DEFAULT_ENABLE_THINKING,
                        )
                        template_kwargs.setdefault("preserve_thinking", False)
                    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                else:
                    payload = request.get_data()
                headers = {
                    key: value
                    for key, value in request.headers.items()
                    if key.lower() in {"authorization", "content-type", "accept"}
                }
                response = requests.post(
                    f"{worker.url}/v1/{path}",
                    data=payload,
                    headers=headers,
                    timeout=(30, REQUEST_TIMEOUT),
                    stream=bool(isinstance(body, dict) and body.get("stream")),
                    proxies={"http": None, "https": None},
                )
                if isinstance(body, dict) and body.get("stream"):
                    def stream() -> Any:
                        try:
                            yield from response.iter_content(chunk_size=8192)
                        finally:
                            response.close()
                            self._release(worker)

                    return Response(stream(), status=response.status_code, content_type=response.headers.get("content-type"))
                return Response(
                    response.content,
                    status=response.status_code,
                    content_type=response.headers.get("content-type", "application/json"),
                )
            except requests.RequestException as error:
                return jsonify({"error": {"message": str(error), "type": "backend_error"}}), 502
            finally:
                if not (isinstance(locals().get("body"), dict) and locals()["body"].get("stream")):
                    self._release(worker)


def serve_pool() -> int:
    if not LLAMA_SERVER.is_file() or not os.access(LLAMA_SERVER, os.X_OK):
        raise RuntimeError(f"llama-server is not executable: {LLAMA_SERVER}")
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Local text model file is missing: {MODEL_PATH}")
    if MMPROJ_PATH is not None and not MMPROJ_PATH.is_file():
        raise RuntimeError(f"Local text mmproj file is missing: {MMPROJ_PATH}")
    workers = configured_workers()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    files = []

    def cleanup(*_args: object) -> None:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in files:
            handle.close()
        clear_state()

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_args: sys.exit(0))

    for worker in workers:
        handle = (LOG_DIR / f"worker-gpu{worker.gpu_id}.log").open("ab")
        files.append(handle)
        processes.append(
            subprocess.Popen(
                worker_command(worker),
                cwd=ROOT,
                env=worker_env(worker),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        )
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if all(worker_ready(worker) for worker in workers):
            break
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A BONSAI worker exited before startup completed")
        time.sleep(1)
    else:
        raise RuntimeError("BONSAI workers did not become ready before timeout")
    write_state(workers)
    PoolServer(workers).app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status", "serve"))
    args = parser.parse_args()
    if args.command == "start":
        return start_pool()
    if args.command == "stop":
        return stop_pool()
    if args.command == "status":
        print(json.dumps(pool_health() or {"ok": False, "running": process_alive(read_pid())}, ensure_ascii=False))
        return 0 if pool_health() else 1
    return serve_pool()


if __name__ == "__main__":
    raise SystemExit(main())
