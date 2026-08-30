#!/usr/bin/env python3
"""Ray-coordinated OpenAI-compatible pool for the local Qwen3.8 Q4 model."""

from __future__ import annotations

import argparse
import atexit
import asyncio
import hashlib
import heapq
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from flask import Flask, Response, jsonify, request

try:
    import ray
except ImportError:  # pragma: no cover - service startup validates Ray.
    ray = None


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
GPU_SELECTION = _setting("BONSAI_LOCAL_GPU_SELECTION", "auto").strip().lower()
GPU_IDS = tuple(
    item.strip()
    for item in _setting("BONSAI_LOCAL_GPU_IDS", "").split(",")
    if item.strip()
)
WORKER_COUNT_VALUE = _setting("BONSAI_LOCAL_WORKER_COUNT", "auto").strip().lower()
MODEL_PATH = Path(
    _setting(
        "BONSAI_LOCAL_MODEL",
        "/home/ai/model-sources/huihui-ai-Huihui-Qwen3.8-27B-abliterated-GGUF/"
        "Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf",
    )
)
DRAFT_MODEL_PATH = Path(
    _setting(
        "BONSAI_LOCAL_DRAFT_MODEL",
        "/home/ai/model-sources/Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
    )
)
LLAMA_SERVER = Path(
    _setting(
        "BONSAI_LOCAL_LLAMA_SERVER",
        "/home/ai/llama.cpp-github/build-cuda-dflash2-pr27342/bin/llama-server",
    )
)
MODEL_ALIAS = _setting(
    "BONSAI_LOCAL_MODEL_ALIAS",
    "huihui/Qwen3.8-27B-Q4-DFlash2",
)
CONTEXT_SIZE = int(_setting("BONSAI_LOCAL_CONTEXT_SIZE", "65536"))
SPEC_DRAFT_N_MAX = int(_setting("BONSAI_LOCAL_SPEC_DRAFT_N_MAX", "5"))
V100_32_CACHE_TYPE = _setting("BONSAI_LOCAL_V100_32_CACHE_TYPE", "f16")
V100_16_P40_CACHE_TYPE = _setting(
    "BONSAI_LOCAL_V100_16_P40_CACHE_TYPE",
    "q8_0",
)
P40_CACHE_TYPE = _setting("BONSAI_LOCAL_P40_CACHE_TYPE", "q8_0")
V100_16_P40_TENSOR_SPLIT = _setting(
    "BONSAI_LOCAL_V100_16_P40_TENSOR_SPLIT",
    "2,3",
)
V100_32_MIN_FREE_MIB = int(
    _setting("BONSAI_LOCAL_V100_32_MIN_FREE_MIB", "30000")
)
V100_16_MIN_FREE_MIB = int(
    _setting("BONSAI_LOCAL_V100_16_MIN_FREE_MIB", "15000")
)
P40_MIN_FREE_MIB = int(_setting("BONSAI_LOCAL_P40_MIN_FREE_MIB", "22000"))
RECONCILE_SECONDS = float(_setting("BONSAI_LOCAL_RECONCILE_SECONDS", "10"))
DEFAULT_ENABLE_THINKING = (
    _setting("BONSAI_LOCAL_ENABLE_THINKING", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
EXTRA_ARGS = tuple(shlex.split(_setting("BONSAI_LOCAL_EXTRA_ARGS", "")))
REQUEST_TIMEOUT = float(_setting("BONSAI_LOCAL_REQUEST_TIMEOUT", "1800"))
ACQUIRE_TIMEOUT = float(_setting("BONSAI_LOCAL_ACQUIRE_TIMEOUT", "900"))
STARTUP_TIMEOUT = float(_setting("BONSAI_LOCAL_STARTUP_TIMEOUT", "900"))

TIER_V100_32 = "v100_32_single"
TIER_V100_16_P40 = "v100_16_p40_pair"
TIER_P40 = "p40_single"
TIER_PRIORITY = {
    TIER_V100_32: 0,
    TIER_V100_16_P40: 1,
    TIER_P40: 2,
}


@dataclass(frozen=True)
class GPU:
    index: str
    uuid: str
    name: str
    memory_total_mib: int
    memory_free_mib: int

    @property
    def kind(self) -> str:
        if "V100" in self.name and self.memory_total_mib >= 30000:
            return "v100_32"
        if "V100" in self.name:
            return "v100_16"
        if "P40" in self.name:
            return "p40"
        return "unsupported"


@dataclass(frozen=True)
class Worker:
    worker_id: str
    tier: str
    priority: int
    gpu_ids: tuple[str, ...]
    gpu_uuids: tuple[str, ...]
    names: tuple[str, ...]
    port: int
    cache_type_k: str
    cache_type_v: str
    tensor_split: str = ""

    @property
    def gpu_id(self) -> str:
        return ",".join(self.gpu_ids)

    @property
    def gpu_uuid(self) -> str:
        return ",".join(self.gpu_uuids)

    @property
    def name(self) -> str:
        return " + ".join(self.names)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def public_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "gpu_ids": list(self.gpu_ids),
            "gpu_uuids": list(self.gpu_uuids),
            "names": list(self.names),
            "url": self.url,
        }


def parse_gpu_inventory(output: str) -> list[GPU]:
    gpus: list[GPU] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",", 4)]
        if len(fields) != 5:
            raise ValueError(f"Unexpected nvidia-smi GPU row: {line}")
        gpus.append(
            GPU(
                index=fields[0],
                uuid=fields[1],
                name=fields[2],
                memory_total_mib=int(fields[3]),
                memory_free_mib=int(fields[4]),
            )
        )
    return gpus


def parse_compute_processes(output: str) -> dict[str, list[dict[str, Any]]]:
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",", 3)]
        if len(fields) != 4:
            raise ValueError(f"Unexpected nvidia-smi process row: {line}")
        by_uuid.setdefault(fields[0], []).append(
            {
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": (
                    None if fields[3] in {"", "N/A", "[N/A]"} else int(fields[3])
                ),
            }
        )
    return by_uuid


def gpu_inventory() -> list[GPU]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return parse_gpu_inventory(result.stdout)


def gpu_compute_processes() -> dict[str, list[dict[str, Any]]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return parse_compute_processes(result.stdout or "")


def selected_manual_gpu_ids() -> tuple[str, ...]:
    if GPU_SELECTION != "manual":
        return ()
    if not GPU_IDS:
        raise RuntimeError("Manual local text GPU selection requires GPU IDs")
    if WORKER_COUNT_VALUE == "auto":
        return GPU_IDS
    count = int(WORKER_COUNT_VALUE)
    if not 1 <= count <= len(GPU_IDS):
        raise RuntimeError(
            "BONSAI_LOCAL_WORKER_COUNT must fit the configured GPU ID list"
        )
    return GPU_IDS[:count]


def available_gpus(
    gpus: list[GPU],
    processes: dict[str, list[dict[str, Any]]],
    *,
    allowed_gpu_ids: tuple[str, ...] = (),
    claimed_gpu_uuids: set[str] | None = None,
) -> list[GPU]:
    allowed = set(allowed_gpu_ids)
    claimed = claimed_gpu_uuids or set()
    available: list[GPU] = []
    for gpu in gpus:
        if allowed and gpu.index not in allowed:
            continue
        if gpu.kind == "unsupported" or gpu.uuid in claimed:
            continue
        if processes.get(gpu.uuid):
            continue
        if gpu.kind == "v100_32" and gpu.memory_free_mib < V100_32_MIN_FREE_MIB:
            continue
        if gpu.kind == "v100_16" and gpu.memory_free_mib < V100_16_MIN_FREE_MIB:
            continue
        if gpu.kind == "p40" and gpu.memory_free_mib < P40_MIN_FREE_MIB:
            continue
        available.append(gpu)
    return available


def plan_workers(
    gpus: list[GPU],
    *,
    base_port: int = BACKEND_BASE_PORT,
    used_ports: set[int] | None = None,
) -> list[Worker]:
    used = set(used_ports or set())
    next_port = base_port

    def allocate_port() -> int:
        nonlocal next_port
        while next_port in used:
            next_port += 1
        port = next_port
        used.add(port)
        next_port += 1
        return port

    v100_32 = sorted(
        (gpu for gpu in gpus if gpu.kind == "v100_32"),
        key=lambda gpu: (-gpu.memory_free_mib, int(gpu.index)),
    )
    v100_16 = sorted(
        (gpu for gpu in gpus if gpu.kind == "v100_16"),
        key=lambda gpu: (-gpu.memory_free_mib, int(gpu.index)),
    )
    p40 = sorted(
        (gpu for gpu in gpus if gpu.kind == "p40"),
        key=lambda gpu: (-gpu.memory_free_mib, int(gpu.index)),
    )
    workers: list[Worker] = []
    for gpu in v100_32:
        workers.append(
            Worker(
                worker_id=f"v10032-{gpu.uuid}",
                tier=TIER_V100_32,
                priority=TIER_PRIORITY[TIER_V100_32],
                gpu_ids=(gpu.index,),
                gpu_uuids=(gpu.uuid,),
                names=(gpu.name,),
                port=allocate_port(),
                cache_type_k=V100_32_CACHE_TYPE,
                cache_type_v=V100_32_CACHE_TYPE,
            )
        )
    while v100_16 and p40:
        v100 = v100_16.pop(0)
        partner = p40.pop(0)
        workers.append(
            Worker(
                worker_id=f"v10016-p40-{v100.uuid}-{partner.uuid}",
                tier=TIER_V100_16_P40,
                priority=TIER_PRIORITY[TIER_V100_16_P40],
                gpu_ids=(v100.index, partner.index),
                gpu_uuids=(v100.uuid, partner.uuid),
                names=(v100.name, partner.name),
                port=allocate_port(),
                cache_type_k=V100_16_P40_CACHE_TYPE,
                cache_type_v=V100_16_P40_CACHE_TYPE,
                tensor_split=V100_16_P40_TENSOR_SPLIT,
            )
        )
    for gpu in p40:
        workers.append(
            Worker(
                worker_id=f"p40-{gpu.uuid}",
                tier=TIER_P40,
                priority=TIER_PRIORITY[TIER_P40],
                gpu_ids=(gpu.index,),
                gpu_uuids=(gpu.uuid,),
                names=(gpu.name,),
                port=allocate_port(),
                cache_type_k=P40_CACHE_TYPE,
                cache_type_v=P40_CACHE_TYPE,
            )
        )
    return sorted(workers, key=lambda worker: (worker.priority, worker.port))


def configured_workers() -> list[Worker]:
    processes = gpu_compute_processes()
    allowed_ids = selected_manual_gpu_ids()
    candidates = available_gpus(
        gpu_inventory(),
        processes,
        allowed_gpu_ids=allowed_ids,
    )
    workers = plan_workers(candidates)
    if not workers:
        raise RuntimeError("No compatible idle GPU topology is available for Qwen3.8 Q4")
    return workers


def worker_command(worker: Worker) -> list[str]:
    command = [
        str(LLAMA_SERVER),
        "--host",
        "127.0.0.1",
        "--port",
        str(worker.port),
        "--model",
        str(MODEL_PATH),
        "--alias",
        MODEL_ALIAS,
        "--ctx-size",
        str(CONTEXT_SIZE),
        "--n-gpu-layers",
        "all",
        "--fit",
        "off",
        "--parallel",
        "1",
        "--batch-size",
        "1024",
        "--ubatch-size",
        "1024",
        "--cache-type-k",
        worker.cache_type_k,
        "--cache-type-v",
        worker.cache_type_v,
        "--flash-attn",
        "on",
        "--kv-offload",
        "--jinja",
        "--no-webui",
        "--reasoning",
        "off",
        "--timeout",
        "7200",
        "--spec-type",
        "draft-dflash",
        "--spec-draft-model",
        str(DRAFT_MODEL_PATH),
        "--spec-draft-n-max",
        str(SPEC_DRAFT_N_MAX),
        "--spec-draft-ngl",
        "all",
    ]
    if len(worker.gpu_uuids) == 1:
        command.extend(
            [
                "--device",
                "CUDA0",
                "--split-mode",
                "none",
                "--main-gpu",
                "0",
            ]
        )
    else:
        command.extend(
            [
                "--split-mode",
                "layer",
                "--tensor-split",
                worker.tensor_split,
                "--main-gpu",
                "0",
            ]
        )
    command.extend(EXTRA_ARGS)
    return command


def worker_env(worker: Worker) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(worker.gpu_uuids),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "BONSAI_LOCAL_POOL_WORKER": worker.worker_id,
        }
    )
    return env


def worker_ready(worker: Worker) -> bool:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(f"{worker.url}/health", timeout=2)
        return response.ok
    except requests.RequestException:
        return False
    finally:
        session.close()


def runtime_fingerprint() -> str:
    payload = {
        "model": str(MODEL_PATH),
        "draft_model": str(DRAFT_MODEL_PATH),
        "llama_server": str(LLAMA_SERVER),
        "alias": MODEL_ALIAS,
        "context_size": CONTEXT_SIZE,
        "spec_draft_n_max": SPEC_DRAFT_N_MAX,
        "v100_32_cache": V100_32_CACHE_TYPE,
        "v100_16_p40_cache": V100_16_P40_CACHE_TYPE,
        "p40_cache": P40_CACHE_TYPE,
        "v100_16_p40_tensor_split": V100_16_P40_TENSOR_SPLIT,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


if ray is not None:

    @ray.remote(max_concurrency=128, num_cpus=0.2, num_gpus=0)
    class TextPoolCoordinatorActor:
        def __init__(self, worker_payloads: list[dict[str, Any]]) -> None:
            self.condition = asyncio.Condition()
            self.workers: dict[str, dict[str, Any]] = {}
            self.available: list[tuple[int, int, str]] = []
            self.available_ids: set[str] = set()
            self.leased_ids: set[str] = set()
            self.sequence = 0
            self._add_workers(worker_payloads)

        def _release_locked(self, worker_id: str) -> None:
            if worker_id not in self.workers or worker_id in self.available_ids:
                return
            worker = self.workers[worker_id]
            self.sequence += 1
            heapq.heappush(
                self.available,
                (int(worker["priority"]), self.sequence, worker_id),
            )
            self.available_ids.add(worker_id)
            self.leased_ids.discard(worker_id)

        def _add_workers(self, worker_payloads: list[dict[str, Any]]) -> None:
            for worker in worker_payloads:
                worker_id = str(worker["worker_id"])
                self.workers[worker_id] = dict(worker)
                self._release_locked(worker_id)

        async def add_workers(self, worker_payloads: list[dict[str, Any]]) -> None:
            async with self.condition:
                self._add_workers(worker_payloads)
                self.condition.notify_all()

        async def remove_workers(self, worker_ids: list[str]) -> None:
            async with self.condition:
                for worker_id in worker_ids:
                    self.workers.pop(worker_id, None)
                    self.available_ids.discard(worker_id)
                    self.leased_ids.discard(worker_id)
                self.available = [
                    item for item in self.available if item[2] in self.workers
                ]
                heapq.heapify(self.available)
                self.condition.notify_all()

        async def acquire(self, timeout: float) -> dict[str, Any] | None:
            deadline = time.monotonic() + max(0.0, timeout)
            async with self.condition:
                while True:
                    while self.available:
                        _priority, _sequence, worker_id = heapq.heappop(
                            self.available
                        )
                        if (
                            worker_id not in self.workers
                            or worker_id not in self.available_ids
                        ):
                            continue
                        self.available_ids.remove(worker_id)
                        self.leased_ids.add(worker_id)
                        return dict(self.workers[worker_id])
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    try:
                        await asyncio.wait_for(self.condition.wait(), remaining)
                    except asyncio.TimeoutError:
                        return None

        async def release(self, worker_id: str) -> None:
            async with self.condition:
                self._release_locked(worker_id)
                self.condition.notify_all()

        async def snapshot(self) -> dict[str, Any]:
            async with self.condition:
                return {
                    "workers": list(self.workers.values()),
                    "available_worker_ids": sorted(self.available_ids),
                    "leased_worker_ids": sorted(self.leased_ids),
                }


class LocalCoordinator:
    """Test-only coordinator with the same lease contract as the Ray actor."""

    def __init__(self, workers: list[Worker]) -> None:
        self.condition = threading.Condition()
        self.workers = {worker.worker_id: worker for worker in workers}
        self.available: list[tuple[int, int, str]] = []
        self.available_ids: set[str] = set()
        self.leased_ids: set[str] = set()
        self.sequence = 0
        self.add_workers(workers)

    def add_workers(self, workers: list[Worker]) -> None:
        with self.condition:
            for worker in workers:
                self.workers[worker.worker_id] = worker
                if worker.worker_id in self.available_ids:
                    continue
                self.sequence += 1
                heapq.heappush(
                    self.available,
                    (worker.priority, self.sequence, worker.worker_id),
                )
                self.available_ids.add(worker.worker_id)
            self.condition.notify_all()

    def remove_workers(self, worker_ids: list[str]) -> None:
        with self.condition:
            for worker_id in worker_ids:
                self.workers.pop(worker_id, None)
                self.available_ids.discard(worker_id)
                self.leased_ids.discard(worker_id)
            self.available = [
                item for item in self.available if item[2] in self.workers
            ]
            heapq.heapify(self.available)
            self.condition.notify_all()

    def acquire(self, timeout: float) -> Worker | None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self.condition:
            while True:
                while self.available:
                    _priority, _sequence, worker_id = heapq.heappop(
                        self.available
                    )
                    if (
                        worker_id not in self.workers
                        or worker_id not in self.available_ids
                    ):
                        continue
                    self.available_ids.remove(worker_id)
                    self.leased_ids.add(worker_id)
                    return self.workers[worker_id]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(timeout=remaining)

    def release(self, worker_id: str) -> None:
        with self.condition:
            if worker_id not in self.workers or worker_id in self.available_ids:
                return
            worker = self.workers[worker_id]
            self.sequence += 1
            heapq.heappush(
                self.available,
                (worker.priority, self.sequence, worker_id),
            )
            self.available_ids.add(worker_id)
            self.leased_ids.discard(worker_id)
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {
                "workers": [
                    worker.public_dict() for worker in self.workers.values()
                ],
                "available_worker_ids": sorted(self.available_ids),
                "leased_worker_ids": sorted(self.leased_ids),
            }


class RayCoordinatorClient:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    def add_workers(self, workers: list[Worker]) -> None:
        ray.get(
            self.actor.add_workers.remote(
                [worker.public_dict() for worker in workers]
            )
        )

    def remove_workers(self, worker_ids: list[str]) -> None:
        ray.get(self.actor.remove_workers.remote(worker_ids))

    def acquire(self, timeout: float) -> Worker | None:
        payload = ray.get(
            self.actor.acquire.remote(timeout),
            timeout=max(5.0, timeout + 5.0),
        )
        return worker_from_payload(payload) if payload else None

    def release(self, worker_id: str) -> None:
        ray.get(self.actor.release.remote(worker_id), timeout=10)

    def snapshot(self) -> dict[str, Any]:
        return ray.get(self.actor.snapshot.remote(), timeout=10)


def worker_from_payload(payload: dict[str, Any]) -> Worker:
    return Worker(
        worker_id=str(payload["worker_id"]),
        tier=str(payload["tier"]),
        priority=int(payload["priority"]),
        gpu_ids=tuple(str(item) for item in payload["gpu_ids"]),
        gpu_uuids=tuple(str(item) for item in payload["gpu_uuids"]),
        names=tuple(str(item) for item in payload["names"]),
        port=int(payload["port"]),
        cache_type_k=str(payload["cache_type_k"]),
        cache_type_v=str(payload["cache_type_v"]),
        tensor_split=str(payload.get("tensor_split") or ""),
    )


class PoolRuntime:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.workers: dict[str, Worker] = {}
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_files: dict[str, Any] = {}
        self.failures: dict[str, str] = {}
        self.stop_event = threading.Event()
        self.coordinator: RayCoordinatorClient | None = None
        self.monitor_thread: threading.Thread | None = None

    def _launch_worker(self, worker: Worker) -> None:
        log_path = LOG_DIR / f"worker-{worker.worker_id.replace(':', '_')}.log"
        handle = log_path.open("ab")
        try:
            process = subprocess.Popen(
                worker_command(worker),
                cwd=ROOT,
                env=worker_env(worker),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            handle.close()
            raise
        self.workers[worker.worker_id] = worker
        self.processes[worker.worker_id] = process
        self.log_files[worker.worker_id] = handle

    def _wait_for_workers(self, worker_ids: list[str]) -> list[Worker]:
        pending = set(worker_ids)
        ready: list[Worker] = []
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while pending and time.monotonic() < deadline:
            for worker_id in list(pending):
                process = self.processes[worker_id]
                worker = self.workers[worker_id]
                if process.poll() is not None:
                    self.failures[worker_id] = (
                        f"backend exited with code {process.returncode}"
                    )
                    pending.remove(worker_id)
                    self._discard_worker(worker_id)
                    continue
                if worker_ready(worker):
                    ready.append(worker)
                    pending.remove(worker_id)
            if pending:
                time.sleep(1)
        for worker_id in list(pending):
            self.failures[worker_id] = "backend startup timed out"
            self._discard_worker(worker_id)
        return ready

    def _discard_worker(self, worker_id: str) -> None:
        process = self.processes.pop(worker_id, None)
        if process is not None and process.poll() is None:
            stop_process_group(process.pid)
        handle = self.log_files.pop(worker_id, None)
        if handle is not None:
            handle.close()
        self.workers.pop(worker_id, None)

    def start_initial(self) -> list[Worker]:
        candidates = configured_workers()
        with self.lock:
            for worker in candidates:
                self._launch_worker(worker)
            ready = self._wait_for_workers(
                [worker.worker_id for worker in candidates]
            )
        if not ready:
            raise RuntimeError(
                f"No Qwen3.8 Q4 workers became ready; see {LOG_DIR}"
            )
        return ready

    def attach_coordinator(self, coordinator: RayCoordinatorClient) -> None:
        self.coordinator = coordinator
        self.monitor_thread = threading.Thread(
            target=self._monitor,
            name="qwen38-ray-pool-monitor",
            daemon=True,
        )
        self.monitor_thread.start()

    def _monitor(self) -> None:
        while not self.stop_event.wait(max(1.0, RECONCILE_SECONDS)):
            try:
                self.reconcile()
            except Exception as exc:
                self.failures["reconcile"] = str(exc)

    def reconcile(self) -> None:
        if self.coordinator is None:
            return
        removed: list[str] = []
        with self.lock:
            for worker_id, process in list(self.processes.items()):
                if process.poll() is not None:
                    self.failures[worker_id] = (
                        f"backend exited with code {process.returncode}"
                    )
                    self._discard_worker(worker_id)
                    removed.append(worker_id)
            if removed:
                self.coordinator.remove_workers(removed)

            claimed = {
                gpu_uuid
                for worker in self.workers.values()
                for gpu_uuid in worker.gpu_uuids
            }
            used_ports = {worker.port for worker in self.workers.values()}
            candidates = available_gpus(
                gpu_inventory(),
                gpu_compute_processes(),
                allowed_gpu_ids=selected_manual_gpu_ids(),
                claimed_gpu_uuids=claimed,
            )
            planned = plan_workers(
                candidates,
                used_ports=used_ports,
            )
            new_workers = [
                worker
                for worker in planned
                if not set(worker.gpu_uuids) & claimed
                and worker.worker_id not in self.workers
            ]
            for worker in new_workers:
                self._launch_worker(worker)
            ready = self._wait_for_workers(
                [worker.worker_id for worker in new_workers]
            )
            if ready:
                self.coordinator.add_workers(ready)
        self.write_state()

    def health(self) -> dict[str, Any]:
        snapshot = (
            self.coordinator.snapshot()
            if self.coordinator is not None
            else {
                "workers": [],
                "available_worker_ids": [],
                "leased_worker_ids": [],
            }
        )
        available_ids = set(snapshot["available_worker_ids"])
        leased_ids = set(snapshot["leased_worker_ids"])
        workers = []
        with self.lock:
            for worker in sorted(
                self.workers.values(),
                key=lambda item: (item.priority, item.port),
            ):
                workers.append(
                    {
                        **worker.public_dict(),
                        "ready": worker_ready(worker),
                        "state": (
                            "leased"
                            if worker.worker_id in leased_ids
                            else "available"
                            if worker.worker_id in available_ids
                            else "starting"
                        ),
                    }
                )
        ready_count = sum(bool(worker["ready"]) for worker in workers)
        return {
            "ok": ready_count > 0 and self.coordinator is not None,
            "orchestrator": "ray_actor",
            "model": MODEL_ALIAS,
            "model_path": str(MODEL_PATH),
            "draft_model_path": str(DRAFT_MODEL_PATH),
            "runtime_fingerprint": runtime_fingerprint(),
            "worker_count": len(workers),
            "ready_workers": ready_count,
            "available_workers": len(available_ids),
            "available_worker_ids": sorted(available_ids),
            "leased_worker_ids": sorted(leased_ids),
            "available_gpu_ids": [
                gpu_id
                for worker in workers
                if worker["worker_id"] in available_ids
                for gpu_id in worker["gpu_ids"]
            ],
            "context_size": CONTEXT_SIZE,
            "speculative": {
                "type": "draft-dflash",
                "draft_n_max": SPEC_DRAFT_N_MAX,
            },
            "workers": workers,
            "failures": dict(self.failures),
        }

    def write_state(self) -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(
            json.dumps(self.health(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        with self.lock:
            for worker_id in list(self.workers):
                self._discard_worker(worker_id)
        clear_state()


class PoolServer:
    def __init__(
        self,
        workers: list[Worker] | None = None,
        *,
        runtime: PoolRuntime | None = None,
        coordinator: Any | None = None,
    ) -> None:
        self.runtime = runtime
        if coordinator is not None:
            self.coordinator = coordinator
        elif workers is not None:
            self.coordinator = LocalCoordinator(workers)
        else:
            raise ValueError("PoolServer requires a runtime coordinator or workers")
        self.workers = list(workers or [])
        self.app = Flask("qwen38_ray_pool")
        self._register_routes()

    def _health(self) -> dict[str, Any]:
        if self.runtime is not None:
            return self.runtime.health()
        snapshot = self.coordinator.snapshot()
        workers = [
            {
                **worker,
                "ready": True,
                "state": (
                    "leased"
                    if worker["worker_id"] in snapshot["leased_worker_ids"]
                    else "available"
                ),
            }
            for worker in snapshot["workers"]
        ]
        return {
            "ok": bool(workers),
            "orchestrator": "local_test",
            "model": MODEL_ALIAS,
            "worker_count": len(workers),
            "ready_workers": len(workers),
            "available_workers": len(snapshot["available_worker_ids"]),
            "available_worker_ids": snapshot["available_worker_ids"],
            "leased_worker_ids": snapshot["leased_worker_ids"],
            "context_size": CONTEXT_SIZE,
            "workers": workers,
            "failures": {},
        }

    def _release(self, worker: Worker) -> None:
        self.coordinator.release(worker.worker_id)

    def _register_routes(self) -> None:
        @self.app.get("/api/health")
        @self.app.get("/health")
        def health() -> Response:
            return jsonify(self._health())

        @self.app.get("/v1/models")
        def models() -> Response:
            return jsonify(
                {
                    "object": "list",
                    "data": [{"id": MODEL_ALIAS, "object": "model"}],
                }
            )

        @self.app.route("/v1/<path:path>", methods=["POST"])
        def forward(path: str) -> Response:
            acquire_timeout = ACQUIRE_TIMEOUT
            requested_timeout = (
                request.headers.get("X-Qwen38-Acquire-Timeout")
                or request.headers.get("X-Bonsai-Acquire-Timeout")
            )
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
                                "message": "invalid worker acquire timeout",
                                "type": "invalid_request_error",
                            }
                        }
                    ), 400
            worker = self.coordinator.acquire(acquire_timeout)
            if worker is None:
                return jsonify(
                    {
                        "error": {
                            "message": "Qwen3.8 Q4 worker pool is busy",
                            "type": "server_error",
                        }
                    }
                ), 503
            body = request.get_json(silent=True)
            response: requests.Response | None = None
            session: requests.Session | None = None
            streaming = bool(isinstance(body, dict) and body.get("stream"))
            try:
                if isinstance(body, dict) and path == "chat/completions":
                    requested_model = str(body.get("model") or MODEL_ALIAS)
                    if requested_model != MODEL_ALIAS:
                        return jsonify(
                            {
                                "error": {
                                    "message": (
                                        f"model {requested_model} is not served; "
                                        f"use {MODEL_ALIAS}"
                                    ),
                                    "type": "invalid_request_error",
                                }
                            }
                        ), 400
                    body["model"] = MODEL_ALIAS
                    template_kwargs = body.setdefault(
                        "chat_template_kwargs",
                        {},
                    )
                    if isinstance(template_kwargs, dict):
                        template_kwargs.setdefault(
                            "enable_thinking",
                            DEFAULT_ENABLE_THINKING,
                        )
                        template_kwargs.setdefault("preserve_thinking", False)
                    payload = json.dumps(
                        body,
                        ensure_ascii=False,
                    ).encode("utf-8")
                else:
                    payload = request.get_data()
                headers = {
                    key: value
                    for key, value in request.headers.items()
                    if key.lower()
                    in {"authorization", "content-type", "accept"}
                }
                session = requests.Session()
                session.trust_env = False
                response = session.post(
                    f"{worker.url}/v1/{path}",
                    data=payload,
                    headers=headers,
                    timeout=(30, REQUEST_TIMEOUT),
                    stream=streaming,
                )
                response_headers = {
                    "X-Video-Analyzer-Worker-Id": worker.worker_id,
                    "X-Video-Analyzer-Worker-Tier": worker.tier,
                    "X-Video-Analyzer-Worker-GPUs": ",".join(worker.gpu_ids),
                }
                if streaming:

                    def stream() -> Any:
                        try:
                            yield from response.iter_content(chunk_size=8192)
                        finally:
                            response.close()
                            session.close()
                            self._release(worker)

                    return Response(
                        stream(),
                        status=response.status_code,
                        content_type=response.headers.get("content-type"),
                        headers=response_headers,
                    )
                content = response.content
                status_code = response.status_code
                content_type = response.headers.get(
                    "content-type",
                    "application/json",
                )
                response.close()
                session.close()
                return Response(
                    content,
                    status=status_code,
                    content_type=content_type,
                    headers=response_headers,
                )
            except requests.RequestException as error:
                return jsonify(
                    {
                        "error": {
                            "message": str(error),
                            "type": "backend_error",
                        }
                    }
                ), 502
            finally:
                if not streaming:
                    if response is not None:
                        response.close()
                    if session is not None:
                        session.close()
                    self._release(worker)


def pool_health() -> dict[str, Any] | None:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(
            f"http://{HOST}:{PORT}/api/health",
            timeout=2,
        )
        return response.json() if response.ok else None
    except requests.RequestException:
        return None
    finally:
        session.close()


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


def clear_state() -> None:
    for path in (PID_PATH, STATE_PATH):
        path.unlink(missing_ok=True)


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


def _spawn_pool() -> subprocess.Popen[bytes]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "serve"],
        cwd=ROOT,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=(LOG_DIR / "pool.log").open("ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_for_pool(process: subprocess.Popen[bytes]) -> dict[str, Any] | None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        health = pool_health()
        if health and health.get("ok"):
            return health
        if process.poll() is not None:
            break
        time.sleep(1)
    return None


def start_pool() -> int:
    health = pool_health()
    if health and health.get("runtime_fingerprint") == runtime_fingerprint():
        print(json.dumps(health, ensure_ascii=False))
        return 0
    pid = read_pid()
    if process_alive(pid):
        raise RuntimeError(
            f"Qwen3.8 pool pid={pid} is running but health check failed"
        )
    clear_state()
    process = _spawn_pool()
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    health = _wait_for_pool(process)
    if health:
        print(json.dumps(health, ensure_ascii=False))
        return 0
    stop_process_group(process.pid)
    clear_state()
    raise RuntimeError(f"Qwen3.8 pool did not become ready; see {LOG_DIR}")


def stop_pool() -> int:
    pid = read_pid()
    if process_alive(pid):
        stop_process_group(pid)
    clear_state()
    print("Qwen3.8 Ray pool stopped")
    return 0


def _require_runtime_files() -> None:
    if ray is None:
        raise RuntimeError("Ray is required for the Qwen3.8 worker pool")
    if not LLAMA_SERVER.is_file() or not os.access(LLAMA_SERVER, os.X_OK):
        raise RuntimeError(f"llama-server is not executable: {LLAMA_SERVER}")
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Local text model file is missing: {MODEL_PATH}")
    if not DRAFT_MODEL_PATH.is_file():
        raise RuntimeError(
            f"Local DFlash2 draft model file is missing: {DRAFT_MODEL_PATH}"
        )


def serve_pool() -> int:
    _require_runtime_files()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    runtime = PoolRuntime()
    started_ray = False

    def cleanup(*_args: object) -> None:
        runtime.shutdown()
        if started_ray and ray is not None and ray.is_initialized():
            ray.shutdown()

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_args: sys.exit(0))

    ready_workers = runtime.start_initial()
    ray.init(
        namespace="video-analyzer-qwen38-text",
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=max(2, len(ready_workers) + 1),
        num_gpus=0,
        logging_level="ERROR",
    )
    started_ray = True
    actor = TextPoolCoordinatorActor.remote(
        [worker.public_dict() for worker in ready_workers]
    )
    coordinator = RayCoordinatorClient(actor)
    runtime.attach_coordinator(coordinator)
    runtime.write_state()
    PoolServer(
        runtime=runtime,
        coordinator=coordinator,
    ).app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
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
        health = pool_health()
        print(
            json.dumps(
                health
                or {
                    "ok": False,
                    "running": process_alive(read_pid()),
                },
                ensure_ascii=False,
            )
        )
        return 0 if health else 1
    return serve_pool()


if __name__ == "__main__":
    raise SystemExit(main())
