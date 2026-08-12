"""Helpers for switching mutually-exclusive local GPU model services."""

from __future__ import annotations

import contextlib
import contextvars
import fcntl
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

from .config import normalize_string_list


REPO_ROOT = Path(__file__).resolve().parents[1]
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_LOCK_PATH = REPO_ROOT / "tmp" / "video-link-status" / "resource-locks" / "local-model-runtime.lock"
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_LOG_INTERVAL_SECONDS = 30.0
_SESSION_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("local_model_session_depth", default=0)


def is_loopback_endpoint(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(str(url))
    host = parsed.hostname
    return host in LOOPBACK_HOSTS


def has_loopback_endpoint(urls: object) -> bool:
    return any(is_loopback_endpoint(url) for url in normalize_string_list(urls))


def local_model_stage_needed(stage: str, config: dict) -> bool:
    if not (config.get("local_model_runtime") or {}).get("enabled", True):
        return False
    if stage == "asr":
        provider = str((config.get("asr") or {}).get("provider") or "").strip().lower()
        if provider in {"firered_3dspeaker", "capswriter"}:
            return False
        vibevoice = (config.get("asr") or {}).get("vibevoice") or {}
        if provider == "qwen3_asr":
            return is_loopback_endpoint(vibevoice.get("qwen3_asr_url"))
        if provider == "firered_asr2":
            return is_loopback_endpoint(vibevoice.get("firered_asr2_url"))
        return has_loopback_endpoint(vibevoice.get("deep_remote_urls") or vibevoice.get("remote_urls"))
    if stage == "ocr":
        ocr = config.get("ocr") or {}
        return has_loopback_endpoint(ocr.get("base_urls") or ocr.get("base_url"))
    if stage == "vl":
        manual = config.get("operation_manual") or {}
        return is_loopback_endpoint(manual.get("vision_base_url") or manual.get("llm_base_url"))
    if stage == "text":
        manual = config.get("operation_manual") or {}
        return is_loopback_endpoint(manual.get("text_base_url") or manual.get("llm_base_url"))
    return False


def local_model_runtime_needed(config: dict) -> bool:
    return any(local_model_stage_needed(stage, config) for stage in ("asr", "ocr", "vl", "text"))


@contextlib.contextmanager
def local_model_runtime_session(config: dict, logger: logging.Logger, owner: str) -> Iterator[None]:
    """Hold the local GPU model runtime for one whole core analysis."""
    if not local_model_runtime_needed(config):
        yield
        return

    with _local_model_lock("core", config, logger, owner):
        token = _SESSION_DEPTH.set(_SESSION_DEPTH.get() + 1)
        try:
            yield
        finally:
            _SESSION_DEPTH.reset(token)


@contextlib.contextmanager
def local_model_runtime_lock(
    config: dict,
    logger: logging.Logger,
    owner: str,
    *,
    stage: str = "text",
) -> Iterator[None]:
    """Hold the shared local-model lock without switching model services."""
    if _SESSION_DEPTH.get() > 0:
        yield
        return
    with _local_model_lock(stage, config, logger, owner):
        yield


@contextlib.contextmanager
def local_model_stage(stage: str, config: dict, logger: logging.Logger, owner: str) -> Iterator[None]:
    """Switch to a local GPU model stage without letting another task preempt it."""
    if not local_model_stage_needed(stage, config):
        yield
        return

    if _SESSION_DEPTH.get() > 0:
        prepare_local_model_stage(stage, config, logger)
        try:
            yield
        finally:
            unload_local_model_stage(config, logger)
        return

    with _local_model_lock(stage, config, logger, owner):
        prepare_local_model_stage(stage, config, logger)
        try:
            yield
        finally:
            unload_local_model_stage(config, logger)


@contextlib.contextmanager
def _local_model_lock(stage: str, config: dict, logger: logging.Logger, owner: str) -> Iterator[None]:
    runtime = config.get("local_model_runtime") or {}
    lock_path = Path(
        os.environ.get("VIDEO_ANALYZER_LOCAL_MODEL_LOCK")
        or runtime.get("lock_path")
        or DEFAULT_LOCK_PATH
    )
    poll_seconds = max(0.1, float(runtime.get("poll_seconds", DEFAULT_POLL_SECONDS)))
    log_interval_seconds = max(
        poll_seconds,
        float(runtime.get("log_interval_seconds", DEFAULT_LOG_INTERVAL_SECONDS)),
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    wait_started = time.monotonic()
    last_wait_log = 0.0
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if last_wait_log == 0.0 or now - last_wait_log >= log_interval_seconds:
                    logger.info(
                        "[local-model-lock] waiting stage=%s owner=%s waited=%.3fs",
                        stage,
                        owner,
                        now - wait_started,
                    )
                    last_wait_log = now
                time.sleep(poll_seconds)
        waited = time.monotonic() - wait_started
        _write_lock_metadata(fd, stage, owner)
        logger.info("[local-model-lock] acquired stage=%s owner=%s waited=%.3fs", stage, owner, waited)
        try:
            yield
        finally:
            logger.info("[local-model-lock] releasing stage=%s owner=%s", stage, owner)
    finally:
        try:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _write_lock_metadata(fd: int, stage: str, owner: str) -> None:
    payload = {
        "resource": "local-model-runtime",
        "stage": stage,
        "owner": owner,
        "pid": os.getpid(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    os.ftruncate(fd, 0)
    os.write(fd, data)
    os.fsync(fd)


def prepare_local_model_stage(stage: str, config: dict, logger: logging.Logger) -> None:
    if stage != "stop" and not local_model_stage_needed(stage, config):
        return

    runtime = config.get("local_model_runtime") or {}
    commands = runtime.get("stage_commands") or {}
    command = commands.get(stage)
    if command:
        command_args = [str(part) for part in command]
    else:
        command_args = [str(REPO_ROOT / "tools" / "prepare_ai_local_model_stage.sh"), stage]

    if not Path(command_args[0]).exists():
        logger.warning("Local model stage switch script missing: %s", command_args[0])
        return

    env = os.environ.copy()
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    if stage == "asr":
        asr = config.get("asr") or {}
        provider = str(asr.get("provider") or "vibevoice")
        vibevoice = asr.get("vibevoice") or {}
        env["ASR_ENGINE"] = provider
        if provider == "vibevoice":
            env["VIBEVOICE_WORKER_COUNT"] = str(
                vibevoice.get("chunk_parallel_workers")
                or vibevoice.get("worker_count")
                or 5
            )
            if vibevoice.get("single_pass_max_duration_sec") is not None:
                env["VIBEVOICE_SINGLE_PASS_MAX_DURATION_SEC"] = str(
                    vibevoice["single_pass_max_duration_sec"]
                )
            if vibevoice.get("chunk_duration_sec") is not None:
                env["VIBEVOICE_CHUNK_DURATION_SEC"] = str(
                    vibevoice["chunk_duration_sec"]
                )
            if vibevoice.get("chunk_overlap_sec") is not None:
                env["VIBEVOICE_CHUNK_OVERLAP_SEC"] = str(
                    vibevoice["chunk_overlap_sec"]
                )
        elif provider == "qwen3_asr":
            options = vibevoice.get("qwen3_asr_options") or {}
            env["QWEN3_ASR_WORKER_COUNT"] = str(options.get("worker_count") or 5)
            gpu_ids = options.get("gpu_ids")
            if gpu_ids:
                env["QWEN3_ASR_GPU_IDS"] = ",".join(str(item) for item in gpu_ids)
            configured_model_path = options.get("model_path")
            if not configured_model_path:
                candidate = str(vibevoice.get("qwen3_asr_model") or "").strip()
                if candidate and Path(candidate).expanduser().is_dir():
                    configured_model_path = candidate
            if configured_model_path:
                env["QWEN3_ASR_MODEL"] = str(Path(configured_model_path).expanduser())
            if options.get("single_pass_max_duration_sec") is not None:
                env["QWEN3_ASR_SINGLE_PASS_SECONDS"] = str(
                    options["single_pass_max_duration_sec"]
                )
            if options.get("chunk_duration_sec") is not None:
                env["QWEN3_ASR_CHUNK_SECONDS"] = str(options["chunk_duration_sec"])
            if options.get("chunk_overlap_sec") is not None:
                env["QWEN3_ASR_CHUNK_OVERLAP_SECONDS"] = str(
                    options["chunk_overlap_sec"]
                )
        elif provider == "firered_asr2":
            options = vibevoice.get("firered_asr2_options") or {}
            env["FIRERED_ASR2_WORKER_COUNT"] = str(options.get("worker_count") or 5)
            gpu_ids = options.get("gpu_ids")
            if gpu_ids:
                env["FIRERED_ASR2_GPU_IDS"] = ",".join(str(item) for item in gpu_ids)
            env["FIRERED_ASR2_CHUNK_SECONDS"] = str(options.get("chunk_duration_sec") or 30)
            env["FIRERED_ASR2_CHUNK_OVERLAP_SECONDS"] = str(
                options.get("chunk_overlap_sec") or 3
            )
            if options.get("single_pass_max_duration_sec") is not None:
                env["FIRERED_ASR2_SINGLE_PASS_SECONDS"] = str(
                    options["single_pass_max_duration_sec"]
                )
            env["FIRERED_ASR2_SEGMENTATION_MODE"] = str(
                options.get("segmentation_mode") or "vad"
            )
            env["FIRERED_VAD_MAX_SEGMENT_SECONDS"] = str(
                options.get("vad_max_segment_sec") or 50
            )
            if options.get("vad_model_path"):
                env["FIRERED_VAD_MODEL"] = str(
                    Path(options["vad_model_path"]).expanduser()
                )
    elif stage == "ocr":
        ocr = config.get("ocr") or {}
        env["OCR_ENGINE"] = str(ocr.get("engine") or ocr.get("provider") or "unlimited")
        if ocr.get("worker_count"):
            env["UNLIMITED_OCR_WORKER_COUNT"] = str(ocr["worker_count"])
            env["DOTS_MOCR_WORKER_COUNT"] = str(ocr["worker_count"])
    elif stage == "vl":
        runtime_options = (config.get("operation_manual") or {}).get("vision_runtime") or {}
        env["VISION_ENGINE"] = str(runtime_options.get("engine") or "minicpm_v45")
        if runtime_options.get("worker_count"):
            env["MINICPM_WORKER_COUNT"] = str(runtime_options["worker_count"])
    timeout = int(runtime.get("stage_timeout_seconds") or 900)
    logger.info("Preparing local GPU model stage '%s' with %s", stage, " ".join(command_args))
    subprocess.run(command_args, cwd=REPO_ROOT, env=env, timeout=timeout, check=True)


def unload_local_model_stage(config: dict, logger: logging.Logger) -> None:
    runtime = config.get("local_model_runtime") or {}
    if not runtime.get("unload_on_stage_exit", False):
        return
    prepare_local_model_stage("stop", config, logger)
