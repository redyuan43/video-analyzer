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
DEFAULT_RECLAIM_STATE_PATH = (
    REPO_ROOT
    / "tmp"
    / "video-link-status"
    / "resource-locks"
    / "reclaimed-gpu-services.json"
)
_SESSION_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("local_model_session_depth", default=0)


def is_loopback_endpoint(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(str(url))
    host = parsed.hostname
    return host in LOOPBACK_HOSTS


def has_loopback_endpoint(urls: object) -> bool:
    return any(is_loopback_endpoint(url) for url in normalize_string_list(urls))


def remote_runtime_profile_owns_endpoint(
    config: dict,
    endpoint: str | None,
    *,
    endpoint_keys: tuple[str, ...],
) -> bool:
    profile_name = str(config.get("active_runtime_profile") or "").strip()
    profile = (config.get("runtime_profiles") or {}).get(profile_name) or {}
    profile_endpoint = next(
        (profile.get(key) for key in endpoint_keys if profile.get(key)),
        None,
    )
    if (
        str(profile_endpoint or "").rstrip("/")
        != str(endpoint or "").rstrip("/")
    ):
        return False
    deployment = str(profile.get("deployment") or "").strip().lower()
    provider = str(profile.get("provider") or "").strip().lower()
    if deployment:
        return deployment == "remote"
    return provider == "trae_local_api"


def local_model_stage_needed(stage: str, config: dict) -> bool:
    if not (config.get("local_model_runtime") or {}).get("enabled", True):
        return False
    if stage == "asr":
        provider = str((config.get("asr") or {}).get("provider") or "").strip().lower()
        if provider in {"firered_3dspeaker", "capswriter", "tencent_hy_asr"}:
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
        endpoint = manual.get("text_base_url") or manual.get("llm_base_url")
        if remote_runtime_profile_owns_endpoint(
            config,
            endpoint,
            endpoint_keys=("text_base_url", "llm_base_url"),
        ):
            return False
        return is_loopback_endpoint(endpoint)
    if stage == "tts":
        tts = config.get("tts") or {}
        return bool(tts.get("enabled", True)) and is_loopback_endpoint(
            tts.get("base_url") or tts.get("endpoint")
        )
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
            try:
                release_reclaimed_gpu_services(config, logger)
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
def try_local_model_runtime_lock(
    config: dict,
    logger: logging.Logger,
    owner: str,
    *,
    stage: str = "text",
) -> Iterator[bool]:
    """Try to reserve the shared local-model runtime without waiting."""
    runtime = config.get("local_model_runtime") or {}
    lock_path = Path(
        os.environ.get("VIDEO_ANALYZER_LOCAL_MODEL_LOCK")
        or runtime.get("lock_path")
        or DEFAULT_LOCK_PATH
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            logger.info(
                "[local-model-lock] busy stage=%s owner=%s; using fallback",
                stage,
                owner,
            )
            yield False
            return
        _write_lock_metadata(fd, stage, owner)
        logger.info(
            "[local-model-lock] acquired without waiting stage=%s owner=%s",
            stage,
            owner,
        )
        yield True
    finally:
        if acquired:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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
            try:
                unload_local_model_stage(config, logger)
            finally:
                release_reclaimed_gpu_services(config, logger)


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


def _reclaim_state_path(config: dict) -> Path:
    runtime = config.get("local_model_runtime") or {}
    return Path(
        runtime.get("reclaim_state_path")
        or DEFAULT_RECLAIM_STATE_PATH
    ).expanduser()


def _apply_reclaim_environment(env: dict[str, str], config: dict) -> None:
    runtime = config.get("local_model_runtime") or {}
    services = runtime.get("reclaimable_gpu_services") or []
    env["VIDEO_ANALYZER_RECLAIMABLE_GPU_SERVICES_JSON"] = json.dumps(
        services,
        ensure_ascii=False,
    )
    env["VIDEO_ANALYZER_GPU_RECLAIM_STATE"] = str(_reclaim_state_path(config))


def prepare_local_model_stage(stage: str, config: dict, logger: logging.Logger) -> None:
    if stage not in {"stop", "release"} and not local_model_stage_needed(stage, config):
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
    _apply_reclaim_environment(env, config)
    if stage == "asr":
        asr = config.get("asr") or {}
        provider = str(asr.get("provider") or "vibevoice")
        vibevoice = asr.get("vibevoice") or {}
        env["ASR_ENGINE"] = provider
        if provider == "vibevoice":
            env["VIBEVOICE_WORKER_COUNT"] = str(
                vibevoice.get("worker_count")
                or vibevoice.get("chunk_parallel_workers")
                or "auto"
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
            if options.get("gpu_selection") == "manual" and gpu_ids:
                env["QWEN3_ASR_GPU_SELECTION"] = "manual"
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
            if options.get("gpu_selection") == "manual" and gpu_ids:
                env["FIRERED_ASR2_GPU_SELECTION"] = "manual"
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
        worker_count = ocr.get("worker_count")
        if worker_count not in {None, "", "auto"}:
            env["UNLIMITED_OCR_GPU_SELECTION"] = "manual"
            env["UNLIMITED_OCR_WORKER_COUNT"] = str(ocr["worker_count"])
            env["DOTS_MOCR_WORKER_COUNT"] = str(ocr["worker_count"])
        gpu_ids = ocr.get("gpu_ids")
        if gpu_ids and gpu_ids != "auto":
            env["UNLIMITED_OCR_GPU_SELECTION"] = "manual"
            env["UNLIMITED_OCR_GPU_IDS"] = ",".join(str(item) for item in gpu_ids)
            env["DOTS_MOCR_GPU_SELECTION"] = "manual"
            env["DOTS_MOCR_GPU_IDS"] = ",".join(str(item) for item in gpu_ids)
        if ocr.get("min_gpu_memory_mib") is not None:
            env["UNLIMITED_OCR_MIN_TOTAL_MIB"] = str(ocr["min_gpu_memory_mib"])
            env["DOTS_MOCR_MIN_TOTAL_MIB"] = str(ocr["min_gpu_memory_mib"])
        if ocr.get("min_gpu_free_mib") is not None:
            env["UNLIMITED_OCR_MIN_FREE_MIB"] = str(ocr["min_gpu_free_mib"])
            env["DOTS_MOCR_MIN_FREE_MIB"] = str(ocr["min_gpu_free_mib"])
    elif stage == "vl":
        runtime_options = (config.get("operation_manual") or {}).get("vision_runtime") or {}
        env["VISION_ENGINE"] = str(runtime_options.get("engine") or "minicpm_v45")
        worker_count = runtime_options.get("worker_count")
        if worker_count not in {None, "", "auto"}:
            env["MINICPM_GPU_SELECTION"] = "manual"
            env["MINICPM_WORKER_COUNT"] = str(runtime_options["worker_count"])
        gpu_ids = runtime_options.get("gpu_ids")
        if gpu_ids and gpu_ids != "auto":
            env["MINICPM_GPU_SELECTION"] = "manual"
            env["MINICPM_GPU_IDS"] = ",".join(str(item) for item in gpu_ids)
        if runtime_options.get("min_gpu_memory_mib") is not None:
            env["MINICPM_MIN_TOTAL_MIB"] = str(runtime_options["min_gpu_memory_mib"])
        if runtime_options.get("min_gpu_free_mib") is not None:
            env["MINICPM_MIN_FREE_MIB"] = str(runtime_options["min_gpu_free_mib"])
    elif stage == "text":
        manual = config.get("operation_manual") or {}
        env["BONSAI_LOCAL_PORT"] = str(manual.get("text_port") or 18103)
        if manual.get("text_gpu_selection") == "manual":
            gpu_ids = list(manual.get("text_gpu_ids") or [])
            configured_worker_count = manual.get("text_worker_count")
            worker_count = (
                len(gpu_ids)
                if str(configured_worker_count or "auto").strip().lower() == "auto"
                else int(configured_worker_count)
            )
            if not 1 <= worker_count <= len(gpu_ids):
                raise ValueError(
                    "manual text_worker_count must fit configured text_gpu_ids"
                )
            env["BONSAI_LOCAL_GPU_SELECTION"] = "manual"
            env["BONSAI_LOCAL_WORKER_COUNT"] = str(worker_count)
            env["BONSAI_LOCAL_GPU_IDS"] = ",".join(
                str(item) for item in gpu_ids[:worker_count]
            )
        else:
            env["BONSAI_LOCAL_GPU_SELECTION"] = "auto"
        context_length = int(
            manual.get("text_context_length")
            or manual.get("context_length")
            or 65536
        )
        if not 1024 <= context_length <= 262144:
            raise ValueError("text_context_length must be between 1024 and 262144")
        env["BONSAI_LOCAL_CONTEXT_SIZE"] = str(context_length)
        text_runtime_env = {
            "BONSAI_LOCAL_MODEL": "text_model_path",
            "BONSAI_LOCAL_DRAFT_MODEL": "text_draft_model_path",
            "BONSAI_LOCAL_LLAMA_SERVER": "text_llama_server",
            "BONSAI_LOCAL_MODEL_ALIAS": "text_model_alias",
            "BONSAI_LOCAL_SPEC_DRAFT_N_MAX": "text_spec_draft_n_max",
            "BONSAI_LOCAL_V100_32_CACHE_TYPE": "text_v100_32_cache_type",
            "BONSAI_LOCAL_V100_16_P40_CACHE_TYPE": (
                "text_v100_16_p40_cache_type"
            ),
            "BONSAI_LOCAL_P40_CACHE_TYPE": "text_p40_cache_type",
            "BONSAI_LOCAL_V100_16_P40_TENSOR_SPLIT": (
                "text_v100_16_p40_tensor_split"
            ),
            "BONSAI_LOCAL_V100_32_MIN_FREE_MIB": (
                "text_v100_32_min_free_mib"
            ),
            "BONSAI_LOCAL_V100_16_MIN_FREE_MIB": (
                "text_v100_16_min_free_mib"
            ),
            "BONSAI_LOCAL_P40_MIN_FREE_MIB": "text_p40_min_free_mib",
            "BONSAI_LOCAL_RECONCILE_SECONDS": "text_reconcile_seconds",
        }
        for environment_name, config_name in text_runtime_env.items():
            value = manual.get(config_name)
            if value not in {None, ""}:
                env[environment_name] = str(value)
    elif stage == "tts":
        tts = config.get("tts") or {}
        endpoint = str(tts.get("base_url") or tts.get("endpoint") or "")
        parsed = urlparse(endpoint)
        if parsed.port:
            env["INDEXTTS_PORT"] = str(parsed.port)
    timeout = int(runtime.get("stage_timeout_seconds") or 900)
    logger.info("Preparing local GPU model stage '%s' with %s", stage, " ".join(command_args))
    subprocess.run(command_args, cwd=REPO_ROOT, env=env, timeout=timeout, check=True)


def unload_local_model_stage(config: dict, logger: logging.Logger) -> None:
    runtime = config.get("local_model_runtime") or {}
    if not runtime.get("unload_on_stage_exit", False):
        return
    prepare_local_model_stage("stop", config, logger)


def release_reclaimed_gpu_services(config: dict, logger: logging.Logger) -> None:
    state_path = _reclaim_state_path(config)
    if not state_path.is_file():
        return
    logger.info("Releasing dynamic GPU workers and restoring reclaimed services")
    try:
        prepare_local_model_stage("release", config, logger)
    except Exception:
        logger.exception(
            "Could not restore one or more reclaimed GPU services; state kept at %s",
            state_path,
        )
