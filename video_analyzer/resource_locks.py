"""Cross-process resource locks for expensive analyzer stages."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_LOCK_DIR = Path("tmp/video-link-status/resource-locks")
DEFAULT_LIMITS = {
    "asr": 1,
    "ocr": 1,
    "vl": 1,
}
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_LOG_INTERVAL_SECONDS = 30.0


def bool_from_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def int_from_env(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def resource_lock_settings(config: dict[str, Any], resource: str) -> dict[str, Any]:
    locks_config = config.get("resource_locks") or {}
    limits_config = locks_config.get("limits") or {}
    default_limit = int(limits_config.get(resource, DEFAULT_LIMITS.get(resource, 1)) or 1)
    env_prefix = f"VIDEO_ANALYZER_{resource.upper()}_LOCK"
    return {
        "enabled": bool_from_env(
            os.environ.get("VIDEO_ANALYZER_RESOURCE_LOCKS"),
            bool(locks_config.get("enabled", True)),
        ),
        "lock_dir": Path(
            os.environ.get("VIDEO_ANALYZER_RESOURCE_LOCK_DIR")
            or locks_config.get("dir")
            or DEFAULT_LOCK_DIR
        ),
        "limit": int_from_env(os.environ.get(f"{env_prefix}_LIMIT"), default_limit),
        "poll_seconds": float(locks_config.get("poll_seconds", DEFAULT_POLL_SECONDS)),
        "log_interval_seconds": float(
            locks_config.get("log_interval_seconds", DEFAULT_LOG_INTERVAL_SECONDS)
        ),
    }


@contextlib.contextmanager
def analyzer_resource_lock(
    config: dict[str, Any],
    resource: str,
    owner: str,
    logger: logging.Logger,
) -> Iterator[None]:
    settings = resource_lock_settings(config, resource)
    if not settings["enabled"]:
        yield
        return

    lease = FileResourceLease(
        resource=resource,
        limit=settings["limit"],
        lock_dir=settings["lock_dir"],
        owner=owner,
        poll_seconds=settings["poll_seconds"],
        log_interval_seconds=settings["log_interval_seconds"],
        logger=logger,
    )
    with lease:
        yield


class FileResourceLease:
    def __init__(
        self,
        *,
        resource: str,
        limit: int,
        lock_dir: Path,
        owner: str,
        poll_seconds: float,
        log_interval_seconds: float,
        logger: logging.Logger,
    ) -> None:
        self.resource = resource
        self.limit = max(1, int(limit))
        self.lock_dir = lock_dir
        self.owner = owner
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.log_interval_seconds = max(self.poll_seconds, float(log_interval_seconds))
        self.logger = logger
        self.slot: int | None = None
        self._fd: int | None = None

    def __enter__(self) -> "FileResourceLease":
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        wait_started = time.monotonic()
        last_wait_log = 0.0
        while True:
            for slot in range(self.limit):
                fd = os.open(self.lock_dir / f"{self.resource}.{slot}.lock", os.O_RDWR | os.O_CREAT, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                self._fd = fd
                self.slot = slot
                self._write_metadata()
                waited = time.monotonic() - wait_started
                self.logger.info(
                    "[resource-lock] acquired resource=%s slot=%s limit=%s owner=%s waited=%.3fs",
                    self.resource,
                    slot,
                    self.limit,
                    self.owner,
                    waited,
                )
                return self

            now = time.monotonic()
            if last_wait_log == 0.0 or now - last_wait_log >= self.log_interval_seconds:
                self.logger.info(
                    "[resource-lock] waiting resource=%s limit=%s owner=%s waited=%.3fs",
                    self.resource,
                    self.limit,
                    self.owner,
                    now - wait_started,
                )
                last_wait_log = now
            time.sleep(self.poll_seconds)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is None:
            return
        try:
            os.ftruncate(self._fd, 0)
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self.logger.info(
                "[resource-lock] released resource=%s slot=%s owner=%s",
                self.resource,
                self.slot,
                self.owner,
            )
        finally:
            os.close(self._fd)
            self._fd = None
            self.slot = None

    def _write_metadata(self) -> None:
        if self._fd is None:
            return
        payload = {
            "resource": self.resource,
            "slot": self.slot,
            "limit": self.limit,
            "owner": self.owner,
            "pid": os.getpid(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        os.ftruncate(self._fd, 0)
        os.write(self._fd, data)
        os.fsync(self._fd)
