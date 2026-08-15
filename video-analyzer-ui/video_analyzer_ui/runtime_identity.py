from __future__ import annotations

import hashlib
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


class RuntimeIdentity:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.runtime_id = os.environ.get("VIDEO_LINK_RUNTIME_ID") or uuid.uuid4().hex
        self.supervisor_pid = int(os.environ.get("VIDEO_LINK_SUPERVISOR_PID") or 0) or None
        self.restart_reason = os.environ.get("VIDEO_LINK_RESTART_REASON") or "manual_start"
        self.started_at = iso_now()
        self._baseline: dict[str, str] = {}
        self._lock = threading.Lock()
        self._refresh_loaded_sources()

    def payload(self) -> dict[str, Any]:
        with self._lock:
            current = self._refresh_loaded_sources()
            stale_files = sorted(
                path
                for path, baseline_hash in self._baseline.items()
                if current.get(path) != baseline_hash
            )
            return {
                "runtime_id": self.runtime_id,
                "server_pid": os.getpid(),
                "supervisor_pid": self.supervisor_pid,
                "started_at": self.started_at,
                "startup_fingerprint": fingerprint(self._baseline),
                "current_fingerprint": fingerprint(current),
                "source_stale": bool(stale_files),
                "stale_files": stale_files,
                "source_file_count": len(self._baseline),
                "restart_reason": self.restart_reason,
            }

    def is_stale(self) -> bool:
        return bool(self.payload()["source_stale"])

    def _refresh_loaded_sources(self) -> dict[str, str]:
        current: dict[str, str] = {}
        for module in tuple(sys.modules.values()):
            raw_path = getattr(module, "__file__", None)
            if not raw_path:
                continue
            path = Path(str(raw_path))
            if path.suffix in {".pyc", ".pyo"}:
                path = path.with_suffix(".py")
            try:
                path = path.resolve()
                relative_path = path.relative_to(self.repo_root)
            except (OSError, ValueError):
                continue
            if (
                path.suffix != ".py"
                or not path.is_file()
                or not relative_path.parts
                or relative_path.parts[0] not in {"tools", "video_analyzer", "video-analyzer-ui", "web_debug_console"}
            ):
                continue
            relative = str(relative_path)
            current[relative] = file_hash(path)
        for path in self._ui_assets():
            try:
                relative = str(path.relative_to(self.repo_root))
            except ValueError:
                continue
            current[relative] = file_hash(path)
        for path, digest in current.items():
            self._baseline.setdefault(path, digest)
        for path in self._baseline:
            current.setdefault(path, "")
        return current

    def _ui_assets(self) -> tuple[Path, ...]:
        asset_roots = (
            self.repo_root / "video-analyzer-ui" / "video_analyzer_ui" / "templates",
            self.repo_root / "video-analyzer-ui" / "video_analyzer_ui" / "static" / "js",
            self.repo_root / "video-analyzer-ui" / "video_analyzer_ui" / "static" / "css",
            self.repo_root / "video-analyzer-ui" / "video_analyzer_ui" / "static" / "data",
        )
        assets: list[Path] = []
        for root in asset_roots:
            if not root.is_dir():
                continue
            assets.extend(path for path in root.rglob("*") if path.is_file())
        return tuple(assets)


def file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def fingerprint(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, file_digest in sorted(files.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
