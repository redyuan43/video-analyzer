#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


RESTART_BACKOFF_SECONDS = (1, 2, 5, 10, 30)


class Supervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.status_path = Path(args.status_file).resolve()
        self.stop_requested = False
        self.child: subprocess.Popen | None = None
        self.restart_reason = "manual_start"
        self.failure_count = 0

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        stale_since: float | None = None
        while not self.stop_requested:
            if self.child is None or self.child.poll() is not None:
                if self.child is not None:
                    self.failure_count += 1
                    delay = RESTART_BACKOFF_SECONDS[min(self.failure_count - 1, len(RESTART_BACKOFF_SECONDS) - 1)]
                    if self.stop_requested:
                        break
                    print(f"[supervisor] server exited with code {self.child.returncode}; restarting in {delay}s", flush=True)
                    if self.wait(delay):
                        break
                    self.restart_reason = "server_exit"
                self.start_child()
                stale_since = None

            health = self.health()
            if health:
                self.failure_count = 0
                runtime = health.get("runtime") or {}
                self.write_status(runtime)
                if runtime.get("source_stale"):
                    stale_since = stale_since or time.monotonic()
                    if time.monotonic() - stale_since >= self.args.reload_debounce_seconds:
                        print(
                            f"[supervisor] runtime {runtime.get('runtime_id')} is stale; restarting process group",
                            flush=True,
                        )
                        self.restart_reason = "source_changed"
                        self.stop_child()
                        stale_since = None
                        continue
                else:
                    stale_since = None
            if self.wait(self.args.poll_seconds):
                break
        self.stop_child()
        self.status_path.unlink(missing_ok=True)
        return 0

    def start_child(self) -> None:
        runtime_id = uuid.uuid4().hex
        env = os.environ.copy()
        env["VIDEO_LINK_RUNTIME_ID"] = runtime_id
        env["VIDEO_LINK_SUPERVISOR_PID"] = str(os.getpid())
        env["VIDEO_LINK_RESTART_REASON"] = self.restart_reason
        command = [
            self.args.python,
            "-m",
            "video_analyzer_ui.server",
            "--host",
            self.args.host,
            "--port",
            str(self.args.port),
            "--jobs-dir",
            self.args.jobs_dir,
        ]
        self.child = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=env,
            start_new_session=True,
        )
        self.write_status(
            {
                "runtime_id": runtime_id,
                "server_pid": self.child.pid,
                "supervisor_pid": os.getpid(),
                "source_stale": False,
                "restart_reason": self.restart_reason,
            }
        )
        print(f"[supervisor] started server pid={self.child.pid} runtime_id={runtime_id}", flush=True)

    def stop_child(self) -> None:
        child = self.child
        self.child = None
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=self.args.stop_timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        child.wait(timeout=5)

    def health(self) -> dict[str, Any] | None:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.args.port}/api/video-link/health",
            headers={"Accept": "application/json"},
        )
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=2) as response:
                payload = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def write_status(self, runtime: dict[str, Any]) -> None:
        payload = {
            "supervisor_pid": os.getpid(),
            "server_pid": runtime.get("server_pid") or (self.child.pid if self.child else None),
            "runtime_id": runtime.get("runtime_id"),
            "source_stale": bool(runtime.get("source_stale")),
            "restart_reason": runtime.get("restart_reason") or self.restart_reason,
            "updated_at_epoch": time.time(),
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.status_path.with_name(f".{self.status_path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.status_path)

    def request_stop(self, _signum: int, _frame: Any) -> None:
        self.stop_requested = True

    def wait(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while not self.stop_requested and time.monotonic() < deadline:
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        return self.stop_requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--reload-debounce-seconds", type=float, default=5.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(Supervisor(parse_args()).run())
