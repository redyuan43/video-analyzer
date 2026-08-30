#!/usr/bin/env python3
"""Temporarily unload registered idle model services and restore them later."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener


def _load_services(raw: str) -> list[dict]:
    if not raw.strip():
        return []
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("services JSON must be a list")
    return [item for item in payload if isinstance(item, dict)]


def _read_json(url: str, timeout: float = 3.0) -> object:
    opener = build_opener(ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return json.load(response)


def _slots_idle(
    url: str,
    settle_seconds: float,
    readiness_seconds: float = 30.0,
) -> bool:
    for attempt in range(2):
        deadline = time.monotonic() + max(0.0, readiness_seconds)
        while True:
            try:
                payload = _read_json(url)
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        if not isinstance(payload, list) or not payload:
            return False
        if any(bool(slot.get("is_processing")) for slot in payload if isinstance(slot, dict)):
            return False
        if attempt == 0:
            time.sleep(settle_seconds)
    return True


def _systemd_active(unit: str) -> bool:
    return (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            check=False,
        ).returncode
        == 0
    )


def _listener_pid(port: int) -> int | None:
    result = subprocess.run(
        ["fuser", "-n", "tcp", str(port)],
        check=False,
        capture_output=True,
        text=True,
    )
    values = [item for item in result.stdout.split() if item.isdigit()]
    return int(values[0]) if len(values) == 1 else None


def _snapshot_process(service: dict) -> dict | None:
    pid = _listener_pid(int(service["port"]))
    if pid is None:
        return None
    command_line = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    argv = [item.decode(errors="surrogateescape") for item in command_line if item]
    expected = str(service.get("command_contains") or "")
    if not argv or (expected and expected not in " ".join(argv)):
        return None
    cwd = str(Path(f"/proc/{pid}/cwd").resolve())
    environment = {}
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        decoded_key = key.decode(errors="ignore")
        if decoded_key in {
            "CUDA_DEVICE_ORDER",
            "CUDA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
            "PATH",
            "HOME",
        }:
            environment[decoded_key] = value.decode(errors="surrogateescape")
    return {
        "kind": "process",
        "name": service["name"],
        "pid": pid,
        "port": int(service["port"]),
        "argv": argv,
        "cwd": cwd,
        "environment": environment,
    }


def _stop_process(snapshot: dict, timeout: float = 30.0) -> bool:
    pid = int(snapshot["pid"])
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.25)
    return False


def reclaim(
    state_path: Path,
    services: list[dict],
    settle_seconds: float,
    readiness_seconds: float = 30.0,
) -> int:
    state = {"services": []}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    reclaimed_names = {
        str(item.get("name"))
        for item in state.get("services", [])
        if isinstance(item, dict)
    }
    for service in services:
        name = str(service.get("name") or "")
        slots_url = str(service.get("slots_url") or "")
        kind = str(service.get("kind") or "systemd_user")
        if not name or not slots_url or name in reclaimed_names:
            continue
        try:
            if kind == "systemd_user":
                unit = str(service.get("unit") or "")
                if not unit or not _systemd_active(unit):
                    continue
                if not _slots_idle(slots_url, settle_seconds, readiness_seconds):
                    print(f"Keeping busy GPU service: {name}")
                    continue
                subprocess.run(["systemctl", "--user", "stop", unit], check=True)
                snapshot = {"kind": kind, "name": name, "unit": unit}
            elif kind == "process":
                snapshot = _snapshot_process(service)
                if snapshot is None:
                    continue
                if not _slots_idle(slots_url, settle_seconds, readiness_seconds):
                    print(f"Keeping busy GPU service: {name}")
                    continue
                if not _stop_process(snapshot):
                    print(f"Could not safely stop GPU process: {name}", file=sys.stderr)
                    continue
            else:
                print(f"Unknown reclaimable service kind for {name}: {kind}", file=sys.stderr)
                continue
        except Exception as exc:
            print(f"Skipping reclaimable GPU service {name}: {exc}", file=sys.stderr)
            continue
        state.setdefault("services", []).append(snapshot)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Reclaimed idle GPU service: {name}")
    return 0


def restore(state_path: Path) -> int:
    if not state_path.is_file():
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    failures = []
    remaining = []
    log_dir = state_path.parent / "restored-services"
    log_dir.mkdir(parents=True, exist_ok=True)
    for service in state.get("services", []):
        try:
            if service.get("kind") == "systemd_user":
                subprocess.run(
                    ["systemctl", "--user", "start", str(service["unit"])],
                    check=True,
                )
            elif service.get("kind") == "process":
                if _listener_pid(int(service["port"])) is None:
                    env = os.environ.copy()
                    env.update(service.get("environment") or {})
                    log_path = log_dir / f"{service['name']}.log"
                    with log_path.open("ab") as log_file:
                        subprocess.Popen(
                            service["argv"],
                            cwd=service["cwd"],
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
            print(f"Restored GPU service: {service.get('name')}")
        except Exception as exc:
            failures.append(f"{service.get('name')}: {exc}")
            remaining.append(service)
    if failures:
        state_path.write_text(
            json.dumps({"services": remaining}, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print("Failed to restore GPU services: " + "; ".join(failures), file=sys.stderr)
        return 1
    state_path.unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["reclaim", "restore"])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--services-json",
        default=os.environ.get("VIDEO_ANALYZER_RECLAIMABLE_GPU_SERVICES_JSON", "[]"),
    )
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--readiness-seconds", type=float, default=30.0)
    args = parser.parse_args()
    try:
        services = _load_services(args.services_json)
        if args.action == "reclaim":
            return reclaim(
                args.state,
                services,
                max(0.0, args.settle_seconds),
                max(0.0, args.readiness_seconds),
            )
        return restore(args.state)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"GPU service reclaim failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
