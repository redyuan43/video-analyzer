#!/usr/bin/env python3
"""Run one Skill workbench project outside the video-link status process."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_analyzer.skill_distillation import (  # noqa: E402
    SkillDistillationPipeline,
    load_state,
    save_state,
)
from video_analyzer.skill_projects import SkillProjectStore  # noqa: E402


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated Skill project worker")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lease-fd", type=int)
    return parser.parse_args()


def write_worker_log(root: Path, event: str, message: str) -> None:
    path = root / "skills" / "cangjie_pack" / "logs" / "worker.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "time": iso_now(),
                    "stage": "worker",
                    "event": event,
                    "message": message,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def acquire_lease(root: Path, inherited_fd: int | None) -> tuple[int, bool]:
    if inherited_fd is not None:
        return inherited_fd, False
    path = root / ".runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError("Skill project is already running in another worker")
    return fd, True


def release_lease(fd: int, owned: bool) -> None:
    if not owned:
        return
    try:
        os.ftruncate(fd, 0)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def update_project(
    store: SkillProjectStore,
    project_id: str,
    run_id: str,
    *,
    state: dict,
    pid: int,
) -> bool:
    project = store.load(project_id)
    runner = project.setdefault("distillation", {}).setdefault("runner", {})
    if str(runner.get("run_id") or "") != run_id:
        return False
    status = str(state.get("status") or "failed")
    project["status"] = {
        "succeeded": "completed",
        "completed_no_skills": "completed",
        "cancelled": "cancelled",
        "failed": "failed",
        "waiting_resource_decision": "waiting_resource_decision",
        "waiting_overview_review": "waiting_overview_review",
        "waiting_candidate_review": "waiting_candidate_review",
    }.get(status, "distilling")
    runner.update(
        {
            "status": status,
            "pid": pid,
            "finished_at": iso_now(),
            "error": state.get("error"),
        }
    )
    store.save(project)
    return True


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    store = SkillProjectStore(repo_root / "var" / "skill-projects")
    root = store.project_dir(args.project_id)
    cancel_event = threading.Event()

    def request_cancel(_signum: int, _frame: object) -> None:
        cancel_event.set()

    signal.signal(signal.SIGTERM, request_cancel)
    signal.signal(signal.SIGINT, request_cancel)
    lease_fd, owns_lease = acquire_lease(root, args.lease_fd)
    try:
        write_worker_log(root, "start", f"pid={os.getpid()} run_id={args.run_id}")
        state = load_state(root) or {}
        terminal_or_review = {
            "waiting_overview_review",
            "waiting_candidate_review",
            "waiting_resource_decision",
            "succeeded",
            "completed_no_skills",
            "cancelled",
        }
        if str(state.get("status") or "") not in terminal_or_review:
            state = SkillDistillationPipeline(root).run_until_pause(
                cancel_event=cancel_event
            )
        update_project(
            store,
            args.project_id,
            args.run_id,
            state=state,
            pid=os.getpid(),
        )
        write_worker_log(root, "done", f"status={state.get('status') or 'unknown'}")
        return 0
    except Exception as exc:
        state = load_state(root) or {}
        state.update(
            {
                "status": "failed",
                "retryable": True,
                "error": str(exc),
                "updated_at": iso_now(),
            }
        )
        save_state(root, state)
        update_project(
            store,
            args.project_id,
            args.run_id,
            state=state,
            pid=os.getpid(),
        )
        write_worker_log(root, "error", str(exc))
        return 1
    finally:
        release_lease(lease_fd, owns_lease)


if __name__ == "__main__":
    raise SystemExit(main())
