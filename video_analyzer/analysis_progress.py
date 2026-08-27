"""Durable analyze-core progress shared by CLI entrypoints."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "progress.json"
ANALYSIS_PROGRESS_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def write_analysis_progress(
    output_dir: Path,
    current_step: str,
    status: str = "running",
    message: str | None = None,
    artifacts: dict[str, str] | None = None,
    details: dict[str, Any] | None = None,
    node_updates: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Best-effort durable progress for status UIs; never fail analysis work."""
    try:
        with ANALYSIS_PROGRESS_LOCK:
            output_dir.mkdir(parents=True, exist_ok=True)
            progress_path = output_dir / PROGRESS_FILENAME
            existing = {}
            if progress_path.is_file():
                try:
                    loaded = json.loads(progress_path.read_text(encoding="utf-8"))
                    existing = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    existing = {}
            now = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
            node_states = dict(existing.get("node_states") or {})
            for node_id, raw_update in (node_updates or {}).items():
                update = dict(raw_update or {})
                previous = dict(node_states.get(node_id) or {})
                node_status = str(
                    update.get("status")
                    or previous.get("status")
                    or "pending"
                )
                if node_status == "running" and not update.get("started_at"):
                    update["started_at"] = previous.get("started_at") or now
                    update.pop("finished_at", None)
                    update.pop("duration_seconds", None)
                elif node_status in {
                    "succeeded",
                    "failed",
                    "skipped",
                    "stopped",
                }:
                    started_at = update.get("started_at") or previous.get("started_at")
                    update["started_at"] = started_at
                    update["finished_at"] = update.get("finished_at") or now
                    if started_at and update.get("duration_seconds") is None:
                        try:
                            started = datetime.fromisoformat(str(started_at))
                            finished = datetime.fromisoformat(
                                str(update["finished_at"])
                            )
                            update["duration_seconds"] = round(
                                max(0.0, (finished - started).total_seconds()),
                                3,
                            )
                        except Exception:
                            pass
                node_states[node_id] = {
                    **previous,
                    **update,
                    "status": node_status,
                    "updated_at": now,
                }
            payload = {
                "version": 3,
                "stage": "analyze-core",
                "current_step": current_step,
                "status": status,
                "message": message,
                "artifacts": artifacts or {},
                "details": details or {},
                "node_states": node_states,
                "updated_at": now,
            }
            tmp_path = output_dir / f".{PROGRESS_FILENAME}.tmp"
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(progress_path)
    except Exception:
        logger.debug("Could not write analysis progress", exc_info=True)
