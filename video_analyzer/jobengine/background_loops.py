"""video-link 状态引擎后台循环片：自动重试、定时调度与后台 TTS 循环。

主类 VideoLinkStatusServer 继承 BackgroundLoopsMixin，保持对外 API 不变。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from http import HTTPStatus
from typing import Any

from video_analyzer.jobengine._shared import (
    AUTO_RETRY_DELAY_SECONDS,
    AUTO_RETRY_POLL_SECONDS,
    AUTO_RETRY_REASONS,
    AUDIO_PIPELINE_KIND_TRANSCRIPTION,
    SCHEDULE_POLL_SECONDS,
    iso_now,
    job_stage_resource,
    normalize_audio_pipeline_profile,
    normalize_stage_name,
    parse_iso_timestamp,
    parse_schedule_datetime,
)
from video_analyzer.jobengine.errors import BridgeError

logger = logging.getLogger(__name__)


class BackgroundLoopsMixin:
    """后台循环相关方法片（由主类继承）。"""

    def start_auto_retry_loop(self) -> None:
        if self.auto_retry_thread and self.auto_retry_thread.is_alive():
            return
        thread = threading.Thread(target=self._auto_retry_loop, daemon=True)
        self.auto_retry_thread = thread
        thread.start()

    def _auto_retry_loop(self) -> None:
        while not self.auto_retry_stop.wait(max(1.0, AUTO_RETRY_POLL_SECONDS)):
            try:
                self.auto_retry_queued_jobs_once()
            except Exception:
                continue

    def auto_retry_queued_jobs_once(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        candidates: list[dict[str, Any]] = []
        for path in sorted(self.jobs_dir.glob("*/job.json")):
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            retry = self.auto_retry_info(job, now)
            if retry.get("ready"):
                job["_auto_retry"] = retry
                candidates.append(job)

        candidates.sort(key=lambda item: ((item.get("_auto_retry") or {}).get("queued_at_ts") or 0, item.get("job_id") or ""))
        started: list[str] = []
        started_resources: set[str] = set()
        for job in candidates:
            retry = job.get("_auto_retry") or {}
            resource = str(retry.get("resource") or "")
            if not resource or resource in started_resources or self.resource_has_running_work(resource):
                continue
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            if isinstance(job.get("collection"), dict):
                self.start_run(job_id, resume_collection=False)
            else:
                self.start_run(job_id)
            started.append(job_id)
            started_resources.add(resource)
        return started

    def start_schedule_loop(self) -> None:
        if self.schedule_thread and self.schedule_thread.is_alive():
            return
        thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self.schedule_thread = thread
        thread.start()

    def _schedule_loop(self) -> None:
        while not self.schedule_stop.wait(max(1.0, SCHEDULE_POLL_SECONDS)):
            try:
                self.start_due_scheduled_jobs_once()
            except Exception:
                continue

    def start_due_scheduled_jobs_once(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        started: list[str] = []
        for path in sorted(self.jobs_dir.glob("*/job.json")):
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            schedule = job.get("schedule") or {}
            if schedule.get("status") != "scheduled":
                continue
            start_at = parse_schedule_datetime(schedule.get("start_at"))
            if start_at is None or start_at.timestamp() > now:
                continue
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            job["schedule"] = {**schedule, "status": "triggered", "triggered_at": iso_now()}
            job["updated_at"] = iso_now()
            self.save_job(job)
            try:
                self.start_run(job_id)
                started.append(job_id)
            except BridgeError as exc:
                job = self.load_job(job_id)
                job["schedule"] = {**(job.get("schedule") or {}), "status": "error", "error": exc.message}
                job["updated_at"] = iso_now()
                self.save_job(job)
        return started

    def is_auto_retry_job(self, job: dict[str, Any]) -> bool:
        return bool(self.auto_retry_info(job).get("auto_retry"))

    def auto_retry_info(self, job: dict[str, Any], now: float | None = None) -> dict[str, Any]:
        runner = job.get("runner") or {}
        stage = normalize_stage_name(runner.get("current_stage") or self.current_stage(job) or "")
        if job.get("status") != "queued" or runner.get("status") != "queued" or not stage:
            return {}
        stage_info = (job.get("stages") or {}).get(stage) or {}
        retry_reason = stage_info.get("retry_reason") or runner.get("error")
        if retry_reason not in AUTO_RETRY_REASONS:
            return {}
        queued_at = stage_info.get("queued_at") or runner.get("updated_at") or job.get("updated_at")
        queued_at_ts = parse_iso_timestamp(queued_at)
        if queued_at_ts is None:
            queued_at_ts = time.time()
        now = time.time() if now is None else now
        retry_info = dict(stage_info.get("retry") or {})
        next_retry_at_ts = parse_iso_timestamp(retry_info.get("next_retry_at"))
        delay = max(0.0, AUTO_RETRY_DELAY_SECONDS)
        ready_at = next_retry_at_ts if next_retry_at_ts is not None else queued_at_ts + delay
        retry_after = max(0.0, ready_at - now)
        resource = runner.get("queued_for") or stage_info.get("queued_for") or job_stage_resource(job, stage)
        return {
            "auto_retry": True,
            "ready": retry_after <= 0,
            "retry_after_seconds": int(round(retry_after)),
            "retry_delay_seconds": int(round(delay)),
            "queued_at_ts": queued_at_ts,
            "resource": resource,
            "stage": stage,
        }

    def resource_has_running_work(self, resource: str) -> bool:
        resources = self.resource_summary()
        info = resources.get(resource) or {}
        return int(info.get("running_count") or 0) > 0

    def queue_audio_tts(self, job_id: str) -> dict[str, Any] | None:
        job = self.load_job(job_id)
        pipeline_kind = normalize_audio_pipeline_profile(
            job.get("audio_pipeline_kind")
            or job.get("audio_pipeline_profile")
        )
        if (
            not self.is_tenant_mobile_audio_job(job)
            or pipeline_kind == AUDIO_PIPELINE_KIND_TRANSCRIPTION
            or job.get("status") != "succeeded"
        ):
            return None
        run_dir = self.discover_run_dir(job)
        manual = run_dir / "operation_manual.md" if run_dir else None
        if not manual or not manual.is_file() or manual.stat().st_size < 120:
            return None
        tasks = dict(job.get("background_tasks") or {})
        current = dict(tasks.get("tts_summary") or {})
        if current.get("status") in {
            "queued",
            "waiting_for_idle",
            "running",
            "succeeded",
        }:
            return current
        tasks["tts_summary"] = {
            "status": "queued",
            "attempt": int(current.get("attempt") or 0),
            "queued_at": iso_now(),
            "started_at": None,
            "finished_at": None,
            "error": "",
            "artifacts": {},
        }
        job["background_tasks"] = tasks
        job["updated_at"] = iso_now()
        self.save_job(job)
        return tasks["tts_summary"]

    def recover_interrupted_audio_tts(self) -> None:
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            tasks = dict(job.get("background_tasks") or {})
            tts = dict(tasks.get("tts_summary") or {})
            explicit_tenant = bool(str(job.get("tenant_id") or "").strip())
            if (
                tts.get("status") in {"queued", "waiting_for_idle"}
                and not explicit_tenant
                and int(tts.get("attempt") or 0) == 0
            ):
                tts.update(
                    {
                        "status": "skipped",
                        "finished_at": iso_now(),
                        "error": "legacy audio job has no tenant binding",
                    }
                )
                tasks["tts_summary"] = tts
                job["background_tasks"] = tasks
                job["updated_at"] = iso_now()
                self.save_job(job)
                continue
            if (
                not tts
                and job.get("status") == "succeeded"
                and self.is_tenant_mobile_audio_job(job)
                and explicit_tenant
            ):
                self.queue_audio_tts(job["job_id"])
                continue
            if tts.get("status") != "running":
                continue
            tts["status"] = "queued"
            tts["error"] = "AI service restarted; waiting for idle capacity"
            tts["queued_at"] = iso_now()
            tasks["tts_summary"] = tts
            job["background_tasks"] = tasks
            self.save_job(job)

    def acknowledge_mobile_audio_tts(
        self,
        job_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        job = self.load_job(job_id)
        if not self.is_mobile_audio_job(job) or not self.mobile_audio_tenant_matches(
            job,
            tenant_id,
        ):
            raise BridgeError(HTTPStatus.NOT_FOUND, "audio job not found")
        tasks = dict(job.get("background_tasks") or {})
        tts = dict(tasks.get("tts_summary") or {})
        if tts.get("status") != "succeeded":
            raise BridgeError(
                HTTPStatus.CONFLICT,
                "audio TTS cannot be acknowledged before it succeeds",
            )
        tts["synced_at"] = iso_now()
        tasks["tts_summary"] = tts
        job["background_tasks"] = tasks
        job["updated_at"] = iso_now()
        self.save_job(job)
        return {
            "acknowledged": True,
            "job_id": job_id,
            "synced_at": tts["synced_at"],
        }

    def start_audio_tts_loop(self) -> None:
        if self.audio_tts_thread and self.audio_tts_thread.is_alive():
            return
        self.audio_tts_stop.clear()
        self.audio_tts_heartbeat_at = time.monotonic()
        self.audio_tts_thread = threading.Thread(
            target=self._audio_tts_loop,
            daemon=True,
            name="audio-tts-background",
        )
        self.audio_tts_thread.start()

    def _audio_tts_loop(self) -> None:
        while not self.audio_tts_stop.wait(10):
            self.audio_tts_heartbeat_at = time.monotonic()
            try:
                if self.production_audio_local_busy():
                    self.audio_tts_idle_since = None
                    continue
                now = time.monotonic()
                if self.audio_tts_idle_since is None:
                    self.audio_tts_idle_since = now
                    continue
                if now - self.audio_tts_idle_since < 60:
                    continue
                candidate = self.next_audio_tts_job()
                if candidate:
                    self.audio_tts_current_job_id = candidate["job_id"]
                    try:
                        self.run_audio_tts(candidate["job_id"])
                    finally:
                        self.audio_tts_current_job_id = ""
                        self.audio_tts_idle_since = None
                self.audio_tts_last_error = ""
            except Exception as exc:
                self.audio_tts_last_error = str(exc)
                logger.exception("audio TTS scheduler iteration failed")
                time.sleep(5)

    def next_audio_tts_job(self) -> dict[str, Any] | None:
        candidates = []
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            tts = ((job.get("background_tasks") or {}).get("tts_summary") or {})
            if (
                str(job.get("tenant_id") or "").strip()
                and tts.get("status") in {"queued", "waiting_for_idle"}
            ):
                candidates.append(job)
        candidates.sort(
            key=lambda item: (
                (
                    (
                        (item.get("background_tasks") or {})
                        .get("tts_summary", {})
                        .get("queued_at")
                    )
                    or ""
                ),
                item.get("job_id") or "",
            )
        )
        return candidates[0] if candidates else None

    def run_audio_tts(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        tasks = dict(job.get("background_tasks") or {})
        tts = dict(tasks.get("tts_summary") or {})
        run_dir = self.require_run_dir(job)
        tts.update(
            {
                "status": "running",
                "attempt": int(tts.get("attempt") or 0) + 1,
                "started_at": iso_now(),
                "finished_at": None,
                "error": "",
            }
        )
        tasks["tts_summary"] = tts
        job["background_tasks"] = tasks
        self.save_job(job)
        log_path = str(
            self.job_dir(job_id) / "logs" / "tts-summary-background.log"
        )
        command = [
            "tools/pipelines/run_audio_narration_stage.sh",
            str(run_dir),
            "--profile",
            os.environ.get("VIDEO_ANALYZER_AUDIO_TTS_PROFILE", "local_new"),
            "--config",
            "config",
        ]
        try:
            self.run_command(command, log_path)
            artifacts = {
                name: str(path)
                for name, path in {
                    "narration_audio": (
                        run_dir
                        / "audio_narration"
                        / "audio_output"
                        / "narration_full.wav"
                    ),
                    "narration_script": (
                        run_dir / "audio_narration" / "narration_script.md"
                    ),
                    "narration_metadata": (
                        run_dir / "audio_narration" / "narration_metadata.json"
                    ),
                    "narration_timeline": (
                        run_dir / "audio_narration" / "narration_timeline.json"
                    ),
                }.items()
                if path.is_file() and path.stat().st_size > 0
            }
            if "narration_audio" not in artifacts:
                raise RuntimeError(
                    "background TTS did not produce narration audio"
                )
            tts.update(
                {
                    "status": "succeeded",
                    "finished_at": iso_now(),
                    "error": "",
                    "artifacts": artifacts,
                }
            )
        except Exception as exc:
            tts.update(
                {
                    "status": "failed",
                    "finished_at": iso_now(),
                    "error": str(exc),
                }
            )
        current = self.load_job(job_id)
        current_tasks = dict(current.get("background_tasks") or {})
        current_tasks["tts_summary"] = tts
        current["background_tasks"] = current_tasks
        current["updated_at"] = iso_now()
        self.save_job(current)
        return tts

