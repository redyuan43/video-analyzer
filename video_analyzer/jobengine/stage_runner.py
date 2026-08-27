"""video-link 状态引擎 stage runner 片：阶段执行、资源等待、失败重试与阶段命令构建。

主类 VideoLinkStatusServer 继承 StageRunnerMixin，保持对外 API 不变。
"""

from __future__ import annotations

import os
import re
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from video_analyzer.jobengine._shared import (
    AUDIO_PIPELINE_PROFILE_NX1,
    AUTO_RETRY_DELAY_SECONDS,
    BAOYU_IMAGE_GENERATION_ENABLED,
    MAX_INTERRUPTED_RETRIES,
    MAX_TRANSIENT_API_RETRIES,
    MAX_YOUTUBE_FORMAT_RETRIES,
    ORPHANED_PROCESS_REQUEUE_MESSAGE,
    RESOURCE_LIMITS,
    RESOURCE_WAIT_SECONDS,
    TRANSIENT_API_REQUEUE_MESSAGE,
    TRANSIENT_RESOURCE_BUSY_PATTERNS,
    TRANSIENT_RESOURCE_REQUEUE_MESSAGE,
    YOUTUBE_FORMAT_REQUEUE_MESSAGE,
    YOUTUBE_FORMAT_UNAVAILABLE_PATTERN,
    YOUTUBE_RATE_LIMIT_PATTERNS,
    is_youtube_url,
    iso_from_timestamp,
    iso_now,
    job_stage_resource,
    normalize_audio_pipeline_profile,
    normalize_stage_name,
    parse_iso_timestamp,
    process_alive,
)
from video_analyzer.failures import read_failure_envelope
from video_analyzer.jobengine.errors import BridgeError
from video_analyzer.tencent_hy_asr import missing_tencent_credentials


class StageRunnerMixin:
    """stage runner 相关方法片（由主类继承）。"""

    def _run_remaining_stages(self, job_id: str) -> None:
        self._run_remaining_stages_serial(job_id)

    def _run_remaining_stages_serial(self, job_id: str) -> None:
        transition_count = 0
        try:
            while True:
                job = self.load_job(job_id)
                if job.get("status") == "no_speech":
                    self.update_runner(job, "no_speech", current_stage=None, finished=True)
                    return
                stage = self.next_stage(job)
                if not stage:
                    job["status"] = "succeeded"
                    self.update_runner(job, "succeeded", current_stage=None, finished=True)
                    self.queue_audio_tts(job_id)
                    return
                if transition_count > len(self.stage_order_for_job(job)):
                    raise BridgeError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "state_machine_invariant_violation: runner exceeded the maximum stage transitions",
                    )
                self.update_runner(job, "running", current_stage=stage)
                result = self.run_stage(job_id, stage, continue_runner=True)
                result_runner = result.get("runner") or {}
                if result.get("status") == "queued" or result_runner.get("status") == "queued":
                    return
                transition_count += 1
                next_stage = self.next_stage(result)
                if next_stage == stage:
                    raise BridgeError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"state_machine_invariant_violation: stage {stage} did not converge",
                    )
                current = self.load_job(job_id)
                runner = dict(current.get("runner") or {})
                runner["transition_count"] = transition_count
                current["runner"] = runner
                self.save_job(current)
        except BridgeError as exc:
            job = self.load_job(job_id)
            if self.runner_failure_can_finish_with_warning(job):
                self.add_warning(job, "runner", exc.message)
                job["status"] = "succeeded"
                self.update_runner(job, "succeeded", error=None, current_stage=None, finished=True)
            else:
                job["status"] = "failed"
                self.update_runner(job, "failed", error=exc.message, finished=True)
        except Exception as exc:
            job = self.load_job(job_id)
            if self.runner_failure_can_finish_with_warning(job):
                self.add_warning(job, "runner", str(exc))
                job["status"] = "succeeded"
                self.update_runner(job, "succeeded", error=None, current_stage=None, finished=True)
            else:
                job["status"] = "failed"
                self.update_runner(job, "failed", error=str(exc), finished=True)
        finally:
            with self.runner_lock:
                self.active_runners.pop(job_id, None)
            self.advance_collection_after_job(job_id)

    def prepare_collection_job_start(
        self,
        job: dict[str, Any],
        *,
        resume_paused: bool = True,
    ) -> None:
        collection = job.get("collection")
        if not isinstance(collection, dict):
            return
        collection_id = str(collection.get("id") or "")
        with self.collection_lock:
            manifest = self.load_collection(collection_id)
            if job.get("collection_rerun"):
                failed_job_ids = {
                    str(item.get("job_id") or "")
                    for item in manifest.get("failures") or []
                    if isinstance(item, dict)
                }
                if job["job_id"] not in failed_job_ids:
                    raise BridgeError(
                        HTTPStatus.CONFLICT,
                        "collection part is not eligible for a failure rerun",
                    )
                return
            if str(manifest.get("current_job_id") or "") != job["job_id"]:
                raise BridgeError(
                    HTTPStatus.CONFLICT,
                    "another Bilibili collection part must run first",
                )
            if manifest.get("status") in {"succeeded", "completed_with_errors"}:
                raise BridgeError(HTTPStatus.CONFLICT, "Bilibili collection is already complete")
            if manifest.get("status") == "paused" and not resume_paused:
                raise BridgeError(HTTPStatus.CONFLICT, "Bilibili collection is paused")
            if manifest.get("status") in {"created", "paused"}:
                manifest["status"] = "running"
                manifest["paused_reason"] = None
                manifest["updated_at"] = iso_now()
                self.save_collection(manifest)

    def pause_collection_for_job(self, job: dict[str, Any]) -> None:
        collection = job.get("collection")
        if not isinstance(collection, dict):
            return
        if job.get("collection_rerun"):
            return
        collection_id = str(collection.get("id") or "")
        with self.collection_lock:
            manifest = self.load_collection(collection_id)
            if str(manifest.get("current_job_id") or "") != job["job_id"]:
                return
            manifest["status"] = "paused"
            manifest["paused_reason"] = "stopped by user"
            manifest["updated_at"] = iso_now()
            self.save_collection(manifest)

    def advance_collection_after_job(self, job_id: str) -> None:
        try:
            job = self.load_job(job_id)
        except BridgeError:
            return
        collection = job.get("collection")
        if not isinstance(collection, dict):
            return
        if job.get("status") not in {"succeeded", "failed", "no_speech"}:
            return
        collection_id = str(collection.get("id") or "")
        next_job_id = ""
        with self.collection_lock:
            manifest = self.load_collection(collection_id)
            if job.get("collection_rerun"):
                failures = [
                    item
                    for item in manifest.get("failures") or []
                    if isinstance(item, dict)
                    and str(item.get("job_id") or "") != job_id
                ]
                if job.get("status") == "failed":
                    failures.append(
                        {
                            "job_id": job_id,
                            "index": int(collection.get("index") or 1),
                            "error": str((job.get("runner") or {}).get("error") or "job failed"),
                            "failed_at": iso_now(),
                        }
                    )
                manifest["failures"] = failures
                if manifest.get("status") in {"succeeded", "completed_with_errors"}:
                    manifest["status"] = (
                        "completed_with_errors"
                        if failures
                        else "succeeded"
                    )
                    manifest["finished_at"] = iso_now()
                manifest["updated_at"] = iso_now()
                self.save_collection(manifest)
                job.pop("collection_rerun", None)
                self.save_job(job)
                return
            if manifest.get("status") == "paused":
                return
            if str(manifest.get("current_job_id") or "") != job_id:
                return
            children = list(manifest.get("children") or [])
            current_index = int(manifest.get("current_index") or collection.get("index") or 1)
            if job.get("status") == "failed":
                failures = list(manifest.get("failures") or [])
                if not any(item.get("job_id") == job_id for item in failures if isinstance(item, dict)):
                    failures.append(
                        {
                            "job_id": job_id,
                            "index": current_index,
                            "error": str((job.get("runner") or {}).get("error") or "job failed"),
                            "failed_at": iso_now(),
                        }
                    )
                manifest["failures"] = failures
            next_child = next(
                (
                    child
                    for child in children
                    if int(child.get("index") or 0) > current_index
                ),
                None,
            )
            if next_child:
                manifest["status"] = "running"
                manifest["current_index"] = int(next_child["index"])
                manifest["current_job_id"] = str(next_child["job_id"])
                manifest["updated_at"] = iso_now()
                self.save_collection(manifest)
                next_job_id = str(next_child["job_id"])
            else:
                manifest["status"] = (
                    "completed_with_errors"
                    if manifest.get("failures")
                    else "succeeded"
                )
                manifest["finished_at"] = iso_now()
                manifest["updated_at"] = manifest["finished_at"]
                self.save_collection(manifest)
        if next_job_id:
            try:
                self.start_run(
                    next_job_id,
                    resume_collection=False,
                )
            except BridgeError as exc:
                with self.collection_lock:
                    manifest = self.load_collection(collection_id)
                    if manifest.get("status") != "paused":
                        manifest["status"] = "paused"
                        manifest["paused_reason"] = exc.message
                        manifest["updated_at"] = iso_now()
                        self.save_collection(manifest)

    def update_runner(
        self,
        job: dict[str, Any],
        status: str,
        current_stage: str | None = None,
        error: str | None = None,
        finished: bool = False,
    ) -> None:
        runner = dict(job.get("runner") or {})
        runner["status"] = status
        runner["updated_at"] = iso_now()
        runner["server_pid"] = os.getpid()
        if "started_at" not in runner:
            runner["started_at"] = runner["updated_at"]
        runner["current_stage"] = current_stage
        runner.pop("wait_reason", None)
        if error is not None:
            runner["error"] = error
        elif status != "failed":
            runner["error"] = None
        if finished:
            runner["finished_at"] = runner["updated_at"]
        job["runner"] = runner
        job["updated_at"] = runner["updated_at"]
        if status == "succeeded":
            job["status"] = "succeeded"
        elif status == "failed":
            job["status"] = "failed"
        elif status == "no_speech":
            job["status"] = "no_speech"
        self.save_job(job)

    def run_stage(self, job_id: str, stage: str, continue_runner: bool = False) -> dict[str, Any]:
        stage = normalize_stage_name(stage)
        job = self.load_job(job_id)
        if stage not in self.stage_order_for_job(job):
            raise BridgeError(HTTPStatus.NOT_FOUND, f"unknown stage: {stage}")
        self.ensure_dependencies(job, stage)
        current_status = job.get("stages", {}).get(stage, {}).get("status")
        if stage == "final-publish" and current_status == "skipped" and self.export_outputs_complete(job):
            now = iso_now()
            stage_info = dict(job.get("stages", {}).get(stage) or {})
            stage_info["status"] = "succeeded"
            stage_info["exit_code"] = 0
            stage_info["finished_at"] = now
            stage_info.pop("error", None)
            stage_info.pop("warning", None)
            stage_info.pop("soft_failed", None)
            job.setdefault("stages", {})[stage] = stage_info
            runner = dict(job.get("runner") or {})
            runner["status"] = "succeeded"
            runner["current_stage"] = None
            runner["error"] = None
            runner["finished_at"] = now
            runner["updated_at"] = now
            job["runner"] = runner
            job["status"] = "succeeded"
            job["summary"] = self.collect_summary(job)
            job["updated_at"] = now
            self.save_job(job)
            return self.public_job(job)
        if stage == "final-publish" and current_status == "skipped" and not self.export_outputs_complete(job):
            current_status = None
        if current_status == "skipped" and self.skipped_stage_outputs_incomplete(job, stage):
            current_status = None
        if current_status in {"succeeded", "skipped"}:
            return self.public_job(job)
        if stage == "image-prompts" and (job["options"].get("skip_images") or not BAOYU_IMAGE_GENERATION_ENABLED):
            return self.mark_stage_skipped(job, stage, "baoyu image generation is disabled", continue_runner=continue_runner)
        if stage == "tts-narration" and not self.tts_narration_enabled(job):
            return self.mark_stage_skipped(
                job,
                stage,
                "runtime profile has no enabled TTS model",
                continue_runner=continue_runner,
            )

        job = self.select_audio_compute_route(job, stage)
        job = self.select_audio_tts_route(job, stage)
        resource = job_stage_resource(job, stage)
        self.mark_stage_queued(job, stage, resource)
        self.wait_for_resource_slot(resource, job_id)
        lock = self.resource_locks[resource]
        lock.acquire()
        try:
            return self._run_stage_locked(job_id, stage, continue_runner=continue_runner)
        finally:
            lock.release()

    def wait_for_resource_slot(self, resource: str, job_id: str) -> None:
        limit = max(1, int(RESOURCE_LIMITS.get(resource, 1)))
        while True:
            job = self.load_job(job_id)
            runner = job.get("runner") or {}
            if job.get("status") == "failed" or runner.get("status") == "failed":
                raise BridgeError(HTTPStatus.CONFLICT, runner.get("error") or "job stopped")
            blockers = self.live_resource_users(resource, exclude_job_id=job_id)
            if len(blockers) < limit:
                return
            self.touch_queued_runner(job_id, resource, len(blockers), limit)
            time.sleep(RESOURCE_WAIT_SECONDS)

    def select_audio_compute_route(
        self,
        job: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        raw_pipeline_kind = (
            job.get("audio_pipeline_kind")
            or job.get("audio_pipeline_profile")
        )
        if (
            normalize_stage_name(stage) != "analyze-core"
            or not raw_pipeline_kind
            or normalize_audio_pipeline_profile(raw_pipeline_kind)
            != AUDIO_PIPELINE_PROFILE_NX1
        ):
            return job
        if job.get("compute_route") in {"local", "cloud_fallback"}:
            return job
        fallback = (job.get("runtime_profile_snapshot") or {}).get(
            "audio_cloud_fallback"
        ) or {}
        content_fallback = (job.get("runtime_profile_snapshot") or {}).get(
            "content_cloud_fallback"
        ) or {}
        fallback_credentials_ready = self.audio_cloud_fallback_credentials_ready(
            fallback,
            content_fallback,
        )
        fallback_enabled = bool(
            fallback.get("enabled") or content_fallback.get("enabled")
        )
        local_busy = self.production_audio_local_busy(job.get("job_id"))
        job["compute_route"] = (
            "cloud_fallback"
            if local_busy and fallback_enabled and fallback_credentials_ready
            else "local"
        )
        job["compute_route_reason"] = (
            "local_resource_busy"
            if job["compute_route"] == "cloud_fallback"
            else (
                "cloud_fallback_credentials_missing"
                if local_busy and fallback_enabled
                else "local_first"
            )
        )
        cloud = job["compute_route"] == "cloud_fallback"
        job["execution_routes"] = {
            "asr": {
                "route": "cloud" if cloud else "local",
                "provider": "tencent_hy_asr" if cloud else "vibevoice",
                "reason": job["compute_route_reason"],
                "local_wait_seconds": 0,
            },
            "diarization": {
                "route": "cloud" if cloud else "local",
                "provider": "asr_embedded" if cloud else "3dspeaker",
                "degraded": cloud,
            },
            "template_selector": {
                "route": "pending",
                "provider": "local_qwen_or_trae",
                "reason": "checked_immediately_before_text_phase",
                "local_wait_seconds": 0,
            },
            "summary": {
                "route": "pending",
                "provider": "local_qwen_or_trae",
                "reason": "checked_immediately_before_text_phase",
                "local_wait_seconds": 0,
            },
        }
        job["updated_at"] = iso_now()
        self.save_job(job)
        return job

    def production_audio_local_busy(self, exclude_job_id: str | None = None) -> bool:
        for path in self.jobs_dir.glob("*/job.json"):
            if exclude_job_id and path.parent.name == exclude_job_id:
                continue
            try:
                candidate = self.load_job(path.parent.name)
            except Exception:
                continue
            runner = candidate.get("runner") or {}
            if (
                candidate.get("status") in {"running", "queued"}
                or runner.get("status") in {"running", "queued"}
            ):
                return True
        return any(
            self.live_resource_users(resource, exclude_job_id=exclude_job_id)
            for resource in ("core", "audio-analysis", "asr", "ocr", "vl", "tts")
        )

    def select_audio_tts_route(
        self,
        job: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        if normalize_stage_name(stage) != "tts-narration":
            return job
        if job.get("tts_route") in {"local", "cloud_fallback"}:
            return job
        snapshot = job.get("runtime_profile_snapshot") or {}
        fallback = snapshot.get("tts_cloud_fallback") or {}
        if not fallback.get("enabled"):
            job["tts_route"] = "local"
            job["tts_route_reason"] = "no_cloud_fallback"
            self.save_job(job)
            return job
        key_env = str(fallback.get("api_key_env") or "").strip()
        credentials_ready = not key_env or bool(
            str(os.environ.get(key_env) or "").strip()
        )
        local_busy = bool(
            self.live_resource_users("tts", exclude_job_id=job.get("job_id"))
        )
        if local_busy and credentials_ready:
            job["tts_route"] = "cloud_fallback"
            job["tts_route_reason"] = "local_tts_queue_detected"
        else:
            job["tts_route"] = "local"
            job["tts_route_reason"] = (
                "cloud_tts_credentials_missing"
                if local_busy
                else "local_tts_slot_available"
            )
        job["updated_at"] = iso_now()
        self.save_job(job)
        return job

    @staticmethod
    def audio_cloud_fallback_credentials_ready(
        fallback: dict[str, Any],
        content_fallback: dict[str, Any] | None = None,
    ) -> bool:
        asr = fallback.get("asr") or {}
        if (
            str(asr.get("protocol") or "") == "tencent_hy_asr_ws"
            and missing_tencent_credentials(dict(asr.get("options") or {}))
        ):
            return False
        for stage in (content_fallback or {}).values():
            if not isinstance(stage, dict) or not stage.get("enabled"):
                continue
            key_env = str(stage.get("api_key_env") or "").strip()
            if key_env and not str(os.environ.get(key_env) or "").strip():
                return False
        return True

    def live_resource_users(self, resource: str, exclude_job_id: str | None = None) -> list[dict[str, Any]]:
        users = []
        for path in self.jobs_dir.glob("*/job.json"):
            if exclude_job_id and path.parent.name == exclude_job_id:
                continue
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            runner = job.get("runner") or {}
            if runner.get("status") != "running":
                continue
            stage = normalize_stage_name(runner.get("current_stage") or self.current_stage(job) or "")
            if not stage or job_stage_resource(job, stage) != resource:
                continue
            stage_info = (job.get("stages") or {}).get(stage) or {}
            process_info = stage_info.get("process") or {}
            pid = process_info.get("pid")
            if self.stage_is_live(job, stage, stage_info) or (pid and process_alive(pid)):
                users.append(job)
        return users

    def _run_stage_locked(self, job_id: str, stage: str, continue_runner: bool = False) -> dict[str, Any]:
        stage = normalize_stage_name(stage)
        job = self.load_job(job_id)
        previous_stage_info = dict(job.get("stages", {}).get(stage, {}) or {})
        current_status = previous_stage_info.get("status")
        if current_status in {"succeeded", "skipped"}:
            return self.public_job(job)

        start = time.time()
        started_at = iso_now()
        attempt = max(1, int(previous_stage_info.get("attempt") or 0) + 1)
        log_path, attempt_log_paths = self.prepare_stage_log_attempt(
            job_id,
            stage,
            previous_stage_info,
            attempt,
        )
        stage_info = {
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "exit_code": None,
            "attempt": attempt,
            "attempt_log_paths": attempt_log_paths,
            "log_path": str(log_path),
            "artifacts": {},
            "queued_for": job_stage_resource(job, stage),
        }
        queued_at = previous_stage_info.get("queued_at")
        if queued_at:
            stage_info["queued_at"] = queued_at
            queued_timestamp = parse_iso_timestamp(queued_at)
            started_timestamp = parse_iso_timestamp(started_at)
            if queued_timestamp and started_timestamp:
                stage_info["queue_duration_seconds"] = round(
                    max(0.0, started_timestamp - queued_timestamp),
                    3,
                )
        failure_path = self.stage_failure_path(job_id, stage, attempt)
        if failure_path.exists():
            failure_path.unlink()
        stage_info["failure_path"] = str(failure_path)
        for key in ("auto_retry_attempts", "first_error", "retry"):
            if previous_stage_info.get(key):
                stage_info[key] = previous_stage_info[key]
        job["status"] = "running"
        job["updated_at"] = iso_now()
        job["stages"][stage] = stage_info
        self.save_job(job)

        try:
            if stage == "probe":
                result = self.stage_probe(job)
            elif stage == "prepare":
                result = self.stage_prepare(job, stage_info["log_path"], stage_info)
            elif stage == "analyze-core":
                result = self.stage_analyze_core(job, stage_info["log_path"], stage_info)
            elif stage == "verify-core":
                result = self.stage_verify_core(
                    job,
                    stage_info["log_path"],
                    stage_info,
                )
            elif stage == "multidoc":
                result = self.run_command_stage(job, stage, self.multidoc_command(job), stage_info["log_path"], stage_info)
            elif stage == "deep-v2":
                result = self.stage_deep_v2(job, stage_info["log_path"], stage_info)
            elif stage == "study-guide":
                result = self.run_command_stage(job, stage, self.study_guide_command(job), stage_info["log_path"], stage_info)
            elif stage == "evidence-review":
                result = self.run_command_stage(job, stage, self.evidence_review_command(job), stage_info["log_path"], stage_info)
            elif stage == "web-evidence":
                result = self.run_command_stage(job, stage, self.web_evidence_command(job), stage_info["log_path"], stage_info)
            elif stage == "qa-index":
                result = self.run_command_stage(job, stage, self.qa_index_command(job), stage_info["log_path"], stage_info)
            elif stage == "image-prompts":
                result = self.run_command_stage(job, stage, self.image_prompts_command(job), stage_info["log_path"], stage_info)
            elif stage == "tts-narration":
                result = self.stage_tts_narration(job, stage_info["log_path"], stage_info)
            else:
                result = self.run_command_stage(job, stage, self.final_publish_command(job), stage_info["log_path"], stage_info)
            stage_info.update(result)
            stage_info.pop("process", None)
            self.update_job_artifacts(job, stage, result.get("artifacts", {}))
            stage_info["status"] = "succeeded"
            stage_info["exit_code"] = 0
            stage_info["duration_seconds"] = round(time.time() - start, 3)
            stage_info["finished_at"] = iso_now()
            job["status"] = "succeeded" if self.next_stage(job) is None else "running"
        except Exception as exc:
            no_speech = (
                stage == "analyze-core"
                and self.is_mobile_audio_job(job)
                and "NO_SPEECH:" in self.exception_text(exc)
            )
            failure = self.stage_failure(stage_info, exc)
            retry_reason = self.retryable_stage_failure_reason(
                job,
                stage,
                exc,
                stage_info["log_path"],
                previous_stage_info,
                failure,
            )
            stage_info["status"] = "queued" if retry_reason else "failed"
            stage_info["exit_code"] = getattr(exc, "returncode", 1)
            stage_info["duration_seconds"] = round(time.time() - start, 3)
            stage_info["finished_at"] = iso_now()
            stage_info.pop("process", None)
            stage_info["failure"] = failure
            if retry_reason:
                stage_info["queued_at"] = stage_info["finished_at"]
                stage_info.pop("queue_duration_seconds", None)
                stage_info["queued_for"] = job_stage_resource(job, stage)
                stage_info["retry_reason"] = retry_reason
                stage_info["last_error"] = self.exception_text(exc) or str(exc)
                previous_retry = dict(previous_stage_info.get("retry") or {})
                max_attempts = self.max_auto_retries_for_reason(retry_reason)
                stage_info["retry"] = {
                    "auto_attempts": int(previous_retry.get("auto_attempts") or 0) + 1,
                    "max_auto_attempts": max_attempts,
                    "next_retry_at": iso_from_timestamp(time.time() + max(0.0, AUTO_RETRY_DELAY_SECONDS)),
                }
                stage_info["auto_retry_attempts"] = stage_info["retry"]["auto_attempts"]
                stage_info["first_error"] = previous_stage_info.get("first_error") or stage_info["last_error"]
                stage_info.pop("error", None)
                job["status"] = "queued"
            elif self.stage_can_soft_fail(job, stage, failure):
                visible_error = str(failure.get("message") or str(exc))
                stage_info["error"] = visible_error
                warning = self.add_warning(job, stage, visible_error)
                stage_info["status"] = "skipped"
                stage_info["warning"] = warning["message"]
                stage_info["soft_failed"] = True
                job["status"] = "running"
            elif no_speech:
                stage_info["status"] = "skipped"
                stage_info["error"] = "未检测到可转写的人声"
                stage_info["error_code"] = "no_speech"
                stage_info["no_speech"] = True
                job["status"] = "no_speech"
            else:
                stage_info["error"] = str(failure.get("message") or str(exc))
                job["status"] = "failed"
        job["updated_at"] = iso_now()
        job["stages"][stage] = stage_info
        job["summary"] = self.collect_summary(job)
        self.finalize_stage_runner(job, stage, stage_info, continue_runner=continue_runner)
        self.save_job(job)
        if stage_info["status"] == "failed":
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, f"{stage} failed: {stage_info.get('error')}")
        return self.public_job(job)

    def retryable_stage_failure_reason(
        self,
        job: dict[str, Any],
        stage: str,
        exc: Exception,
        log_path: str,
        previous_stage_info: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
    ) -> str | None:
        previous_stage_info = previous_stage_info or {}
        failure = failure or {}
        retry = dict(previous_stage_info.get("retry") or {})
        normalized_stage = normalize_stage_name(stage)
        if failure.get("kind") == "transient_resource":
            if int(retry.get("auto_attempts") or 0) < MAX_TRANSIENT_API_RETRIES:
                return TRANSIENT_RESOURCE_REQUEUE_MESSAGE
            return None
        if failure.get("retryable"):
            if int(retry.get("auto_attempts") or 0) < MAX_TRANSIENT_API_RETRIES:
                return TRANSIENT_API_REQUEUE_MESSAGE
            return None
        text = self.exception_text(exc)
        output = getattr(exc, "output", None)
        if not output:
            try:
                text += "\n" + Path(log_path).read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if (
            normalized_stage == "prepare"
            and is_youtube_url(str(job.get("video_url") or ""))
            and any(pattern in text for pattern in YOUTUBE_RATE_LIMIT_PATTERNS)
        ):
            if int(retry.get("auto_attempts") or 0) < MAX_TRANSIENT_API_RETRIES:
                return TRANSIENT_API_REQUEUE_MESSAGE
            return None
        if normalized_stage != "analyze-core":
            return None
        retry_reason = self.retryable_stage_failure_text(stage, text)
        if retry_reason:
            return retry_reason
        if not self.youtube_format_retry_allowed(job, previous_stage_info, text):
            return None
        return YOUTUBE_FORMAT_REQUEUE_MESSAGE

    def stage_failure(self, stage_info: dict[str, Any], exc: Exception) -> dict[str, Any]:
        envelope = read_failure_envelope(stage_info.get("failure_path"))
        if envelope:
            return {
                "kind": str(envelope.get("kind") or "unknown"),
                "retryable": bool(envelope.get("retryable")),
                "status_code": envelope.get("status_code"),
                "provider_code": envelope.get("provider_code"),
                "message": str(envelope.get("message") or str(exc)),
            }
        text = self.exception_text(exc)
        if self.retryable_stage_failure_text("", text):
            return {
                "kind": "transient_resource",
                "retryable": True,
                "status_code": None,
                "provider_code": None,
                "message": str(exc),
            }
        return {
            "kind": "unknown",
            "retryable": False,
            "status_code": None,
            "provider_code": None,
            "message": str(exc),
        }

    def max_auto_retries_for_reason(self, reason: str) -> int:
        if reason == YOUTUBE_FORMAT_REQUEUE_MESSAGE:
            return MAX_YOUTUBE_FORMAT_RETRIES
        if reason == ORPHANED_PROCESS_REQUEUE_MESSAGE:
            return MAX_INTERRUPTED_RETRIES
        return MAX_TRANSIENT_API_RETRIES

    def retryable_stage_failure_text(self, stage: str, text: str) -> str | None:
        if "Ray frame driver failed" not in text and "run_frame_worker" not in text and "Jetson" not in text:
            return None
        if any(pattern in text for pattern in TRANSIENT_RESOURCE_BUSY_PATTERNS):
            return TRANSIENT_RESOURCE_REQUEUE_MESSAGE
        return None

    def youtube_format_retry_allowed(self, job: dict[str, Any], previous_stage_info: dict[str, Any], text: str) -> bool:
        if not is_youtube_url(str(job.get("video_url") or "")):
            return False
        if YOUTUBE_FORMAT_UNAVAILABLE_PATTERN not in text or "[youtube]" not in text.lower():
            return False
        if int(previous_stage_info.get("auto_retry_attempts") or 0) >= MAX_YOUTUBE_FORMAT_RETRIES:
            return False
        return not self.core_artifacts_exist(job)

    def core_artifacts_exist(self, job: dict[str, Any]) -> bool:
        run_dir_value = str(job.get("run_dir") or "")
        if not run_dir_value:
            return False
        run_dir = Path(run_dir_value)
        return any((run_dir / name).is_file() for name in ("analysis.json", "operation_manual.md", "manual_evidence.md"))

    def exception_text(self, exc: Exception) -> str:
        text = str(exc)
        output = getattr(exc, "output", None)
        if output:
            text += "\n" + str(output)
        return text

    def touch_queued_runner(self, job_id: str, resource: str, blocker_count: int, limit: int) -> None:
        try:
            job = self.load_job(job_id)
        except BridgeError:
            return
        runner = dict(job.get("runner") or {})
        if runner.get("status") != "queued":
            return
        now = iso_now()
        runner["updated_at"] = now
        runner["wait_reason"] = f"waiting for {resource}: {blocker_count}/{limit} slot(s) in use"
        runner["server_pid"] = os.getpid()
        job["runner"] = runner
        job["updated_at"] = now
        self.save_job(job)

    def mark_stage_queued(
        self,
        job: dict[str, Any],
        stage: str,
        resource: str,
        *,
        retry_reason: str | None = None,
    ) -> None:
        now = iso_now()
        stage_info = dict((job.get("stages") or {}).get(stage) or {})
        stage_info.update(
            {
                "status": "queued",
                "queued_at": now,
                "queued_for": resource,
                "log_path": stage_info.get("log_path") or str(self.stage_log_path(job["job_id"], stage)),
            }
        )
        if retry_reason:
            previous_error = stage_info.pop("error", None)
            if previous_error:
                stage_info["last_error"] = previous_error
            previous_failure = stage_info.pop("failure", None)
            if previous_failure:
                stage_info["last_failure"] = previous_failure
            stage_info["retry_reason"] = retry_reason
            stage_info["retry"] = {
                "auto_attempts": 0,
                "max_auto_attempts": 0,
                "next_retry_at": now,
            }
        stage_info.pop("finished_at", None)
        stage_info.pop("exit_code", None)
        stage_info.pop("queue_duration_seconds", None)
        stage_info.pop("process", None)
        job.setdefault("stages", {})[stage] = stage_info
        job["status"] = "queued"
        runner = dict(job.get("runner") or {})
        runner["status"] = "queued"
        runner["current_stage"] = stage
        runner["queued_for"] = resource
        runner["updated_at"] = now
        runner["server_pid"] = os.getpid()
        runner["error"] = None
        runner.pop("wait_reason", None)
        runner.pop("finished_at", None)
        if "started_at" not in runner:
            runner["started_at"] = now
        job["runner"] = runner
        job["updated_at"] = now
        self.save_job(job)

