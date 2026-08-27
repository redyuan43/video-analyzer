"""video-link 状态引擎 mobile_audio 片：移动端音频任务视图、租户匹配、确认与清理。

主类 VideoLinkStatusServer 继承 MobileAudioMixin，保持对外 API 不变。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from video_analyzer.jobengine._shared import (
    AUDIO_JOB_RETENTION_DAYS,
    AUDIO_PIPELINE_KIND_TRANSCRIPTION,
    AUDIO_PIPELINE_PROFILE_NX1,
    AUDIO_TEMPLATE_CATALOG,
    UPLOAD_SOURCE_TYPE,
    iso_now,
    normalize_audio_pipeline_profile,
    parse_iso_timestamp,
)
from video_analyzer.jobengine.errors import BridgeError
from video_analyzer.model_settings import AUDIO_WORKFLOW_ID
from video_analyzer.url_context import AUDIO_MEDIA_EXTENSIONS


class MobileAudioMixin:
    """mobile audio 任务相关方法片（由主类继承）。"""

    def mobile_audio_job_by_attempt(
        self,
        external_attempt_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            if (
                self.is_mobile_audio_job(job)
                and self.mobile_audio_tenant_matches(job, tenant_id)
                and str(job.get("external_attempt_id") or "") == external_attempt_id
            ):
                return job
        return None

    def acknowledge_mobile_audio_job(
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
        if job.get("status") != "succeeded":
            raise BridgeError(
                HTTPStatus.CONFLICT,
                "audio job cannot be acknowledged before it succeeds",
            )
        job["consumer_acknowledged_at"] = iso_now()
        job["updated_at"] = iso_now()
        self.save_job(job)
        return {
            "acknowledged": True,
            "job_id": job_id,
            "retention_days": AUDIO_JOB_RETENTION_DAYS,
            "consumer_acknowledged_at": job["consumer_acknowledged_at"],
        }

    def cleanup_acknowledged_mobile_audio_jobs(
        self,
        now: float | None = None,
    ) -> list[str]:
        now = time.time() if now is None else now
        cutoff = now - AUDIO_JOB_RETENTION_DAYS * 86400
        deleted: list[str] = []
        for path in sorted(self.jobs_dir.glob("*/job.json")):
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            if not self.is_mobile_audio_job(job):
                continue
            tts = ((job.get("background_tasks") or {}).get("tts_summary") or {})
            if tts.get("status") in {"queued", "waiting_for_idle", "running"}:
                continue
            acknowledged = parse_iso_timestamp(job.get("consumer_acknowledged_at"))
            if acknowledged is None or acknowledged > cutoff:
                continue
            try:
                self.delete_job(job["job_id"])
            except BridgeError:
                continue
            deleted.append(job["job_id"])
        return deleted

    def mobile_audio_templates(self) -> dict[str, Any]:
        try:
            raw = AUDIO_TEMPLATE_CATALOG.read_bytes()
            templates = json.loads(raw.decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BridgeError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"audio template catalog is unavailable: {exc}",
            ) from exc
        if not isinstance(templates, list):
            raise BridgeError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "audio template catalog must be a list",
            )
        public_fields = (
            "id",
            "title",
            "title_zh",
            "first_category",
            "first_category_zh",
        )
        public_templates = []
        for item in templates:
            if not isinstance(item, dict):
                raise BridgeError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "audio template catalog entries must be objects",
                )
            public_templates.append({key: item.get(key, "") for key in public_fields})
        return {
            "pipeline_profile": AUDIO_PIPELINE_PROFILE_NX1,
            "templates": public_templates,
            "total": len(public_templates),
            "version": hashlib.sha256(raw).hexdigest(),
        }

    def is_mobile_audio_job(self, job: dict[str, Any]) -> bool:
        opts = job.get("options") or {}
        source_name = str(job.get("source_name") or "")
        return bool(
            job.get("source_type") == UPLOAD_SOURCE_TYPE
            and (
                opts.get("run_name") == "audio-summary"
                or opts.get("run_name") == "audio-transcription"
                or re.fullmatch(r"\d{14}\.mp3", source_name)
                or str(job.get("upload_suffix") or "").lower() in AUDIO_MEDIA_EXTENSIONS
            )
        )

    def is_tenant_mobile_audio_job(self, job: dict[str, Any]) -> bool:
        return self.is_mobile_audio_job(job) and bool(
            job.get("audio_pipeline") or job.get("tenant_id")
        )

    @staticmethod
    def mobile_audio_tenant_id(job: dict[str, Any]) -> str:
        return str(job.get("tenant_id") or "nx1").strip().lower()

    def mobile_audio_tenant_matches(
        self,
        job: dict[str, Any],
        tenant_id: str | None,
    ) -> bool:
        return tenant_id is None or self.mobile_audio_tenant_id(job) == tenant_id

    def mobile_audio_job(self, job: dict[str, Any], include_resources: bool = False) -> dict[str, Any]:
        public = self.public_job(job)
        prompt = public.get("prompt_template") or {}
        requested = self.mobile_prompt_template((prompt.get("requested") or {}))
        actual = self.mobile_prompt_template((prompt.get("actual") or {}))
        item = {
            "job_id": public["job_id"],
            "status": public.get("status"),
            "title": public.get("title"),
            "source_name": public.get("source_name"),
            "created_at": public.get("created_at"),
            "updated_at": public.get("updated_at"),
            "current_stage": public.get("current_stage"),
            "progress": public.get("progress"),
            "queue": self.mobile_audio_queue_info(job),
            "error": ((public.get("runner") or {}).get("error") or ""),
            "error_code": ((public.get("error_summary") or {}).get("code") or ""),
            "external_attempt_id": job.get("external_attempt_id"),
            "source_sha256": job.get("source_sha256"),
            "source_device": job.get("source_device"),
            "source_file_id": job.get("source_file_id"),
            "provided_transcript": bool(job.get("provided_transcript")),
            "source_transcription_id": job.get("source_transcription_id"),
            "source_transcript_sha256": job.get("source_transcript_sha256"),
            "profile": ((job.get("options") or {}).get("profile")),
            "workflow_id": (
                (job.get("runtime_profile_snapshot") or {}).get("workflow_id")
                or (job.get("options") or {}).get("workflow_id")
            ),
            "pipeline_kind": normalize_audio_pipeline_profile(
                job.get("audio_pipeline_kind")
                or job.get("audio_pipeline_profile")
            ),
            "pipeline_profile": normalize_audio_pipeline_profile(
                job.get("audio_pipeline_profile")
                or job.get("audio_pipeline_kind")
            ),
            "asr_provider": job.get("asr_provider"),
            "compute_route": job.get("compute_route") or "local",
            "compute_route_reason": job.get("compute_route_reason") or "",
            "consumer_acknowledged_at": job.get("consumer_acknowledged_at"),
            "summary": {"study": (public.get("summary") or {}).get("study") or {}},
            "prompt_template": {
                "requested": requested,
                "actual": actual,
            },
            "background_tasks": copy.deepcopy(job.get("background_tasks") or {}),
            "execution_routes": copy.deepcopy(job.get("execution_routes") or {}),
        }
        if include_resources:
            item["result_resources"] = public.get("result_resources") or {}
            item["result"] = self.mobile_audio_result(job)
        return item

    def mobile_audio_result(self, job: dict[str, Any]) -> dict[str, Any]:
        run_dir_value = job.get("run_dir")
        if not run_dir_value:
            return {}
        run_dir = Path(str(run_dir_value))
        pipeline_kind = normalize_audio_pipeline_profile(
            job.get("audio_pipeline_kind")
            or job.get("audio_pipeline_profile")
        )
        result_path = (
            run_dir / "transcription.json"
            if pipeline_kind == AUDIO_PIPELINE_KIND_TRANSCRIPTION
            else run_dir / "analysis.json"
        )
        if not result_path.is_file():
            return {}
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if pipeline_kind == AUDIO_PIPELINE_KIND_TRANSCRIPTION:
            return payload
        return {
            "pipeline_profile": str(
                payload.get("pipeline_profile") or AUDIO_PIPELINE_PROFILE_NX1
            ),
            "workflow_id": (
                (job.get("runtime_profile_snapshot") or {}).get("workflow_id")
                or AUDIO_WORKFLOW_ID
            ),
            "pipeline_version": payload.get("pipeline_version"),
            "audio_template_analysis": payload.get("audio_template_analysis") or {},
            "speaker_diarization": payload.get("speaker_diarization") or {},
            "speaker_count": (
                (payload.get("speaker_diarization") or {}).get("final_speaker_count")
                or (payload.get("speaker_diarization") or {}).get("detected_speaker_count")
                or (payload.get("speaker_diarization") or {}).get("original_speaker_count")
                or 0
            ),
            "asr": payload.get("asr") or {},
            "providers_run": list((payload.get("asr") or {}).get("providers_run") or []),
            "transcript": payload.get("transcript") or {},
            "provided_transcript": bool(job.get("provided_transcript")),
            "source_transcription_id": job.get("source_transcription_id"),
            "source_transcript_sha256": job.get("source_transcript_sha256"),
        }

    def mobile_audio_queue_info(self, job: dict[str, Any]) -> dict[str, Any]:
        runner = job.get("runner") or {}
        if runner.get("status") == "running":
            return {"state": "running", "position": 0}
        queued: list[dict[str, Any]] = []
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                candidate = self.load_job(path.parent.name)
            except Exception:
                continue
            candidate_runner = candidate.get("runner") or {}
            if (
                self.is_mobile_audio_job(candidate)
                and candidate_runner.get("status") == "queued"
            ):
                queued.append(candidate)
        queued.sort(
            key=lambda item: (
                item.get("created_at") or "",
                item.get("job_id") or "",
            )
        )
        for index, candidate in enumerate(queued, start=1):
            if candidate.get("job_id") == job.get("job_id"):
                return {"state": "queued", "position": index}
        return {"state": str(runner.get("status") or job.get("status") or "idle"), "position": None}

    def mobile_prompt_template(self, value: dict[str, Any]) -> dict[str, Any]:
        classification = value.get("classification") or {}
        return {
            "id": value.get("id"),
            "title": value.get("title"),
            "title_zh": value.get("title_zh"),
            "category": value.get("category"),
            "classification": {
                "method": classification.get("method"),
                "content_form": classification.get("content_form"),
                "domain": classification.get("domain"),
                "confidence": classification.get("confidence"),
                "runner_up_id": classification.get("runner_up_id"),
                "margin": classification.get("margin"),
                "warnings": list(classification.get("warnings") or []),
                "audit_path": classification.get("audit_path"),
            },
        }
