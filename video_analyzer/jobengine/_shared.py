"""video-link 状态引擎跨切片共享常量与纯工具函数。

主模块与各 mixin 片共用；主模块从这里导入并保留原名，向后兼容。
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from http import HTTPStatus

from video_analyzer.jobengine.errors import BridgeError

REPO_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_SOURCE_TYPE = "upload"

AUDIO_TEMPLATE_CATALOG = (
    REPO_ROOT
    / "video-analyzer-ui"
    / "video_analyzer_ui"
    / "static"
    / "data"
    / "audio_prompt_templates.json"
)
AUDIO_JOB_RETENTION_DAYS = max(
    1,
    int(os.environ.get("VIDEO_ANALYZER_AUDIO_RETENTION_DAYS", "7")),
)
AUDIO_PIPELINE_PROFILE_NX1 = "audio_nx1"
AUDIO_PRODUCTION_PROFILE = "audio_nx1_deepseek_flash"
AUDIO_PIPELINE_KIND_TRANSCRIPTION = "transcription"
AUDIO_PIPELINE_PROFILE_ALIASES = {
    "": AUDIO_PIPELINE_PROFILE_NX1,
    "analysis": AUDIO_PIPELINE_PROFILE_NX1,
    AUDIO_PIPELINE_PROFILE_NX1: AUDIO_PIPELINE_PROFILE_NX1,
    AUDIO_PIPELINE_KIND_TRANSCRIPTION: AUDIO_PIPELINE_KIND_TRANSCRIPTION,
}


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_iso_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError:
        return None


def normalize_audio_pipeline_profile(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    profile = AUDIO_PIPELINE_PROFILE_ALIASES.get(normalized)
    if profile:
        return profile
    allowed = ", ".join(
        (AUDIO_PIPELINE_PROFILE_NX1, AUDIO_PIPELINE_KIND_TRANSCRIPTION)
    )
    raise BridgeError(
        HTTPStatus.BAD_REQUEST,
        f"audio pipeline profile must be one of: {allowed}",
    )


ORPHANED_PROCESS_GONE_MESSAGE = (
    "server stopped while this stage was running; process is gone and artifacts are incomplete; retry to continue"
)
ORPHANED_PROCESS_REQUEUE_MESSAGE = (
    "server stopped while this stage was running; process is gone and artifacts are incomplete; queued for retry"
)
TRANSIENT_RESOURCE_REQUEUE_MESSAGE = "remote/system resource is temporarily busy; queued for retry"
TRANSIENT_API_REQUEUE_MESSAGE = "text API is temporarily unavailable; queued for one automatic retry"
YOUTUBE_FORMAT_REQUEUE_MESSAGE = "YouTube returned no downloadable formats; queued for one automatic retry"
MANUAL_RERUN_REQUEUE_MESSAGE = "queued by user to rerun with the current profile"
AUTO_RETRY_REASONS = {
    ORPHANED_PROCESS_REQUEUE_MESSAGE,
    TRANSIENT_RESOURCE_REQUEUE_MESSAGE,
    TRANSIENT_API_REQUEUE_MESSAGE,
    YOUTUBE_FORMAT_REQUEUE_MESSAGE,
    MANUAL_RERUN_REQUEUE_MESSAGE,
}
AUTO_RETRY_DELAY_SECONDS = float(os.environ.get("VIDEO_LINK_AUTO_RETRY_DELAY_SECONDS", "60"))
AUTO_RETRY_POLL_SECONDS = float(os.environ.get("VIDEO_LINK_AUTO_RETRY_POLL_SECONDS", "5"))
SCHEDULE_POLL_SECONDS = float(os.environ.get("VIDEO_LINK_SCHEDULE_POLL_SECONDS", "5"))

STAGE_ALIASES = {
    "operation": "analyze-core",
    "verify_core": "verify-core",
    "deep_v2": "deep-v2",
    "study": "study-guide",
    "study_guide": "study-guide",
    "review": "evidence-review",
    "evidence_review": "evidence-review",
    "web": "web-evidence",
    "web_evidence": "web-evidence",
    "qa": "qa-index",
    "qa_index": "qa-index",
    "export": "final-publish",
    "export_docs": "final-publish",
    "image_prompts": "image-prompts",
    "final_publish": "final-publish",
    "tts": "tts-narration",
    "tts_narration": "tts-narration",
    "audio_narration": "tts-narration",
}
STAGE_RESOURCES = {
    "probe": "prepare",
    "prepare": "prepare",
    "analyze-core": "core",
    "verify-core": "verify",
    "study-guide": "study-guide",
    "multidoc": "multidoc",
    "deep-v2": "deep-v2",
    "evidence-review": "study-guide",
    "web-evidence": "qa-index",
    "qa-index": "qa-index",
    "image-prompts": "image-prompts",
    "final-publish": "final-publish",
    "tts-narration": "tts",
}


def normalize_stage_name(stage: str) -> str:
    value = str(stage or "").strip()
    return STAGE_ALIASES.get(value, value)


def stage_resource(stage: str) -> str:
    return STAGE_RESOURCES.get(normalize_stage_name(stage), "core")


def job_stage_resource(job: dict[str, Any], stage: str) -> str:
    if (
        normalize_stage_name(stage) == "tts-narration"
        and job.get("tts_route") == "cloud_fallback"
    ):
        return "tts-cloud"
    if normalize_stage_name(stage) == "analyze-core":
        raw_pipeline_kind = (
            job.get("audio_pipeline_kind")
            or job.get("audio_pipeline_profile")
        )
        if not job.get("audio_pipeline") and not raw_pipeline_kind:
            return stage_resource(stage)
        pipeline_kind = normalize_audio_pipeline_profile(
            raw_pipeline_kind
        )
        if pipeline_kind == AUDIO_PIPELINE_KIND_TRANSCRIPTION:
            return "asr"
        if pipeline_kind == AUDIO_PIPELINE_PROFILE_NX1:
            return (
                "audio-cloud-analysis"
                if job.get("compute_route") == "cloud_fallback"
                else "audio-analysis"
            )
    return stage_resource(stage)


def parse_schedule_datetime(value: Any) -> "datetime | None":
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


BAOYU_IMAGE_GENERATION_ENABLED = os.environ.get("VIDEO_LINK_ENABLE_BAOYU_IMAGES", "").strip().lower() in {"1", "true", "yes", "on"}
TRANSIENT_RESOURCE_BUSY_PATTERNS = (
    "ray.exceptions.OutOfMemoryError",
    "Task was killed due to the node running low on memory",
    "exceeds the memory usage threshold",
    "Ray killed this worker",
)
YOUTUBE_FORMAT_UNAVAILABLE_PATTERN = "Requested format is not available"
YOUTUBE_RATE_LIMIT_PATTERNS = ("HTTP Error 429", "Too Many Requests")
MAX_YOUTUBE_FORMAT_RETRIES = 1
MAX_TRANSIENT_API_RETRIES = 1
MAX_INTERRUPTED_RETRIES = 1
RESOURCE_WAIT_SECONDS = 5.0
RESOURCE_LIMITS = {
    "prepare": 2,
    "core": 1,
    "audio-analysis": 1,
    "audio-cloud-analysis": 4,
    "asr": 1,
    "ocr": 1,
    "vl": 1,
    "verify": 3,
    "study-guide": 3,
    "multidoc": 3,
    "deep-v2": 3,
    "qa-index": 2,
    "image-prompts": 3,
    "final-publish": 3,
    "tts": 1,
    "tts-cloud": 3,
}


def is_youtube_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname == "youtu.be" or hostname.endswith(".youtube.com")


def process_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except OSError:
        return False
    return True


def iso_from_timestamp(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(value))
