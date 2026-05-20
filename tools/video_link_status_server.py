#!/usr/bin/env python3
"""Local status server for running the video-link workflow."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_DIR = REPO_ROOT / "tmp" / "video-link-status" / "jobs"
BAOYU_PROMPT_SCRIPT = Path.home() / ".codex" / "skills" / "video-link" / "scripts" / "prepare_baoyu_image_prompts.py"
ALLOWED_ANALYSIS_MODES = ("auto", "fast", "balanced", "deep", "long-talk-fast")
ALLOWED_COOKIE_BROWSERS = ("", "chrome", "none", "edge", "firefox", "chromium", "brave")
DEFAULT_COOKIE_BROWSER = "chrome"
DEFAULT_PROFILE = "deepseek_v4_pro"
DEFAULT_RUN_NAME = "operation-manual"
DEFAULT_SUBTITLE_LANGS = "zh-CN,zh-Hans,zh,en"
MODULE_ORDER = [
    "probe",
    "prepare",
    "analyze-core",
    "verify-core",
    "multidoc",
    "deep-v2",
    "image-prompts",
    "final-publish",
]
MODULE_LABELS = {
    "probe": "探测时长",
    "prepare": "下载/上下文",
    "analyze-core": "核心分析",
    "verify-core": "校验产物",
    "multidoc": "多文档分析",
    "deep-v2": "章节深度报告",
    "image-prompts": "生成配图提示词",
    "final-publish": "最终定稿/发布",
}
MODULE_SPECS = {
    "probe": {"requires": [], "produces": ["duration", "resolved_mode"]},
    "prepare": {"requires": ["resolved_mode"], "produces": ["video_path", "page_context"]},
    "analyze-core": {
        "requires": ["video_path", "page_context"],
        "produces": ["run_dir", "analysis_json", "operation_manual", "transcript", "frames", "ocr_events", "frame_analyses"],
    },
    "verify-core": {"requires": ["run_dir"], "produces": ["verified_core"]},
    "multidoc": {"requires": ["run_dir", "verified_core"], "produces": ["docs_analysis"]},
    "deep-v2": {"requires": ["run_dir", "verified_core"], "produces": ["chapter_deep_report"]},
    "image-prompts": {"requires": ["run_dir"], "produces": ["image_prompts"]},
    "final-publish": {"requires": ["run_dir", "verified_core"], "produces": ["exports"]},
}
STAGE_ORDER = MODULE_ORDER
STAGE_LABELS = MODULE_LABELS
STAGE_ALIASES = {
    "operation": "analyze-core",
    "verify_core": "verify-core",
    "deep_v2": "deep-v2",
    "export": "final-publish",
    "export_docs": "final-publish",
    "image_prompts": "image-prompts",
    "final_publish": "final-publish",
}
STAGE_RESOURCES = {
    "probe": "prepare",
    "prepare": "prepare",
    "analyze-core": "core",
    "verify-core": "verify",
    "multidoc": "multidoc",
    "deep-v2": "deep-v2",
    "image-prompts": "image-prompts",
    "final-publish": "final-publish",
}
RESOURCE_LIMITS = {
    "prepare": 2,
    "core": 1,
    "verify": 2,
    "multidoc": 1,
    "deep-v2": 1,
    "image-prompts": 1,
    "final-publish": 1,
}
EXPECTED_FINAL_EXPORTS = (
    "operation_manual.pdf",
    "knowledge_notes_v2.pdf",
    "deep_report_v2.pdf",
    "manual_evidence.pdf",
)
ORPHANED_PROCESS_GONE_MESSAGE = (
    "server stopped while this stage was running; process is gone and artifacts are incomplete; retry to continue"
)
ORPHANED_PROCESS_REQUEUE_MESSAGE = (
    "server stopped while this stage was running; process is gone and artifacts are incomplete; queued for retry"
)
CORE_PROGRESS_STEPS = [
    ("ray", "Ray 集群准备", (r"\[jetson-ray\]", r"Ray runtime started")),
    ("context", "页面/评论/素材准备", (r"Extracting cookies", r"\[youtube\]", r"Writing video metadata")),
    ("audio", "音频提取", (r"Extracting audio from video",)),
    ("asr", "ASR 转写", (r"Transcribing audio",)),
    ("asr_done", "ASR 完成", (r"ASR succeeded", r"Using existing transcript file")),
    ("frames", "候选帧抽取", (r"Extracting frames from video", r"Jetson video cache", r"frame worker")),
    ("frames_done", "候选帧就绪", (r"Extracted \d+ screen keyframes",)),
    ("ocr", "OCR 执行", (r"Running OCR",)),
    ("ocr_ready", "OCR 服务/请求", (r"DotsMOCR endpoint not ready", r"DotsMOCR endpoint ready", r"OpenAI-compatible vision OCR")),
    ("vl", "VL 帧选择/分析", (r"Selecting and analyzing VL frames",)),
    ("manual", "操作手册生成", (r"Generating operation manual",)),
    ("write", "结果写出", (r"Operation manual saved", r"Analysis complete", r"\[done\] run_dir:")),
]
CORE_PROGRESS_WEIGHTS = {
    "ray": 3,
    "context": 7,
    "audio": 5,
    "asr": 15,
    "asr_done": 5,
    "frames": 10,
    "frames_done": 5,
    "ocr": 5,
    "ocr_ready": 20,
    "vl": 15,
    "manual": 7,
    "write": 3,
}
STAGE_PROGRESS_STEPS = {
    "probe": [
        ("probe", "探测视频信息", (r"probe stage started", r"duration", r"video duration")),
        ("resolve", "选择分析模式", (r"resolved mode:",)),
    ],
    "prepare": [
        ("cookies", "读取浏览器 Cookie", (r"Extracting cookies from", r"Extracted \d+ cookies")),
        ("metadata", "读取页面元数据", (r"\[youtube\].*Extracting URL", r"Downloading webpage", r"Downloading .*API JSON")),
        ("js", "处理 YouTube JS", (r"Solving JS challenges", r"Downloading player")),
        ("download", "下载视频/音频", (r"^\[download\]\s+\d", r"Destination:")),
        ("merge", "合并媒体文件", (r"\[Merger\]", r"Deleting original file")),
        ("context", "写出页面上下文", (r"\[download\] video:", r"\[download\] description:", r"\[download\] context:")),
        ("transcript", "准备字幕 Transcript", (r"subtitle transcript",)),
    ],
    "verify-core": [
        ("check", "检查核心产物", (r"verifying core artifacts", r"analysis\.json", r"operation_manual\.md", r"manual_evidence\.md")),
        ("complete", "核心产物可用", (r"core artifacts verified", r"missing core artifact\(s\):")),
    ],
    "multidoc": [
        ("load", "读取核心手册/证据", (r"operation_manual\.md", r"manual_evidence\.md", r"docs_analysis")),
        ("analyze", "生成多文档分析", (r"run_multidoc_analysis", r"knowledge_notes", r"deep_report")),
        ("write", "写出多文档产物", (r"docs_analysis", r"\[summary\]", r"saved", r"written")),
    ],
    "deep-v2": [
        ("load", "读取章节证据", (r"load", r"chapters", r"chapter_assets")),
        ("chapters", "逐章深度报告", (r"\[run\] chapter", r"\[skip\] chapter", r"chapter concurrency")),
        ("synthesis", "综合/格式化", (r"\[run\] final synthesis", r"\[run\] markdown format", r"\[format\] block")),
        ("review", "校验深度报告", (r"deep_report_review", r"validate", r"review")),
        ("write", "写出 deep_report_v2", (r"deep_report_v2\.md", r"deep_report_v2\.pre_format\.md")),
    ],
    "image-prompts": [
        ("load", "读取文档内容", (r"operation_manual", r"knowledge_notes", r"deep_report", r"manual_evidence")),
        ("prompt", "生成配图提示词", (r"prompt", r"baoyu", r"cover", r"image")),
        ("write", "写出提示词文件", (r"prompts", r"\.md", r"saved", r"written")),
    ],
    "final-publish": [
        ("images", "生成/复用最终图片", (r"\[images\]", r"augment_video_docs_images")),
        ("docs", "补齐最终文档", (r"\[docs\]", r"multidoc", r"deep-v2")),
        ("augment", "插入配图", (r"augment", r"image-augmented", r"baoyu_images")),
        ("export", "导出 PDF/长图", (r"\[pdf\]", r"export_video_docs", r"\[long-png\]")),
        ("verify", "校验发布产物", (r"\[verify\]", r"pdf=", r"long_png=")),
        ("summary", "写出发布摘要", (r"\[summary\]", r"final_publish_summary\.json")),
        ("send", "发送/跳过发送", (r"\[send\]", r"skipped")),
    ],
}


class BridgeError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class VideoLinkStatusServer:
    def __init__(self, jobs_dir: Path = DEFAULT_JOBS_DIR, repo_root: Path = REPO_ROOT, auto_resume: bool = False):
        self.jobs_dir = jobs_dir
        self.repo_root = repo_root
        self.runner_lock = threading.Lock()
        self.active_runners: dict[str, threading.Thread] = {}
        self.resource_locks = {name: threading.BoundedSemaphore(limit) for name, limit in RESOURCE_LIMITS.items()}
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        if auto_resume:
            self.recover_interrupted_jobs(auto_start=True)

    def options(self) -> dict[str, Any]:
        profiles = runtime_profile_names()
        default_profile = DEFAULT_PROFILE if DEFAULT_PROFILE in profiles else active_runtime_profile(profiles)
        return {
            "defaults": {
                "analysis_mode": "auto",
                "profile": default_profile,
                "run_name": DEFAULT_RUN_NAME,
                "cookies_from_browser": DEFAULT_COOKIE_BROWSER,
                "skip_images": False,
                "keep_existing": True,
                "include_subtitles": True,
                "prefer_subtitle_transcript": False,
                "include_comments": True,
                "max_comments": 30,
                "subtitle_langs": DEFAULT_SUBTITLE_LANGS,
                "refresh_context": False,
            },
            "choices": {
                "analysis_modes": list(ALLOWED_ANALYSIS_MODES),
                "profiles": profiles,
                "cookie_browsers": [item for item in ALLOWED_COOKIE_BROWSERS if item],
            },
        }

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        video_url = str(payload.get("video_url") or payload.get("videoUrl") or "").strip()
        if not video_url.startswith(("http://", "https://")):
            raise BridgeError(HTTPStatus.BAD_REQUEST, "video_url must be an http(s) URL")

        analysis_mode = str(payload.get("analysis_mode") or payload.get("analysisMode") or "auto").strip() or "auto"
        if analysis_mode not in ALLOWED_ANALYSIS_MODES:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"analysis_mode must be one of {sorted(ALLOWED_ANALYSIS_MODES)}")

        cookie_browser = normalize_cookie_browser(payload.get("cookies_from_browser") or payload.get("cookiesFromBrowser"))
        if cookie_browser not in ALLOWED_COOKIE_BROWSERS:
            raise BridgeError(
                HTTPStatus.BAD_REQUEST,
                f"cookies_from_browser must be one of {sorted(ALLOWED_COOKIE_BROWSERS)} or none",
            )

        defaults = self.options()["defaults"]
        run_name = sanitize_run_name(str(payload.get("run_name") or payload.get("runName") or defaults["run_name"]))
        profile = str(payload.get("profile") or defaults["profile"]).strip() or defaults["profile"]
        profiles = runtime_profile_names()
        if profiles and profile not in profiles:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"profile must be one of {profiles}")
        skip_images = parse_bool(normalize_optional_template(payload.get("skip_images") if "skip_images" in payload else payload.get("skipImages", False)))
        auto_start = parse_bool(normalize_optional_template(payload.get("auto_start") if "auto_start" in payload else payload.get("autoStart", False)))
        keep_existing = parse_bool_option(payload, "keep_existing", "keepExisting", defaults["keep_existing"])
        include_subtitles = parse_bool_option(payload, "include_subtitles", "includeSubtitles", defaults["include_subtitles"])
        prefer_subtitle_transcript = parse_bool_option(
            payload,
            "prefer_subtitle_transcript",
            "preferSubtitleTranscript",
            defaults["prefer_subtitle_transcript"],
        )
        include_comments = parse_bool_option(payload, "include_comments", "includeComments", defaults["include_comments"])
        refresh_context = parse_bool_option(payload, "refresh_context", "refreshContext", defaults["refresh_context"])
        max_comments = parse_int_option(payload.get("max_comments") if "max_comments" in payload else payload.get("maxComments"), defaults["max_comments"])
        subtitle_langs = str(payload.get("subtitle_langs") or payload.get("subtitleLangs") or defaults["subtitle_langs"]).strip()

        job_id = uuid.uuid4().hex
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        job = {
            "job_id": job_id,
            "status": "created",
            "created_at": iso_now(),
            "updated_at": iso_now(),
            "video_url": video_url,
            "options": {
                "analysis_mode": analysis_mode,
                "profile": profile,
                "run_name": run_name,
                "cookies_from_browser": cookie_browser,
                "skip_images": skip_images,
                "keep_existing": keep_existing,
                "include_subtitles": include_subtitles,
                "prefer_subtitle_transcript": prefer_subtitle_transcript,
                "include_comments": include_comments,
                "max_comments": max_comments,
                "subtitle_langs": subtitle_langs,
                "refresh_context": refresh_context,
            },
            "resolved_mode": None,
            "run_dir": None,
            "artifacts": {},
            "stages": {},
            "modules": {},
            "runner": {"status": "idle", "current_stage": None, "error": None},
        }
        self.save_job(job)
        if auto_start:
            return self.start_run(job_id)
        return self.public_job(job)

    def create_jobs(self, payload: dict[str, Any]) -> dict[str, Any]:
        urls = extract_batch_urls(payload)
        if not urls:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "video_urls must include at least one http(s) URL")

        auto_start = parse_bool(normalize_optional_template(payload.get("auto_start") if "auto_start" in payload else payload.get("autoStart", True)))
        base_run_name = sanitize_run_name(str(payload.get("run_name") or payload.get("runName") or self.options()["defaults"]["run_name"]))
        created: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        seen: dict[str, int] = {}

        for index, url in enumerate(urls, start=1):
            if not url.startswith(("http://", "https://")):
                errors.append({"index": index, "video_url": url, "error": "video_url must be an http(s) URL"})
                continue
            seen[url] = seen.get(url, 0) + 1
            batch_payload = dict(payload)
            batch_payload["video_url"] = url
            batch_payload["run_name"] = f"{base_run_name}-{index:03d}"
            batch_payload["auto_start"] = False
            try:
                job = self.create_job(batch_payload)
                created.append(job)
            except BridgeError as exc:
                errors.append({"index": index, "video_url": url, "error": exc.message})

        if auto_start:
            started = []
            for job in created:
                try:
                    started.append(self.start_run(job["job_id"]))
                except BridgeError as exc:
                    errors.append({"index": None, "video_url": job.get("video_url"), "job_id": job.get("job_id"), "error": exc.message})
            created = started

        return {
            "jobs": created,
            "errors": errors,
            "created": len(created),
            "failed": len(errors),
            "total": len(urls),
            "duplicates": {url: count for url, count in seen.items() if count > 1},
        }

    def list_jobs(self, limit: int = 50) -> dict[str, Any]:
        jobs = []
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                jobs.append(self.public_job(self.load_job(path.parent.name)))
            except Exception:
                continue
        jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {
            "jobs": jobs[: max(1, min(limit, 200))],
            "total": len(jobs),
            "summary": self.jobs_summary(jobs),
            "resources": self.resource_summary(jobs),
        }

    def jobs_summary(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {status: 0 for status in ("created", "running", "queued", "succeeded", "failed")}
        progress_values = []
        for job in jobs:
            status = job.get("status") or "created"
            counts[status] = counts.get(status, 0) + 1
            progress = job.get("progress") or {}
            if "percent" in progress:
                progress_values.append(progress.get("percent") or 0)
        return {
            "total": len(jobs),
            "counts": counts,
            "average_progress": int(round(sum(progress_values) / len(progress_values))) if progress_values else 0,
        }

    def resource_summary(self, jobs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if jobs is None:
            jobs = []
            for path in self.jobs_dir.glob("*/job.json"):
                try:
                    jobs.append(self.public_job(self.load_job(path.parent.name)))
                except Exception:
                    continue

        resources = {
            name: {"limit": limit, "running": [], "queued": []}
            for name, limit in RESOURCE_LIMITS.items()
        }
        for job in jobs:
            runner = job.get("runner") or {}
            status = runner.get("status")
            stage = runner.get("current_stage") or job.get("current_stage")
            if status not in {"running", "queued"} or not stage:
                continue
            resource = runner.get("queued_for") or stage_resource(stage)
            entry = {
                "job_id": job.get("job_id"),
                "video_url": job.get("video_url"),
                "stage": normalize_stage_name(stage),
                "stage_label": STAGE_LABELS.get(normalize_stage_name(stage), stage),
                "updated_at": job.get("updated_at"),
                "progress_percent": (job.get("progress") or {}).get("percent"),
            }
            resources.setdefault(resource, {"limit": RESOURCE_LIMITS.get(resource, 1), "running": [], "queued": []})
            if status == "queued":
                entry["position"] = (job.get("queue") or {}).get("position")
                resources[resource]["queued"].append(entry)
            else:
                process = job.get("process")
                if process:
                    entry["pid"] = process.get("pid")
                resources[resource]["running"].append(entry)

        for info in resources.values():
            info["running"].sort(key=lambda item: item.get("updated_at") or "")
            info["queued"].sort(key=lambda item: (item.get("position") or 999999, item.get("updated_at") or ""))
            info["running_count"] = len(info["running"])
            info["queued_count"] = len(info["queued"])
            info["available"] = max(0, int(info.get("limit") or 0) - len(info["running"]))
        return resources

    def start_run(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        with self.runner_lock:
            active = self.active_runners.get(job_id)
            if active and active.is_alive():
                return self.public_job(job)
            self.active_runners.pop(job_id, None)

            now = iso_now()
            job["status"] = "running"
            job["updated_at"] = now
            job["runner"] = {
                "status": "running",
                "started_at": now,
                "updated_at": now,
                "finished_at": None,
                "current_stage": self.next_stage(job),
                "error": None,
                "server_pid": os.getpid(),
            }
            self.save_job(job)
            thread = threading.Thread(target=self._run_remaining_stages, args=(job_id,), daemon=True)
            self.active_runners[job_id] = thread
        thread.start()
        return self.public_job(self.load_job(job_id))

    def recover_interrupted_jobs(self, auto_start: bool = False) -> None:
        for path in sorted(self.jobs_dir.glob("*/job.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            runner = raw.get("runner") or {}
            stages = raw.get("stages") or {}
            interrupted = runner.get("status") in {"running", "queued"} or any(
                info.get("status") in {"running", "queued"} for info in stages.values()
            )
            if not interrupted:
                continue
            try:
                recovered = self.load_job(raw["job_id"])
            except BridgeError:
                continue
            recovered_runner = recovered.get("runner") or {}
            if auto_start and recovered.get("status") == "queued" and recovered_runner.get("status") == "queued":
                self.start_run(raw["job_id"])

    def _run_remaining_stages(self, job_id: str) -> None:
        try:
            while True:
                job = self.load_job(job_id)
                stage = self.next_stage(job)
                if not stage:
                    job["status"] = "succeeded"
                    self.update_runner(job, "succeeded", current_stage=None, finished=True)
                    return
                self.update_runner(job, "running", current_stage=stage)
                self.run_stage(job_id, stage)
        except BridgeError as exc:
            job = self.load_job(job_id)
            job["status"] = "failed"
            self.update_runner(job, "failed", error=exc.message, finished=True)
        except Exception as exc:
            job = self.load_job(job_id)
            job["status"] = "failed"
            self.update_runner(job, "failed", error=str(exc), finished=True)
        finally:
            with self.runner_lock:
                self.active_runners.pop(job_id, None)

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
        self.save_job(job)

    def run_stage(self, job_id: str, stage: str) -> dict[str, Any]:
        stage = normalize_stage_name(stage)
        if stage not in STAGE_ORDER:
            raise BridgeError(HTTPStatus.NOT_FOUND, f"unknown stage: {stage}")
        job = self.load_job(job_id)
        self.ensure_dependencies(job, stage)
        current_status = job.get("stages", {}).get(stage, {}).get("status")
        if current_status in {"succeeded", "skipped"}:
            return self.public_job(job)
        if stage == "image-prompts" and job["options"].get("skip_images"):
            return self.mark_stage_skipped(job, stage, "skip_images is true")

        resource = stage_resource(stage)
        self.mark_stage_queued(job, stage, resource)
        lock = self.resource_locks[resource]
        lock.acquire()
        try:
            return self._run_stage_locked(job_id, stage)
        finally:
            lock.release()

    def _run_stage_locked(self, job_id: str, stage: str) -> dict[str, Any]:
        stage = normalize_stage_name(stage)
        job = self.load_job(job_id)
        current_status = job.get("stages", {}).get(stage, {}).get("status")
        if current_status in {"succeeded", "skipped"}:
            return self.public_job(job)

        start = time.time()
        stage_info = {
            "status": "running",
            "started_at": iso_now(),
            "finished_at": None,
            "exit_code": None,
            "log_path": str(self.stage_log_path(job_id, stage)),
            "artifacts": {},
            "queued_for": stage_resource(stage),
        }
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
                result = self.stage_verify_core(job)
            elif stage == "multidoc":
                result = self.run_command_stage(job, stage, self.multidoc_command(job), stage_info["log_path"], stage_info)
            elif stage == "deep-v2":
                result = self.run_command_stage(job, stage, self.deep_v2_command(job), stage_info["log_path"], stage_info)
            elif stage == "image-prompts":
                result = self.run_command_stage(job, stage, self.image_prompts_command(job), stage_info["log_path"], stage_info)
            else:
                result = self.run_command_stage(job, stage, self.final_publish_command(job), stage_info["log_path"], stage_info)
            stage_info.update(result)
            stage_info.pop("process", None)
            self.update_job_artifacts(job, stage, result.get("artifacts", {}))
            stage_info["status"] = "succeeded"
            stage_info["exit_code"] = 0
            stage_info["duration_seconds"] = round(time.time() - start, 3)
            stage_info["finished_at"] = iso_now()
            job["status"] = "succeeded" if stage == STAGE_ORDER[-1] else "running"
        except Exception as exc:
            stage_info["status"] = "failed"
            stage_info["exit_code"] = getattr(exc, "returncode", 1)
            stage_info["duration_seconds"] = round(time.time() - start, 3)
            stage_info["finished_at"] = iso_now()
            stage_info["error"] = str(exc)
            stage_info.pop("process", None)
            job["status"] = "failed"
        job["updated_at"] = iso_now()
        job["stages"][stage] = stage_info
        job["summary"] = self.collect_summary(job)
        self.save_job(job)
        if stage_info["status"] == "failed":
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, f"{stage} failed: {stage_info.get('error')}")
        return self.public_job(job)

    def mark_stage_queued(self, job: dict[str, Any], stage: str, resource: str) -> None:
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
        stage_info.pop("finished_at", None)
        stage_info.pop("exit_code", None)
        job.setdefault("stages", {})[stage] = stage_info
        job["status"] = "queued"
        runner = dict(job.get("runner") or {})
        runner["status"] = "queued"
        runner["current_stage"] = stage
        runner["queued_for"] = resource
        runner["updated_at"] = now
        runner["server_pid"] = os.getpid()
        runner["error"] = None
        if "started_at" not in runner:
            runner["started_at"] = now
        job["runner"] = runner
        job["updated_at"] = now
        self.save_job(job)

    def stage_probe(self, job: dict[str, Any]) -> dict[str, Any]:
        duration = probe_duration_seconds(job["video_url"])
        requested_mode = job["options"]["analysis_mode"]
        if requested_mode == "auto":
            resolved_mode = "long-talk-fast" if duration is not None and duration >= 2700 else "balanced"
        else:
            resolved_mode = requested_mode
        job["resolved_mode"] = resolved_mode
        return {"artifacts": {"duration_seconds": duration, "resolved_mode": resolved_mode}}

    def stage_operation(self, job: dict[str, Any], log_path: str) -> dict[str, Any]:
        return self.stage_analyze_core(job, log_path)

    def stage_prepare(self, job: dict[str, Any], log_path: str, stage_info: dict[str, Any] | None = None) -> dict[str, Any]:
        if not job.get("resolved_mode"):
            self.stage_probe(job)
        command = self.prepare_command(job)
        result = self.run_command(command, log_path, on_start=self.record_stage_process(job, "prepare", stage_info))
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        video_path = parse_prefixed_path(text, "[download] video:")
        page_context = parse_prefixed_path(text, "[download] context:")
        if not video_path or not page_context:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, "prepare stage did not produce video/context paths")
        job["video_path"] = str(self.resolve_output_path(video_path))
        job["page_context_path"] = str(self.resolve_output_path(page_context))
        job["video_dir"] = str(Path(job["video_path"]).parent)
        artifacts = {
            "video_path": job["video_path"],
            "page_context": job["page_context_path"],
            "video_dir": job["video_dir"],
            "command": command,
        }
        return {"artifacts": artifacts, "stdout_tail": result["stdout_tail"]}

    def stage_analyze_core(self, job: dict[str, Any], log_path: str, stage_info: dict[str, Any] | None = None) -> dict[str, Any]:
        if not job.get("resolved_mode"):
            self.stage_probe(job)
        command = self.operation_command(job)
        result = self.run_command(command, log_path, on_start=self.record_stage_process(job, "analyze-core", stage_info))
        run_dir = parse_run_dir(Path(log_path).read_text(encoding="utf-8", errors="replace"))
        if not run_dir:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, "operation stage did not print a run directory")
        job["run_dir"] = str(self.resolve_output_path(run_dir))
        artifacts = {"run_dir": job["run_dir"], "command": command, **self.collect_core_artifacts(Path(job["run_dir"]))}
        return {"artifacts": artifacts, "stdout_tail": result["stdout_tail"]}

    def stage_verify_core(self, job: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.require_run_dir(job)
        required = ["analysis.json", "operation_manual.md", "manual_evidence.md"]
        missing = [name for name in required if not (run_dir / name).exists()]
        if missing:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, f"missing core artifact(s): {', '.join(missing)}")
        return {"artifacts": {"required": required, "missing": []}}

    def run_command_stage(
        self,
        job: dict[str, Any],
        stage: str,
        command: list[str],
        log_path: str,
        stage_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.run_command(command, log_path, on_start=self.record_stage_process(job, stage, stage_info))
        return {
            "artifacts": {
                "stage": stage,
                "command": command,
                **self.collect_summary(job),
            },
            "stdout_tail": result["stdout_tail"],
        }

    def run_command(
        self,
        command: list[str],
        log_path: str,
        on_start: Any | None = None,
    ) -> dict[str, Any]:
        env = operation_env()
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            if on_start:
                on_start(process)
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        if return_code != 0:
            error = subprocess.CalledProcessError(return_code, command)
            error.output = text
            raise error
        return {"stdout_tail": tail_lines(text)}

    def record_stage_process(self, job: dict[str, Any], stage: str, stage_info: dict[str, Any] | None = None):
        def _record(process: subprocess.Popen) -> None:
            process_info = {
                "pid": process.pid,
                "command": list(process.args) if isinstance(process.args, (list, tuple)) else process.args,
                "started_at": iso_now(),
                "alive": True,
            }
            if stage_info is not None:
                stage_info["process"] = process_info
            try:
                current = self.load_job(job["job_id"])
            except BridgeError:
                return
            current_stage = dict((current.get("stages") or {}).get(stage) or {})
            current_stage["process"] = process_info
            current_stage["status"] = "running"
            current.setdefault("stages", {})[stage] = current_stage
            runner = dict(current.get("runner") or {})
            runner["status"] = "running"
            runner["current_stage"] = stage
            runner["updated_at"] = process_info["started_at"]
            runner["server_pid"] = os.getpid()
            current["runner"] = runner
            current["status"] = "running"
            current["updated_at"] = process_info["started_at"]
            self.save_job(current)

        return _record

    def operation_command(self, job: dict[str, Any]) -> list[str]:
        opts = job["options"]
        resolved_mode = job.get("resolved_mode") or opts["analysis_mode"]
        if resolved_mode == "long-talk-fast":
            command = [
                "tools/run_long_talk_fast_from_url.sh",
                job["video_url"],
                "--profile",
                opts["profile"],
                "--run-name",
                opts["run_name"],
            ]
        else:
            command = [
                "tools/run_operation_manual_from_url.sh",
                job["video_url"],
                "--profile",
                opts["profile"],
                "--run-name",
                opts["run_name"],
                "--pipeline-mode",
                pipeline_mode_for(resolved_mode),
            ]
        self.append_url_options(command, opts)
        return command

    def prepare_command(self, job: dict[str, Any]) -> list[str]:
        opts = job["options"]
        resolved_mode = job.get("resolved_mode") or opts["analysis_mode"]
        command = [
            "tools/run_operation_manual_from_url.sh",
            job["video_url"],
            "--profile",
            opts["profile"],
            "--run-name",
            opts["run_name"],
            "--pipeline-mode",
            pipeline_mode_for(resolved_mode),
            "--download-only",
        ]
        self.append_url_options(command, opts)
        return command

    def append_url_options(self, command: list[str], opts: dict[str, Any]) -> None:
        if opts.get("keep_existing"):
            command.append("--keep-existing")
        if opts.get("cookies_from_browser"):
            command.extend(["--cookies-from-browser", opts["cookies_from_browser"]])
        command.append("--include-subtitles" if opts.get("include_subtitles") else "--no-include-subtitles")
        command.append("--include-comments" if opts.get("include_comments") else "--no-include-comments")
        if opts.get("prefer_subtitle_transcript"):
            command.append("--prefer-subtitle-transcript")
        if opts.get("refresh_context"):
            command.append("--refresh-context")
        if opts.get("max_comments") is not None:
            command.extend(["--max-comments", str(opts["max_comments"])])
        if opts.get("subtitle_langs"):
            command.extend(["--subtitle-langs", opts["subtitle_langs"]])

    def multidoc_command(self, job: dict[str, Any]) -> list[str]:
        return ["tools/run_multidoc_analysis.sh", str(self.require_run_dir(job)), "--profile", job["options"]["profile"]]

    def deep_v2_command(self, job: dict[str, Any]) -> list[str]:
        return [
            sys.executable,
            "tools/generate_chapter_deep_report.py",
            str(self.require_run_dir(job)),
            "--profile",
            job["options"]["profile"],
            "--deep-v2",
            "--no-final-synthesis",
            "--no-format-markdown-final",
        ]

    def export_command(self, job: dict[str, Any]) -> list[str]:
        return ["tools/export_video_docs.sh", str(self.require_run_dir(job)), "--final-only", "--jobs", "3"]

    def final_publish_command(self, job: dict[str, Any]) -> list[str]:
        command = [
            "tools/run_video_doc_final_publish.sh",
            str(self.require_run_dir(job)),
            "--profile",
            job["options"].get("profile") or DEFAULT_PROFILE,
            "--jobs",
            "3",
            "--finalize-only",
            "--skip-send",
        ]
        if job["options"].get("skip_images"):
            command.append("--skip-images")
        return command

    def image_prompts_command(self, job: dict[str, Any]) -> list[str]:
        if not BAOYU_PROMPT_SCRIPT.exists():
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, f"missing Baoyu prompt script: {BAOYU_PROMPT_SCRIPT}")
        return [sys.executable, str(BAOYU_PROMPT_SCRIPT), str(self.require_run_dir(job))]

    def mark_stage_skipped(self, job: dict[str, Any], stage: str, reason: str) -> dict[str, Any]:
        stage = normalize_stage_name(stage)
        job["stages"][stage] = {
            "status": "skipped",
            "reason": reason,
            "started_at": iso_now(),
            "finished_at": iso_now(),
            "exit_code": 0,
            "log_path": None,
            "artifacts": self.collect_summary(job),
        }
        job["status"] = "succeeded" if stage == STAGE_ORDER[-1] else "running"
        job["updated_at"] = iso_now()
        job["summary"] = self.collect_summary(job)
        self.save_job(job)
        return self.public_job(job)

    def update_job_artifacts(self, job: dict[str, Any], stage: str, artifacts: dict[str, Any]) -> None:
        artifact_store = dict(job.get("artifacts") or {})
        produced = MODULE_SPECS.get(stage, {}).get("produces", [])
        for name in produced:
            if name in artifacts:
                artifact_store[name] = {
                    "value": artifacts[name],
                    "module": stage,
                    "updated_at": iso_now(),
                }
        if stage == "probe":
            artifact_store["resolved_mode"] = {"value": job.get("resolved_mode"), "module": stage, "updated_at": iso_now()}
        job["artifacts"] = artifact_store

    def collect_core_artifacts(self, run_dir: Path) -> dict[str, Any]:
        analysis_path = run_dir / "analysis.json"
        manual_path = run_dir / "operation_manual.md"
        quality_failed_path = run_dir / "operation_manual.quality_failed.md"
        artifacts: dict[str, Any] = {
            "analysis_json": str(analysis_path),
            "operation_manual": str(manual_path if manual_path.exists() else quality_failed_path),
        }
        orin_dir = run_dir / "orin"
        candidate_paths = {
            "transcript": [run_dir / "transcript.md", orin_dir / "transcript.md"],
            "ocr_events": [orin_dir / "ocr_events.json"],
            "frame_analyses": [orin_dir / "frame_analyses.json"],
            "frames": [run_dir / "frames"],
        }
        for name, paths in candidate_paths.items():
            path = next((candidate for candidate in paths if candidate.exists()), None)
            if path:
                artifacts[name] = str(path)
        if analysis_path.exists():
            try:
                payload = json.loads(analysis_path.read_text(encoding="utf-8"))
                metadata = payload.get("metadata") or {}
                artifacts["frames"] = artifacts.get("frames") or metadata.get("frame_extraction", {}).get("frame_manifest")
                artifacts["transcript"] = artifacts.get("transcript") or metadata.get("transcript_markdown")
                artifacts["core_counts"] = {
                    "frames_extracted": metadata.get("frames_extracted"),
                    "ocr_events": len(payload.get("ocr_events") or []),
                    "frame_analyses": len(payload.get("frame_analyses") or []),
                    "timings": metadata.get("timings") or {},
                }
            except Exception:
                pass
        return artifacts

    def resolve_output_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def ensure_dependencies(self, job: dict[str, Any], stage: str) -> None:
        stage = normalize_stage_name(stage)
        if stage == "probe":
            return
        stage_index = STAGE_ORDER.index(stage)
        for previous in STAGE_ORDER[:stage_index]:
            previous_status = job["stages"].get(previous, {}).get("status")
            if previous_status not in {"succeeded", "skipped"}:
                raise BridgeError(HTTPStatus.CONFLICT, f"stage {previous} must succeed before {stage}")

    def require_run_dir(self, job: dict[str, Any]) -> Path:
        run_dir = job.get("run_dir")
        if not run_dir:
            raise BridgeError(HTTPStatus.CONFLICT, "run_dir is not available yet")
        path = Path(run_dir).expanduser().resolve()
        if not path.is_dir():
            raise BridgeError(HTTPStatus.CONFLICT, f"run_dir does not exist: {path}")
        return path

    def collect_summary(self, job: dict[str, Any]) -> dict[str, Any]:
        run_dir_value = job.get("run_dir")
        if not run_dir_value:
            return {}
        run_dir = Path(run_dir_value)
        return {
            "run_dir": str(run_dir),
            "markdown_files": sorted(str(path.relative_to(run_dir)) for path in run_dir.glob("**/*.md") if path.is_file()),
            "export_files": sorted(str(path.relative_to(run_dir)) for path in (run_dir / "exports").glob("*") if path.is_file())
            if (run_dir / "exports").is_dir()
            else [],
            "prompt_files": sorted(str(path.relative_to(run_dir)) for path in (run_dir / "baoyu_images" / "prompts").glob("*") if path.is_file())
            if (run_dir / "baoyu_images" / "prompts").is_dir()
            else [],
            "final_images": sorted(str(path.relative_to(run_dir)) for path in (run_dir / "baoyu_images" / "final").glob("*") if path.is_file())
            if (run_dir / "baoyu_images" / "final").is_dir()
            else [],
        }

    def public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        public = dict(job)
        public["stages"] = dict(job.get("stages") or {})
        public["summary"] = self.collect_summary(job)
        public["stage_order"] = STAGE_ORDER
        public["progress"] = self.progress(job)
        public["current_stage"] = self.current_stage(job)
        public["next_stage"] = self.next_stage(job)
        public["error_summary"] = self.error_summary(job)
        public["dashboard_url"] = self.dashboard_url(job["job_id"])
        public["queue"] = self.queue_info(public)
        public["core_progress"] = self.core_progress(public)
        public["stage_progress"] = self.stage_progress(public)
        queued_stage = public["queue"].get("stage")
        if queued_stage and queued_stage in public["stages"]:
            public["stages"][queued_stage] = dict(public["stages"][queued_stage])
            public["stages"][queued_stage]["queue_position"] = public["queue"].get("position")
        current_stage = public.get("current_stage")
        current_info = public["stages"].get(current_stage or "", {})
        public["process"] = self.public_process_info(current_info.get("process"))
        return public

    def public_process_info(self, process_info: dict[str, Any] | None) -> dict[str, Any] | None:
        if not process_info:
            return None
        info = dict(process_info)
        pid = info.get("pid")
        info["alive"] = process_alive(pid) if pid else False
        return info

    def export_outputs_complete(self, job: dict[str, Any]) -> bool:
        run_dir_value = job.get("run_dir")
        if not run_dir_value:
            return False
        export_dir = Path(run_dir_value) / "exports"
        return all((export_dir / name).is_file() and (export_dir / name).stat().st_size > 0 for name in EXPECTED_FINAL_EXPORTS)

    def queue_info(self, job: dict[str, Any]) -> dict[str, Any]:
        runner = job.get("runner") or {}
        stage = runner.get("current_stage") or self.current_stage(job)
        if runner.get("status") != "queued" or not stage:
            return {}
        resource = runner.get("queued_for") or stage_resource(stage)
        queued = []
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            candidate_runner = candidate.get("runner") or {}
            if candidate_runner.get("status") == "queued" and (candidate_runner.get("queued_for") or stage_resource(candidate_runner.get("current_stage") or "")) == resource:
                queued.append(candidate)
        queued.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "")
        position = next((index + 1 for index, item in enumerate(queued) if item.get("job_id") == job.get("job_id")), None)
        return {"stage": stage, "resource": resource, "position": position, "size": len(queued)}

    def core_progress(self, job: dict[str, Any]) -> dict[str, Any] | None:
        stage_info = (job.get("stages") or {}).get("analyze-core") or {}
        if not stage_info and not self.stage_log_path(job["job_id"], "analyze-core").exists():
            return None
        log_path = Path(stage_info.get("log_path") or self.stage_log_path(job["job_id"], "analyze-core"))
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        progress = parse_core_progress(text, stage_info.get("status") or "pending")
        return self.annotate_stage_progress(job, "analyze-core", progress, stage_info)

    def stage_progress(self, job: dict[str, Any]) -> dict[str, Any] | None:
        stage = self.current_stage(job) or (self.error_summary(job) or {}).get("stage") or self.next_stage(job)
        stage = normalize_stage_name(stage or "")
        if stage == "analyze-core":
            return self.core_progress(job)
        if stage not in STAGE_PROGRESS_STEPS:
            return None
        stage_info = (job.get("stages") or {}).get(stage) or {}
        log_path = Path(stage_info.get("log_path") or self.stage_log_path(job["job_id"], stage))
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if not text:
            text = stage_progress_text(stage, job, stage_info)
        progress = parse_stage_progress(stage, text, stage_info.get("status") or "pending")
        return self.annotate_stage_progress(job, stage, progress, stage_info)

    def annotate_stage_progress(
        self,
        job: dict[str, Any],
        stage: str,
        progress: dict[str, Any],
        stage_info: dict[str, Any],
    ) -> dict[str, Any]:
        progress = dict(progress)
        steps = list(progress.get("steps") or [])
        status = stage_info.get("status") or progress.get("status") or "pending"
        live = self.stage_is_live(job, stage, stage_info)
        current_step = progress.get("current_step") if live else None
        current_label = next((step.get("label") for step in steps if step.get("id") == current_step), None)
        visible_steps = [step for step in steps if step.get("status") != "pending"]
        last_signal = visible_steps[-1] if visible_steps else None
        progress.update(
            {
                "stage": stage,
                "stage_label": STAGE_LABELS.get(stage, stage),
                "status": status,
                "live": live,
                "current_step": current_step,
                "current_label": current_label,
                "last_signal_label": last_signal.get("label") if last_signal else None,
                "stale": bool(status == "queued" and visible_steps and not live),
                "summary": self.stage_progress_summary(job, stage, status, live, current_label),
            }
        )
        return progress

    def stage_is_live(self, job: dict[str, Any], stage: str, stage_info: dict[str, Any]) -> bool:
        runner = job.get("runner") or {}
        if runner.get("status") != "running" or stage_info.get("status") != "running":
            return False
        active = self.active_runners.get(job.get("job_id"))
        if active and active.is_alive():
            return True
        process_info = stage_info.get("process") or {}
        pid = process_info.get("pid")
        if pid:
            return process_alive(pid)
        return runner.get("server_pid") == os.getpid()

    def stage_progress_summary(
        self,
        job: dict[str, Any],
        stage: str,
        status: str,
        live: bool,
        current_label: str | None,
    ) -> str:
        if status == "queued":
            queue = job.get("queue") or self.queue_info(job)
            if queue.get("resource"):
                return (
                    f"等待 {queue.get('resource')} #{queue.get('position') or '-'}/{queue.get('size') or '-'}；"
                    "下方为上次尝试日志信号"
                )
            return "等待资源；下方为上次尝试日志信号"
        if live:
            return f"正在{current_label or STAGE_LABELS.get(stage, stage)}"
        if status == "running":
            return "运行状态待确认；下方为最近日志信号"
        if status == "failed":
            return "阶段失败；下方为失败前日志信号"
        if status in {"succeeded", "skipped"}:
            return "阶段已完成"
        return "等待开始"

    def error_summary(self, job: dict[str, Any]) -> dict[str, Any] | None:
        runner = job.get("runner") or {}
        failed_stage = None
        failed_info: dict[str, Any] = {}
        for stage in STAGE_ORDER:
            info = job.get("stages", {}).get(stage, {})
            if info.get("status") == "failed":
                failed_stage = stage
                failed_info = info
                break
        message = failed_info.get("error") or runner.get("error")
        if runner.get("status") == "queued" and message == ORPHANED_PROCESS_REQUEUE_MESSAGE:
            return None
        if not failed_stage and not message:
            return None
        return {
            "stage": failed_stage,
            "stage_label": STAGE_LABELS.get(failed_stage or "", failed_stage),
            "message": message,
            "log_path": failed_info.get("log_path"),
        }

    def progress(self, job: dict[str, Any]) -> dict[str, Any]:
        statuses = [job.get("stages", {}).get(stage, {}).get("status") for stage in STAGE_ORDER]
        completed = sum(1 for status in statuses if status in {"succeeded", "skipped"})
        failed = sum(1 for status in statuses if status == "failed")
        running = sum(1 for status in statuses if status == "running")
        queued = sum(1 for status in statuses if status == "queued")
        return {
            "total": len(STAGE_ORDER),
            "completed": completed,
            "running": running,
            "queued": queued,
            "failed": failed,
            "percent": int(round((completed / len(STAGE_ORDER)) * 100)),
        }

    def current_stage(self, job: dict[str, Any]) -> str | None:
        runner = job.get("runner") or {}
        if runner.get("status") in {"running", "queued"} and runner.get("current_stage"):
            return normalize_stage_name(runner["current_stage"])
        for stage in STAGE_ORDER:
            if job.get("stages", {}).get(stage, {}).get("status") in {"running", "queued"}:
                return stage
        return None

    def next_stage(self, job: dict[str, Any]) -> str | None:
        for stage in STAGE_ORDER:
            if job.get("stages", {}).get(stage, {}).get("status") not in {"succeeded", "skipped"}:
                return stage
        return None

    def dashboard_url(self, job_id: str) -> str:
        return f"/?job={job_id}"

    def stage_log(self, job_id: str, stage: str, limit: int = 80, full: bool = False) -> dict[str, Any]:
        stage = normalize_stage_name(stage)
        if stage not in STAGE_ORDER:
            raise BridgeError(HTTPStatus.NOT_FOUND, f"unknown stage: {stage}")
        self.load_job(job_id)
        limit = max(1, min(limit, 500))
        log_path = self.stage_log_path(job_id, stage)
        if not log_path.exists():
            return {"job_id": job_id, "stage": stage, "log_path": str(log_path), "lines": [], "text": ""}
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return {
            "job_id": job_id,
            "stage": stage,
            "log_path": str(log_path),
            "lines": text.splitlines() if full else tail_lines(text, limit),
            "text": text if full else "",
            "full": full,
        }

    def tail_stage_log(self, job_id: str, stage: str, limit: int = 80) -> dict[str, Any]:
        return self.stage_log(job_id, stage, limit=limit, full=False)

    def load_job(self, job_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise BridgeError(HTTPStatus.NOT_FOUND, "job not found")
        path = self.job_path(job_id)
        if not path.exists():
            raise BridgeError(HTTPStatus.NOT_FOUND, "job not found")
        job = json.loads(path.read_text(encoding="utf-8"))
        return self.recover_orphaned_running_job(job)

    def recover_orphaned_running_job(self, job: dict[str, Any]) -> dict[str, Any]:
        runner = job.get("runner") or {}
        if self.should_requeue_legacy_interrupted_job(job):
            return self.requeue_interrupted_job(job)
        if runner.get("status") not in {"running", "queued"}:
            return job
        active = self.active_runners.get(job["job_id"])
        if active and active.is_alive():
            return job
        if runner.get("server_pid") == os.getpid():
            return job

        raw_stage = runner.get("current_stage") or self.current_stage(job) or self.next_stage(job)
        stage = normalize_stage_name(raw_stage or "") if raw_stage else None
        stage_info = dict((job.get("stages") or {}).get(stage or "") or (job.get("stages") or {}).get(raw_stage or "") or {})
        process_info = dict(stage_info.get("process") or {})
        pid = process_info.get("pid")
        now = iso_now()

        if pid and process_alive(pid):
            process_info["alive"] = True
            process_info["orphaned"] = True
            process_info["checked_at"] = now
            stage_info["process"] = process_info
            stage_info["status"] = "running"
            stage_info["error"] = "server restarted; external process is still running"
            runner["status"] = "running"
            runner["error"] = stage_info["error"]
            runner["updated_at"] = now
            runner["current_stage"] = stage
            runner.pop("queued_for", None)
            job["runner"] = runner
            job["status"] = "running"
            if stage:
                job.setdefault("stages", {})[stage] = stage_info
            job["updated_at"] = now
            self.save_job(job)
            return job

        if stage == "final-publish" and self.export_outputs_complete(job):
            stage_info["status"] = "succeeded"
            stage_info["exit_code"] = 0
            stage_info["finished_at"] = now
            stage_info["error"] = None
            stage_info.pop("process", None)
            job.setdefault("stages", {})[stage] = stage_info
            next_stage = self.next_stage(job)
            runner["status"] = "queued" if next_stage else "succeeded"
            runner["current_stage"] = next_stage
            runner["updated_at"] = now
            runner["error"] = None
            if not next_stage:
                runner["finished_at"] = now
            job["runner"] = runner
            job["status"] = "queued" if next_stage else "succeeded"
            job["summary"] = self.collect_summary(job)
            job["updated_at"] = now
            self.save_job(job)
            return job

        return self.requeue_interrupted_job(job, stage, stage_info)

    def should_requeue_legacy_interrupted_job(self, job: dict[str, Any]) -> bool:
        runner = job.get("runner") or {}
        if runner.get("status") != "failed" or ORPHANED_PROCESS_GONE_MESSAGE not in str(runner.get("error") or ""):
            return False
        stage = normalize_stage_name(runner.get("current_stage") or self.next_stage(job) or "")
        return stage in STAGE_ORDER

    def requeue_interrupted_job(
        self,
        job: dict[str, Any],
        stage: str | None = None,
        stage_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = iso_now()
        raw_stage = stage or (job.get("runner") or {}).get("current_stage") or self.current_stage(job) or self.next_stage(job)
        stage = normalize_stage_name(raw_stage or "") if raw_stage else None
        if not stage or stage not in STAGE_ORDER:
            return job
        stage_info = dict(stage_info or (job.get("stages") or {}).get(stage) or {})
        resource = stage_resource(stage)
        stage_info["status"] = "queued"
        stage_info["queued_at"] = now
        stage_info["queued_for"] = resource
        stage_info["retry_reason"] = ORPHANED_PROCESS_REQUEUE_MESSAGE
        stage_info["log_path"] = stage_info.get("log_path") or str(self.stage_log_path(job["job_id"], stage))
        stage_info.pop("process", None)
        stage_info.pop("finished_at", None)
        stage_info.pop("exit_code", None)
        stage_info.pop("error", None)
        job.setdefault("stages", {})[stage] = stage_info
        runner = dict(job.get("runner") or {})
        runner["status"] = "queued"
        runner["error"] = ORPHANED_PROCESS_REQUEUE_MESSAGE
        runner["updated_at"] = now
        runner["current_stage"] = stage
        runner["queued_for"] = resource
        runner["server_pid"] = os.getpid()
        runner.pop("finished_at", None)
        job["runner"] = runner
        job["status"] = "queued"
        job["updated_at"] = now
        self.save_job(job)
        return job

    def save_job(self, job: dict[str, Any]) -> None:
        path = self.job_path(job["job_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def job_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def stage_log_path(self, job_id: str, stage: str) -> Path:
        return self.job_dir(job_id) / "logs" / f"{stage}.log"


def sanitize_run_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
    return name or "operation-manual"


def extract_batch_urls(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("video_urls")
    if raw is None:
        raw = payload.get("videoUrls")
    if raw is None:
        raw = payload.get("video_urls_text")
    if raw is None:
        raw = payload.get("videoUrlsText")
    if isinstance(raw, str):
        candidates = re.split(r"[\n\r\t ,]+", raw)
    elif isinstance(raw, list):
        candidates = [str(item or "") for item in raw]
    else:
        candidates = []
    return [item.strip() for item in candidates if item and item.strip()]


def normalize_stage_name(stage: str) -> str:
    value = str(stage or "").strip()
    return STAGE_ALIASES.get(value, value)


def stage_resource(stage: str) -> str:
    return STAGE_RESOURCES.get(normalize_stage_name(stage), "core")


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


def parse_core_progress(text: str, stage_status: str) -> dict[str, Any]:
    return parse_progress_steps(text, stage_status, CORE_PROGRESS_STEPS, CORE_PROGRESS_WEIGHTS)


def parse_stage_progress(stage: str, text: str, stage_status: str) -> dict[str, Any]:
    return parse_progress_steps(text, stage_status, STAGE_PROGRESS_STEPS.get(stage, []))


def stage_progress_text(stage: str, job: dict[str, Any], stage_info: dict[str, Any]) -> str:
    lines = []
    if stage_info and stage == "probe":
        lines.append("probe stage started")
    if stage == "probe" and job.get("resolved_mode"):
        lines.append(f"resolved mode: {job['resolved_mode']}")
    if stage == "verify-core":
        lines.append("verifying core artifacts")
        if stage_info.get("status") == "succeeded":
            lines.append("core artifacts verified")
        if stage_info.get("error"):
            lines.append(stage_info["error"])
    return "\n".join(lines)


def parse_progress_steps(
    text: str,
    stage_status: str,
    step_specs: list[tuple[str, str, tuple[str, ...]]],
    weights: dict[str, int] | None = None,
) -> dict[str, Any]:
    lines = text.splitlines()
    matches: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        for index, (step_id, _label, patterns) in enumerate(step_specs):
            if any(re.search(pattern, line) for pattern in patterns):
                if step_id in matches:
                    matches[step_id]["line"] = line_number
                    matches[step_id]["message"] = line.strip()
                    if matches[step_id].get("timestamp") is None:
                        matches[step_id]["timestamp"] = parse_log_timestamp(line)
                else:
                    matches[step_id] = {
                        "index": index,
                        "line": line_number,
                        "message": line.strip(),
                        "timestamp": parse_log_timestamp(line),
                    }
                break

    current_index = max((item["index"] for item in matches.values()), default=None)
    current_step = None
    steps = []
    for index, (step_id, label, _patterns) in enumerate(step_specs):
        match = matches.get(step_id)
        status = "pending"
        if match:
            if stage_status == "failed" and index == current_index:
                status = "failed"
            elif stage_status == "running" and index == current_index:
                status = "running"
                current_step = step_id
            else:
                status = "succeeded"
        elif current_index is not None and index < current_index:
            status = "succeeded"

        started_at = match.get("timestamp") if match else None
        finished_at = None
        duration = None
        if started_at:
            next_ts = next(
                (
                    candidate.get("timestamp")
                    for candidate in sorted(matches.values(), key=lambda item: item["index"])
                    if candidate["index"] > index and candidate.get("timestamp")
                ),
                None,
            )
            if status == "running":
                duration = max(0.0, time.time() - started_at)
            elif next_ts:
                finished_at = next_ts
                duration = max(0.0, next_ts - started_at)
        steps.append(
            {
                "id": step_id,
                "label": label,
                "status": status,
                "line": match.get("line") if match else None,
                "message": match.get("message") if match else None,
                "started_at": format_epoch(started_at) if started_at else None,
                "finished_at": format_epoch(finished_at) if finished_at else None,
                "duration_seconds": round(duration, 3) if duration is not None else None,
            }
        )
    if stage_status == "failed" and current_index is not None:
        current_step = step_specs[current_index][0]
    percent = progress_percent_from_steps(steps, step_specs, stage_status, weights)
    return {
        "status": stage_status,
        "current_step": current_step,
        "current_label": next((step["label"] for step in steps if step["id"] == current_step), None),
        "percent": percent,
        "steps": steps,
    }


def progress_percent_from_steps(
    steps: list[dict[str, Any]],
    step_specs: list[tuple[str, str, tuple[str, ...]]],
    stage_status: str,
    weights: dict[str, int] | None = None,
) -> int:
    if not step_specs:
        return 100 if stage_status in {"succeeded", "skipped"} else 0
    if stage_status in {"succeeded", "skipped"}:
        return 100
    resolved_weights = {step_id: max(0, weights.get(step_id, 0)) for step_id, _label, _patterns in step_specs} if weights else {
        step_id: 1 for step_id, _label, _patterns in step_specs
    }
    total = sum(resolved_weights.values()) or len(step_specs)
    completed = 0.0
    for step in steps:
        weight = resolved_weights.get(step["id"], 1)
        if step["status"] == "succeeded":
            completed += weight
        elif step["status"] in {"running", "failed"}:
            completed += weight * 0.5
    percent = int(round((completed / total) * 100))
    if stage_status == "running":
        return min(99, max(1 if completed else 0, percent))
    return max(0, min(100, percent))


def parse_log_timestamp(line: str) -> float | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})", line)
    if not match:
        return None
    try:
        parsed = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return time.mktime(parsed.timetuple()) + int(match.group(2)) / 1000


def format_epoch(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(value))


def pipeline_mode_for(analysis_mode: str) -> str:
    return "fast" if analysis_mode == "long-talk-fast" else analysis_mode


def operation_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    venv_bin = REPO_ROOT / ".venv" / "bin"
    venv_python = venv_bin / "python"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    if venv_python.exists():
        env["PYTHON"] = str(venv_python)
    deepseek_env = Path(os.environ.get("VIDEO_ANALYZER_DEEPSEEK_ENV", Path.home() / ".config" / "video-analyzer" / "deepseek.env"))
    if deepseek_env.is_file():
        for line in deepseek_env.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].strip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                continue
            env[key] = value.strip().strip("\"'")
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_bool_option(payload: dict[str, Any], snake_key: str, camel_key: str, default: bool) -> bool:
    if snake_key in payload:
        return parse_bool(normalize_optional_template(payload.get(snake_key)))
    if camel_key in payload:
        return parse_bool(normalize_optional_template(payload.get(camel_key)))
    return default


def parse_int_option(value: Any, default: int) -> int:
    value = normalize_optional_template(value)
    if value in {None, ""}:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BridgeError(HTTPStatus.BAD_REQUEST, "max_comments must be an integer") from exc
    return max(0, parsed)


def normalize_optional_template(value: Any) -> Any:
    if isinstance(value, str) and re.fullmatch(r"\s*\{\{\s*[^{}]+?\s*\}\}\s*", value):
        return ""
    return value


def normalize_cookie_browser(value: Any) -> str:
    browser = str(normalize_optional_template(value) or "").strip().lower()
    if not browser or browser == "auto":
        return DEFAULT_COOKIE_BROWSER
    if browser in {"none", "no", "off", "false", "disabled"}:
        return ""
    return browser


def runtime_config() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in (REPO_ROOT / "video_analyzer" / "config" / "default_config.json", REPO_ROOT / "config" / "config.json"):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "active_runtime_profile" in data:
            merged["active_runtime_profile"] = data["active_runtime_profile"]
        profiles = data.get("runtime_profiles")
        if isinstance(profiles, dict):
            merged.setdefault("runtime_profiles", {}).update(profiles)
    return merged


def runtime_profile_names() -> list[str]:
    profiles = runtime_config().get("runtime_profiles") or {}
    return sorted(profiles)


def active_runtime_profile(profiles: list[str]) -> str:
    active = str(runtime_config().get("active_runtime_profile") or "").strip()
    if active and active in profiles:
        return active
    return profiles[0] if profiles else DEFAULT_PROFILE


def probe_duration_seconds(video_url: str) -> int | None:
    command = ["yt-dlp", "--dump-single-json", "--skip-download", "--no-playlist", video_url]
    if shutil.which("node"):
        command[1:1] = ["--js-runtimes", "node"]
    if can_connect_local_proxy():
        command[1:1] = ["--proxy", "http://127.0.0.1:10808"]
    try:
        completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=operation_env())
        data = json.loads(completed.stdout)
        duration = data.get("duration") or 0
        return int(float(duration)) if duration else None
    except Exception:
        return None


def can_connect_local_proxy() -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", 10808), timeout=1):
            return True
    except OSError:
        return False


def parse_run_dir(text: str) -> str | None:
    patterns = [
        re.compile(r"^\[done\] RUN_DIR=(?P<path>.+)$"),
        re.compile(r"^RUN_DIR=(?P<path>.+)$"),
        re.compile(r"^\[done\] run_dir: (?P<path>.+)$"),
    ]
    for line in reversed(text.splitlines()):
        for pattern in patterns:
            match = pattern.match(line.strip())
            if match:
                return match.group("path").strip()
    return None


def parse_prefixed_path(text: str, prefix: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return None


def tail_lines(text: str, limit: int = 40) -> list[str]:
    return text.splitlines()[-limit:]


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def render_create_page(options: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>video-link 新建任务</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f7f8fa; color: #202124; }}
    main {{ max-width: 960px; margin: 0 auto; padding: 28px 20px 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ color: #5f6368; font-size: 13px; margin-bottom: 20px; }}
    .panel {{ background: #fff; border: 1px solid #dadce0; border-radius: 8px; padding: 16px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }}
    label {{ display: block; color: #5f6368; font-size: 12px; margin-bottom: 6px; }}
    input, select {{ box-sizing: border-box; width: 100%; border: 1px solid #c9d1d9; border-radius: 6px; padding: 9px 10px; font-size: 14px; background: #fff; }}
    input[type="checkbox"] {{ width: auto; margin-right: 8px; }}
    .wide {{ grid-column: 1 / -1; }}
    .check {{ display: flex; align-items: center; min-height: 38px; color: #202124; font-size: 14px; }}
    details summary {{ cursor: pointer; color: #174ea6; font-weight: 600; }}
    button {{ border: 0; border-radius: 6px; background: #1a73e8; color: #fff; padding: 10px 16px; font-size: 14px; cursor: pointer; }}
    button:disabled {{ background: #9aa0a6; cursor: wait; }}
    .error {{ display: none; margin-top: 12px; color: #a50e0e; background: #fce8e6; border: 1px solid #f5c2c7; border-radius: 6px; padding: 10px; }}
    .hint {{ color: #5f6368; font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
  <main>
    <h1>video-link 新建任务</h1>
    <div class="meta">提交后会启动后台流程，并自动跳转到状态页查看进度、日志和产物。</div>
    <form class="panel" id="jobForm">
      <div class="grid">
        <div class="wide">
          <label for="video_url">视频链接</label>
          <input id="video_url" name="video_url" type="url" placeholder="https://..." required autofocus>
        </div>
        <div>
          <label for="analysis_mode">分析模式</label>
          <select id="analysis_mode" name="analysis_mode"></select>
        </div>
        <div>
          <label for="profile">运行配置</label>
          <select id="profile" name="profile"></select>
        </div>
        <div>
          <label for="run_name">运行名称</label>
          <input id="run_name" name="run_name">
        </div>
        <div>
          <label for="cookies_from_browser">Cookie 来源</label>
          <select id="cookies_from_browser" name="cookies_from_browser"></select>
        </div>
        <label class="check"><input id="skip_images" name="skip_images" type="checkbox">跳过配图/提示词</label>
      </div>
      <details class="panel">
        <summary>采集选项</summary>
        <div class="grid" style="margin-top: 14px;">
          <label class="check"><input id="keep_existing" name="keep_existing" type="checkbox">复用已有下载</label>
          <label class="check"><input id="include_subtitles" name="include_subtitles" type="checkbox">下载并纳入字幕</label>
          <label class="check"><input id="prefer_subtitle_transcript" name="prefer_subtitle_transcript" type="checkbox">优先用字幕作为 transcript</label>
          <label class="check"><input id="include_comments" name="include_comments" type="checkbox">下载并纳入评论</label>
          <label class="check"><input id="refresh_context" name="refresh_context" type="checkbox">刷新页面上下文</label>
          <div>
            <label for="max_comments">最多评论数</label>
            <input id="max_comments" name="max_comments" type="number" min="0" step="1">
          </div>
          <div class="wide">
            <label for="subtitle_langs">字幕语言优先级</label>
            <input id="subtitle_langs" name="subtitle_langs">
          </div>
        </div>
      </details>
      <button id="submitBtn" type="submit">启动任务</button>
      <div class="hint">模型和 endpoint 由 profile 控制，不在页面临时覆盖。</div>
      <div class="error" id="errorBox"></div>
    </form>
  </main>
  <script>
    const options = {json.dumps(options, ensure_ascii=False)};
    const defaults = options.defaults || {{}};
    const choices = options.choices || {{}};
    function fillSelect(id, values, selected) {{
      const node = document.getElementById(id);
      node.innerHTML = (values || []).map(value => `<option value="${{value}}">${{value}}</option>`).join("");
      node.value = selected || "";
    }}
    function setChecked(id, value) {{ document.getElementById(id).checked = Boolean(value); }}
    fillSelect("analysis_mode", choices.analysis_modes, defaults.analysis_mode);
    fillSelect("profile", choices.profiles, defaults.profile);
    fillSelect("cookies_from_browser", choices.cookie_browsers, defaults.cookies_from_browser);
    document.getElementById("run_name").value = defaults.run_name || "operation-manual";
    document.getElementById("max_comments").value = defaults.max_comments ?? 30;
    document.getElementById("subtitle_langs").value = defaults.subtitle_langs || "";
    setChecked("skip_images", defaults.skip_images);
    setChecked("keep_existing", defaults.keep_existing);
    setChecked("include_subtitles", defaults.include_subtitles);
    setChecked("prefer_subtitle_transcript", defaults.prefer_subtitle_transcript);
    setChecked("include_comments", defaults.include_comments);
    setChecked("refresh_context", defaults.refresh_context);
    document.getElementById("jobForm").addEventListener("submit", async event => {{
      event.preventDefault();
      const button = document.getElementById("submitBtn");
      const errorBox = document.getElementById("errorBox");
      button.disabled = true;
      errorBox.style.display = "none";
      const payload = {{
        video_url: document.getElementById("video_url").value.trim(),
        analysis_mode: document.getElementById("analysis_mode").value,
        profile: document.getElementById("profile").value,
        run_name: document.getElementById("run_name").value.trim(),
        cookies_from_browser: document.getElementById("cookies_from_browser").value,
        skip_images: document.getElementById("skip_images").checked,
        keep_existing: document.getElementById("keep_existing").checked,
        include_subtitles: document.getElementById("include_subtitles").checked,
        prefer_subtitle_transcript: document.getElementById("prefer_subtitle_transcript").checked,
        include_comments: document.getElementById("include_comments").checked,
        max_comments: Number(document.getElementById("max_comments").value || 0),
        subtitle_langs: document.getElementById("subtitle_langs").value.trim(),
        refresh_context: document.getElementById("refresh_context").checked,
        auto_start: true
      }};
      try {{
        const response = await fetch("/api/video-link/jobs", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify(payload)
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `HTTP ${{response.status}}`);
        window.location.href = data.dashboard_url || `/video-link/jobs/${{data.job_id}}`;
      }} catch (error) {{
        errorBox.textContent = error.message;
        errorBox.style.display = "block";
        button.disabled = false;
      }}
    }});
  </script>
</body>
</html>"""


def render_job_dashboard(job: dict[str, Any]) -> str:
    job_id = escape_text(job["job_id"])
    api_url = f"/api/video-link/jobs/{job_id}"
    initial_stage = job.get("current_stage") or job.get("next_stage") or STAGE_ORDER[-1]
    log_url = f"/api/video-link/jobs/{job_id}/logs/{initial_stage}?tail=80"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>video-link 状态 {job_id}</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f7f8fa; color: #202124; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 20px 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ color: #5f6368; font-size: 13px; margin-bottom: 20px; }}
    .panel {{ background: #fff; border: 1px solid #dadce0; border-radius: 8px; padding: 16px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .key {{ color: #5f6368; font-size: 12px; margin-bottom: 4px; }}
    .value {{ font-size: 14px; overflow-wrap: anywhere; }}
    .bar {{ height: 10px; background: #eceff3; border-radius: 999px; overflow: hidden; margin-top: 10px; }}
    .bar > div {{ height: 100%; width: 0%; background: #1a73e8; transition: width .25s ease; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #edf0f2; text-align: left; padding: 10px 8px; vertical-align: top; }}
    th {{ color: #5f6368; font-weight: 600; }}
    .status {{ display: inline-block; min-width: 70px; padding: 3px 8px; border-radius: 999px; background: #edf0f2; font-size: 12px; text-align: center; }}
    .succeeded, .skipped {{ background: #e6f4ea; color: #137333; }}
    .running {{ background: #e8f0fe; color: #174ea6; }}
    .failed {{ background: #fce8e6; color: #a50e0e; }}
    .errorBox {{ display: none; border-color: #f5c2c7; background: #fff5f5; color: #842029; }}
    .errorBox strong {{ display: block; margin-bottom: 8px; }}
    .hint {{ margin-top: 8px; color: #5f6368; font-size: 13px; }}
    .actions {{ margin-top: 14px; display: flex; gap: 10px; align-items: center; }}
    button {{ border: 0; border-radius: 6px; padding: 9px 14px; background: #202124; color: #fff; font-size: 14px; cursor: pointer; }}
    button.secondary {{ background: #eef2f7; color: #202124; }}
    button:disabled {{ opacity: .55; cursor: default; }}
    .logLink {{ border: 0; background: transparent; color: #0969da; padding: 0; cursor: pointer; font: inherit; }}
    pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; line-height: 1.5; max-height: 420px; overflow: auto; }}
    a {{ color: #0969da; }}
  </style>
</head>
<body>
  <main>
    <h1>video-link 后台流程状态</h1>
    <div class="meta">任务 ID: <code>{job_id}</code> · 每 5 秒自动刷新</div>
    <section class="panel">
      <div class="grid">
        <div><div class="key">总体状态</div><div class="value" id="status">-</div></div>
        <div><div class="key">当前阶段</div><div class="value" id="current">-</div></div>
        <div><div class="key">下一阶段</div><div class="value" id="next">-</div></div>
        <div><div class="key">进度</div><div class="value" id="progressText">-</div></div>
      </div>
      <div class="bar"><div id="progressBar"></div></div>
      <div class="actions">
        <button id="runButton" type="button">继续运行</button>
        <span class="hint" id="runMessage"></span>
      </div>
    </section>
    <section class="panel errorBox" id="errorBox">
      <strong id="errorTitle">流程失败</strong>
      <div class="value" id="errorMessage">-</div>
      <div class="hint" id="errorHint">请查看下方当前日志尾部定位原因。</div>
    </section>
    <section class="panel">
      <div class="grid">
        <div><div class="key">视频链接</div><div class="value" id="videoUrl">-</div></div>
        <div><div class="key">运行目录</div><div class="value" id="runDir">-</div></div>
        <div><div class="key">分析模式</div><div class="value" id="mode">-</div></div>
        <div><div class="key">更新时间</div><div class="value" id="updatedAt">-</div></div>
      </div>
    </section>
    <section class="panel">
      <h2>阶段</h2>
      <table>
        <thead><tr><th>阶段</th><th>状态</th><th>耗时</th><th>日志</th></tr></thead>
        <tbody id="stages"></tbody>
      </table>
    </section>
    <section class="panel" id="corePanel" style="display:none">
      <h2>核心分析子项</h2>
      <table>
        <thead><tr><th>子项</th><th>状态</th><th>耗时</th><th>最近信号</th></tr></thead>
        <tbody id="coreSteps"></tbody>
      </table>
    </section>
    <section class="panel">
      <h2>产物</h2>
      <div id="artifacts" class="value">-</div>
    </section>
    <section class="panel">
      <h2>当前日志</h2>
      <div class="hint" id="logHint">显示当前阶段或失败阶段的日志尾部。</div>
      <div class="actions">
        <button id="copyLogButton" class="secondary" type="button">复制完整日志</button>
        <span class="hint" id="copyLogMessage"></span>
      </div>
      <pre id="logs">-</pre>
    </section>
  </main>
  <script>
    const apiUrl = {json.dumps(api_url)};
    let logUrl = {json.dumps(log_url)};
    let selectedLogStage = null;
    let currentJobId = {json.dumps(job_id)};
    const stageNames = {json.dumps(STAGE_LABELS, ensure_ascii=False)};
    function text(id, value) {{ document.getElementById(id).textContent = value || "-"; }}
    function statusClass(status) {{ return "status " + (status || "pending"); }}
    function duration(value) {{ return value == null ? "-" : `${{value}}s`; }}
    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => {{
        if (char === "&") return "&amp;";
        if (char === "<") return "&lt;";
        if (char === ">") return "&gt;";
        if (char === '"') return "&quot;";
        return "&#39;";
      }});
    }}
    function chooseLogStage(job) {{
      if (selectedLogStage) return selectedLogStage;
      return job.current_stage || job.error_summary?.stage || job.next_stage || [...(job.stage_order || [])].reverse().find(stage => job.stages?.[stage]?.log_path);
    }}
    async function refresh() {{
      const job = await fetch(apiUrl).then(r => r.json());
      currentJobId = job.job_id;
      const progress = job.progress || {{}};
      text("status", job.status);
      const runButton = document.getElementById("runButton");
      runButton.disabled = job.status === "running" || job.runner?.status === "running";
      runButton.textContent = job.status === "failed" ? "重试失败阶段" : "继续运行";
      text("current", stageNames[job.current_stage] || job.current_stage);
      text("next", stageNames[job.next_stage] || job.next_stage);
      text("progressText", `${{progress.completed || 0}}/${{progress.total || 0}} · ${{progress.percent || 0}}%`);
      document.getElementById("progressBar").style.width = (progress.percent || 0) + "%";
      text("videoUrl", job.video_url);
      text("runDir", job.summary?.run_dir || job.run_dir);
      text("mode", `${{job.options?.analysis_mode || "-"}} -> ${{job.resolved_mode || "-"}}`);
      text("updatedAt", job.updated_at);
      const error = job.error_summary;
      const errorBox = document.getElementById("errorBox");
      if (error) {{
        errorBox.style.display = "block";
        text("errorTitle", `流程失败：${{error.stage_label || error.stage || "未知阶段"}}`);
        text("errorMessage", error.message || "未提供错误信息");
      }} else {{
        errorBox.style.display = "none";
      }}
      document.getElementById("stages").innerHTML = (job.stage_order || []).map(stage => {{
        const info = job.stages?.[stage] || {{}};
        const errorText = info.error ? `<div class="hint">${{info.error}}</div>` : "";
        const logCell = info.log_path ? `<button class="logLink" type="button" data-stage="${{stage}}">查看日志</button>` : "-";
        return `<tr><td>${{stageNames[stage] || stage}}${{errorText}}</td><td><span class="${{statusClass(info.status)}}">${{info.status || "pending"}}</span></td><td>${{duration(info.duration_seconds)}}</td><td>${{logCell}}</td></tr>`;
      }}).join("");
      document.querySelectorAll(".logLink").forEach(button => {{
        button.addEventListener("click", async () => {{
          selectedLogStage = button.dataset.stage;
          await loadLog(job, selectedLogStage);
        }});
      }});
      renderCoreProgress(job.core_progress);
      const summary = job.summary || {{}};
      document.getElementById("artifacts").innerHTML = [
        `Markdown: ${{(summary.markdown_files || []).length}}`,
        `导出文件: ${{(summary.export_files || []).length}}`,
        `配图提示词: ${{(summary.prompt_files || []).length}}`,
        `最终图片: ${{(summary.final_images || []).length}}`
      ].join("<br>");
      const stageForLog = chooseLogStage(job);
      await loadLog(job, stageForLog);
    }}
    function renderCoreProgress(core) {{
      const panel = document.getElementById("corePanel");
      if (!core || !(core.steps || []).some(step => step.status !== "pending")) {{
        panel.style.display = "none";
        return;
      }}
      panel.style.display = "block";
      document.getElementById("coreSteps").innerHTML = core.steps.map(step => {{
        const message = step.message ? escapeHtml(step.message) : "-";
        return `<tr><td>${{escapeHtml(step.label)}}</td><td><span class="${{statusClass(step.status)}}">${{step.status}}</span></td><td>${{duration(step.duration_seconds)}}</td><td>${{message}}</td></tr>`;
      }}).join("");
    }}
    async function loadLog(job, stageForLog) {{
      text("logHint", stageForLog ? `显示：${{stageNames[stageForLog] || stageForLog}} 的日志尾部` : "暂无日志");
      if (stageForLog) logUrl = `/api/video-link/jobs/${{job.job_id}}/logs/${{stageForLog}}?tail=80`;
      const log = await fetch(logUrl).then(r => r.json()).catch(() => ({{lines: []}}));
      text("logs", (log.lines || []).join("\\n"));
    }}
    document.getElementById("copyLogButton").addEventListener("click", async () => {{
      const message = document.getElementById("copyLogMessage");
      const stage = selectedLogStage || logUrl.match(/\\/logs\\/([a-z0-9-]+)/)?.[1];
      if (!stage) {{
        message.textContent = "暂无日志";
        return;
      }}
      try {{
        const log = await fetch(`/api/video-link/jobs/${{currentJobId}}/logs/${{stage}}?full=1`).then(r => r.json());
        await navigator.clipboard.writeText(log.text || (log.lines || []).join("\\n"));
        message.textContent = "已复制";
      }} catch (error) {{
        message.textContent = `复制失败：${{error.message}}`;
      }}
    }});
    document.getElementById("runButton").addEventListener("click", async () => {{
      const button = document.getElementById("runButton");
      const message = document.getElementById("runMessage");
      button.disabled = true;
      message.textContent = "已发送";
      try {{
        await fetch(`${{apiUrl}}/run`, {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: "{{}}" }}).then(async r => {{
          if (!r.ok) throw new Error((await r.json()).error || `HTTP ${{r.status}}`);
          return r.json();
        }});
        message.textContent = "运行中";
        await refresh();
      }} catch (error) {{
        message.textContent = error.message;
        button.disabled = false;
      }}
    }});
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>"""


def escape_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


class StatusRequestHandler(BaseHTTPRequestHandler):
    server_app: VideoLinkStatusServer

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/video-link":
                self.write_html(render_create_page(self.server_app.options()))
                return
            if path == "/api/video-link/health":
                self.write_json({"ok": True, "stages": STAGE_ORDER})
                return
            if path == "/api/video-link/options":
                self.write_json(self.server_app.options())
                return
            if path == "/api/video-link/jobs":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["50"])[0])
                self.write_json(self.server_app.list_jobs(limit))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})", path)
            if match:
                self.write_json(self.server_app.public_job(self.server_app.load_job(match.group(1))))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/logs/([a-z0-9-]+)", path)
            if match:
                query = parse_qs(parsed.query)
                limit = int(query.get("tail", ["80"])[0])
                full = parse_bool(query.get("full", ["false"])[0])
                self.write_json(self.server_app.stage_log(match.group(1), match.group(2), limit, full))
                return
            match = re.fullmatch(r"/video-link/jobs/([a-f0-9]{32})", path)
            if match:
                self.write_redirect(f"/?job={match.group(1)}")
                return
            if path == "/":
                query = parse_qs(parsed.query)
                job_id = (query.get("job") or [""])[0]
                if re.fullmatch(r"[a-f0-9]{32}", job_id):
                    self.write_html(render_job_dashboard(self.server_app.public_job(self.server_app.load_job(job_id))))
                else:
                    self.write_redirect("/video-link")
                return
            raise BridgeError(HTTPStatus.NOT_FOUND, "not found")
        except ValueError as exc:
            self.write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except BridgeError as exc:
            self.write_json({"error": exc.message}, exc.status)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            payload = self.read_json_body()
            if path == "/api/video-link/jobs":
                self.write_json(self.server_app.create_job(payload), HTTPStatus.CREATED)
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/run", path)
            if match:
                self.write_json(self.server_app.start_run(match.group(1)), HTTPStatus.ACCEPTED)
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/stages/([a-z0-9-]+)", path)
            if match:
                self.write_json(self.server_app.run_stage(match.group(1), match.group(2)))
                return
            raise BridgeError(HTTPStatus.NOT_FOUND, "not found")
        except BridgeError as exc:
            self.write_json({"error": exc.message}, exc.status)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise BridgeError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
        return payload

    def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_html(self, body_text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = body_text.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_redirect(self, location: str) -> None:
        self.send_response(int(HTTPStatus.FOUND))
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))


def run_server(args: argparse.Namespace) -> None:
    StatusRequestHandler.server_app = VideoLinkStatusServer(Path(args.jobs_dir), REPO_ROOT, auto_resume=True)
    server = ThreadingHTTPServer((args.host, args.port), StatusRequestHandler)
    print(f"video-link status server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=18120)
    serve.add_argument("--jobs-dir", default=str(DEFAULT_JOBS_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "serve":
        run_server(args)


if __name__ == "__main__":
    main()
