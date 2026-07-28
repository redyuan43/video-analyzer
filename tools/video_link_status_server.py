#!/usr/bin/env python3
"""Local status server for running the video-link workflow."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import tempfile
from email import policy
from email.parser import BytesParser
from urllib.parse import quote, unquote, urlencode
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature
from video_analyzer.doc_chat import ask_video_docs_result
from video_analyzer.failures import FAILURE_FILE_ENV, read_failure_envelope
from video_analyzer.qa_index import ANSWER_INDEX_NAME, CHUNKS_NAME, QA_DIR_NAME
from video_analyzer.skill_candidate import (
    build_tool_skill_candidate,
    candidate_summary,
    enable_tool_skill_candidate,
)
from video_analyzer.resource_locks import DEFAULT_LOCK_DIR
from video_analyzer.url_context import (
    AUDIO_MEDIA_EXTENSIONS,
    FALLBACK_OUTPUT_ROOT,
    MEDIA_EXTENSIONS,
    apply_runtime_profile,
    build_analyzer_command,
    infer_video_id_from_url,
    materialize_analysis_context,
    safe_slug,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_DIR = REPO_ROOT / "tmp" / "video-link-status" / "jobs"
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
BAOYU_PROMPT_SCRIPT = REPO_ROOT / "tools" / "prepare_baoyu_image_prompts.py"
ALLOWED_ANALYSIS_MODES = ("auto", "fast", "balanced", "deep", "operation-fast", "long-talk-fast")
ALLOWED_ANALYSIS_DEPTHS = ("light", "full")
ALLOWED_COOKIE_BROWSERS = ("", "chrome", "none", "edge", "firefox", "chromium", "brave")
ALLOWED_DOWNLOAD_DEVICES = ("local", "mi")
DEFAULT_COOKIE_BROWSER = ""
DEFAULT_PROFILE = "deepseek_v4_flash"
DEFAULT_FRAME_EXTRACTOR = "jetson"
DEFAULT_JETSON_FRAME_HOSTS = "agx,agx"
DEFAULT_JETSON_FRAME_BACKEND = "ray"
DEFAULT_JETSON_SAMPLE_FPS = "0.5"
BAOYU_IMAGE_GENERATION_ENABLED = os.environ.get("VIDEO_LINK_ENABLE_BAOYU_IMAGES", "").strip().lower() in {"1", "true", "yes", "on"}
CORE_ANALYSIS_ERROR_PATTERNS = (
    "Error analyzing frame",
    "model-resource-busy",
    "ActorDiedError",
)
CORE_DIAGNOSTIC_ERROR_PATTERNS = (
    "Traceback",
    "CUDA out of memory",
    "out of memory",
    "ActorDiedError",
    "model-resource-busy",
    "Error analyzing frame",
)
CORE_DIAGNOSTIC_NOT_READY_PATTERNS = (
    "endpoint not ready",
    "DotsMOCR endpoint not ready",
    "no MiniCPM worker is ready",
    "Connection refused",
)
CORE_DIAGNOSTIC_STALE_SECONDS = 600
CORE_DIAGNOSTIC_QUEUE_WARN_SECONDS = 300
CORE_DIAGNOSTIC_EXPECTED_MINICPM_CONCURRENCY = 5
CORE_DIAGNOSTIC_GPU_TTL_SECONDS = 120
CORE_DIAGNOSTIC_GPU_TIMEOUT_SECONDS = 0.8
AUTO_MODE_LONG_SECONDS = 2700
AUTO_MODE_FAST_KEYWORDS = (
    "快速",
    "先看",
    "大概",
    "概览",
    "摘要",
    "简短",
    "粗略",
    "测试",
    "quick",
    "fast",
    "summary",
    "overview",
    "smoke",
    "tldr",
    "tl;dr",
)
AUTO_MODE_DEEP_KEYWORDS = (
    "深度",
    "详细",
    "完整",
    "最终",
    "发布",
    "不能漏",
    "不要漏",
    "全量",
    "严谨",
    "复核",
    "精确",
    "逐步",
    "每一步",
    "操作手册",
    "教程",
    "排错",
    "风险",
    "参数",
    "代码",
    "命令",
    "高质量",
    "deep",
    "detailed",
    "complete",
    "exhaustive",
    "final",
    "precise",
    "troubleshoot",
    "debug",
    "production",
)
AUTO_MODE_LONG_TALK_KEYWORDS = (
    "长视频",
    "播客",
    "访谈",
    "演讲",
    "讲座",
    "会议",
    "字幕",
    "转写",
    "章节",
    "podcast",
    "talk",
    "lecture",
    "transcript",
    "subtitle",
    "chapter",
)
AUTO_MODE_OPERATION_KEYWORDS = (
    "操作",
    "教程",
    "演示",
    "屏幕",
    "界面",
    "配置",
    "步骤",
    "命令",
    "安装",
    "部署",
    "debug",
    "setup",
    "install",
    "configure",
    "tutorial",
    "screen",
    "ui",
    "cli",
)
DEFAULT_RUN_NAME = "operation-manual"
DEFAULT_SUBTITLE_LANGS = "zh-CN,zh-Hans,zh,en"
UPLOAD_SOURCE_TYPE = "upload"
UPLOAD_OUTPUT_PREFIX = "upload-"
MODULE_ORDER = [
    "probe",
    "prepare",
    "analyze-core",
    "verify-core",
    "study-guide",
    "multidoc",
    "deep-v2",
    "evidence-review",
    "web-evidence",
    "qa-index",
    "image-prompts",
    "final-publish",
]
MODULE_LABELS = {
    "probe": "探测时长",
    "prepare": "下载/上下文",
    "analyze-core": "核心分析",
    "verify-core": "校验产物",
    "study-guide": "学习证据账本",
    "multidoc": "多文档分析",
    "deep-v2": "章节深度报告",
    "evidence-review": "证据复核/发布门禁",
    "web-evidence": "联网补证据",
    "qa-index": "问答证据索引",
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
    "study-guide": {
        "requires": ["run_dir", "verified_core"],
        "produces": ["study_guide", "evidence_gaps", "evidence_triage"],
    },
    "multidoc": {"requires": ["run_dir", "verified_core", "study_guide"], "produces": ["docs_analysis"]},
    "deep-v2": {"requires": ["run_dir", "verified_core", "study_guide"], "produces": ["chapter_deep_report"]},
    "evidence-review": {
        "requires": ["run_dir", "verified_core", "study_guide"],
        "produces": ["evidence_review", "publish_decision"],
    },
    "web-evidence": {"requires": ["run_dir", "verified_core", "evidence_review"], "produces": ["web_evidence"]},
    "qa-index": {"requires": ["run_dir", "verified_core"], "produces": ["qa_index"]},
    "image-prompts": {"requires": ["run_dir"], "produces": ["image_prompts"]},
    "final-publish": {"requires": ["run_dir", "verified_core"], "produces": ["exports"]},
}
STAGE_ORDER = MODULE_ORDER
STAGE_LABELS = MODULE_LABELS
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
}
RESOURCE_LIMITS = {
    "prepare": 2,
    "core": 1,
    "audio-analysis": 1,
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
}
EXPECTED_FINAL_EXPORTS = (
    "operation_manual.pdf",
    "knowledge_notes_v2.pdf",
    "deep_report_v2.pdf",
    "manual_evidence.pdf",
)
EXPECTED_FINAL_DOCUMENTS = (
    "operation_manual.md",
    "docs_analysis_chapters/knowledge_notes_v2.md",
    "docs_analysis_chapters/deep_report_v2.md",
    "manual_evidence.md",
)
DOCUMENT_PREVIEW_PRIMARY = (
    ("operation_manual.md", "操作手册", "优先阅读：核心结论、流程步骤和关键截图。"),
    ("docs_analysis_chapters/knowledge_notes_v2.md", "逐章知识笔记", "第二阅读：按章节整理概念、背景和要点。"),
    ("docs_analysis_chapters/deep_report_v2.md", "深度报告", "第三阅读：综合分析、风险点和深入判断。"),
)
DOCUMENT_PREVIEW_EVIDENCE = (
    ("manual_evidence.md", "证据审计表", "核查 OCR/VL/帧证据时使用，内容较重。"),
    ("evidence_index.md", "证据索引", "按结论和证据位置整理的索引。"),
    ("visual_review.html", "视觉复核页", "快速检查抽帧、截图和视觉证据覆盖。"),
    ("evidence_review.json", "证据复核结果", "发布门禁和证据质量复核数据。"),
    ("publish_decision.json", "发布决策", "是否可发布及阻断原因。"),
    ("web_evidence.md", "联网补证据摘要", "需要外部资料时的补证据结果。"),
    ("web_evidence.json", "联网补证据数据", "联网补证据的结构化原始数据。"),
)
DOCUMENT_PREVIEW_PROCESS = (
    ("transcript.md", "转写文本", "ASR 生成的全文转写。"),
    ("analysis.json", "核心分析 JSON", "核心分析的结构化总产物。"),
    ("study_guide.json", "学习证据账本", "学习卡片和证据分诊的结构化输入。"),
    ("study_overview.md", "学习概览", "轻量脑图/概要式学习入口。"),
    ("study_cards.md", "学习卡片", "拆分后的学习卡片。"),
    ("evidence_gaps.json", "证据缺口", "模型发现的证据不足点。"),
    ("evidence_triage.json", "证据分诊", "证据缺口的处理路由。"),
    ("docs_analysis/knowledge_notes.md", "知识笔记草稿", "旧版多文档分析中间产物。"),
    ("docs_analysis/deep_report.md", "深度报告草稿", "旧版深度报告中间产物。"),
    ("docs_analysis/operation_manual_review.md", "操作手册复核", "多文档分析对操作手册的复核。"),
    ("docs_analysis_chapters/deep_report_v2.review.md", "深度报告复核", "新版深度报告质量复核。"),
    ("docs_analysis_chapters/deep_report_v2.review.json", "深度报告复核数据", "新版深度报告复核结构化数据。"),
    ("RUN_MANIFEST.md", "运行清单", "本次运行的关键命令和环境记录。"),
    ("frame_dedup_audit.json", "抽帧去重审计", "候选帧去重和覆盖情况。"),
)
DOCUMENT_PREVIEW_ASSETS = (
    ("frames", "原始抽帧", "Jetson/Ray 抽出的候选帧目录。"),
    ("manual_assets", "手册截图", "写入 Markdown 的真实截图资源。"),
    ("orin", "OCR/VL 原始结果", "ASR、OCR、VL 的结构化中间结果。"),
    ("visual_review", "视觉复核素材", "视觉复核页面使用的素材目录。"),
)
SOFT_FAILURE_STAGES = {"deep-v2", "web-evidence", "qa-index", "image-prompts"}
VIDEO_PREVIEW_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VSCODE_PORT_RANGE = tuple(
    int(part)
    for part in os.environ.get("VIDEO_LINK_VSCODE_PORT_RANGE", "19000-19100").split("-", 1)
)
VSCODE_PORT = int(os.environ.get("VIDEO_LINK_VSCODE_PORT", str(VSCODE_PORT_RANGE[0])))
ORPHANED_PROCESS_GONE_MESSAGE = (
    "server stopped while this stage was running; process is gone and artifacts are incomplete; retry to continue"
)
ORPHANED_PROCESS_REQUEUE_MESSAGE = (
    "server stopped while this stage was running; process is gone and artifacts are incomplete; queued for retry"
)
TRANSIENT_RESOURCE_REQUEUE_MESSAGE = "remote/system resource is temporarily busy; queued for retry"
TRANSIENT_API_REQUEUE_MESSAGE = "text API is temporarily unavailable; queued for one automatic retry"
YOUTUBE_FORMAT_REQUEUE_MESSAGE = "YouTube returned no downloadable formats; queued for one automatic retry"
AUTO_RETRY_REASONS = {
    ORPHANED_PROCESS_REQUEUE_MESSAGE,
    TRANSIENT_RESOURCE_REQUEUE_MESSAGE,
    TRANSIENT_API_REQUEUE_MESSAGE,
    YOUTUBE_FORMAT_REQUEUE_MESSAGE,
}
TRANSIENT_RESOURCE_BUSY_PATTERNS = (
    "ray.exceptions.OutOfMemoryError",
    "Task was killed due to the node running low on memory",
    "exceeds the memory usage threshold",
    "Ray killed this worker",
)
YOUTUBE_FORMAT_UNAVAILABLE_PATTERN = "Requested format is not available"
MAX_YOUTUBE_FORMAT_RETRIES = 1
MAX_TRANSIENT_API_RETRIES = 1
MAX_INTERRUPTED_RETRIES = 1
RESOURCE_WAIT_SECONDS = 5.0
AUTO_RETRY_DELAY_SECONDS = float(os.environ.get("VIDEO_LINK_AUTO_RETRY_DELAY_SECONDS", "60"))
AUTO_RETRY_POLL_SECONDS = float(os.environ.get("VIDEO_LINK_AUTO_RETRY_POLL_SECONDS", "5"))
ANALYSIS_PROGRESS_FILENAME = "progress.json"
CORE_PROGRESS_STEPS = [
    ("ray", "Ray 集群准备", (r"\[jetson-ray\]", r"Ray runtime started")),
    ("context", "页面/评论/素材准备", (r"Extracting cookies", r"\[youtube\]", r"Writing video metadata")),
    ("audio", "音频提取", (r"Extracting audio from video",)),
    ("local_model", "本机模型资源等待", (r"\[local-model-lock\] (waiting|acquired) stage=core",)),
    ("asr", "ASR 转写", (r"Transcribing audio", r"\[resource-lock\] (waiting|acquired) resource=asr")),
    ("asr_done", "ASR 完成", (r"ASR succeeded", r"Using existing transcript file")),
    ("frames", "扫描/候选帧抽取", (r"Extracting frames from video", r"Jetson video cache", r"frame worker")),
    ("frames_done", "候选帧就绪", (r"Extracted \d+ screen keyframes",)),
    ("ocr", "OCR关键帧选择/执行", (r"Selected \d+ OCR keyframes", r"Running OCR", r"\[resource-lock\] (waiting|acquired) resource=ocr")),
    ("ocr_ready", "OCR文本事件就绪", (r"DotsMOCR endpoint not ready", r"DotsMOCR endpoint ready", r"OpenAI-compatible vision OCR", r"OCR results ready")),
    ("vl", "VL解释帧选择/分析", (r"Selecting and analyzing VL frames", r"\[resource-lock\] (waiting|acquired) resource=vl")),
    ("manual", "操作手册生成", (r"Generating operation manual",)),
    ("write", "结果写出", (r"Operation manual saved", r"Analysis complete", r"\[done\] run_dir:")),
]
CORE_PROGRESS_WEIGHTS = {
    "ray": 3,
    "context": 7,
    "audio": 5,
    "local_model": 1,
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
    "study-guide": [
        ("load", "读取多模态证据", (r"study_guide", r"evidence", r"analysis\.json")),
        ("gaps", "检查证据缺口", (r"evidence_gaps", r"gap", r"missing")),
        ("write", "写出学习视图产物", (r"study_overview\.md", r"study_cards\.md", r"evidence_index\.md")),
    ],
    "evidence-review": [
        ("load", "读取学习证据账本", (r"study_guide", r"evidence_gaps", r"study_chapters")),
        ("review", "模型复核缺口影响", (r"evidence_review", r"review", r"publish_decision")),
        ("gate", "写出发布门禁", (r"publish_decision", r"publishable", r"blocked")),
    ],
    "web-evidence": [
        ("load", "读取证据缺口", (r"evidence_gaps", r"study_guide", r"web_evidence")),
        ("search", "联网搜索外部证据", (r"search", r"external", r"web")),
        ("write", "写出联网证据账本", (r"web_evidence\.json", r"web_evidence\.md", r"processed_gaps")),
    ],
    "qa-index": [
        ("load", "读取最终文档与证据", (r"operation_manual", r"manual_evidence", r"transcript")),
        ("chunk", "生成问答证据切片", (r"source_chunks\.jsonl", r"chunk_count")),
        ("write", "写出问答索引", (r"answer_index\.json", r"QA index")),
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
        ("export", "导出发布文件", (r"\[pdf\]", r"\[export\]", r"export_video_docs", r"\[long-png\]")),
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
        self.vscode_sessions: dict[str, dict[str, Any]] = {}
        self.vscode_lock = threading.Lock()
        self.auto_retry_stop = threading.Event()
        self.auto_retry_thread: threading.Thread | None = None
        self.gpu_snapshot_cache: dict[str, Any] | None = None
        self.gpu_snapshot_cache_time = 0.0
        self.resource_locks = {name: threading.BoundedSemaphore(limit) for name, limit in RESOURCE_LIMITS.items()}
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        if auto_resume:
            self.recover_interrupted_jobs(auto_start=True)
            self.start_auto_retry_loop()

    def options(self) -> dict[str, Any]:
        profiles = runtime_profile_names()
        default_profile = active_runtime_profile(profiles)
        return {
            "defaults": {
                "analysis_mode": "auto",
                "analysis_depth": "full",
                "profile": default_profile,
                "run_name": DEFAULT_RUN_NAME,
                "cookies_from_browser": "none",
                "download_device": "local",
                "skip_images": True,
                "keep_existing": True,
                "include_subtitles": True,
                "prefer_subtitle_transcript": True,
                "include_comments": True,
                "max_comments": 3000,
                "subtitle_langs": DEFAULT_SUBTITLE_LANGS,
                "refresh_context": True,
                "focus_prompt": "",
            },
            "choices": {
                "analysis_modes": list(ALLOWED_ANALYSIS_MODES),
                "analysis_depths": list(ALLOWED_ANALYSIS_DEPTHS),
                "profiles": profiles,
                "cookie_browsers": [item for item in ALLOWED_COOKIE_BROWSERS if item],
                "download_devices": list(ALLOWED_DOWNLOAD_DEVICES),
            },
        }

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        video_url = str(payload.get("video_url") or payload.get("videoUrl") or "").strip()
        if not video_url.startswith(("http://", "https://")):
            raise BridgeError(HTTPStatus.BAD_REQUEST, "video_url must be an http(s) URL")
        return self._create_job(payload, video_url=video_url)

    def create_uploaded_media_job(self, payload: dict[str, Any], media_path: Path, source_filename: str) -> dict[str, Any]:
        source_name = sanitize_upload_filename(source_filename)
        suffix = Path(source_name).suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"media file must be one of {sorted(MEDIA_EXTENSIONS)}")
        if not media_path.is_file():
            raise BridgeError(HTTPStatus.BAD_REQUEST, "media file is not available")
        if media_path.stat().st_size <= 0:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "uploaded media file is empty")

        create_payload = dict(payload)
        create_payload["auto_start"] = False
        job = self._create_job(
            create_payload,
            video_url=f"upload://{source_name}",
            source_type=UPLOAD_SOURCE_TYPE,
            source_name=source_name,
            upload_suffix=suffix,
        )
        loaded = self.load_job(job["job_id"])
        video_dir = self.upload_video_dir(loaded["job_id"])
        video_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"audio{suffix}" if suffix in AUDIO_MEDIA_EXTENSIONS else f"video{suffix}"
        target_path = video_dir / target_name
        shutil.copy2(media_path, target_path)
        context_path = video_dir / "page_context.md"
        info_path = video_dir / "info.json"
        context_path.write_text(upload_page_context(source_name, target_path), encoding="utf-8")
        info_path.write_text(
            json.dumps(
                {
                    "id": f"{UPLOAD_OUTPUT_PREFIX}{loaded['job_id']}",
                    "title": source_name,
                    "source": UPLOAD_SOURCE_TYPE,
                    "filename": source_name,
                    "media_path": str(target_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        loaded.update(
            {
                "media_path": str(target_path),
                "video_path": str(target_path),
                "page_context_path": str(context_path),
                "video_dir": str(video_dir),
                "title": source_name,
                "source_name": source_name,
            }
        )
        loaded["run_dir"] = str(video_dir / loaded["options"]["run_name"])
        self.save_job(loaded)
        if parse_bool(normalize_optional_template(payload.get("auto_start") if "auto_start" in payload else payload.get("autoStart", False))):
            return self.start_run(loaded["job_id"])
        return self.public_job(loaded)

    def _create_job(
        self,
        payload: dict[str, Any],
        *,
        video_url: str,
        source_type: str = "url",
        source_name: str | None = None,
        upload_suffix: str | None = None,
    ) -> dict[str, Any]:

        analysis_mode = str(payload.get("analysis_mode") or payload.get("analysisMode") or "auto").strip() or "auto"
        if analysis_mode not in ALLOWED_ANALYSIS_MODES:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"analysis_mode must be one of {sorted(ALLOWED_ANALYSIS_MODES)}")
        default_depth = "light" if source_type == UPLOAD_SOURCE_TYPE else "full"
        analysis_depth = str(payload.get("analysis_depth") or payload.get("analysisDepth") or default_depth).strip() or default_depth
        if analysis_depth not in ALLOWED_ANALYSIS_DEPTHS:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"analysis_depth must be one of {sorted(ALLOWED_ANALYSIS_DEPTHS)}")

        cookie_browser = "" if source_type == UPLOAD_SOURCE_TYPE else normalize_cookie_browser(payload.get("cookies_from_browser") or payload.get("cookiesFromBrowser"))
        if cookie_browser not in ALLOWED_COOKIE_BROWSERS:
            raise BridgeError(
                HTTPStatus.BAD_REQUEST,
                f"cookies_from_browser must be one of {sorted(ALLOWED_COOKIE_BROWSERS)} or none",
            )
        download_device = "local" if source_type == UPLOAD_SOURCE_TYPE else str(payload.get("download_device") or payload.get("downloadDevice") or "local").strip() or "local"
        if download_device not in ALLOWED_DOWNLOAD_DEVICES:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"download_device must be one of {sorted(ALLOWED_DOWNLOAD_DEVICES)}")

        defaults = self.options()["defaults"]
        run_name = sanitize_run_name(str(payload.get("run_name") or payload.get("runName") or defaults["run_name"]))
        profile = str(payload.get("profile") or defaults["profile"]).strip() or defaults["profile"]
        profiles = runtime_profile_names()
        if profiles and profile not in profiles:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"profile must be one of {profiles}")
        skip_images = True
        if BAOYU_IMAGE_GENERATION_ENABLED:
            skip_images = parse_bool_option(payload, "skip_images", "skipImages", defaults["skip_images"])
        auto_start = parse_bool(normalize_optional_template(payload.get("auto_start") if "auto_start" in payload else payload.get("autoStart", False)))
        keep_existing = parse_bool_option(payload, "keep_existing", "keepExisting", defaults["keep_existing"])
        include_subtitles = False if source_type == UPLOAD_SOURCE_TYPE else parse_bool_option(payload, "include_subtitles", "includeSubtitles", defaults["include_subtitles"])
        prefer_subtitle_transcript = parse_bool_option(
            payload,
            "prefer_subtitle_transcript",
            "preferSubtitleTranscript",
            False if source_type == UPLOAD_SOURCE_TYPE else defaults["prefer_subtitle_transcript"],
        )
        include_comments = False if source_type == UPLOAD_SOURCE_TYPE else parse_bool_option(payload, "include_comments", "includeComments", defaults["include_comments"])
        refresh_context = False if source_type == UPLOAD_SOURCE_TYPE else parse_bool_option(payload, "refresh_context", "refreshContext", defaults["refresh_context"])
        max_comments = parse_int_option(payload.get("max_comments") if "max_comments" in payload else payload.get("maxComments"), defaults["max_comments"])
        subtitle_langs = str(payload.get("subtitle_langs") or payload.get("subtitleLangs") or defaults["subtitle_langs"]).strip()
        focus_prompt = normalize_focus_prompt(payload.get("focus_prompt") if "focus_prompt" in payload else payload.get("focusPrompt", ""))
        template_id = normalize_optional_template(payload.get("template_id") if "template_id" in payload else payload.get("templateId", ""))
        template_title = normalize_optional_template(payload.get("template_title") if "template_title" in payload else payload.get("templateTitle", ""))
        template_title_zh = normalize_optional_template(payload.get("template_title_zh") if "template_title_zh" in payload else payload.get("templateTitleZh", ""))
        template_category = normalize_optional_template(payload.get("template_category") if "template_category" in payload else payload.get("templateCategory", ""))

        job_id = uuid.uuid4().hex
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        job = {
            "job_id": job_id,
            "status": "created",
            "created_at": iso_now(),
            "updated_at": iso_now(),
            "video_url": video_url,
            "source_type": source_type,
            "source_name": source_name,
            "upload_suffix": upload_suffix,
            "options": {
                "analysis_mode": analysis_mode,
                "analysis_depth": analysis_depth,
                "profile": profile,
                "run_name": run_name,
                "cookies_from_browser": cookie_browser,
                "download_device": download_device,
                "skip_images": skip_images,
                "keep_existing": keep_existing,
                "include_subtitles": include_subtitles,
                "prefer_subtitle_transcript": prefer_subtitle_transcript,
                "include_comments": include_comments,
                "max_comments": max_comments,
                "subtitle_langs": subtitle_langs,
                "refresh_context": refresh_context,
                "focus_prompt": focus_prompt,
                "template_id": template_id,
                "template_title": template_title,
                "template_title_zh": template_title_zh,
                "template_category": template_category,
            },
            "resolved_mode": None,
            "run_dir": None,
            "artifacts": {},
            "stages": {},
            "modules": {},
            "warnings": [],
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
            batch_payload["focus_prompt"] = focus_prompt_for_url(payload, url, index)
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
                jobs.append(self.public_job_summary(self.load_job(path.parent.name)))
            except Exception:
                continue
        self.annotate_failure_dispositions(jobs)
        jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {
            "jobs": jobs[: max(1, min(limit, 200))],
            "total": len(jobs),
            "summary": self.jobs_summary(jobs),
            "resources": self.resource_summary(jobs),
        }

    def list_mobile_audio_jobs(self, limit: int = 50) -> dict[str, Any]:
        self.cleanup_acknowledged_mobile_audio_jobs()
        jobs = []
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            if self.is_mobile_audio_job(job):
                jobs.append(self.mobile_audio_job(job))
        jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {"jobs": jobs[: max(1, min(limit, 100))], "total": len(jobs)}

    def get_mobile_audio_job(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        if not self.is_mobile_audio_job(job):
            raise BridgeError(HTTPStatus.NOT_FOUND, "audio job not found")
        return self.mobile_audio_job(job, include_resources=True)

    def get_mobile_audio_job_by_attempt(self, external_attempt_id: str) -> dict[str, Any]:
        external_attempt_id = normalize_external_attempt_id(external_attempt_id)
        job = self.mobile_audio_job_by_attempt(external_attempt_id)
        if not job:
            raise BridgeError(HTTPStatus.NOT_FOUND, "audio job not found")
        return self.mobile_audio_job(job, include_resources=True)

    def create_mobile_audio_job(
        self,
        payload: dict[str, Any],
        media_path: Path,
        source_filename: str,
        *,
        pipeline_kind: str = "analysis",
    ) -> dict[str, Any]:
        pipeline_kind = str(pipeline_kind or "analysis").strip().lower()
        if pipeline_kind not in {"analysis", "transcription"}:
            raise BridgeError(
                HTTPStatus.BAD_REQUEST,
                "audio pipeline kind must be analysis or transcription",
            )
        external_attempt_id = normalize_external_attempt_id(
            payload.get("external_attempt_id")
            or payload.get("externalAttemptId")
            or ""
        )
        source_sha256 = str(
            payload.get("source_sha256")
            or payload.get("sourceSha256")
            or ""
        ).strip().lower()
        actual_sha256 = sha256_file(media_path)
        if source_sha256 and source_sha256 != actual_sha256:
            raise BridgeError(
                HTTPStatus.BAD_REQUEST,
                "uploaded media sha256 does not match source_sha256",
            )
        source_sha256 = actual_sha256

        if external_attempt_id:
            existing = self.mobile_audio_job_by_attempt(external_attempt_id)
            if existing:
                existing_kind = str(
                    existing.get("audio_pipeline_kind") or "analysis"
                )
                if existing_kind != pipeline_kind:
                    raise BridgeError(
                        HTTPStatus.CONFLICT,
                        "external_attempt_id is already bound to another audio pipeline",
                    )
                if str(existing.get("source_sha256") or "") != source_sha256:
                    raise BridgeError(
                        HTTPStatus.CONFLICT,
                        "external_attempt_id is already bound to another audio file",
                    )
                return self.mobile_audio_job(existing, include_resources=True)

        create_payload = dict(payload)
        create_payload.update(
            {
                "analysis_mode": "auto",
                "analysis_depth": "light",
                "run_name": (
                    "audio-transcription"
                    if pipeline_kind == "transcription"
                    else "audio-summary"
                ),
                "skip_images": True,
                "keep_existing": True,
                "auto_start": False,
                "profile": str(
                    payload.get("profile")
                    or payload.get("analysis_profile")
                    or "deepseek_v4_pro"
                ).strip(),
            }
        )
        created = self.create_uploaded_media_job(
            create_payload,
            media_path,
            source_filename,
        )
        job = self.load_job(created["job_id"])
        job["external_attempt_id"] = external_attempt_id or uuid.uuid4().hex
        job["source_sha256"] = source_sha256
        job["source_device"] = str(payload.get("source_device") or "external-audio")
        job["source_file_id"] = str(payload.get("source_file_id") or "")
        job["consumer_acknowledged_at"] = None
        job["audio_pipeline"] = True
        job["audio_pipeline_kind"] = pipeline_kind
        if pipeline_kind == "transcription":
            job["asr_provider"] = str(
                payload.get("asr_provider")
                or payload.get("asrProvider")
                or "firered_3dspeaker"
            ).strip()
        self.save_job(job)
        self.cleanup_acknowledged_mobile_audio_jobs()
        return self.start_run(job["job_id"])

    def create_mobile_transcript_job(
        self,
        payload: dict[str, Any],
        transcript_path: Path,
        source_filename: str,
    ) -> dict[str, Any]:
        external_attempt_id = normalize_external_attempt_id(payload.get("external_attempt_id") or "")
        required = {
            "source_sha256": str(payload.get("source_sha256") or "").strip().lower(),
            "source_transcription_id": str(payload.get("source_transcription_id") or "").strip(),
            "source_transcript_sha256": str(payload.get("source_transcript_sha256") or "").strip().lower(),
        }
        if not external_attempt_id or any(not value for value in required.values()):
            raise BridgeError(HTTPStatus.BAD_REQUEST, "external_attempt_id and transcript source identifiers are required")
        if not re.fullmatch(r"[a-f0-9]{64}", required["source_sha256"]):
            raise BridgeError(HTTPStatus.BAD_REQUEST, "source_sha256 must be a lowercase sha256")
        if not transcript_path.is_file() or transcript_path.stat().st_size <= 0:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "transcript JSON file is required")
        try:
            transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"transcript file is invalid JSON: {exc}") from exc
        if not isinstance(transcript_payload, dict):
            raise BridgeError(HTTPStatus.BAD_REQUEST, "transcript JSON must be an object")
        actual_transcript_sha256 = sha256_file(transcript_path)
        if required["source_transcript_sha256"] != actual_transcript_sha256:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "transcript sha256 does not match source_transcript_sha256")

        existing = self.mobile_audio_job_by_attempt(external_attempt_id)
        if existing:
            matches = bool(existing.get("provided_transcript")) and all(
                str(existing.get(key) or "") == value for key, value in required.items()
            )
            if not matches:
                raise BridgeError(HTTPStatus.CONFLICT, "external_attempt_id is already bound to another transcript source")
            return self.mobile_audio_job(existing, include_resources=True)

        create_payload = dict(payload)
        create_payload.update({
            "analysis_mode": "auto", "analysis_depth": "light", "run_name": "audio-summary",
            "skip_images": True, "keep_existing": True, "auto_start": False,
            "profile": str(payload.get("profile") or "deepseek_v4_pro").strip(),
        })
        source_name = sanitize_upload_filename(source_filename or "transcript.json")
        created = self._create_job(
            create_payload,
            video_url=f"upload://{source_name}",
            source_type=UPLOAD_SOURCE_TYPE,
            source_name=source_name,
            upload_suffix=".json",
        )
        job = self.load_job(created["job_id"])
        video_dir = self.upload_video_dir(job["job_id"])
        video_dir.mkdir(parents=True, exist_ok=True)
        provided_path = video_dir / "provided_transcript.json"
        shutil.copy2(transcript_path, provided_path)
        context_path = video_dir / "page_context.md"
        context_path.write_text(f"# Provided transcript\n\n- source: `{source_name}`\n", encoding="utf-8")
        job.update({
            "external_attempt_id": external_attempt_id,
            **required,
            "provided_transcript": True,
            "provided_transcript_path": str(provided_path),
            "media_path": str(provided_path),
            "video_path": str(provided_path),
            "page_context_path": str(context_path),
            "video_dir": str(video_dir),
            "run_dir": str(video_dir / job["options"]["run_name"]),
            "audio_pipeline": True,
            "audio_pipeline_kind": "analysis",
            "consumer_acknowledged_at": None,
        })
        self.save_job(job)
        return self.start_run(job["job_id"])

    def mobile_audio_job_by_attempt(self, external_attempt_id: str) -> dict[str, Any] | None:
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            if (
                self.is_mobile_audio_job(job)
                and str(job.get("external_attempt_id") or "") == external_attempt_id
            ):
                return job
        return None

    def acknowledge_mobile_audio_job(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        if not self.is_mobile_audio_job(job):
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
            "pipeline_kind": str(job.get("audio_pipeline_kind") or "analysis"),
            "asr_provider": job.get("asr_provider"),
            "consumer_acknowledged_at": job.get("consumer_acknowledged_at"),
            "summary": {"study": (public.get("summary") or {}).get("study") or {}},
            "prompt_template": {
                "requested": requested,
                "actual": actual,
            },
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
        pipeline_kind = str(job.get("audio_pipeline_kind") or "analysis")
        result_path = (
            run_dir / "transcription.json"
            if pipeline_kind == "transcription"
            else run_dir / "analysis.json"
        )
        if not result_path.is_file():
            return {}
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if pipeline_kind == "transcription":
            return payload
        return {
            "audio_template_analysis": payload.get("audio_template_analysis") or {},
            "speaker_diarization": payload.get("speaker_diarization") or {},
            "speaker_count": (
                (payload.get("speaker_diarization") or {}).get("final_speaker_count")
                or (payload.get("speaker_diarization") or {}).get("detected_speaker_count")
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
        return {
            "id": value.get("id"),
            "title": value.get("title"),
            "title_zh": value.get("title_zh"),
            "category": value.get("category"),
        }

    def delete_job(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        active = self.active_runners.get(job_id)
        if active and active.is_alive():
            raise BridgeError(HTTPStatus.CONFLICT, "job is still running")

        stage = self.current_stage(job) or self.next_stage(job)
        process = job.get("process") or ((job.get("stages") or {}).get(stage or "") or {}).get("process") or {}
        pid = process.get("pid")
        if pid and process_alive(pid):
            raise BridgeError(HTTPStatus.CONFLICT, f"job process is still running: {pid}")

        with self.vscode_lock:
            self._stop_vscode_session_locked(job_id)
        shutil.rmtree(self.job_dir(job_id))
        return {"deleted": True, "job_id": job_id}

    def jobs_summary(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {status: 0 for status in ("created", "running", "queued", "succeeded", "failed")}
        failure_counts: dict[str, int] = {}
        rerun_required = 0
        progress_values = []
        for job in jobs:
            status = job.get("status") or "created"
            counts[status] = counts.get(status, 0) + 1
            disposition = job.get("failure_disposition") or {}
            category = str(disposition.get("category") or "")
            if status == "failed" and category:
                failure_counts[category] = failure_counts.get(category, 0) + 1
                if disposition.get("rerun_recommended"):
                    rerun_required += 1
            progress = job.get("progress") or {}
            if "percent" in progress:
                progress_values.append(progress.get("percent") or 0)
        return {
            "total": len(jobs),
            "counts": counts,
            "failure_counts": failure_counts,
            "rerun_required": rerun_required,
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
            resource = runner.get("queued_for") or job_stage_resource(job, stage)
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

        for resource, entry in self.lock_file_resource_users(jobs):
            resources.setdefault(resource, {"limit": RESOURCE_LIMITS.get(resource, 1), "running": [], "queued": []})
            if not any(item.get("pid") == entry.get("pid") for item in resources[resource]["running"]):
                resources[resource]["running"].append(entry)

        for info in resources.values():
            info["running"].sort(key=lambda item: item.get("updated_at") or "")
            info["queued"].sort(key=lambda item: (item.get("position") or 999999, item.get("updated_at") or ""))
            info["running_count"] = len(info["running"])
            info["queued_count"] = len(info["queued"])
            info["available"] = max(0, int(info.get("limit") or 0) - len(info["running"]))
        return resources

    def lock_file_resource_users(self, jobs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        lock_dir_value = os.environ.get("VIDEO_ANALYZER_RESOURCE_LOCK_DIR") or str(DEFAULT_LOCK_DIR)
        lock_dir = Path(lock_dir_value)
        if not lock_dir.is_absolute():
            lock_dir = self.repo_root / lock_dir
        if not lock_dir.exists():
            return []

        run_dir_to_job = {str(job.get("run_dir")): job for job in jobs if job.get("run_dir")}
        users = []
        for path in lock_dir.glob("*.lock"):
            try:
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                payload = json.loads(text)
            except Exception:
                continue
            pid = payload.get("pid")
            if not process_alive(pid):
                continue
            resource = str(payload.get("resource") or path.name.split(".", 1)[0])
            owner = str(payload.get("owner") or "")
            job = run_dir_to_job.get(owner)
            users.append(
                (
                    resource,
                    {
                        "job_id": job.get("job_id") if job else "",
                        "video_url": job.get("video_url") if job else owner,
                        "stage": "analyze-core",
                        "stage_label": f"{resource.upper()} 锁",
                        "updated_at": job.get("updated_at") if job else payload.get("acquired_at"),
                        "progress_percent": (job.get("progress") or {}).get("percent") if job else 0,
                        "pid": pid,
                        "owner": owner,
                    },
                )
            )
        return users

    def start_run(self, job_id: str, profile: str | None = None) -> dict[str, Any]:
        job = self.load_job(job_id)
        with self.runner_lock:
            active = self.active_runners.get(job_id)
            if active and active.is_alive():
                return self.public_job(job)
            self.active_runners.pop(job_id, None)

            now = iso_now()
            if profile:
                profile = str(profile).strip()
                profiles = runtime_profile_names()
                if profile not in profiles:
                    raise BridgeError(HTTPStatus.BAD_REQUEST, f"profile must be one of {profiles}")
                options = dict(job.get("options") or {})
                previous_profile = str(options.get("profile") or "")
                if previous_profile != profile:
                    job.setdefault("profile_history", []).append(
                        {
                            "profile": previous_profile or None,
                            "replaced_by": profile,
                            "changed_at": now,
                            "reason": "resume_with_current_profile",
                        }
                    )
                    options["profile"] = profile
                    job["options"] = options
            job["status"] = "running"
            job["updated_at"] = now
            job["runner"] = {
                "status": "running",
                "run_id": uuid.uuid4().hex,
                "started_at": now,
                "updated_at": now,
                "finished_at": None,
                "current_stage": self.next_stage(job),
                "error": None,
                "server_pid": os.getpid(),
                "transition_count": 0,
            }
            self.save_job(job)
            thread = threading.Thread(target=self._run_remaining_stages, args=(job_id,), daemon=True)
            self.active_runners[job_id] = thread
        thread.start()
        return self.public_job(self.load_job(job_id))

    def rerun_from_stage(self, job_id: str, stage: str) -> dict[str, Any]:
        stage = normalize_stage_name(stage)
        job = self.load_job(job_id)
        if stage not in self.stage_order_for_job(job):
            raise BridgeError(HTTPStatus.NOT_FOUND, f"unknown stage: {stage}")
        if self.current_stage(job):
            raise BridgeError(HTTPStatus.CONFLICT, "job already has a running stage")

        stage_order = self.stage_order_for_job(job)
        stage_index = stage_order.index(stage)
        for invalidated_stage in stage_order[stage_index:]:
            job.setdefault("stages", {}).pop(invalidated_stage, None)
            for artifact_name in MODULE_SPECS.get(invalidated_stage, {}).get("produces", []):
                job.setdefault("artifacts", {}).pop(artifact_name, None)
        job["warnings"] = [
            warning
            for warning in job.get("warnings", [])
            if warning.get("stage") not in stage_order[stage_index:]
        ]
        job["status"] = "queued"
        job["updated_at"] = iso_now()
        job["summary"] = self.collect_summary(job)
        self.save_job(job)
        return self.start_run(job_id)

    def stop_job(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        stage = normalize_stage_name((job.get("runner") or {}).get("current_stage") or self.current_stage(job) or self.next_stage(job) or "")
        stage_info = dict((job.get("stages") or {}).get(stage) or {})
        process_info = dict(stage_info.get("process") or job.get("process") or {})
        pid = process_info.get("pid")
        stopped_pids: list[int] = []
        if pid and process_alive(pid):
            stopped_pids = terminate_process_tree(pid)
        now = iso_now()
        if stage:
            stage_info.update(
                {
                    "status": "failed",
                    "finished_at": now,
                    "exit_code": -15 if stopped_pids else None,
                    "error": "stopped by user",
                }
            )
            if process_info:
                process_info["alive"] = False
                process_info["stopped_at"] = now
                process_info["stopped_pids"] = stopped_pids
                stage_info["process"] = process_info
            job.setdefault("stages", {})[stage] = stage_info
        job["status"] = "failed"
        job["updated_at"] = now
        runner = dict(job.get("runner") or {})
        runner.update(
            {
                "status": "failed",
                "current_stage": stage or runner.get("current_stage"),
                "error": "stopped by user",
                "finished_at": now,
                "updated_at": now,
                "server_pid": os.getpid(),
            }
        )
        job["runner"] = runner
        self.save_job(job)
        return {"stopped": True, "job_id": job_id, "stage": stage, "pid": pid, "stopped_pids": stopped_pids}

    def recover_interrupted_jobs(self, auto_start: bool = False) -> None:
        recovered_jobs: list[dict[str, Any]] = []
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
            recovered_jobs.append(recovered)

        if not auto_start:
            return
        for recovered in recovered_jobs:
            recovered_runner = recovered.get("runner") or {}
            if not (auto_start and recovered.get("status") == "queued" and recovered_runner.get("status") == "queued"):
                continue
            if self.is_auto_retry_job(recovered):
                continue
            self.start_run(recovered["job_id"])

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
            self.start_run(job_id)
            started.append(job_id)
            started_resources.add(resource)
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

    def _run_remaining_stages(self, job_id: str) -> None:
        self._run_remaining_stages_serial(job_id)

    def _run_remaining_stages_serial(self, job_id: str) -> None:
        transition_count = 0
        try:
            while True:
                job = self.load_job(job_id)
                stage = self.next_stage(job)
                if not stage:
                    job["status"] = "succeeded"
                    self.update_runner(job, "succeeded", current_stage=None, finished=True)
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
        attempt = max(1, int(previous_stage_info.get("attempt") or 0) + 1)
        log_path, attempt_log_paths = self.prepare_stage_log_attempt(
            job_id,
            stage,
            previous_stage_info,
            attempt,
        )
        stage_info = {
            "status": "running",
            "started_at": iso_now(),
            "finished_at": None,
            "exit_code": None,
            "attempt": attempt,
            "attempt_log_paths": attempt_log_paths,
            "log_path": str(log_path),
            "artifacts": {},
            "queued_for": job_stage_resource(job, stage),
        }
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
                result = self.stage_verify_core(job)
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
        if failure.get("kind") == "transient_resource":
            if int(retry.get("auto_attempts") or 0) < MAX_TRANSIENT_API_RETRIES:
                return TRANSIENT_RESOURCE_REQUEUE_MESSAGE
            return None
        if failure.get("retryable"):
            if int(retry.get("auto_attempts") or 0) < MAX_TRANSIENT_API_RETRIES:
                return TRANSIENT_API_REQUEUE_MESSAGE
            return None
        if normalize_stage_name(stage) != "analyze-core":
            return None
        text = self.exception_text(exc)
        output = getattr(exc, "output", None)
        if not output:
            try:
                text += "\n" + Path(log_path).read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
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
        runner.pop("wait_reason", None)
        if "started_at" not in runner:
            runner["started_at"] = now
        job["runner"] = runner
        job["updated_at"] = now
        self.save_job(job)

    def stage_probe(self, job: dict[str, Any]) -> dict[str, Any]:
        duration = probe_media_duration_seconds(job["media_path"]) if self.uploaded_media_job(job) else probe_duration_seconds(job["video_url"])
        requested_mode = job["options"]["analysis_mode"]
        resolved_mode, reason = resolve_auto_analysis_mode(
            requested_mode=requested_mode,
            duration_seconds=duration,
            focus_prompt=(job.get("options") or {}).get("focus_prompt") or "",
        )
        if self.uploaded_media_job(job) and resolved_mode == "long-talk-fast":
            resolved_mode = "fast"
            reason = f"{reason}; upload media uses direct fast pipeline"
        job["resolved_mode"] = resolved_mode
        job["resolved_mode_reason"] = reason
        return {"artifacts": {"duration_seconds": duration, "resolved_mode": resolved_mode, "resolved_mode_reason": reason}}

    def stage_operation(self, job: dict[str, Any], log_path: str) -> dict[str, Any]:
        return self.stage_analyze_core(job, log_path)

    def stage_prepare(self, job: dict[str, Any], log_path: str, stage_info: dict[str, Any] | None = None) -> dict[str, Any]:
        if not job.get("resolved_mode"):
            self.stage_probe(job)
        if self.uploaded_media_job(job):
            return self.stage_prepare_uploaded_media(job)
        command = self.prepare_command(job)
        result = self.run_command(command, log_path, on_start=self.record_stage_process(job, "prepare", stage_info))
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        video_path = parse_prefixed_path(text, "[download] video:")
        page_context = parse_prefixed_path(text, "[download] context:")
        if not video_path:
            video_path = parse_prefixed_path(text, "[download] reusing")
        if video_path and not page_context:
            page_context = str(Path(video_path).parent / "page_context.md")
        if not video_path or not page_context:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, "prepare stage did not produce video/context paths")
        job["video_path"] = str(self.resolve_output_path(video_path))
        job["page_context_path"] = str(self.resolve_output_path(page_context))
        job["video_dir"] = str(Path(job["video_path"]).parent)
        title = self.resolve_job_title(job)
        if title:
            job["title"] = title
        artifacts = {
            "video_path": job["video_path"],
            "page_context": job["page_context_path"],
            "video_dir": job["video_dir"],
            "title": title,
            "command": command,
        }
        return {"artifacts": artifacts, "stdout_tail": result["stdout_tail"]}

    def stage_analyze_core(self, job: dict[str, Any], log_path: str, stage_info: dict[str, Any] | None = None) -> dict[str, Any]:
        if not job.get("resolved_mode"):
            self.stage_probe(job)
        command = self.operation_command(job)
        jetson_ray = self.ensure_jetson_ray_ready(command, log_path)
        run_kwargs = {"on_start": self.record_stage_process(job, "analyze-core", stage_info)}
        if jetson_ray:
            run_kwargs["append_log"] = True
        run_kwargs["env_overrides"] = self.stage_failure_env(stage_info)
        result = self.run_command(command, log_path, **run_kwargs)
        run_dir = str(job.get("run_dir") or "") if self.uploaded_media_job(job) else parse_run_dir(Path(log_path).read_text(encoding="utf-8", errors="replace"))
        if not run_dir:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, "operation stage did not print a run directory")
        job["run_dir"] = str(self.resolve_output_path(run_dir))
        run_dir_path = Path(job["run_dir"])
        artifacts = {
            "run_dir": job["run_dir"],
            "command": command,
            **({"jetson_ray": jetson_ray} if jetson_ray else {}),
            **self.collect_core_artifacts(run_dir_path),
        }
        generation_error = self.core_manual_generation_error(run_dir_path)
        if generation_error:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, generation_error)
        actual_template = self.audio_prompt_template_actual(run_dir_path)
        if actual_template:
            job["prompt_template_actual"] = actual_template
        warning = self.core_quality_warning(run_dir_path)
        if warning:
            self.add_warning(job, "analyze-core", warning)
        return {"artifacts": artifacts, "stdout_tail": result["stdout_tail"]}

    def ensure_jetson_ray_ready(self, command: list[str], log_path: str) -> dict[str, Any] | None:
        if "--jetson-frame-backend" not in command:
            return None
        backend_index = command.index("--jetson-frame-backend") + 1
        if backend_index >= len(command) or command[backend_index] != "ray":
            return None
        script = self.repo_root / "tools" / "start_jetson_frame_ray.sh"
        if not script.is_file():
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, f"missing Jetson Ray startup script: {script}")
        result = subprocess.run(
            [str(script)],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            env=operation_env(),
        )
        preflight_log = "\n".join(
            line
            for line in (
                "[jetson-ray] ensuring cluster readiness",
                result.stdout.strip(),
                result.stderr.strip(),
            )
            if line
        )
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(preflight_log + "\n", encoding="utf-8")
        if result.returncode != 0:
            error = subprocess.CalledProcessError(result.returncode, [str(script)])
            error.output = preflight_log
            raise error
        return {"command": [str(script)], "stdout_tail": tail_lines(preflight_log)}

    def stage_verify_core(self, job: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.require_run_dir(job)
        generation_error = self.core_manual_generation_error(run_dir)
        if generation_error:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, generation_error)
        missing = self.missing_core_artifacts(run_dir)
        if missing:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, f"missing core artifact(s): {', '.join(missing)}")
        warning = self.core_quality_warning(run_dir)
        warnings = []
        if warning:
            warnings.append(self.add_warning(job, "verify-core", warning))
        return {"artifacts": {"required": ["analysis.json", "operation_manual.md|operation_manual.quality_failed.md", "manual_evidence.md"], "missing": [], "warnings": warnings}}

    def stage_deep_v2(self, job: dict[str, Any], log_path: str, stage_info: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.require_run_dir(job)
        outputs = (
            run_dir / "docs_analysis_chapters" / "knowledge_notes_v2.md",
            run_dir / "docs_analysis_chapters" / "deep_report_v2.md",
        )
        if all(path.is_file() and path.stat().st_size > 0 for path in outputs):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("[docs] reusing chapter documents from multidoc\n", encoding="utf-8")
            return {
                "artifacts": {
                    "stage": "deep-v2",
                    "reused_from": "multidoc",
                    "knowledge_notes_v2": str(outputs[0]),
                    "deep_report_v2": str(outputs[1]),
                },
                "stdout_tail": ["[docs] reusing chapter documents from multidoc"],
            }
        return self.run_command_stage(job, "deep-v2", self.deep_v2_command(job), log_path, stage_info)

    def skipped_stage_outputs_incomplete(self, job: dict[str, Any], stage: str) -> bool:
        if stage not in {"multidoc", "deep-v2"}:
            return False
        run_dir = self.require_run_dir(job)
        expected = {
            "multidoc": (
                run_dir / "docs_analysis" / "analysis.json",
                run_dir / "docs_analysis" / "knowledge_notes.md",
                run_dir / "docs_analysis" / "deep_report.md",
                run_dir / "docs_analysis" / "operation_manual_review.md",
            ),
            "deep-v2": (
                run_dir / "docs_analysis_chapters" / "knowledge_notes_v2.md",
                run_dir / "docs_analysis_chapters" / "deep_report_v2.md",
            ),
        }
        return any(not path.is_file() or path.stat().st_size == 0 for path in expected[stage])

    def run_command_stage(
        self,
        job: dict[str, Any],
        stage: str,
        command: list[str],
        log_path: str,
        stage_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage in {"study-guide", "multidoc", "deep-v2", "evidence-review", "web-evidence"}:
            command = self.local_text_command(job, command)
        result = self.run_command(
            command,
            log_path,
            on_start=self.record_stage_process(job, stage, stage_info),
            env_overrides=self.stage_failure_env(stage_info),
        )
        summary = self.collect_summary(job)
        artifacts = {
            "stage": stage,
            "command": command,
            **summary,
        }
        if stage == "multidoc":
            multidoc_summary = summary.get("multidoc") or {}
            if multidoc_summary.get("analysis"):
                artifacts["docs_analysis"] = multidoc_summary["analysis"]
            artifacts["generation_metrics"] = multidoc_summary.get("metrics") or {}
        return {
            "artifacts": artifacts,
            "stdout_tail": result["stdout_tail"],
        }

    def local_text_command(self, job: dict[str, Any], command: list[str]) -> list[str]:
        return [
            sys.executable,
            "tools/run_local_model_stage.py",
            "--stage",
            "text",
            "--config",
            "config",
            "--profile",
            job["options"].get("profile") or DEFAULT_PROFILE,
            "--",
            *command,
        ]

    def run_command(
        self,
        command: list[str],
        log_path: str,
        on_start: Any | None = None,
        append_log: bool = False,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        env = operation_env()
        env.update(env_overrides or {})
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("a" if append_log else "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            if on_start:
                on_start(process)
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
        if self.uploaded_media_job(job):
            return self.uploaded_media_operation_command(job)
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
            self.append_default_frame_extractor_options(command, job)
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
            self.append_default_frame_extractor_options(command, job)
            if resolved_mode == "operation-fast":
                command.extend(["--vl-frame-policy", "auto", "--min-vl-frames", "8", "--max-vl-frames", "16"])
        command.append("--resume-existing-core")
        self.append_url_options(command, opts)
        return command

    def uploaded_media_operation_command(self, job: dict[str, Any]) -> list[str]:
        opts = job["options"]
        media_path = Path(str(job.get("media_path") or job.get("video_path") or ""))
        context_path = Path(str(job.get("page_context_path") or ""))
        run_dir = Path(str(job.get("run_dir") or ""))
        if not media_path.is_file():
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"uploaded media file does not exist: {media_path}")
        if not context_path.is_file():
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"uploaded media context does not exist: {context_path}")
        if not run_dir:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "uploaded media job is missing run_dir")
        if str(job.get("audio_pipeline_kind") or "analysis") == "transcription":
            return [
                os.environ.get("PYTHON") or sys.executable,
                "tools/run_audio_transcription.py",
                str(media_path),
                "--output",
                str(run_dir),
                "--config",
                "config",
                "--profile",
                opts["profile"],
                "--source-name",
                str(job.get("source_name") or media_path.name),
                "--asr-provider",
                str(job.get("asr_provider") or "firered_3dspeaker"),
            ]
        command = [
            os.environ.get("PYTHON") or sys.executable,
            "tools/run_audio_template_analysis.py",
            str(media_path),
            "--output",
            str(run_dir),
            "--config",
            "config",
            "--profile",
            opts["profile"],
            "--source-name",
            str(job.get("source_name") or media_path.name),
        ]
        if opts.get("template_id"):
            command.extend(["--template-id", opts["template_id"]])
        if opts.get("focus_prompt"):
            command.extend(["--focus-prompt", opts["focus_prompt"]])
        if job.get("provided_transcript"):
            command.extend(["--transcript-json", str(job["provided_transcript_path"])])
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

    def stage_prepare_uploaded_media(self, job: dict[str, Any]) -> dict[str, Any]:
        media_path = Path(str(job.get("media_path") or job.get("video_path") or ""))
        page_context = Path(str(job.get("page_context_path") or ""))
        if not media_path.is_file():
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"uploaded media file does not exist: {media_path}")
        if not job.get("provided_transcript") and media_path.suffix.lower() not in MEDIA_EXTENSIONS:
            raise BridgeError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "uploaded media file is not supported")
        if not page_context.is_file():
            page_context.write_text(upload_page_context(str(job.get("source_name") or media_path.name), media_path), encoding="utf-8")
        job["video_path"] = str(media_path)
        job["media_path"] = str(media_path)
        job["page_context_path"] = str(page_context)
        job["video_dir"] = str(media_path.parent)
        job["run_dir"] = str(media_path.parent / job["options"]["run_name"])
        title = self.resolve_job_title(job)
        artifacts = {
            "video_path": job["video_path"],
            "page_context": job["page_context_path"],
            "video_dir": job["video_dir"],
            "title": title,
            "command": ["prepare-uploaded-media", str(media_path)],
        }
        return {"artifacts": artifacts, "stdout_tail": []}

    def append_url_options(self, command: list[str], opts: dict[str, Any]) -> None:
        if opts.get("keep_existing"):
            command.append("--keep-existing")
        if opts.get("cookies_from_browser"):
            command.extend(["--cookies-from-browser", opts["cookies_from_browser"]])
        if opts.get("download_device") and opts.get("download_device") != "local":
            command.extend(["--download-device", opts["download_device"]])
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
        if opts.get("focus_prompt"):
            command.extend(["--focus-prompt", opts["focus_prompt"]])

    def append_default_frame_extractor_options(self, command: list[str], job: dict[str, Any] | None = None) -> None:
        profile_name = str(((job or {}).get("options") or {}).get("profile") or "")
        profile = (runtime_config().get("runtime_profiles") or {}).get(profile_name) or {}
        command.extend(
            [
                "--frame-extractor",
                str(profile.get("frame_extractor") or os.environ.get("VIDEO_LINK_FRAME_EXTRACTOR", DEFAULT_FRAME_EXTRACTOR)),
                "--jetson-frame-hosts",
                str(
                    profile.get("jetson_frame_hosts")
                    or os.environ.get("VIDEO_LINK_JETSON_FRAME_HOSTS", os.environ.get("JETSON_FRAME_HOSTS", DEFAULT_JETSON_FRAME_HOSTS))
                ),
                "--jetson-frame-backend",
                str(profile.get("jetson_frame_backend") or os.environ.get("VIDEO_LINK_JETSON_FRAME_BACKEND", DEFAULT_JETSON_FRAME_BACKEND)),
                "--jetson-sample-fps",
                str(profile.get("jetson_sample_fps") or os.environ.get("VIDEO_LINK_JETSON_SAMPLE_FPS", DEFAULT_JETSON_SAMPLE_FPS)),
                "--jetson-require-hwdec",
            ]
        )

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

    def study_guide_command(self, job: dict[str, Any]) -> list[str]:
        return [
            sys.executable,
            "tools/run_study_guide.py",
            str(self.require_run_dir(job)),
            "--profile",
            job["options"].get("profile") or DEFAULT_PROFILE,
            "--skip-review",
        ]

    def evidence_review_command(self, job: dict[str, Any]) -> list[str]:
        return [
            sys.executable,
            "tools/run_study_guide.py",
            str(self.require_run_dir(job)),
            "--profile",
            job["options"].get("profile") or DEFAULT_PROFILE,
        ]

    def web_evidence_command(self, job: dict[str, Any]) -> list[str]:
        return [
            sys.executable,
            "-m",
            "video_analyzer.web_evidence",
            str(self.require_run_dir(job)),
            "--profile",
            job["options"].get("profile") or DEFAULT_PROFILE,
        ]

    def qa_index_command(self, job: dict[str, Any]) -> list[str]:
        return [
            sys.executable,
            "-m",
            "video_analyzer.doc_chat",
            str(self.require_run_dir(job)),
            "--profile",
            job["options"].get("profile") or DEFAULT_PROFILE,
            "--build-index",
        ]

    def export_command(self, job: dict[str, Any]) -> list[str]:
        return ["tools/export_video_docs.sh", str(self.require_run_dir(job)), "--final-only", "--jobs", "3"]

    def final_publish_command(self, job: dict[str, Any]) -> list[str]:
        self.ensure_publish_not_blocked(job)
        command = [
            "tools/run_video_doc_final_publish.sh",
            str(self.require_run_dir(job)),
            "--profile",
            job["options"].get("profile") or DEFAULT_PROFILE,
            "--jobs",
            "3",
            "--finalize-only",
            "--skip-pdf",
            "--skip-send",
        ]
        if job["options"].get("skip_images") or not BAOYU_IMAGE_GENERATION_ENABLED:
            command.append("--skip-images")
        return command

    def image_prompts_command(self, job: dict[str, Any]) -> list[str]:
        self.ensure_publish_not_blocked(job)
        if not BAOYU_PROMPT_SCRIPT.exists():
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, f"missing Baoyu prompt script: {BAOYU_PROMPT_SCRIPT}")
        return [sys.executable, str(BAOYU_PROMPT_SCRIPT), str(self.require_run_dir(job))]

    def ensure_publish_not_blocked(self, job: dict[str, Any]) -> None:
        run_dir = self.require_run_dir(job)
        decision_path = run_dir / "publish_decision.json"
        if not decision_path.is_file():
            return
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if decision.get("status") == "blocked":
            reason = decision.get("reason") or "evidence review blocked publishing"
            raise BridgeError(HTTPStatus.CONFLICT, f"publish blocked by evidence gate: {reason}")

    def mark_stage_skipped(self, job: dict[str, Any], stage: str, reason: str, continue_runner: bool = False) -> dict[str, Any]:
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
        job["status"] = "succeeded" if self.next_stage(job) is None else "running"
        job["updated_at"] = iso_now()
        job["summary"] = self.collect_summary(job)
        self.finalize_stage_runner(job, stage, job["stages"][stage], continue_runner=continue_runner)
        self.save_job(job)
        return self.public_job(job)

    def finalize_stage_runner(
        self,
        job: dict[str, Any],
        stage: str,
        stage_info: dict[str, Any],
        continue_runner: bool = False,
    ) -> None:
        next_stage = self.next_stage(job)
        now = job.get("updated_at") or iso_now()
        runner = dict(job.get("runner") or {})
        runner["updated_at"] = now
        runner["server_pid"] = os.getpid()
        runner.pop("wait_reason", None)
        if job["status"] == "failed":
            runner["status"] = "failed"
            runner["current_stage"] = stage
            runner["queued_for"] = job_stage_resource(job, stage)
            runner["error"] = stage_info.get("error")
            runner["finished_at"] = now
        elif job["status"] == "queued":
            runner["status"] = "queued"
            runner["current_stage"] = stage
            runner["queued_for"] = job_stage_resource(job, stage)
            runner["error"] = stage_info.get("retry_reason")
            runner.pop("finished_at", None)
        elif next_stage is None:
            job["status"] = "succeeded"
            runner["status"] = "succeeded"
            runner["current_stage"] = None
            runner["queued_for"] = None
            runner["error"] = None
            runner["finished_at"] = now
        elif continue_runner:
            job["status"] = "running"
            runner["status"] = "running"
            runner["current_stage"] = next_stage
            runner["queued_for"] = job_stage_resource(job, next_stage)
            runner["error"] = None
            runner.pop("finished_at", None)
        else:
            resource = job_stage_resource(job, next_stage)
            next_info = dict((job.get("stages") or {}).get(next_stage) or {})
            next_info.setdefault("log_path", str(self.stage_log_path(job["job_id"], next_stage)))
            next_info["status"] = "queued"
            next_info["queued_at"] = now
            next_info["queued_for"] = resource
            next_info.pop("finished_at", None)
            next_info.pop("exit_code", None)
            next_info.pop("process", None)
            job.setdefault("stages", {})[next_stage] = next_info
            job["status"] = "queued"
            runner["status"] = "queued"
            runner["current_stage"] = next_stage
            runner["queued_for"] = resource
            runner["error"] = None
            runner.pop("finished_at", None)
        job["runner"] = runner

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
            artifact_store["resolved_mode_reason"] = {
                "value": job.get("resolved_mode_reason"),
                "module": stage,
                "updated_at": iso_now(),
            }
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
            "frame_dedup_audit": [run_dir / "frame_dedup_audit.json"],
            "visual_review": [run_dir / "visual_review.html"],
            "run_manifest": [run_dir / "RUN_MANIFEST.md"],
        }
        for name, paths in candidate_paths.items():
            path = next((candidate for candidate in paths if candidate.exists()), None)
            if path:
                artifacts[name] = str(path)
        study_candidates = {
            "study_guide": run_dir / "study_guide.json",
            "evidence_gaps": run_dir / "evidence_gaps.json",
            "evidence_triage": run_dir / "evidence_triage.json",
            "evidence_review": run_dir / "evidence_review.json",
            "web_evidence": run_dir / "web_evidence.json",
            "publish_decision": run_dir / "publish_decision.json",
        }
        for name, path in study_candidates.items():
            if path.exists():
                artifacts[name] = str(path)
        if analysis_path.exists():
            try:
                payload = json.loads(analysis_path.read_text(encoding="utf-8"))
                metadata = payload.get("metadata") or {}
                ocr_keyframes = metadata.get("ocr_keyframes") or {}
                artifacts["frames"] = artifacts.get("frames") or metadata.get("frame_extraction", {}).get("frame_manifest")
                artifacts["transcript"] = artifacts.get("transcript") or metadata.get("transcript_markdown")
                artifacts["core_counts"] = {
                    "frames_extracted": metadata.get("frames_extracted"),
                    "scan_frames": ocr_keyframes.get("scan_frames_count"),
                    "ocr_candidate_frames": ocr_keyframes.get("ocr_candidate_frames_count"),
                    "ocr_keyframes": ocr_keyframes.get("ocr_frames_count"),
                    "ocr_text_events": ocr_keyframes.get("ocr_text_events_count"),
                    "ocr_events": len(payload.get("ocr_events") or []),
                    "frame_analyses": len(payload.get("frame_analyses") or []),
                    "frame_dedup_audit": (metadata.get("frame_dedup_audit") or {}).get("summary", {}),
                    "visual_review": metadata.get("visual_review") or {},
                    "run_manifest": metadata.get("run_manifest") or {},
                    "timings": metadata.get("timings") or {},
                }
            except Exception:
                pass
        return artifacts

    def core_manual_path(self, run_dir: Path) -> Path | None:
        for path in (run_dir / "operation_manual.md", run_dir / "operation_manual.quality_failed.md"):
            if path.is_file() and path.stat().st_size > 0:
                return path
        return None

    def missing_core_artifacts(self, run_dir: Path) -> list[str]:
        missing = []
        if not (run_dir / "analysis.json").is_file() or (run_dir / "analysis.json").stat().st_size <= 0:
            missing.append("analysis.json")
        if not self.core_manual_path(run_dir):
            missing.append("operation_manual.md or operation_manual.quality_failed.md")
        if not (run_dir / "manual_evidence.md").is_file() or (run_dir / "manual_evidence.md").stat().st_size <= 0:
            missing.append("manual_evidence.md")
        if "analysis.json" not in missing:
            try:
                json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
            except Exception:
                missing.append("analysis.json (invalid JSON)")
        core_errors = self.core_analysis_errors(run_dir)
        if core_errors:
            missing.append(f"core analysis errors: {len(core_errors)} VL/resource failure(s)")
        return missing

    def core_quality_warning(self, run_dir: Path) -> str | None:
        quality_failed_path = run_dir / "operation_manual.quality_failed.md"
        if quality_failed_path.is_file() and not (run_dir / "operation_manual.md").is_file():
            return f"operation manual failed quality gate; review artifact: {quality_failed_path}"
        core_errors = self.core_analysis_errors(run_dir)
        if core_errors:
            sample = "; ".join(core_errors[:3])
            return f"core analysis contains VL/resource failure(s): {sample}"
        return None

    def core_manual_generation_error(self, run_dir: Path) -> str | None:
        quality_failed_path = run_dir / "operation_manual.quality_failed.md"
        if not quality_failed_path.is_file() or (run_dir / "operation_manual.md").is_file():
            return None
        try:
            text = quality_failed_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        match = re.search(r"^Error generating operation manual:\s*(.+)$", text, flags=re.MULTILINE)
        if not match:
            return None
        return f"operation manual generation failed: {match.group(1).strip()}"

    def core_analysis_errors(self, run_dir: Path) -> list[str]:
        candidates = [
            run_dir / "analysis.json",
            run_dir / "orin" / "frame_analyses.json",
            run_dir / "orin" / "visual_events.json",
        ]
        errors: list[str] = []
        seen: set[str] = set()
        for path in candidates:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for text in iter_nested_strings(payload):
                if any(pattern in text for pattern in CORE_ANALYSIS_ERROR_PATTERNS):
                    normalized = " ".join(text.split())
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        errors.append(normalized[:240])
        return errors

    def core_markdown_available_for_job(self, job: dict[str, Any]) -> bool:
        run_dir = self.discover_run_dir(job)
        return bool(run_dir and self.core_manual_path(run_dir))

    def stage_can_soft_fail(
        self,
        job: dict[str, Any],
        stage: str,
        failure: dict[str, Any] | None = None,
    ) -> bool:
        if str((failure or {}).get("kind") or "").startswith("permanent_"):
            return False
        return normalize_stage_name(stage) in SOFT_FAILURE_STAGES and self.core_markdown_available_for_job(job)

    def runner_failure_can_finish_with_warning(self, job: dict[str, Any]) -> bool:
        stage = normalize_stage_name((job.get("runner") or {}).get("current_stage") or self.current_stage(job) or self.next_stage(job) or "")
        return stage in SOFT_FAILURE_STAGES and self.core_markdown_available_for_job(job)

    def add_warning(self, job: dict[str, Any], stage: str, message: str) -> dict[str, Any]:
        warning = {"stage": normalize_stage_name(stage), "message": str(message), "updated_at": iso_now()}
        warnings = list(job.get("warnings") or [])
        if not any(item.get("stage") == warning["stage"] and item.get("message") == warning["message"] for item in warnings):
            warnings.append(warning)
        job["warnings"] = warnings
        return warning

    def discover_run_dir(self, job: dict[str, Any]) -> Path | None:
        candidates: list[Any] = [
            job.get("run_dir"),
            ((job.get("artifacts") or {}).get("run_dir") or {}).get("value"),
        ]
        run_name = (job.get("options") or {}).get("run_name")
        for base in (job.get("video_dir"), Path(job["video_path"]).parent if job.get("video_path") else None):
            if base and run_name:
                candidates.append(Path(str(base)) / str(run_name))
        for value in candidates:
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.repo_root / path
            path = path.resolve()
            if path.is_dir():
                return path
        return None

    def core_progress_snapshot(self, job: dict[str, Any]) -> dict[str, Any] | None:
        run_dir = self.discover_run_dir(job)
        if not run_dir:
            return None

        snapshot: dict[str, Any] = {}
        progress_path = run_dir / ANALYSIS_PROGRESS_FILENAME
        if progress_path.is_file():
            try:
                loaded = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    snapshot.update(loaded)
            except Exception:
                pass

        inferred = infer_core_progress_from_artifacts(run_dir)
        if is_later_core_step(inferred.get("current_step"), snapshot.get("current_step")):
            snapshot.update(inferred)
        elif inferred and not snapshot:
            snapshot.update(inferred)
        return snapshot or None

    def core_artifacts_complete(self, run_dir: Path) -> bool:
        if self.missing_core_artifacts(run_dir):
            return False
        return True

    def reconcile_completed_core_stage(
        self,
        job: dict[str, Any],
        stage_info: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        run_dir = self.discover_run_dir(job)
        if not run_dir or not self.core_artifacts_complete(run_dir):
            return None

        now = iso_now()
        job["run_dir"] = str(run_dir)
        stage_info = dict(stage_info or (job.get("stages") or {}).get("analyze-core") or {})
        artifacts = {"run_dir": str(run_dir), **self.collect_core_artifacts(run_dir)}
        actual_template = self.audio_prompt_template_actual(run_dir)
        if actual_template:
            job["prompt_template_actual"] = actual_template
        warning = self.core_quality_warning(run_dir)
        if warning:
            self.add_warning(job, "analyze-core", warning)
        stage_info.update(
            {
                "status": "succeeded",
                "exit_code": 0,
                "finished_at": stage_info.get("finished_at") or now,
                "recovered_at": now,
                "recovery_reason": "core artifacts already exist in run_dir",
                "artifacts": {**dict(stage_info.get("artifacts") or {}), **artifacts},
            }
        )
        stage_info.pop("process", None)
        stage_info.pop("error", None)
        stage_info.pop("retry_reason", None)
        job.setdefault("stages", {})["analyze-core"] = stage_info
        self.update_job_artifacts(job, "analyze-core", artifacts)

        job["summary"] = self.collect_summary(job)
        job["updated_at"] = now
        next_stage = self.next_stage(job)
        if next_stage:
            self.mark_stage_queued(job, next_stage, job_stage_resource(job, next_stage))
            job["summary"] = self.collect_summary(job)
            self.save_job(job)
            return job

        runner = dict(job.get("runner") or {})
        runner["status"] = "succeeded"
        runner["current_stage"] = None
        runner["queued_for"] = None
        runner["error"] = None
        runner["updated_at"] = now
        runner["server_pid"] = os.getpid()
        runner["finished_at"] = now
        job["runner"] = runner
        job["status"] = "succeeded"
        self.save_job(job)
        return job

    def resolve_output_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def ensure_dependencies(self, job: dict[str, Any], stage: str) -> None:
        stage = normalize_stage_name(stage)
        if stage == "probe":
            return
        stage_order = self.stage_order_for_job(job)
        stage_index = stage_order.index(stage)
        for previous in stage_order[:stage_index]:
            previous_status = job["stages"].get(previous, {}).get("status")
            if previous_status not in {"succeeded", "skipped"}:
                if previous in SOFT_FAILURE_STAGES and self.core_markdown_available_for_job(job):
                    continue
                raise BridgeError(HTTPStatus.CONFLICT, f"stage {previous} must succeed before {stage}")

    def require_run_dir(self, job: dict[str, Any]) -> Path:
        run_dir = job.get("run_dir")
        if not run_dir:
            raise BridgeError(HTTPStatus.CONFLICT, "run_dir is not available yet")
        path = Path(run_dir).expanduser().resolve()
        if not path.is_dir():
            raise BridgeError(HTTPStatus.CONFLICT, f"run_dir does not exist: {path}")
        return path

    def open_run_dir(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        if job.get("status") != "succeeded":
            raise BridgeError(HTTPStatus.CONFLICT, "job must be succeeded before opening generated resources")
        run_dir = self.require_run_dir(job)
        code_bin = shutil.which("code")
        if not code_bin:
            raise BridgeError(HTTPStatus.SERVICE_UNAVAILABLE, "code command is not available on PATH")
        command = [code_bin, str(run_dir)]
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"opened": True, "run_dir": str(run_dir), "command": ["code", str(run_dir)]}

    def start_vscode_session(
        self,
        job_id: str,
        public_host: str | None = None,
        restart: bool = False,
    ) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        code_server = find_code_server_binary()
        if not code_server:
            raise BridgeError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Web VS Code server is not available; install code-server/openvscode-server or keep npx available",
            )

        with self.vscode_lock:
            existing = self.vscode_sessions.get(job_id)
            if restart and existing:
                stop_managed_vscode_sessions(self.jobs_dir)
                self.vscode_sessions.clear()
                existing = None
            if existing and process_alive(existing.get("pid")):
                existing["run_dir"] = str(run_dir)
                return self.public_vscode_session(existing, public_host)

            discovered = discover_global_vscode_session(self.jobs_dir)
            if discovered:
                discovered["job_id"] = job_id
                discovered["run_dir"] = str(run_dir)
                self.vscode_sessions[job_id] = discovered
                return self.public_vscode_session(discovered, public_host)

            stop_managed_vscode_sessions(self.jobs_dir)
            port = allocate_vscode_port(VSCODE_PORT)
            host = os.environ.get("VIDEO_LINK_VSCODE_BIND_HOST", "0.0.0.0")
            data_dir = self.jobs_dir / "_vscode-server-data"
            user_dir = self.jobs_dir / "_vscode-user-data"
            extensions_dir = self.jobs_dir / "_vscode-extensions"
            data_dir.mkdir(parents=True, exist_ok=True)
            user_dir.mkdir(parents=True, exist_ok=True)
            extensions_dir.mkdir(parents=True, exist_ok=True)
            socket_path = user_dir / "code-server-ipc.sock"
            if socket_path.exists():
                socket_path.unlink()
            command = build_code_server_command(code_server, host, port, user_dir, extensions_dir, None)
            log_path = self.jobs_dir / "_vscode-server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("wb") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=self.repo_root,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            session = {
                "job_id": job_id,
                "pid": process.pid,
                "port": port,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "server": code_server.get("server"),
                "command": command[:3],
                "started_at": iso_now(),
            }
            self.vscode_sessions[job_id] = session
            return self.public_vscode_session(session, public_host)

    def stop_vscode_session(self, job_id: str) -> dict[str, Any]:
        with self.vscode_lock:
            stopped = self._stop_vscode_session_locked(job_id)
        return {"stopped": stopped, "job_id": job_id}

    def _stop_vscode_session_locked(self, job_id: str) -> bool:
        session = self.vscode_sessions.pop(job_id, None)
        if not session:
            return False
        pid = session.get("pid")
        if pid and process_alive(pid):
            try:
                os.killpg(os.getpgid(int(pid)), 15)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    os.kill(int(pid), 15)
                except Exception:
                    pass
        return True

    def public_vscode_session(self, session: dict[str, Any], public_host: str | None = None) -> dict[str, Any]:
        host = public_vscode_host(public_host)
        port = int(session["port"])
        query = urlencode({"folder": str(session.get("run_dir") or "")})
        url = f"http://{host}:{port}/?{query}" if query else f"http://{host}:{port}/"
        return {
            "ready": process_alive(session.get("pid")),
            "url": url,
            "pid": session.get("pid"),
            "port": port,
            "run_dir": session.get("run_dir"),
            "server": session.get("server"),
            "started_at": session.get("started_at"),
            "log_path": session.get("log_path"),
        }

    def vscode_preview_metadata(self, job: dict[str, Any], public_host: str | None = None) -> dict[str, Any]:
        run_dir = job.get("run_dir") or (((job.get("artifacts") or {}).get("run_dir") or {}).get("value"))
        session = self.vscode_sessions.get(job["job_id"])
        if not session and run_dir:
            try:
                discovered = discover_global_vscode_session(self.jobs_dir)
            except Exception:
                discovered = None
            if discovered:
                discovered["job_id"] = job["job_id"]
                discovered["run_dir"] = str(Path(run_dir).expanduser().resolve())
                self.vscode_sessions[job["job_id"]] = discovered
                session = discovered
        if not session:
            return {"ready": False, "url": None, "run_dir": run_dir}
        return self.public_vscode_session(session, public_host)

    def collect_summary(self, job: dict[str, Any]) -> dict[str, Any]:
        run_dir_value = job.get("run_dir")
        if not run_dir_value:
            return {}
        run_dir = Path(run_dir_value)
        core_counts = {}
        analysis_path = run_dir / "analysis.json"
        if analysis_path.is_file():
            try:
                metadata = (json.loads(analysis_path.read_text(encoding="utf-8")).get("metadata") or {})
                ocr_keyframes = metadata.get("ocr_keyframes") or {}
                core_counts = {
                    "frames_extracted": metadata.get("frames_extracted"),
                    "scan_frames": ocr_keyframes.get("scan_frames_count"),
                    "ocr_candidate_frames": ocr_keyframes.get("ocr_candidate_frames_count"),
                    "ocr_keyframes": ocr_keyframes.get("ocr_frames_count"),
                    "ocr_text_events": ocr_keyframes.get("ocr_text_events_count"),
                    "vl_frames": metadata.get("vl_frames_processed"),
                }
            except Exception:
                core_counts = {}
        qa_summary = self.qa_summary(run_dir)
        return {
            "run_dir": str(run_dir),
            "core_counts": core_counts,
            "study": self.study_summary(run_dir),
            "multidoc": self.multidoc_summary(run_dir),
            "qa": qa_summary,
            "qa_index": qa_summary.get("answer_index"),
            "skill_candidate": self.skill_candidate_summary(run_dir),
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

    def multidoc_summary(self, run_dir: Path) -> dict[str, Any]:
        analysis_path = run_dir / "docs_analysis" / "analysis.json"
        if not analysis_path.is_file():
            return {"available": False, "analysis": None, "metrics": {}}
        try:
            payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "available": False,
                "analysis": str(analysis_path.relative_to(run_dir)),
                "metrics": {},
                "error": "docs_analysis/analysis.json is invalid",
            }
        generation = payload.get("generation") or {}
        checkpoint_value = generation.get("chapter_checkpoints")
        checkpoint_path = str(checkpoint_value or "")
        if checkpoint_path:
            try:
                checkpoint_path = str(Path(checkpoint_path).resolve().relative_to(run_dir.resolve()))
            except (OSError, ValueError):
                pass
        return {
            "available": True,
            "analysis": str(analysis_path.relative_to(run_dir)),
            "chapter_count": generation.get("chapter_count"),
            "resumable": bool(generation.get("resumable")),
            "chapter_checkpoints": checkpoint_path or None,
            "metrics": generation.get("metrics") or {},
        }

    def qa_summary(self, run_dir: Path) -> dict[str, Any]:
        qa_dir = run_dir / QA_DIR_NAME
        index_path = qa_dir / ANSWER_INDEX_NAME
        chunks_path = qa_dir / CHUNKS_NAME
        if not index_path.is_file():
            return {"available": False, "answer_index": None, "source_chunks": None, "warnings": []}
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"available": False, "error": "qa index is invalid", "answer_index": str(index_path.relative_to(run_dir))}
        return {
            "available": chunks_path.is_file(),
            "answer_index": str(index_path.relative_to(run_dir)),
            "source_chunks": str(chunks_path.relative_to(run_dir)) if chunks_path.is_file() else None,
            "source_count": index.get("source_count"),
            "chunk_count": index.get("chunk_count"),
            "generated_at": index.get("generated_at"),
            "warnings": index.get("warnings") or [],
        }

    def skill_candidate_summary(self, run_dir: Path) -> dict[str, Any]:
        try:
            return candidate_summary(run_dir)
        except Exception as exc:
            return {"available": False, "error": str(exc), "warnings": []}

    def study_summary(self, run_dir: Path) -> dict[str, Any]:
        guide_path = run_dir / "study_guide.json"
        gaps_path = run_dir / "evidence_gaps.json"
        decision_path = run_dir / "publish_decision.json"
        summary: dict[str, Any] = {"available": guide_path.is_file()}
        if guide_path.is_file():
            try:
                guide = json.loads(guide_path.read_text(encoding="utf-8"))
                summary.update(
                    {
                        "title": guide.get("title"),
                        "chapter_count": len(guide.get("chapters") or []),
                        "evidence_count": len(guide.get("evidence") or []),
                    }
                )
            except Exception:
                summary["error"] = "study_guide.json is invalid"
        if gaps_path.is_file():
            try:
                gaps = json.loads(gaps_path.read_text(encoding="utf-8"))
                summary["gaps"] = gaps.get("summary") or {}
            except Exception:
                summary["gaps_error"] = "evidence_gaps.json is invalid"
        web_path = run_dir / "web_evidence.json"
        if web_path.is_file():
            try:
                web_evidence = json.loads(web_path.read_text(encoding="utf-8"))
                summary["web_evidence"] = web_evidence.get("summary") or {}
            except Exception:
                summary["web_evidence_error"] = "web_evidence.json is invalid"
        if decision_path.is_file():
            try:
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                summary["publish_decision"] = {
                    "status": decision.get("status"),
                    "reason": decision.get("reason"),
                    "risk_level": decision.get("risk_level"),
                }
            except Exception:
                summary["publish_decision_error"] = "publish_decision.json is invalid"
        return summary

    def preview_video_candidate(self, job: dict[str, Any]) -> Path | None:
        artifacts = job.get("artifacts") or {}
        stages = job.get("stages") or {}
        values = [
            job.get("video_path"),
            (artifacts.get("video_path") or {}).get("value"),
            ((stages.get("prepare") or {}).get("artifacts") or {}).get("video_path"),
        ]
        for value in values:
            if value:
                return self.resolve_output_path(str(value))
        video_id = infer_video_id_from_url(str(job.get("video_url") or ""))
        if video_id:
            video_dir = self.resolve_output_path(str(Path(FALLBACK_OUTPUT_ROOT) / safe_slug(video_id)))
            for extension in sorted(VIDEO_PREVIEW_EXTENSIONS):
                candidate = video_dir / f"video{extension}"
                if candidate.is_file():
                    return candidate
        return None

    def preview_duration_seconds(self, job: dict[str, Any]) -> int | None:
        artifacts = job.get("artifacts") or {}
        probe_artifacts = ((job.get("stages") or {}).get("probe") or {}).get("artifacts") or {}
        for value in (
            job.get("duration_seconds"),
            (artifacts.get("duration_seconds") or {}).get("value"),
            (artifacts.get("duration") or {}).get("value"),
            probe_artifacts.get("duration_seconds"),
            probe_artifacts.get("duration"),
        ):
            if value in (None, ""):
                continue
            try:
                seconds = int(float(value))
            except (TypeError, ValueError):
                continue
            if seconds > 0:
                return seconds
        return None

    def preview_metadata(self, job: dict[str, Any]) -> dict[str, Any]:
        video_path = self.preview_video_candidate(job)
        valid_extension = bool(video_path and video_path.suffix.lower() in VIDEO_PREVIEW_EXTENSIONS)
        video_ready = bool(valid_extension and video_path and video_path.is_file())
        return {
            "video_ready": video_ready,
            "video_url": f"/api/video-link/jobs/{job['job_id']}/video" if video_ready else None,
            "duration_seconds": self.preview_duration_seconds(job),
        }

    def source_player_metadata(self, job: dict[str, Any]) -> dict[str, Any]:
        if self.uploaded_media_job(job):
            return {
                "source_url": None,
                "provider": "local_media",
                "can_embed": False,
                "supports_timestamp": False,
                "embed_url": None,
                "watch_url": None,
                "duration_seconds": self.preview_duration_seconds(job),
            }
        source_url = str(job.get("video_url") or "").strip()
        duration_seconds = self.preview_duration_seconds(job)
        base = {
            "source_url": source_url,
            "provider": "external",
            "can_embed": False,
            "supports_timestamp": bool(source_url),
            "embed_url": None,
            "watch_url": source_url or None,
            "duration_seconds": duration_seconds,
        }
        if not source_url:
            return base

        parsed = urlparse(source_url)
        host = parsed.netloc.lower()
        video_id = infer_video_id_from_url(source_url)
        if video_id and ("youtube.com" in host or "youtu.be" in host):
            base.update(
                {
                    "provider": "youtube",
                    "can_embed": True,
                    "supports_timestamp": True,
                    "embed_url": f"https://www.youtube.com/embed/{quote(video_id)}",
                    "watch_url": f"https://www.youtube.com/watch?v={quote(video_id)}",
                }
            )
            return base
        if video_id and ("bilibili.com" in host or "b23.tv" in host):
            base.update(
                {
                    "provider": "bilibili",
                    "can_embed": True,
                    "supports_timestamp": True,
                    "embed_url": f"https://player.bilibili.com/player.html?bvid={quote(video_id)}",
                    "watch_url": f"https://www.bilibili.com/video/{quote(video_id)}",
                }
            )
        return base

    def frame_time_map(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        manifest_path = run_dir / "frames_manifest.json"
        if not manifest_path.is_file():
            return {"available": False, "frames": {}, "count": 0}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, "frame manifest is invalid") from exc

        frames: dict[str, dict[str, Any]] = {}
        for item in payload.get("frames") or []:
            if not isinstance(item, dict):
                continue
            frame_number = item.get("frame_number", item.get("number"))
            timestamp = item.get("timestamp", item.get("timestamp_sec"))
            frame_path = item.get("path") or item.get("frame_path")
            if frame_number is None or timestamp is None:
                continue
            try:
                number = int(frame_number)
                seconds = float(timestamp)
            except (TypeError, ValueError):
                continue
            entry = {
                "frame_number": number,
                "timestamp_sec": seconds,
                "timestamp_label": format_seconds_label(seconds),
            }
            keys = set()
            for extension in ("jpg", "jpeg", "png", "webp"):
                keys.add(f"manual_assets/frame_{number:03d}.{extension}")
                keys.add(f"manual_assets/frame_{number}.{extension}")
            if frame_path:
                path_value = str(frame_path).replace("\\", "/")
                keys.update({path_value, Path(path_value).name})
            for key in keys:
                if key:
                    frames[key] = entry
        return {"available": True, "frames": frames, "count": len(frames)}

    def preview_video_file(self, job_id: str) -> tuple[Path, str | None]:
        job = self.load_job(job_id)
        video_path = self.preview_video_candidate(job)
        if not video_path:
            raise BridgeError(HTTPStatus.CONFLICT, "video is not available yet")
        if video_path.suffix.lower() not in VIDEO_PREVIEW_EXTENSIONS:
            raise BridgeError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "preview file is not a supported video type")
        if not video_path.is_file():
            raise BridgeError(HTTPStatus.NOT_FOUND, f"video file does not exist: {video_path}")
        return video_path, mimetypes.guess_type(str(video_path))[0]

    def resource_file(self, job_id: str, relative_path: str) -> tuple[Path, str | None]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        value = str(relative_path or "").strip()
        if not value:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "resource path is required")
        candidate = Path(value)
        if candidate.is_absolute():
            raise BridgeError(HTTPStatus.BAD_REQUEST, "resource path must be relative")
        path = (run_dir / candidate).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise BridgeError(HTTPStatus.FORBIDDEN, "resource path escapes run_dir") from exc
        if not path.is_file():
            raise BridgeError(HTTPStatus.NOT_FOUND, "resource file is not available")
        return path, mimetypes.guess_type(str(path))[0]

    def study_guide(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        path = run_dir / "study_guide.json"
        if not path.is_file():
            raise BridgeError(HTTPStatus.NOT_FOUND, "study guide is not available")
        try:
            guide = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, "study guide is invalid") from exc
        for name in ("evidence_gaps", "evidence_triage", "evidence_review", "web_evidence", "publish_decision"):
            sidecar = run_dir / f"{name}.json"
            if sidecar.is_file():
                try:
                    guide[name] = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    guide[name] = {"status": "invalid"}
        return guide

    def qa_index(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        summary = self.qa_summary(run_dir)
        if not summary.get("available"):
            raise BridgeError(HTTPStatus.NOT_FOUND, summary.get("error") or "QA index is not available")
        return summary

    def qa_history(self, job_id: str, limit: int = 50) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        history_dir = run_dir / QA_DIR_NAME / "chat_history"
        records: list[dict[str, Any]] = []
        if history_dir.is_dir():
            for path in sorted(history_dir.glob("*.jsonl")):
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            record = json.loads(line)
                            record["history_file"] = str(path.relative_to(run_dir))
                            records.append(record)
                except Exception:
                    continue
        records.sort(key=lambda item: str(item.get("created_at") or ""))
        if limit > 0:
            records = records[-limit:]
        return {
            "available": bool(records),
            "history_dir": str(history_dir.relative_to(run_dir)),
            "messages": records,
            "count": len(records),
        }

    def web_evidence(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        path = run_dir / "web_evidence.json"
        if not path.is_file():
            raise BridgeError(HTTPStatus.NOT_FOUND, "web evidence is not available")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, "web evidence is invalid") from exc

    def skill_candidate(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        return self.skill_candidate_summary(run_dir)

    def generate_skill_candidate(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        try:
            summary = build_tool_skill_candidate(run_dir)
        except FileNotFoundError as exc:
            raise BridgeError(HTTPStatus.CONFLICT, str(exc)) from exc
        job["summary"] = self.collect_summary(job)
        job["updated_at"] = iso_now()
        self.save_job(job)
        return summary

    def enable_skill_candidate(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        try:
            summary = enable_tool_skill_candidate(run_dir, self.repo_root)
        except FileNotFoundError as exc:
            raise BridgeError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except (FileExistsError, ValueError) as exc:
            raise BridgeError(HTTPStatus.CONFLICT, str(exc)) from exc
        job["summary"] = self.collect_summary(job)
        job["updated_at"] = iso_now()
        self.save_job(job)
        return summary

    def ask_qa(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        job = self.load_job(job_id)
        run_dir = self.require_run_dir(job)
        question = str(payload.get("question") or "").strip()
        if not question:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "question is required")
        profile_name = (job.get("options") or {}).get("profile") or DEFAULT_PROFILE
        config = Config("config")
        profile = config.get_runtime_profile(profile_name)
        base_url = profile.get("llm_base_url") or (config.get("endpoints") or {}).get("services", {}).get("amd_fast_base_url")
        model = profile.get("text_model")
        if not base_url or not model:
            raise BridgeError(HTTPStatus.BAD_REQUEST, f"profile {profile_name} must provide llm_base_url and text_model")
        client = GenericOpenAIAPIClient(
            resolve_api_key(
                api_key_env=profile.get("text_api_key_env") or profile.get("api_key_env"),
                api_url=base_url,
            ),
            base_url,
            extra_body=build_openai_extra_body(profile, base_url),
        )
        try:
            result = ask_video_docs_result(
                run_dir,
                question,
                client,
                model,
                resolve_temperature(profile, 0.2),
                int(payload.get("max_context_chars") or 60000),
            )
        except FileNotFoundError as exc:
            raise BridgeError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        history_dir = run_dir / QA_DIR_NAME / "chat_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        record = {"question": question, **result, "created_at": iso_now()}
        history_path = history_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        result["history_path"] = str(history_path.relative_to(run_dir))
        result["created_at"] = record["created_at"]
        return result

    def public_job(self, job: dict[str, Any], public_host: str | None = None) -> dict[str, Any]:
        public = dict(job)
        public["stages"] = dict(job.get("stages") or {})
        public["summary"] = self.collect_summary(job)
        title = self.resolve_job_title(public)
        public["title"] = title
        public["display_title"] = title or job.get("source_name") or job.get("video_url") or job.get("job_id")
        public["stage_order"] = self.stage_order_for_job(job)
        public["progress"] = self.progress(job)
        public["current_stage"] = self.current_stage(job)
        public["next_stage"] = self.next_stage(job)
        public["error_summary"] = self.error_summary(job)
        public["failure_disposition"] = self.failure_disposition(job)
        public["dashboard_url"] = self.dashboard_url(job["job_id"])
        public["queue"] = self.queue_info(public)
        public["core_progress"] = self.core_progress(public)
        public["stage_progress"] = self.stage_progress(public)
        public["core_diagnostics"] = self.core_diagnostics(public)
        public["preview"] = self.preview_metadata(public)
        public["source_player"] = self.source_player_metadata(public)
        public["vscode_preview"] = self.vscode_preview_metadata(public, public_host)
        public["warnings"] = self.active_warnings(public)
        public["prompt_template"] = self.prompt_template_metadata(public)
        public["result_resources"] = self.result_resources(public)
        public["document_preview"] = self.document_preview(public)
        queued_stage = public["queue"].get("stage")
        if queued_stage and queued_stage in public["stages"]:
            public["stages"][queued_stage] = dict(public["stages"][queued_stage])
            public["stages"][queued_stage]["queue_position"] = public["queue"].get("position")
        current_stage = public.get("current_stage")
        current_info = public["stages"].get(current_stage or "", {})
        public["process"] = self.public_process_info(current_info.get("process"))
        return public

    def audio_prompt_template_actual(self, run_dir: Path) -> dict[str, Any] | None:
        for path in (run_dir / "audio_template_analysis.json", run_dir / "analysis.json"):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            analysis = payload.get("audio_template_analysis") if path.name == "analysis.json" else payload
            selected = (analysis or {}).get("selected_template") or {}
            classification = (analysis or {}).get("classification") or {}
            if selected:
                return {
                    "id": selected.get("id"),
                    "title": selected.get("title"),
                    "title_zh": selected.get("title_zh") or selected.get("title"),
                    "category": "/".join(
                        item
                        for item in [selected.get("first_category_zh"), selected.get("second_category_zh")]
                        if item
                    ),
                    "classification": classification,
                    "prompt": selected.get("prompt_original"),
                }
        return None

    def prompt_template_metadata(self, job: dict[str, Any]) -> dict[str, Any]:
        opts = job.get("options") or {}
        requested = {
            "id": opts.get("template_id") or None,
            "title": opts.get("template_title") or None,
            "title_zh": opts.get("template_title_zh") or None,
            "category": opts.get("template_category") or None,
            "focus_prompt": opts.get("focus_prompt") or None,
        }
        actual = job.get("prompt_template_actual") if isinstance(job.get("prompt_template_actual"), dict) else None
        run_dir_value = job.get("run_dir") or (((job.get("artifacts") or {}).get("run_dir") or {}).get("value"))
        if actual is None and run_dir_value:
            actual = self.audio_prompt_template_actual(Path(str(run_dir_value)))
        return {"requested": requested, "actual": actual}

    def result_resources(self, job: dict[str, Any]) -> dict[str, str]:
        run_dir_value = job.get("run_dir") or (((job.get("artifacts") or {}).get("run_dir") or {}).get("value"))
        if not run_dir_value:
            return {}
        run_dir = Path(str(run_dir_value)).expanduser().resolve()
        artifacts = job.get("artifacts") or {}
        if str(job.get("audio_pipeline_kind") or "analysis") == "transcription":
            file_candidates = {
                "transcript_markdown": run_dir / "transcript.md",
                "transcript_json": run_dir / "orin" / "transcript.json",
                "asr_json": run_dir / "orin" / "asr.json",
                "transcription_json": run_dir / "transcription.json",
                "speaker_diarization_report": run_dir / "qa" / "speaker_diarization_report.json",
                "transcript_raw": run_dir / "transcript_raw.json",
                "transcript_aligned": run_dir / "transcript_aligned.json",
                "transcription_manifest": run_dir / "transcription_manifest.json",
            }
            return {
                name: path
                for name, value in file_candidates.items()
                if (path := self.resource_relative_path(run_dir, value))
            }
        artifact_candidates: dict[str, Any] = {
            "summary_markdown": ((artifacts.get("operation_manual") or {}).get("value")),
            "transcript_markdown": ((artifacts.get("transcript") or {}).get("value")),
            "analysis_json": ((artifacts.get("analysis_json") or {}).get("value")),
        }
        file_candidates: dict[str, Any] = {
            "transcript_json": run_dir / "orin" / "transcript.json",
            "asr_json": run_dir / "orin" / "asr.json",
            "transcription_json": run_dir / "transcription.json",
            "speaker_diarization_report": run_dir / "qa" / "speaker_diarization_report.json",
            "study_guide": run_dir / "study_guide.json",
            "mindmap_markdown": run_dir / "study_overview.md",
            "study_cards_markdown": run_dir / "study_cards.md",
            "audio_template_analysis": run_dir / "audio_template_analysis.json",
        }
        resources: dict[str, str] = {}
        for name, value in artifact_candidates.items():
            path = self.resource_relative_path(run_dir, value, allow_missing=True)
            if path:
                resources[name] = path
        for name, value in file_candidates.items():
            path = self.resource_relative_path(run_dir, value)
            if path:
                resources[name] = path
        return resources

    def document_preview(self, job: dict[str, Any]) -> dict[str, Any]:
        run_dir_value = job.get("run_dir") or (((job.get("artifacts") or {}).get("run_dir") or {}).get("value"))
        if not run_dir_value:
            return {
                "primary": [],
                "evidence": [],
                "process": [],
                "assets": [],
                "derivation": self.document_derivation([], [], [], []),
            }
        run_dir = Path(str(run_dir_value)).expanduser().resolve()
        job_id = str(job.get("job_id") or "")
        primary = self.document_preview_files(job_id, run_dir, DOCUMENT_PREVIEW_PRIMARY)
        evidence = self.document_preview_files(job_id, run_dir, DOCUMENT_PREVIEW_EVIDENCE)
        process = self.document_preview_files(job_id, run_dir, DOCUMENT_PREVIEW_PROCESS)
        assets = self.document_preview_dirs(run_dir, DOCUMENT_PREVIEW_ASSETS)
        return {
            "primary": primary,
            "evidence": evidence,
            "process": process,
            "assets": assets,
            "derivation": self.document_derivation(primary, evidence, process, assets),
        }

    def document_preview_files(self, job_id: str, run_dir: Path, specs: tuple[tuple[str, str, str], ...]) -> list[dict[str, Any]]:
        items = []
        for relative, title, description in specs:
            path = run_dir / relative
            if not path.is_file():
                continue
            items.append(
                {
                    "type": "file",
                    "path": relative,
                    "title": title,
                    "description": description,
                    "size_bytes": path.stat().st_size,
                    "updated_at": iso_from_timestamp(path.stat().st_mtime),
                    "mime_type": mimetypes.guess_type(str(path))[0],
                    "url": self.resource_url(job_id, relative),
                }
            )
        return items

    def document_preview_dirs(self, run_dir: Path, specs: tuple[tuple[str, str, str], ...]) -> list[dict[str, Any]]:
        items = []
        for relative, title, description in specs:
            path = run_dir / relative
            if not path.is_dir():
                continue
            files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
            items.append(
                {
                    "type": "directory",
                    "path": relative,
                    "title": title,
                    "description": description,
                    "file_count": len(files),
                    "size_bytes": sum(candidate.stat().st_size for candidate in files),
                    "updated_at": iso_from_timestamp(max((candidate.stat().st_mtime for candidate in files), default=path.stat().st_mtime)),
                }
            )
        return items

    def resource_url(self, job_id: str, relative_path: str) -> str:
        return f"/api/video-link/jobs/{job_id}/resources/{quote(relative_path, safe='/')}"

    def document_derivation(
        self,
        primary: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        process: list[dict[str, Any]],
        assets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        available = {item["path"] for item in [*primary, *evidence, *process, *assets]}
        specs = (
            ("input", "输入素材", "视频 / 页面上下文 / 评论", 0, None),
            ("transcript", "ASR 转写", "transcript.md", 1, "transcript.md"),
            ("frames", "抽帧与截图", "frames / manual_assets", 1, "manual_assets"),
            ("visual", "OCR/VL 视觉理解", "orin / manual_evidence.md", 2, "manual_evidence.md"),
            ("manual", "操作手册", "operation_manual.md", 3, "operation_manual.md"),
            ("study", "学习账本与证据分诊", "study_guide / evidence_triage", 4, "study_guide.json"),
            ("notes", "逐章知识笔记", "knowledge_notes_v2.md", 5, "docs_analysis_chapters/knowledge_notes_v2.md"),
            ("report", "深度报告", "deep_report_v2.md", 5, "docs_analysis_chapters/deep_report_v2.md"),
            ("audit", "证据审计与发布判断", "evidence_review / publish_decision", 6, "evidence_review.json"),
        )
        nodes = [
            {"id": node_id, "title": title, "description": description, "tier": tier, "available": required is None or required in available}
            for node_id, title, description, tier, required in specs
        ]
        edges = [
            ("input", "transcript", "音频提取"),
            ("input", "frames", "视频抽帧"),
            ("transcript", "manual", "文本上下文"),
            ("frames", "visual", "视觉证据"),
            ("visual", "manual", "截图/证据"),
            ("manual", "study", "核心文档"),
            ("visual", "study", "证据缺口"),
            ("study", "notes", "章节整理"),
            ("study", "report", "深度分析"),
            ("notes", "audit", "发布材料"),
            ("report", "audit", "发布材料"),
            ("visual", "audit", "证据复核"),
        ]
        return {
            "nodes": nodes,
            "edges": [{"from": left, "to": right, "label": label} for left, right, label in edges],
            "mermaid": self.document_derivation_mermaid(nodes, edges),
        }

    def document_derivation_mermaid(self, nodes: list[dict[str, Any]], edges: list[tuple[str, str, str]]) -> str:
        labels = {node["id"]: f"{node['title']}<br/>{node['description']}" for node in nodes}
        lines = ["flowchart LR"]
        for node_id, label in labels.items():
            lines.append(f"  {node_id}[\"{label}\"]")
        for left, right, label in edges:
            lines.append(f"  {left} -->|{label}| {right}")
        return "\n".join(lines)

    def resource_relative_path(self, run_dir: Path, value: Any, allow_missing: bool = False) -> str | None:
        if not value:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            candidate = (run_dir / path).resolve()
        else:
            candidate = path.expanduser().resolve()
        try:
            relative = candidate.relative_to(run_dir)
        except ValueError:
            return None
        if not allow_missing and not candidate.is_file():
            return None
        return str(relative)

    def public_job_summary(self, job: dict[str, Any]) -> dict[str, Any]:
        public = {
            "job_id": job["job_id"],
            "video_url": job.get("video_url"),
            "source_type": job.get("source_type") or "url",
            "source_name": job.get("source_name"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "options": job.get("options") or {},
            "runner": job.get("runner") or {},
            "stages": dict(job.get("stages") or {}),
            "resolved_mode": job.get("resolved_mode") or ((job.get("artifacts") or {}).get("resolved_mode") or {}).get("value"),
            "resolved_mode_reason": job.get("resolved_mode_reason")
            or ((job.get("artifacts") or {}).get("resolved_mode_reason") or {}).get("value"),
            "run_dir": job.get("run_dir"),
            "video_path": job.get("video_path"),
        }
        title = self.resolve_job_title(public)
        public["title"] = title
        public["display_title"] = title or job.get("source_name") or job.get("video_url") or job.get("job_id")
        public["stage_order"] = self.stage_order_for_job(job)
        public["progress"] = self.progress(job)
        public["current_stage"] = self.current_stage(job)
        public["next_stage"] = self.next_stage(job)
        public["error_summary"] = self.error_summary(job)
        public["failure_disposition"] = self.failure_disposition(job)
        public["dashboard_url"] = self.dashboard_url(job["job_id"])
        public["source_player"] = self.source_player_metadata(public)
        current_info = public["stages"].get(public.get("current_stage") or "", {})
        public["process"] = self.public_process_info(current_info.get("process"))
        return public

    def annotate_failure_dispositions(self, jobs: list[dict[str, Any]]) -> None:
        for job in jobs:
            job["failure_disposition"] = self.failure_disposition(job, jobs)

    def failure_disposition(
        self,
        job: dict[str, Any],
        peers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if job.get("status") != "failed":
            return None

        peers = peers or []
        job_id = str(job.get("job_id") or "")
        run_dir = str(job.get("run_dir") or "")
        video_url = str(job.get("video_url") or "")
        created_at = str(job.get("created_at") or "")
        for peer in peers:
            if peer.get("job_id") == job_id or peer.get("status") != "succeeded":
                continue
            same_run = bool(run_dir and str(peer.get("run_dir") or "") == run_dir)
            newer_same_source = bool(
                video_url
                and str(peer.get("video_url") or "") == video_url
                and str(peer.get("created_at") or "") > created_at
            )
            if same_run or newer_same_source:
                return {
                    "category": "superseded",
                    "label": "已有成功任务",
                    "rerun_recommended": False,
                    "reason": "相同资源目录或来源已有成功任务",
                    "action": "直接查看成功任务或现有产物，不要重复运行",
                    "superseded_by": peer.get("job_id"),
                }

        text = self.failure_text(job)
        lowered = text.lower()
        if any(
            pattern in lowered
            for pattern in (
                "publish blocked by evidence gate",
                "quality gate",
                "content exists risk",
                "关键证据缺口",
                "模型复核建议",
            )
        ):
            return {
                "category": "review_required",
                "label": "需要人工复核",
                "rerun_recommended": False,
                "reason": "失败来自内容安全、质量或证据门禁，盲目重跑不能解决",
                "action": "查看 operation_manual、manual_evidence 和 publish_decision 后决定是否补证据",
            }

        if self.final_documents_present(job):
            return {
                "category": "artifacts_complete",
                "label": "产物已完整",
                "rerun_recommended": False,
                "reason": "当前要求的四份 Markdown 文档均已存在且非空",
                "action": "直接验收现有文档；旧发布失败无需重跑",
            }

        recommended_profile = active_runtime_profile(runtime_profile_names())
        if "insufficient balance" in lowered or "402" in lowered:
            return {
                "category": "external_block",
                "label": "余额恢复后续跑",
                "rerun_recommended": True,
                "reason": "核心素材仍可复用，文本生成被账户余额阻断",
                "action": f"余额可用后从第一个产物不完整阶段继续，并切换到 {recommended_profile}",
                "recommended_profile": recommended_profile,
            }

        run_dir_path = Path(run_dir) if run_dir else None
        if run_dir_path and run_dir_path.is_dir() and not self.missing_core_artifacts(run_dir_path):
            return {
                "category": "resume_required",
                "label": "可以继续生成",
                "rerun_recommended": True,
                "reason": "核心分析产物完整，缺少后续文档或发布产物",
                "action": f"从第一个产物不完整阶段继续，并使用 {recommended_profile}",
                "recommended_profile": recommended_profile,
            }

        created_timestamp = parse_iso_timestamp(job.get("created_at"))
        if created_timestamp and time.time() - created_timestamp > 30 * 24 * 60 * 60:
            return {
                "category": "historical",
                "label": "历史失败",
                "rerun_recommended": False,
                "reason": "任务超过 30 天且没有可直接复用的完整核心产物",
                "action": "仅在仍有业务价值时新建任务，不建议直接续跑旧状态",
            }

        return {
            "category": "rerun_core",
            "label": "需要重跑核心分析",
            "rerun_recommended": True,
            "reason": "核心产物缺失、无效或包含未解决的分析错误",
            "action": f"从核心分析阶段重新运行，并使用 {recommended_profile}",
            "recommended_profile": recommended_profile,
        }

    def failure_text(self, job: dict[str, Any]) -> str:
        values = [str((job.get("runner") or {}).get("error") or "")]
        for stage_info in (job.get("stages") or {}).values():
            failure = stage_info.get("failure") or {}
            values.extend(
                [
                    str(stage_info.get("error") or ""),
                    str(stage_info.get("warning") or ""),
                    str(stage_info.get("last_error") or ""),
                    str(failure.get("message") or ""),
                ]
            )
        return "\n".join(value for value in values if value)

    def final_documents_present(self, job: dict[str, Any]) -> bool:
        run_dir_value = str(job.get("run_dir") or "")
        if not run_dir_value:
            return False
        run_dir = Path(run_dir_value)
        return all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in EXPECTED_FINAL_DOCUMENTS)

    def active_warnings(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        warnings = []
        stages = job.get("stages") or {}
        audio_only = self.audio_only_job(job)
        for warning in job.get("warnings") or []:
            stage = normalize_stage_name(warning.get("stage") or "")
            stage_info = stages.get(stage) or {}
            if stage_info.get("status") == "succeeded":
                continue
            if audio_only and stage == "deep-v2":
                continue
            warnings.append(warning)
        return warnings

    def audio_only_job(self, job: dict[str, Any]) -> bool:
        run_dir_value = job.get("run_dir")
        if not run_dir_value:
            return False
        manifest_path = Path(run_dir_value) / "frames_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return isinstance(payload, dict) and payload.get("source") == "audio_only"

    def resolve_job_title(self, job: dict[str, Any]) -> str:
        for value in (
            job.get("title"),
            job.get("source_name"),
            ((job.get("summary") or {}).get("study") or {}).get("title"),
        ):
            title = clean_display_title(value)
            if title:
                return title

        info_title = self.video_info_title(job)
        if info_title:
            return info_title

        context_title = self.page_context_title(job)
        if context_title:
            return context_title
        return ""

    def video_info_title(self, job: dict[str, Any]) -> str:
        video_dir = job.get("video_dir")
        if not video_dir and job.get("video_path"):
            video_dir = str(Path(job["video_path"]).parent)
        if not video_dir:
            artifact_video = artifact_value(job, "video_path")
            if artifact_video:
                video_dir = str(Path(artifact_video).parent)
        if not video_dir:
            return ""
        for path in (Path(video_dir) / "info.json", *sorted(Path(video_dir).glob("download*.info.json"))):
            if not path.is_file():
                continue
            try:
                title = clean_display_title((json.loads(path.read_text(encoding="utf-8")) or {}).get("title"))
            except Exception:
                title = ""
            if title:
                return title
        return ""

    def page_context_title(self, job: dict[str, Any]) -> str:
        path_value = job.get("page_context_path") or artifact_value(job, "page_context")
        if not path_value:
            return ""
        path = Path(path_value)
        if not path.is_file():
            return ""
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = re.match(r"^#\s+(.+?)\s*$", line)
                if match:
                    return clean_display_title(match.group(1))
        except Exception:
            return ""
        return ""

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
        run_dir = Path(run_dir_value)
        summary_path = run_dir / "final_publish_summary.json"
        if not summary_path.is_file() or summary_path.stat().st_size <= 0:
            return False
        if not all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in EXPECTED_FINAL_DOCUMENTS):
            return False
        if job.get("options", {}).get("skip_images") or not BAOYU_IMAGE_GENERATION_ENABLED:
            return True
        final_dir = run_dir / "baoyu_images" / "final"
        expected_images = ("02-infographic-knowledge-notes.png", "03-infographic-deep-report.png")
        return all((final_dir / name).is_file() and (final_dir / name).stat().st_size > 0 for name in expected_images)

    def queue_info(self, job: dict[str, Any]) -> dict[str, Any]:
        runner = job.get("runner") or {}
        stage = runner.get("current_stage") or self.current_stage(job)
        if runner.get("status") != "queued" or not stage:
            return {}
        resource = runner.get("queued_for") or job_stage_resource(job, stage)
        queued = []
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            candidate_runner = candidate.get("runner") or {}
            if candidate_runner.get("status") == "queued" and (
                candidate_runner.get("queued_for")
                or job_stage_resource(candidate, candidate_runner.get("current_stage") or "")
            ) == resource:
                queued.append(candidate)
        queued.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "")
        position = next((index + 1 for index, item in enumerate(queued) if item.get("job_id") == job.get("job_id")), None)
        info = {"stage": stage, "resource": resource, "position": position, "size": len(queued)}
        retry = self.auto_retry_info(job)
        if retry.get("auto_retry"):
            info.update(
                {
                    "auto_retry": True,
                    "retry_after_seconds": retry.get("retry_after_seconds"),
                    "retry_delay_seconds": retry.get("retry_delay_seconds"),
                }
            )
        return info

    def core_progress(self, job: dict[str, Any]) -> dict[str, Any] | None:
        stage_info = (job.get("stages") or {}).get("analyze-core") or {}
        snapshot = self.core_progress_snapshot(job)
        if not stage_info and not self.stage_log_path(job["job_id"], "analyze-core").exists() and not snapshot:
            return None
        log_path = Path(stage_info.get("log_path") or self.stage_log_path(job["job_id"], "analyze-core"))
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        progress = parse_core_progress(text, stage_info.get("status") or "pending")
        if snapshot:
            progress = merge_core_progress_snapshot(progress, snapshot, stage_info.get("status") or "pending")
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
        for stage in self.stage_order_for_job(job):
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

    def core_diagnostics(self, job: dict[str, Any]) -> dict[str, Any] | None:
        stage_info = (job.get("stages") or {}).get("analyze-core") or {}
        log_path = Path(stage_info.get("log_path") or self.stage_log_path(job["job_id"], "analyze-core"))
        run_dir = self.discover_run_dir(job)
        progress = job.get("core_progress") or self.core_progress(job)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        analysis = read_analysis_payload(run_dir)
        timings = ((analysis.get("metadata") or {}).get("timings") or {}) if analysis else {}
        counts = core_diagnostic_counts(analysis)
        issues: list[dict[str, Any]] = []
        sources = {
            "run_dir": str(run_dir) if run_dir else None,
            "log_path": str(log_path) if log_path.exists() else str(log_path),
            "progress_json": str(run_dir / ANALYSIS_PROGRESS_FILENAME) if run_dir and (run_dir / ANALYSIS_PROGRESS_FILENAME).is_file() else None,
            "analysis_json": str(run_dir / "analysis.json") if run_dir and (run_dir / "analysis.json").is_file() else None,
        }

        if not stage_info and not run_dir and not log_text:
            return None

        add_core_process_issues(job, stage_info, issues)
        add_core_log_issues(log_text, issues)
        add_core_artifact_issues(self, job, run_dir, stage_info, issues)
        add_core_stale_issue(progress, issues)
        add_core_queue_issue(job, issues)
        add_core_concurrency_issue(job, log_text, issues)
        gpu = self.gpu_snapshot() if core_gpu_snapshot_needed(job, stage_info) else None
        if gpu:
            add_core_gpu_issues(gpu, core_command_text(job, log_text), issues)
        add_core_efficiency_issues(timings, issues)

        efficiency = core_efficiency_summary(timings, counts)
        status = diagnostic_status(issues)
        return {
            "status": status,
            "summary": core_diagnostic_summary(status, issues, efficiency, progress),
            "efficiency": efficiency,
            "gpu": gpu,
            "issues": issues,
            "sources": sources,
        }

    def gpu_snapshot(self) -> dict[str, Any]:
        now = time.time()
        if self.gpu_snapshot_cache and now - self.gpu_snapshot_cache_time < CORE_DIAGNOSTIC_GPU_TTL_SECONDS:
            return self.gpu_snapshot_cache
        snapshot = collect_gpu_snapshot()
        self.gpu_snapshot_cache = snapshot
        self.gpu_snapshot_cache_time = now
        return snapshot

    def stage_order_for_job(self, job: dict[str, Any]) -> tuple[str, ...]:
        options = job.get("options") or {}
        if options.get("analysis_depth") == "light":
            return ("probe", "prepare", "analyze-core")
        return tuple(STAGE_ORDER)

    def progress(self, job: dict[str, Any]) -> dict[str, Any]:
        stage_order = self.stage_order_for_job(job)
        statuses = [job.get("stages", {}).get(stage, {}).get("status") for stage in stage_order]
        completed = sum(1 for status in statuses if status in {"succeeded", "skipped"})
        failed = sum(1 for status in statuses if status == "failed")
        running = sum(1 for status in statuses if status == "running")
        queued = sum(1 for status in statuses if status == "queued")
        total = len(stage_order)
        return {
            "total": total,
            "completed": completed,
            "running": running,
            "queued": queued,
            "failed": failed,
            "percent": int(round((completed / total) * 100)) if total else 100,
        }

    def current_stage(self, job: dict[str, Any]) -> str | None:
        runner = job.get("runner") or {}
        stage_order = self.stage_order_for_job(job)
        if runner.get("status") in {"running", "queued"} and runner.get("current_stage"):
            stage = normalize_stage_name(runner["current_stage"])
            if stage in stage_order:
                return stage
        for stage in stage_order:
            if job.get("stages", {}).get(stage, {}).get("status") in {"running", "queued"}:
                return stage
        return None

    def next_stage(self, job: dict[str, Any]) -> str | None:
        for stage in self.stage_order_for_job(job):
            status = job.get("stages", {}).get(stage, {}).get("status")
            if status == "skipped" and self.skipped_stage_outputs_incomplete(job, stage):
                return stage
            if status not in {"succeeded", "skipped"}:
                return stage
        return None

    def dashboard_url(self, job_id: str) -> str:
        return f"/?job={job_id}"

    def stage_log(self, job_id: str, stage: str, limit: int = 80, full: bool = False) -> dict[str, Any]:
        stage = normalize_stage_name(stage)
        job = self.load_job(job_id)
        if stage not in self.stage_order_for_job(job):
            raise BridgeError(HTTPStatus.NOT_FOUND, f"unknown stage: {stage}")
        limit = max(1, min(limit, 500))
        stage_info = (job.get("stages") or {}).get(stage) or {}
        log_path = Path(str(stage_info.get("log_path") or self.stage_log_path(job_id, stage)))
        attempt_log_paths = self.stage_attempt_log_paths(job_id, stage, stage_info)
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        displayed_log_path = log_path
        history_fallback = False
        if not text.strip():
            for attempt_log_path in reversed(attempt_log_paths):
                if attempt_log_path == log_path or not attempt_log_path.is_file():
                    continue
                history_text = attempt_log_path.read_text(encoding="utf-8", errors="replace")
                if history_text.strip():
                    text = history_text
                    displayed_log_path = attempt_log_path
                    history_fallback = True
                    break
        return {
            "job_id": job_id,
            "stage": stage,
            "log_path": str(log_path),
            "displayed_log_path": str(displayed_log_path),
            "attempt_log_paths": [str(path) for path in attempt_log_paths],
            "history_fallback": history_fallback,
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
        job = self.recover_orphaned_running_job(job)
        return self.reconcile_incomplete_false_success(job)

    def reconcile_incomplete_false_success(self, job: dict[str, Any]) -> dict[str, Any]:
        final_stage = dict((job.get("stages") or {}).get("final-publish") or {})
        run_dir_value = str(job.get("run_dir") or "")
        core_error = self.core_manual_generation_error(Path(run_dir_value)) if run_dir_value else None
        if (
            job.get("status") == "failed"
            and final_stage.get("status") == "failed"
            and core_error
            and str((job.get("runner") or {}).get("error") or "") != core_error
        ):
            now = iso_now()
            final_stage["error"] = core_error
            final_stage["root_cause_stage"] = "analyze-core"
            job.setdefault("stages", {})["final-publish"] = final_stage
            runner = dict(job.get("runner") or {})
            runner["error"] = core_error
            runner["updated_at"] = now
            job["runner"] = runner
            job["updated_at"] = now
            self.save_job(job)
            return job
        if (
            job.get("status") != "succeeded"
            or final_stage.get("status") not in {"failed", "skipped"}
            or self.export_outputs_complete(job)
        ):
            return job
        now = iso_now()
        message = str(
            core_error
            or final_stage.get("error")
            or "final publish is incomplete: expected four non-empty final Markdown documents"
        )
        if core_error:
            final_stage["root_cause_stage"] = "analyze-core"
        final_stage["status"] = "failed"
        final_stage["error"] = message
        final_stage["finished_at"] = final_stage.get("finished_at") or now
        final_stage.pop("soft_failed", None)
        job.setdefault("stages", {})["final-publish"] = final_stage
        runner = dict(job.get("runner") or {})
        runner.update(
            {
                "status": "failed",
                "current_stage": "final-publish",
                "queued_for": "final-publish",
                "error": message,
                "updated_at": now,
                "finished_at": now,
                "server_pid": os.getpid(),
            }
        )
        runner.pop("wait_reason", None)
        job["runner"] = runner
        job["status"] = "failed"
        job["updated_at"] = now
        self.save_job(job)
        return job

    def recover_orphaned_running_job(self, job: dict[str, Any]) -> dict[str, Any]:
        runner = job.get("runner") or {}
        if self.should_requeue_legacy_interrupted_job(job):
            recovered = self.reconcile_completed_core_stage(job)
            if recovered:
                return recovered
            return self.requeue_interrupted_job(job)
        if self.should_requeue_legacy_transient_failure(job):
            return self.requeue_interrupted_job(job, reason=TRANSIENT_RESOURCE_REQUEUE_MESSAGE)
        if runner.get("status") not in {"running", "queued"}:
            return job
        active = self.active_runners.get(job["job_id"])
        if active and active.is_alive():
            return job
        if runner.get("server_pid") == os.getpid():
            if normalize_stage_name(runner.get("current_stage") or "") == "analyze-core":
                recovered = self.reconcile_completed_core_stage(job)
                if recovered:
                    return recovered
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

        if stage == "analyze-core":
            recovered = self.reconcile_completed_core_stage(job, stage_info)
            if recovered:
                return recovered

        return self.requeue_interrupted_job(job, stage, stage_info)

    def should_requeue_legacy_interrupted_job(self, job: dict[str, Any]) -> bool:
        runner = job.get("runner") or {}
        if runner.get("status") != "failed" or ORPHANED_PROCESS_GONE_MESSAGE not in str(runner.get("error") or ""):
            return False
        stage = normalize_stage_name(runner.get("current_stage") or self.next_stage(job) or "")
        return stage in self.stage_order_for_job(job)

    def should_requeue_legacy_transient_failure(self, job: dict[str, Any]) -> bool:
        runner = job.get("runner") or {}
        if job.get("status") != "failed" or runner.get("status") != "failed":
            return False
        stage = normalize_stage_name(runner.get("current_stage") or self.current_stage(job) or self.next_stage(job) or "")
        if stage not in self.stage_order_for_job(job):
            return False
        stage_info = (job.get("stages") or {}).get(stage) or {}
        text = "\n".join(
            str(value or "")
            for value in (
                runner.get("error"),
                stage_info.get("error"),
                stage_info.get("last_error"),
            )
        )
        log_path = stage_info.get("log_path")
        if log_path and Path(str(log_path)).is_file():
            try:
                text += "\n" + Path(str(log_path)).read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        return self.retryable_stage_failure_text(stage, text) is not None

    def requeue_interrupted_job(
        self,
        job: dict[str, Any],
        stage: str | None = None,
        stage_info: dict[str, Any] | None = None,
        reason: str = ORPHANED_PROCESS_REQUEUE_MESSAGE,
    ) -> dict[str, Any]:
        now = iso_now()
        raw_stage = stage or (job.get("runner") or {}).get("current_stage") or self.current_stage(job) or self.next_stage(job)
        stage = normalize_stage_name(raw_stage or "") if raw_stage else None
        if not stage or stage not in self.stage_order_for_job(job):
            return job
        stage_info = dict(stage_info or (job.get("stages") or {}).get(stage) or {})
        if stage == "analyze-core":
            recovered = self.reconcile_completed_core_stage(job, stage_info)
            if recovered:
                return recovered
        interrupted_attempts = int(stage_info.get("restart_recovery_attempts") or 0)
        if reason == ORPHANED_PROCESS_REQUEUE_MESSAGE and interrupted_attempts >= MAX_INTERRUPTED_RETRIES:
            message = "stage was interrupted repeatedly after service restart; automatic recovery budget exhausted"
            stage_info.update(
                {
                    "status": "failed",
                    "finished_at": now,
                    "error": message,
                    "failure": {
                        "kind": "interrupted",
                        "retryable": False,
                        "status_code": None,
                        "provider_code": None,
                        "message": message,
                    },
                }
            )
            stage_info.pop("process", None)
            job.setdefault("stages", {})[stage] = stage_info
            runner = dict(job.get("runner") or {})
            runner.update(
                {
                    "status": "failed",
                    "current_stage": stage,
                    "queued_for": job_stage_resource(job, stage),
                    "error": message,
                    "updated_at": now,
                    "finished_at": now,
                    "server_pid": os.getpid(),
                }
            )
            runner.pop("wait_reason", None)
            job["runner"] = runner
            job["status"] = "failed"
            job["updated_at"] = now
            self.save_job(job)
            return job
        resource = job_stage_resource(job, stage)
        stage_info["status"] = "queued"
        stage_info["queued_at"] = now
        stage_info["queued_for"] = resource
        stage_info["retry_reason"] = reason
        if reason == ORPHANED_PROCESS_REQUEUE_MESSAGE:
            stage_info["restart_recovery_attempts"] = interrupted_attempts + 1
        stage_info["retry"] = {
            "auto_attempts": int((stage_info.get("retry") or {}).get("auto_attempts") or 0) + 1,
            "max_auto_attempts": self.max_auto_retries_for_reason(reason),
            "next_retry_at": iso_from_timestamp(time.time() + max(0.0, AUTO_RETRY_DELAY_SECONDS)),
        }
        stage_info["log_path"] = stage_info.get("log_path") or str(self.stage_log_path(job["job_id"], stage))
        previous_error = stage_info.get("error")
        if previous_error and not stage_info.get("last_error"):
            stage_info["last_error"] = previous_error
        stage_info.pop("process", None)
        stage_info.pop("finished_at", None)
        stage_info.pop("exit_code", None)
        stage_info.pop("error", None)
        job.setdefault("stages", {})[stage] = stage_info
        runner = dict(job.get("runner") or {})
        runner["status"] = "queued"
        runner["error"] = reason
        runner["updated_at"] = now
        runner["current_stage"] = stage
        runner["queued_for"] = resource
        runner["server_pid"] = os.getpid()
        runner.pop("wait_reason", None)
        runner.pop("finished_at", None)
        job["runner"] = runner
        job["status"] = "queued"
        job["updated_at"] = now
        self.save_job(job)
        return job

    def save_job(self, job: dict[str, Any]) -> None:
        path = self.job_path(job["job_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def upload_video_dir(self, job_id: str) -> Path:
        return self.resolve_output_path(str(Path(FALLBACK_OUTPUT_ROOT) / f"{UPLOAD_OUTPUT_PREFIX}{job_id}"))

    def uploaded_media_job(self, job: dict[str, Any]) -> bool:
        return (job.get("source_type") or "") == UPLOAD_SOURCE_TYPE

    def job_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def stage_log_path(self, job_id: str, stage: str) -> Path:
        return self.job_dir(job_id) / "logs" / f"{stage}.log"

    def stage_failure_path(self, job_id: str, stage: str, attempt: int) -> Path:
        return self.job_dir(job_id) / "logs" / f"{stage}.attempt-{attempt}.failure.json"

    def stage_failure_env(self, stage_info: dict[str, Any] | None) -> dict[str, str]:
        path = str((stage_info or {}).get("failure_path") or "")
        return {FAILURE_FILE_ENV: path} if path else {}

    def stage_attempt_log_path(self, job_id: str, stage: str, attempt: int) -> Path:
        return self.job_dir(job_id) / "logs" / f"{stage}.attempt-{attempt}.log"

    def stage_attempt_log_paths(self, job_id: str, stage: str, stage_info: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []
        seen: set[Path] = set()
        for value in stage_info.get("attempt_log_paths") or []:
            path = Path(str(value))
            if path not in seen:
                paths.append(path)
                seen.add(path)
        for path in sorted(
            self.job_dir(job_id).joinpath("logs").glob(f"{stage}.attempt-*.log"),
            key=lambda candidate: candidate.stat().st_mtime,
        ):
            if path not in seen:
                paths.append(path)
                seen.add(path)
        return paths

    def prepare_stage_log_attempt(
        self,
        job_id: str,
        stage: str,
        previous_stage_info: dict[str, Any],
        attempt: int,
    ) -> tuple[Path, list[str]]:
        log_path = self.stage_log_path(job_id, stage)
        attempt_log_paths = [str(path) for path in self.stage_attempt_log_paths(job_id, stage, previous_stage_info)]
        previous_attempt = int(previous_stage_info.get("attempt") or 0)
        previous_log_path = Path(str(previous_stage_info.get("log_path") or log_path))
        if attempt > 1 and previous_attempt and previous_log_path == log_path and log_path.is_file():
            archived_path = self.stage_attempt_log_path(job_id, stage, previous_attempt)
            archived_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.replace(archived_path)
            attempt_log_paths = [str(archived_path) if path == str(log_path) else path for path in attempt_log_paths]
            if str(archived_path) not in attempt_log_paths:
                attempt_log_paths.append(str(archived_path))
        if str(log_path) not in attempt_log_paths:
            attempt_log_paths.append(str(log_path))
        return log_path, attempt_log_paths


def sanitize_run_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
    return name or "operation-manual"


def clean_display_title(value: Any) -> str:
    title = re.sub(r"\s+", " ", str(value or "")).strip()
    title = re.sub(r"^Page Context Evidence:\s*", "", title, flags=re.IGNORECASE).strip()
    return title[:180]


def is_youtube_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname == "youtu.be" or hostname.endswith(".youtube.com")


def artifact_value(job: dict[str, Any], name: str) -> str:
    value = ((job.get("artifacts") or {}).get(name) or {}).get("value")
    return str(value or "")


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


def job_stage_resource(job: dict[str, Any], stage: str) -> str:
    if normalize_stage_name(stage) == "analyze-core":
        pipeline_kind = str(job.get("audio_pipeline_kind") or "")
        if pipeline_kind == "transcription":
            return "asr"
        if pipeline_kind == "analysis":
            return "audio-analysis"
    return stage_resource(stage)


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


def child_process_map() -> dict[int, list[int]]:
    try:
        result = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=True)
    except Exception:
        return {}
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    return children


def process_tree_pids(root_pid: Any) -> list[int]:
    try:
        root = int(root_pid)
    except (TypeError, ValueError):
        return []
    children = child_process_map()
    ordered: list[int] = []
    stack = [root]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        stack.extend(children.get(pid, []))
    return ordered


def terminate_process_tree(root_pid: Any, grace_seconds: float = 3.0) -> list[int]:
    pids = [pid for pid in process_tree_pids(root_pid) if pid != os.getpid()]
    if not pids:
        return []
    for pid in reversed(pids):
        if process_alive(pid):
            try:
                os.kill(pid, 15)
            except OSError:
                pass
    deadline = time.time() + grace_seconds
    while time.time() < deadline and any(process_alive(pid) for pid in pids):
        time.sleep(0.1)
    for pid in reversed(pids):
        if process_alive(pid):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
    return pids


def parse_core_progress(text: str, stage_status: str) -> dict[str, Any]:
    return parse_progress_steps(text, stage_status, CORE_PROGRESS_STEPS, CORE_PROGRESS_WEIGHTS)


def core_step_index(step_id: Any) -> int | None:
    for index, (candidate, _label, _patterns) in enumerate(CORE_PROGRESS_STEPS):
        if candidate == step_id:
            return index
    return None


def is_later_core_step(candidate: Any, current: Any) -> bool:
    candidate_index = core_step_index(candidate)
    if candidate_index is None:
        return False
    current_index = core_step_index(current)
    return current_index is None or candidate_index > current_index


def infer_core_progress_from_artifacts(run_dir: Path) -> dict[str, Any]:
    signals = [
        ("asr_done", run_dir / "transcript.md", "transcript file exists"),
        ("asr_done", run_dir / "orin" / "transcript.md", "transcript file exists"),
        ("frames_done", run_dir / "frame_manifest.json", "frame manifest exists"),
        ("ocr_ready", run_dir / "orin" / "ocr_events.json", "OCR artifacts exist"),
        ("vl", run_dir / "orin" / "frame_analyses.json", "VL artifacts exist"),
        ("write", run_dir / "analysis.json", "analysis output exists"),
    ]
    best: dict[str, Any] = {}
    for step_id, path, message in signals:
        if path.is_file() and path.stat().st_size > 0 and is_later_core_step(step_id, best.get("current_step")):
            best = {
                "current_step": step_id,
                "status": "running",
                "message": message,
                "artifacts": {step_id: str(path)},
            }
    required = [run_dir / "analysis.json", run_dir / "operation_manual.md", run_dir / "manual_evidence.md"]
    if all(path.is_file() and path.stat().st_size > 0 for path in required):
        best = {
            "current_step": "write",
            "status": "succeeded",
            "message": "core artifacts complete",
            "artifacts": {"analysis_json": str(run_dir / "analysis.json")},
        }
    return best


def merge_core_progress_snapshot(
    progress: dict[str, Any],
    snapshot: dict[str, Any],
    stage_status: str,
) -> dict[str, Any]:
    current_step = snapshot.get("current_step")
    snapshot_index = core_step_index(current_step)
    if snapshot_index is None:
        return progress
    details = snapshot.get("details") if isinstance(snapshot.get("details"), dict) else {}
    vl_progress = details.get("vl") if isinstance(details.get("vl"), dict) else None

    parsed_index = max(
        (
            core_step_index(step.get("id")) or 0
            for step in progress.get("steps", [])
            if step.get("status") != "pending"
        ),
        default=None,
    )
    if parsed_index is not None and parsed_index > snapshot_index:
        merged = dict(progress)
        if details:
            merged["details"] = details
        if vl_progress:
            merged["vl"] = vl_progress
        return merged

    snapshot_status = str(snapshot.get("status") or "running")
    merged = dict(progress)
    steps = []
    for index, step in enumerate(progress.get("steps") or []):
        item = dict(step)
        if snapshot_status == "succeeded":
            item["status"] = "succeeded"
        elif index < snapshot_index:
            item["status"] = "succeeded"
        elif index == snapshot_index:
            item["status"] = "failed" if snapshot_status == "failed" else "running"
            if current_step == "vl" and vl_progress:
                completed = max(int(vl_progress.get("completed") or 0), 0)
                total = max(int(vl_progress.get("total_selected") or 0), 0)
                reused = max(int(vl_progress.get("reused") or 0), 0)
                eta_seconds = vl_progress.get("eta_seconds")
                eta_text = format_seconds_label(float(eta_seconds)) if isinstance(eta_seconds, (int, float)) else "-"
                item["message"] = f"VL {completed}/{total} · 复用 {reused} · ETA {eta_text}"
            elif snapshot.get("message") and not item.get("message"):
                item["message"] = str(snapshot["message"])
        elif item.get("status") != "pending":
            item["status"] = "pending"
        steps.append(item)

    effective_status = "succeeded" if snapshot_status == "succeeded" else stage_status
    if snapshot_status == "failed":
        effective_status = "failed"
    merged["status"] = effective_status
    merged["current_step"] = None if effective_status in {"succeeded", "skipped"} else current_step
    merged["current_label"] = next((step["label"] for step in steps if step["id"] == merged["current_step"]), None)
    merged["percent"] = progress_percent_from_steps(steps, CORE_PROGRESS_STEPS, effective_status, CORE_PROGRESS_WEIGHTS)
    if current_step == "vl" and vl_progress and effective_status not in {"succeeded", "skipped"}:
        total = max(int(vl_progress.get("total_selected") or 0), 0)
        completed = max(int(vl_progress.get("completed") or 0), 0)
        fraction = min(completed / total, 1.0) if total else 1.0
        total_weight = sum(CORE_PROGRESS_WEIGHTS.values()) or 1
        completed_weight = sum(
            CORE_PROGRESS_WEIGHTS.get(step_id, 0)
            for step_id, _label, _patterns in CORE_PROGRESS_STEPS[:snapshot_index]
        )
        completed_weight += CORE_PROGRESS_WEIGHTS.get("vl", 0) * fraction
        merged["percent"] = min(99, max(0, int(round(completed_weight / total_weight * 100))))
    merged["steps"] = steps
    merged["source"] = "progress_json"
    merged["progress_updated_at"] = snapshot.get("updated_at")
    if details:
        merged["details"] = details
    if vl_progress:
        merged["vl"] = vl_progress
    return merged


def parse_stage_progress(stage: str, text: str, stage_status: str) -> dict[str, Any]:
    return parse_progress_steps(text, stage_status, STAGE_PROGRESS_STEPS.get(stage, []))


def stage_progress_text(stage: str, job: dict[str, Any], stage_info: dict[str, Any]) -> str:
    lines = []
    if stage_info and stage == "probe":
        lines.append("probe stage started")
    if stage == "probe" and job.get("resolved_mode"):
        lines.append(f"resolved mode: {job['resolved_mode']}")
        if job.get("resolved_mode_reason"):
            lines.append(f"mode reason: {job['resolved_mode_reason']}")
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


def read_analysis_payload(run_dir: Path | None) -> dict[str, Any] | None:
    if not run_dir:
        return None
    analysis_path = run_dir / "analysis.json"
    if not analysis_path.is_file():
        return None
    try:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def core_diagnostic_counts(analysis: dict[str, Any] | None) -> dict[str, Any]:
    if not analysis:
        return {}
    metadata = analysis.get("metadata") or {}
    ocr_keyframes = metadata.get("ocr_keyframes") or {}
    frame_selection = metadata.get("frame_selection") or {}
    return {
        "frames_extracted": metadata.get("frames_extracted"),
        "scan_frames": ocr_keyframes.get("scan_frames_count"),
        "ocr_candidate_frames": ocr_keyframes.get("ocr_candidate_frames_count"),
        "ocr_keyframes": ocr_keyframes.get("ocr_frames_count"),
        "ocr_text_events": ocr_keyframes.get("ocr_text_events_count"),
        "vl_frames": frame_selection.get("vl_frames_count") or metadata.get("vl_frames_processed"),
        "video_duration_seconds": frame_selection.get("video_duration_seconds") or ocr_keyframes.get("video_duration_seconds"),
    }


def issue(severity: str, code: str, title: str, detail: str, recommendation: str, evidence: str | None = None) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "title": title,
        "detail": detail,
        "recommendation": recommendation,
        "evidence": evidence,
    }


def add_core_process_issues(job: dict[str, Any], stage_info: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    status = stage_info.get("status")
    if status == "failed":
        issues.append(
            issue(
                "error",
                "core-stage-failed",
                "核心分析阶段失败",
                str(stage_info.get("error") or "analyze-core returned a non-zero result"),
                "查看 analyze-core 日志末尾，优先处理第一个模型/资源错误。",
            )
        )
    if status == "running":
        process_info = stage_info.get("process") or {}
        pid = process_info.get("pid")
        if pid and not process_alive(pid):
            issues.append(
                issue(
                    "error",
                    "dead-core-process",
                    "核心分析进程已退出",
                    f"job 仍标记 running，但 PID {pid} 已不存在。",
                    "将该任务标记为失败或重试当前阶段，避免状态页继续误报运行中。",
                )
            )
        runner = job.get("runner") or {}
        if runner.get("status") == "running" and not pid:
            issues.append(
                issue(
                    "watch",
                    "missing-process-record",
                    "缺少运行进程记录",
                    "任务处于 running，但 analyze-core 没有记录 PID。",
                    "确认 status server 是否在启动子进程前异常重启；必要时根据产物恢复阶段状态。",
                )
            )


def add_core_log_issues(log_text: str, issues: list[dict[str, Any]]) -> None:
    for pattern in CORE_DIAGNOSTIC_ERROR_PATTERNS:
        line = first_line_containing(log_text, pattern)
        if line:
            issues.append(
                issue(
                    "error",
                    "core-log-error",
                    "日志出现模型或运行时错误",
                    f"匹配到 `{pattern}`。",
                    "先处理该错误对应的模型服务或资源冲突，再重试核心分析。",
                    line,
                )
            )
            break
    for pattern in CORE_DIAGNOSTIC_NOT_READY_PATTERNS:
        line = first_line_containing(log_text, pattern)
        if line:
            issues.append(
                issue(
                    "warning",
                    "endpoint-not-ready",
                    "本地模型 endpoint 未就绪",
                    f"匹配到 `{pattern}`。",
                    "检查 stage handoff、端口监听和当前常驻模型，确认 ASR/OCR/VL 服务已切到正确阶段。",
                    line,
                )
            )
            break


def add_core_artifact_issues(
    server: VideoLinkStatusServer,
    job: dict[str, Any],
    run_dir: Path | None,
    stage_info: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    if not run_dir:
        return
    missing = server.missing_core_artifacts(run_dir)
    if missing and stage_info.get("status") in {"succeeded", "failed"}:
        issues.append(
            issue(
                "error",
                "missing-core-artifacts",
                "核心产物不完整",
                ", ".join(missing),
                "优先从 transcript 或已存在帧目录恢复，避免不必要地重跑 ASR。",
            )
        )
    core_errors = server.core_analysis_errors(run_dir)
    if core_errors:
        issues.append(
            issue(
                "warning",
                "core-result-errors",
                "核心分析产物包含错误文本",
                f"发现 {len(core_errors)} 条 VL/resource failure。",
                "打开 frame_analyses/visual_events 核对失败帧；如果比例高，重跑 VL 阶段。",
                core_errors[0],
            )
        )


def add_core_stale_issue(progress: dict[str, Any] | None, issues: list[dict[str, Any]]) -> None:
    if not progress or not progress.get("live"):
        return
    steps = progress.get("steps") or []
    current_step = progress.get("current_step")
    current = next((step for step in steps if step.get("id") == current_step), None)
    started_at = parse_iso_timestamp(current.get("started_at")) if current else None
    if not started_at:
        return
    elapsed = time.time() - started_at
    if elapsed >= CORE_DIAGNOSTIC_STALE_SECONDS:
        issues.append(
            issue(
                "watch",
                "stale-core-step",
                "当前核心子项长时间没有新信号",
                f"{current.get('label') or current_step} 已持续 {int(elapsed)} 秒。",
                "查看当前日志尾部和 GPU/endpoint 状态，确认是正常长请求还是服务卡住。",
                current.get("message"),
            )
        )


def add_core_queue_issue(job: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    queue = job.get("queue") or {}
    if queue.get("resource") != "core":
        return
    stage_info = (job.get("stages") or {}).get("analyze-core") or {}
    queued_at = parse_iso_timestamp(stage_info.get("queued_at") or ((job.get("runner") or {}).get("updated_at")))
    if not queued_at:
        return
    waited = time.time() - queued_at
    if waited >= CORE_DIAGNOSTIC_QUEUE_WARN_SECONDS:
        issues.append(
            issue(
                "watch",
                "core-queue-wait",
                "核心分析等待资源较久",
                f"core 队列等待约 {int(waited)} 秒，位置 #{queue.get('position') or '-'} / {queue.get('size') or '-'}。",
                "这是预期的全局互斥行为；若没有真实运行任务，检查残留 PID 或 lock 文件。",
            )
        )


def add_core_concurrency_issue(job: dict[str, Any], log_text: str, issues: list[dict[str, Any]]) -> None:
    command = core_command_text(job, log_text)
    if not command or "minicpm-v-4.5-v100" not in command:
        return
    match = re.search(r"--vl-concurrency(?:=|\s+)(\d+)", command)
    if not match:
        return
    concurrency = int(match.group(1))
    if concurrency < CORE_DIAGNOSTIC_EXPECTED_MINICPM_CONCURRENCY:
        issues.append(
            issue(
                "warning",
                "low-minicpm-vl-concurrency",
                "MiniCPM VL 并发低于 worker 数",
                f"命令使用 --vl-concurrency {concurrency}，当前期望至少 {CORE_DIAGNOSTIC_EXPECTED_MINICPM_CONCURRENCY}。",
                "后续任务使用更新后的 MiniCPM profile；当前任务如需立刻提速，需要从可恢复点重启。",
                match.group(0),
            )
        )


def core_gpu_snapshot_needed(job: dict[str, Any], stage_info: dict[str, Any]) -> bool:
    runner = job.get("runner") or {}
    status = stage_info.get("status")
    return bool(
        status in {"running", "queued"}
        or (runner.get("status") in {"running", "queued"} and runner.get("current_stage") == "analyze-core")
    )


def add_core_gpu_issues(gpu: dict[str, Any], command: str, issues: list[dict[str, Any]]) -> None:
    if gpu.get("status") != "ok":
        issues.append(
            issue(
                "watch",
                "gpu-snapshot-unavailable",
                "GPU 快照不可用",
                str(gpu.get("error") or "nvidia-smi did not return data"),
                "核心分析不受影响；如需 GPU 视图，检查 nvidia-smi 是否可用。",
            )
        )
        return
    if "minicpm-v-4.5-v100" not in command:
        return
    workers = [
        proc
        for device in gpu.get("devices", [])
        for proc in device.get("processes", [])
        if "llama-server" in str(proc.get("process_name") or "")
    ]
    if len(workers) < CORE_DIAGNOSTIC_EXPECTED_MINICPM_CONCURRENCY:
        issues.append(
            issue(
                "warning",
                "minicpm-gpu-worker-count-low",
                "MiniCPM GPU worker 数低于预期",
                f"当前 nvidia-smi 看到 {len(workers)} 个 llama-server，期望 {CORE_DIAGNOSTIC_EXPECTED_MINICPM_CONCURRENCY} 个。",
                "检查 MiniCPM 代理健康和 worker 日志，确认 5 个 P40 backend 都已加载到 GPU。",
            )
        )


def add_core_efficiency_issues(timings: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    numeric = numeric_timings(timings)
    total = numeric.get("total_seconds")
    if not total:
        return
    bottleneck = bottleneck_timing(numeric)
    if bottleneck and bottleneck[1] / total >= 0.6:
        label = timing_label(bottleneck[0])
        issues.append(
            issue(
                "watch",
                "dominant-core-bottleneck",
                f"{label} 是主要耗时瓶颈",
                f"{label} 用时 {round(bottleneck[1], 1)} 秒，占核心分析约 {round((bottleneck[1] / total) * 100)}%。",
                "结合并发、选帧数量和对应 endpoint 日志判断是否需要调参。",
            )
        )


def core_efficiency_summary(timings: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    numeric = numeric_timings(timings)
    total = numeric.get("total_seconds")
    video_duration = counts.get("video_duration_seconds")
    ratio = None
    if total and video_duration:
        try:
            ratio = round(total / float(video_duration), 3)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = None
    bottleneck = bottleneck_timing(numeric)
    return {
        "total_seconds": total,
        "video_duration_seconds": video_duration,
        "runtime_ratio": ratio,
        "bottleneck": {"key": bottleneck[0], "label": timing_label(bottleneck[0]), "seconds": bottleneck[1]} if bottleneck else None,
        "timings": numeric,
        "counts": counts,
    }


def numeric_timings(timings: dict[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in timings.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        numeric[key] = round(number, 3)
    return numeric


def bottleneck_timing(timings: dict[str, float]) -> tuple[str, float] | None:
    candidates = {
        key: value
        for key, value in timings.items()
        if key != "total_seconds" and key.endswith("_seconds") and value is not None
    }
    if not candidates:
        return None
    return max(candidates.items(), key=lambda item: item[1])


def timing_label(key: str | None) -> str:
    labels = {
        "asr_seconds": "ASR 转写",
        "candidate_frame_extraction_seconds": "候选帧抽取",
        "ocr_seconds": "OCR",
        "frame_selection_seconds": "VL 选帧",
        "vl_seconds": "VL 分析",
        "manual_generation_seconds": "手册生成",
    }
    return labels.get(key or "", key or "-")


def diagnostic_status(issues: list[dict[str, Any]]) -> str:
    severities = {item.get("severity") for item in issues}
    if "error" in severities:
        return "error"
    if "warning" in severities:
        return "warning"
    if "watch" in severities:
        return "watch"
    return "ok"


def core_diagnostic_summary(
    status: str,
    issues: list[dict[str, Any]],
    efficiency: dict[str, Any],
    progress: dict[str, Any] | None,
) -> str:
    if issues:
        first = issues[0]
        return f"{first.get('title')}: {first.get('detail')}"
    bottleneck = efficiency.get("bottleneck")
    if bottleneck:
        ratio = efficiency.get("runtime_ratio")
        ratio_text = f"，耗时比 {ratio}x" if ratio is not None else ""
        return f"核心分析正常；主要耗时在{bottleneck.get('label')}{ratio_text}。"
    if progress and progress.get("summary"):
        return str(progress["summary"])
    return "暂无异常信号。"


def first_line_containing(text: str, pattern: str) -> str | None:
    for line in text.splitlines():
        if pattern in line:
            return line.strip()[:240]
    return None


def core_command_text(job: dict[str, Any], log_text: str) -> str:
    candidates = []
    command = ((job.get("stages") or {}).get("analyze-core") or {}).get("artifacts", {}).get("command")
    if isinstance(command, list):
        candidates.append(" ".join(str(part) for part in command))
    elif isinstance(command, str):
        candidates.append(command)
    artifact_command = ((job.get("artifacts") or {}).get("command") or {}).get("value")
    if isinstance(artifact_command, list):
        candidates.append(" ".join(str(part) for part in artifact_command))
    elif isinstance(artifact_command, str):
        candidates.append(artifact_command)
    for line in log_text.splitlines():
        if "--vl-concurrency" in line or "minicpm-v-4.5-v100" in line:
            candidates.append(line)
    return "\n".join(candidates)


def collect_gpu_snapshot() -> dict[str, Any]:
    started = time.time()
    try:
        gpu_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=CORE_DIAGNOSTIC_GPU_TIMEOUT_SECONDS,
        )
        proc_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=CORE_DIAGNOSTIC_GPU_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {
            "status": "unavailable",
            "error": str(exc),
            "sampled_at": iso_now(),
            "duration_seconds": round(time.time() - started, 3),
            "devices": [],
        }
    devices = parse_gpu_rows(gpu_output)
    processes_by_uuid = parse_gpu_process_rows(proc_output)
    for device in devices:
        processes = processes_by_uuid.get(device.get("uuid"), [])
        device["processes"] = processes
        device["process_count"] = len(processes)
    return {
        "status": "ok",
        "sampled_at": iso_now(),
        "duration_seconds": round(time.time() - started, 3),
        "devices": devices,
        "process_count": sum(len(device.get("processes", [])) for device in devices),
    }


def parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    devices = []
    for row in csv_rows(text, 8):
        index, name, uuid_value, memory_total, memory_used, utilization, power_draw, power_limit = row
        devices.append(
            {
                "index": parse_int(index),
                "name": name,
                "uuid": uuid_value,
                "memory_total_mib": parse_int(memory_total),
                "memory_used_mib": parse_int(memory_used),
                "utilization_gpu_percent": parse_int(utilization),
                "power_draw_w": parse_float(power_draw),
                "power_limit_w": parse_float(power_limit),
                "processes": [],
                "process_count": 0,
            }
        )
    return devices


def parse_gpu_process_rows(text: str) -> dict[str, list[dict[str, Any]]]:
    processes: dict[str, list[dict[str, Any]]] = {}
    for row in csv_rows(text, 4):
        pid, process_name, gpu_uuid, used_memory = row
        if not gpu_uuid:
            continue
        item = {
            "pid": parse_int(pid),
            "process_name": process_name,
            "used_memory_mib": parse_int(used_memory),
        }
        processes.setdefault(gpu_uuid, []).append(item)
    return processes


def csv_rows(text: str, width: int) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) < width:
            continue
        rows.append(values[:width])
    return rows


def parse_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        return round(float(str(value).strip()), 1)
    except (TypeError, ValueError):
        return None


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
    return "fast" if analysis_mode in {"long-talk-fast", "operation-fast"} else analysis_mode


def resolve_auto_analysis_mode(requested_mode: str, duration_seconds: int | float | None, focus_prompt: str) -> tuple[str, str]:
    if requested_mode != "auto":
        return requested_mode, "explicit mode selected"

    prompt = str(focus_prompt or "").strip().lower()
    is_long = duration_seconds is not None and duration_seconds >= AUTO_MODE_LONG_SECONDS
    if not prompt:
        if is_long:
            return "long-talk-fast", f"duration >= {AUTO_MODE_LONG_SECONDS}s"
        return "balanced", "default auto mode"

    fast_score = keyword_score(prompt, AUTO_MODE_FAST_KEYWORDS)
    deep_score = keyword_score(prompt, AUTO_MODE_DEEP_KEYWORDS)
    long_talk_score = keyword_score(prompt, AUTO_MODE_LONG_TALK_KEYWORDS)
    operation_score = keyword_score(prompt, AUTO_MODE_OPERATION_KEYWORDS)

    if is_long:
        if deep_score > max(fast_score, long_talk_score):
            return "deep", "focus prompt asks for deep/high-detail analysis"
        if operation_score > long_talk_score and fast_score:
            return "operation-fast", "long video with fast operation/tutorial intent"
        if fast_score or long_talk_score:
            return "long-talk-fast", "long video with speed/subtitle/talk intent"
        return "long-talk-fast", f"duration >= {AUTO_MODE_LONG_SECONDS}s"

    if deep_score > fast_score:
        return "deep", "focus prompt asks for deep/high-detail analysis"
    if fast_score > 0 and operation_score > 0:
        return "operation-fast", "focus prompt asks for quick operation/tutorial analysis"
    if fast_score > 0:
        return "fast", "focus prompt asks for quick/summary analysis"
    return "balanced", "focus prompt has no strong mode hint"


def keyword_score(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def find_code_server_binary() -> dict[str, Any] | None:
    configured = os.environ.get("VIDEO_LINK_CODE_SERVER_BIN")
    if configured and Path(configured).is_file():
        return {"server": "code-server", "command": [str(Path(configured))]}

    path_code_server = shutil.which("code-server")
    if path_code_server:
        return {"server": "code-server", "command": [path_code_server]}

    path_openvscode = shutil.which("openvscode-server")
    if path_openvscode:
        return {"server": "openvscode-server", "command": [path_openvscode]}

    # Microsoft's ~/.vscode-server/.../server/bin/code-server is only a remote
    # extension-host backend here; it returns 404 for the Web workbench root.
    # Coder code-server is the lightest available direct Web workbench fallback.
    path_npx = shutil.which("npx")
    if path_npx and os.environ.get("VIDEO_LINK_DISABLE_NPX_CODE_SERVER") != "1":
        return {"server": "code-server", "command": [path_npx, "--yes", "code-server"]}
    return None


def build_code_server_command(
    server: dict[str, Any],
    host: str,
    port: int,
    user_dir: Path,
    extensions_dir: Path,
    run_dir: Path | None,
) -> list[str]:
    base_command = [str(part) for part in server["command"]]
    bind_addr = f"{host}:{port}"
    if server["server"] == "openvscode-server":
        command = base_command + [
            "--host",
            host,
            "--port",
            str(port),
            "--without-connection-token",
            "--user-data-dir",
            str(user_dir),
            "--extensions-dir",
            str(extensions_dir),
        ]
        if run_dir:
            command.append(str(run_dir))
        return command
    command = base_command + [
        "--bind-addr",
        bind_addr,
        "--auth",
        "none",
        "--disable-telemetry",
        "--disable-update-check",
        "--disable-workspace-trust",
        "--user-data-dir",
        str(user_dir),
        "--extensions-dir",
        str(extensions_dir),
    ]
    if run_dir:
        command.append(str(run_dir))
    return command


def discover_vscode_session(job_id: str, run_dir: Path) -> dict[str, Any] | None:
    matches = discover_vscode_processes(run_dir)
    if not matches:
        return None
    match = max(matches, key=lambda item: item["pgid"])
    return {
        "job_id": job_id,
        "pid": match["pgid"],
        "port": match["port"],
        "run_dir": str(run_dir),
        "log_path": None,
        "server": "code-server",
        "started_at": iso_now(),
    }


def discover_global_vscode_session(jobs_dir: Path) -> dict[str, Any] | None:
    matches = [
        match for match in discover_managed_vscode_processes(jobs_dir)
        if "_vscode-user-data" in match.get("command", "")
    ]
    if not matches:
        return None
    match = sorted(matches, key=lambda item: (item["port"] != VSCODE_PORT, item["pgid"]))[0]
    return {
        "job_id": None,
        "pid": match["pgid"],
        "port": match["port"],
        "run_dir": None,
        "log_path": str(jobs_dir / "_vscode-server.log"),
        "server": "code-server",
        "started_at": iso_now(),
    }


def stop_discovered_vscode_sessions(run_dir: Path) -> int:
    stopped = 0
    for match in discover_vscode_processes(run_dir):
        try:
            os.killpg(int(match["pgid"]), 15)
            stopped += 1
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.kill(int(match["pid"]), 15)
                stopped += 1
            except Exception:
                pass
    return stopped


def stop_managed_vscode_sessions(jobs_dir: Path) -> int:
    stopped = 0
    for match in discover_managed_vscode_processes(jobs_dir):
        try:
            os.killpg(int(match["pgid"]), 15)
            stopped += 1
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.kill(int(match["pid"]), 15)
                stopped += 1
            except Exception:
                pass
    if stopped:
        time.sleep(0.5)
    return stopped


def discover_vscode_processes(run_dir: Path) -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,pgid=,cmd="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    run_dir_text = str(run_dir)
    groups: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        command = parts[2]
        if "code-server" not in command:
            continue
        try:
            command_args = shlex.split(command)
        except ValueError:
            command_args = command.split()
        if run_dir_text not in command_args:
            continue
        port_match = re.search(r"--bind-addr\s+\S+:(\d+)", command) or re.search(r"--port\s+(\d+)", command)
        if not port_match:
            continue
        groups.setdefault(pgid, {"pid": pid, "pgid": pgid, "port": int(port_match.group(1)), "command": command})
    return list(groups.values())


def discover_managed_vscode_processes(jobs_dir: Path) -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,pgid=,cmd="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    jobs_dir_text = str(jobs_dir)
    groups: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        command = parts[2]
        if "code-server" not in command or jobs_dir_text not in command:
            continue
        port_match = re.search(r"--bind-addr\s+\S+:(\d+)", command) or re.search(r"--port\s+(\d+)", command)
        if not port_match:
            continue
        groups.setdefault(pgid, {"pid": pid, "pgid": pgid, "port": int(port_match.group(1)), "command": command})
    return list(groups.values())


def allocate_vscode_port(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise BridgeError(HTTPStatus.SERVICE_UNAVAILABLE, f"VS Code Server port {port} is busy") from exc
    return port


def public_vscode_host(public_host: str | None = None) -> str:
    configured = os.environ.get("VIDEO_LINK_VSCODE_PUBLIC_HOST")
    if configured:
        return configured
    host = (public_host or "").strip()
    if host and host not in {"127.0.0.1", "localhost", "::1"}:
        return host
    tailscale_host = local_tailscale_host()
    return tailscale_host or host or "127.0.0.1"


def local_tailscale_host() -> str | None:
    try:
        output = subprocess.check_output(
            ["tailscale", "ip", "-4"],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    for line in output.splitlines():
        value = line.strip()
        if value:
            return value
    return None


def allocate_local_port(port_range: tuple[int, int]) -> int:
    start, end = port_range
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise BridgeError(HTTPStatus.SERVICE_UNAVAILABLE, f"no free VS Code Server port in {start}-{end}")


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


def iter_nested_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_nested_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_nested_strings(item)


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


def normalize_focus_prompt(value: Any, max_chars: int = 4000) -> str:
    text = str(normalize_optional_template(value) or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text[:max_chars]


def focus_prompt_for_url(payload: dict[str, Any], url: str, index: int) -> str:
    fallback = normalize_focus_prompt(payload.get("focus_prompt") if "focus_prompt" in payload else payload.get("focusPrompt", ""))
    prompts = payload.get("focus_prompts") if "focus_prompts" in payload else payload.get("focusPrompts")
    if isinstance(prompts, dict):
        return normalize_focus_prompt(prompts.get(url, "")) or fallback
    if isinstance(prompts, list) and index - 1 < len(prompts):
        item = prompts[index - 1]
        if isinstance(item, dict):
            return normalize_focus_prompt(item.get("focus_prompt") if "focus_prompt" in item else item.get("focusPrompt", "")) or fallback
        return normalize_focus_prompt(item) or fallback
    return fallback


def normalize_optional_template(value: Any) -> Any:
    if isinstance(value, str) and re.fullmatch(r"\s*\{\{\s*[^{}]+?\s*\}\}\s*", value):
        return ""
    return value


def normalize_external_attempt_id(value: Any) -> str:
    attempt_id = str(normalize_optional_template(value) or "").strip()
    if not attempt_id:
        return ""
    if len(attempt_id) > 128 or not re.fullmatch(r"[A-Za-z0-9._-]+", attempt_id):
        raise BridgeError(
            HTTPStatus.BAD_REQUEST,
            "external_attempt_id must contain only letters, numbers, dot, dash, or underscore",
        )
    return attempt_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def probe_media_duration_seconds(media_path: str | Path) -> int | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=operation_env())
        duration = completed.stdout.strip()
        return int(float(duration)) if duration else None
    except Exception:
        return None


def sanitize_upload_filename(filename: str) -> str:
    candidate = Path(str(filename or "media")).name.strip()
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-")
    if not candidate:
        candidate = "media"
    suffix = Path(candidate).suffix.lower()
    stem = Path(candidate).stem[:80].strip(".-") or "media"
    return f"{stem}{suffix}"


def upload_page_context(source_name: str, media_path: Path) -> str:
    return "\n".join(
        [
            f"# {source_name}",
            "",
            "## 来源",
            "",
            "- 类型：本地上传媒体文件",
            f"- 文件名：{source_name}",
            f"- 服务端路径：{media_path}",
            "",
        ]
    )


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


def iso_from_timestamp(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(value))


def format_seconds_label(value: float) -> str:
    seconds = max(0, int(value))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining = seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{remaining:02d}"
    return f"{minutes}:{remaining:02d}"


def parse_iso_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError:
        return None


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
        <div>
          <label for="download_device">下载设备</label>
          <select id="download_device" name="download_device"></select>
        </div>
        <label class="check"><input id="skip_images" name="skip_images" type="checkbox" checked disabled>跳过配图/提示词</label>
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
    fillSelect("download_device", choices.download_devices, defaults.download_device);
    document.getElementById("run_name").value = defaults.run_name || "operation-manual";
    document.getElementById("max_comments").value = defaults.max_comments ?? 3000;
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
        download_device: document.getElementById("download_device").value,
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
    .docSection {{ margin-top: 14px; }}
    .docSection h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .docGrid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }}
    .docCard {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fff; min-height: 96px; }}
    .docCard.primary {{ border-color: #b6d4fe; background: #f8fbff; }}
    .docTitle {{ display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; font-weight: 650; }}
    .docBadge {{ flex: 0 0 auto; border-radius: 999px; padding: 2px 8px; font-size: 12px; background: #e8f0fe; color: #174ea6; }}
    .docDescription {{ margin-top: 8px; color: #3c4043; font-size: 13px; line-height: 1.5; }}
    .docMeta {{ margin-top: 8px; color: #5f6368; font-size: 12px; }}
    .docList {{ display: grid; gap: 8px; }}
    .docListItem {{ display: grid; grid-template-columns: minmax(160px, 1fr) minmax(120px, auto); gap: 10px; border-bottom: 1px solid #edf0f2; padding: 8px 0; }}
    .docListItem:last-child {{ border-bottom: 0; }}
    .docEmpty {{ color: #5f6368; font-size: 13px; }}
    .mindmap {{ display: grid; gap: 8px; }}
    .mindmapTier {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .mindmapTier + .mindmapTier::before {{ content: "\\2193"; color: #5f6368; margin-right: 4px; }}
    .mindNode {{ border: 1px solid #dadce0; border-radius: 8px; padding: 8px 10px; background: #fff; min-width: 150px; }}
    .mindNode.missing {{ opacity: .58; background: #f8f9fa; }}
    .mindNode strong {{ display: block; font-size: 13px; }}
    .mindNode span {{ display: block; color: #5f6368; font-size: 12px; margin-top: 3px; }}
    details.docDetails {{ margin-top: 12px; border-top: 1px solid #edf0f2; padding-top: 10px; }}
    details.docDetails summary {{ cursor: pointer; font-weight: 650; }}
	    .actions {{ margin-top: 14px; display: flex; gap: 10px; align-items: center; }}
	    button {{ border: 0; border-radius: 6px; padding: 9px 14px; background: #202124; color: #fff; font-size: 14px; cursor: pointer; }}
	    button.secondary {{ background: #eef2f7; color: #202124; }}
	    button.success-action {{ display: inline-flex; align-items: center; justify-content: center; gap: 6px; background: #16a34a; color: #fff; }}
	    button.success-action::before {{ content: "\\2713"; display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 999px; background: rgba(255,255,255,.22); font-size: 12px; font-weight: 700; line-height: 1; }}
	    button.play-action, button.stop-action {{ display: inline-flex; align-items: center; justify-content: center; gap: 6px; }}
	    button.play-action {{ background: #16a34a; color: #fff; }}
	    button.play-action::before {{ content: ""; width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid currentColor; }}
	    button.stop-action {{ background: #dc2626; color: #fff; }}
	    button.stop-action::before {{ content: ""; width: 11px; height: 11px; border-radius: 2px; background: currentColor; }}
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
        <button id="runButton" class="play-action" type="button">继续运行</button>
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
      <div class="hint" id="stageDurationSummary">原视频长度：- · 阶段总耗时：-</div>
    </section>
    <section class="panel" id="corePanel" style="display:none">
      <h2>核心分析子项</h2>
      <table>
        <thead><tr><th>子项</th><th>状态</th><th>耗时</th><th>最近信号</th></tr></thead>
        <tbody id="coreSteps"></tbody>
      </table>
    </section>
    <section class="panel">
      <h2>文档预览</h2>
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
	    const runActionUrl = `${{apiUrl}}/run`;
	    const stopActionUrl = `${{apiUrl}}/stop`;
	    const openRunDirActionUrl = `${{apiUrl}}/open-run-dir`;
	    let logUrl = {json.dumps(log_url)};
	    let selectedLogStage = null;
	    let currentJobId = {json.dumps(job_id)};
	    let currentJob = null;
	    const stageNames = {json.dumps(STAGE_LABELS, ensure_ascii=False)};
    function text(id, value) {{ document.getElementById(id).textContent = value || "-"; }}
    function statusClass(status) {{ return "status " + (status || "pending"); }}
    function duration(value) {{ return value == null ? "-" : `${{value}}s`; }}
    function durationMinutes(value) {{
      const seconds = Number(value);
      if (!Number.isFinite(seconds) || seconds <= 0) return "-";
      const minutes = seconds / 60;
      if (minutes < 1) return `${{seconds.toFixed(1)}} 秒`;
      return `${{minutes.toFixed(1)}} 分钟`;
    }}
    function totalStageDuration(job) {{
      return (job.stage_order || []).reduce((total, stage) => {{
        const value = Number(job.stages?.[stage]?.duration_seconds);
        return Number.isFinite(value) && value > 0 ? total + value : total;
      }}, 0);
    }}
    function stageDurationSummary(job) {{
      const videoSeconds = Number(job.preview?.duration_seconds);
      const videoText = `原视频长度：${{durationMinutes(videoSeconds)}}`;
      const stageText = `阶段总耗时：${{durationMinutes(totalStageDuration(job))}}`;
      return `${{videoText}} · ${{stageText}}`;
    }}
    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => {{
        if (char === "&") return "&amp;";
        if (char === "<") return "&lt;";
        if (char === ">") return "&gt;";
        if (char === '"') return "&quot;";
        return "&#39;";
      }});
    }}
    function formatBytes(value) {{
      const bytes = Number(value);
      if (!Number.isFinite(bytes) || bytes <= 0) return "-";
      if (bytes < 1024) return `${{bytes}} B`;
      if (bytes < 1024 * 1024) return `${{(bytes / 1024).toFixed(1)}} KB`;
      return `${{(bytes / 1024 / 1024).toFixed(1)}} MB`;
    }}
    function docLink(item) {{
      if (!item?.url) return escapeHtml(item?.path || "");
      return `<a href="${{escapeHtml(item.url)}}" target="_blank" rel="noopener noreferrer">${{escapeHtml(item.title || item.path)}}</a>`;
    }}
    function docMeta(item) {{
      const parts = [];
      if (item?.path) parts.push(escapeHtml(item.path));
      if (item?.type === "directory") parts.push(`${{Number(item.file_count || 0)}} 个文件`);
      if (item?.size_bytes) parts.push(formatBytes(item.size_bytes));
      return parts.join(" · ");
    }}
    function renderDocCards(items, badge) {{
      if (!items?.length) return `<div class="docEmpty">暂无可用文档</div>`;
      return `<div class="docGrid">${{items.map(item => `
        <article class="docCard primary">
          <div class="docTitle"><span>${{docLink(item)}}</span><span class="docBadge">${{escapeHtml(badge)}}</span></div>
          <div class="docDescription">${{escapeHtml(item.description || "")}}</div>
          <div class="docMeta">${{docMeta(item)}}</div>
        </article>
      `).join("")}}</div>`;
    }}
    function renderDocList(items) {{
      if (!items?.length) return `<div class="docEmpty">暂无可用文件</div>`;
      return `<div class="docList">${{items.map(item => `
        <div class="docListItem">
          <div>
            <div>${{item.type === "file" ? docLink(item) : escapeHtml(item.title || item.path)}}</div>
            <div class="hint">${{escapeHtml(item.description || "")}}</div>
          </div>
          <div class="docMeta">${{docMeta(item)}}</div>
        </div>
      `).join("")}}</div>`;
    }}
    function renderMindmap(derivation) {{
      const nodes = derivation?.nodes || [];
      if (!nodes.length) return `<div class="docEmpty">暂无推导关系</div>`;
      const tiers = [...new Set(nodes.map(node => node.tier))].sort((a, b) => a - b);
      const html = tiers.map(tier => `
        <div class="mindmapTier">
          ${{nodes.filter(node => node.tier === tier).map(node => `
            <div class="mindNode ${{node.available ? "" : "missing"}}">
              <strong>${{escapeHtml(node.title)}}</strong>
              <span>${{escapeHtml(node.description)}}</span>
            </div>
          `).join("")}}
        </div>
      `).join("");
      const source = derivation?.mermaid ? `<details class="docDetails"><summary>Mermaid 源码</summary><pre>${{escapeHtml(derivation.mermaid)}}</pre></details>` : "";
      return `<div class="mindmap">${{html}}</div>${{source}}`;
    }}
    function renderDocumentPreview(preview) {{
      if (!preview) return `<div class="docEmpty">暂无文档预览</div>`;
      return `
        <div class="docSection">
          <h3>重点阅读</h3>
          ${{renderDocCards(preview.primary || [], "重点")}}
        </div>
        <div class="docSection">
          <h3>推导脑图</h3>
          ${{renderMindmap(preview.derivation)}}
        </div>
        <div class="docSection">
          <h3>证据审计</h3>
          ${{renderDocList(preview.evidence || [])}}
        </div>
        <details class="docDetails">
          <summary>过程文件</summary>
          ${{renderDocList(preview.process || [])}}
        </details>
        <details class="docDetails">
          <summary>素材与中间目录</summary>
          ${{renderDocList(preview.assets || [])}}
        </details>
      `;
    }}
    function chooseLogStage(job) {{
      if (selectedLogStage) return selectedLogStage;
      return job.current_stage || job.error_summary?.stage || job.next_stage || [...(job.stage_order || [])].reverse().find(stage => job.stages?.[stage]?.log_path);
    }}
    async function refresh() {{
	      const job = await fetch(apiUrl).then(r => r.json());
	      currentJob = job;
	      currentJobId = job.job_id;
	      const progress = job.progress || {{}};
	      text("status", job.status);
	      const runButton = document.getElementById("runButton");
	      const runDir = job.summary?.run_dir || job.run_dir;
	      const isSucceeded = job.status === "succeeded";
	      const process = job.process || job.stages?.[job.current_stage || job.next_stage || ""]?.process;
	      const isActive = Boolean(process?.alive || job.status === "running" || job.status === "queued" || job.runner?.status === "running" || job.runner?.status === "queued");
	      runButton.disabled = isSucceeded ? !runDir : false;
	      runButton.dataset.action = isActive ? "stop" : (isSucceeded ? "open-run-dir" : "run");
	      runButton.classList.toggle("success-action", isSucceeded && !isActive);
	      runButton.classList.toggle("play-action", !isSucceeded && !isActive);
	      runButton.classList.toggle("stop-action", isActive);
	      runButton.textContent = isActive ? "停止" : (isSucceeded ? "成功" : (job.status === "failed" ? "重试失败阶段" : "继续运行"));
	      runButton.title = isActive ? "停止当前运行任务" : (isSucceeded && runDir ? `打开资源目录：${{runDir}}` : "继续运行任务");
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
      text("stageDurationSummary", stageDurationSummary(job));
      document.querySelectorAll(".logLink").forEach(button => {{
        button.addEventListener("click", async () => {{
          selectedLogStage = button.dataset.stage;
          await loadLog(job, selectedLogStage);
        }});
      }});
      renderCoreProgress(job.core_progress);
      document.getElementById("artifacts").innerHTML = renderDocumentPreview(job.document_preview);
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
        await copyText(log.text || (log.lines || []).join("\\n"));
        message.textContent = "已复制";
      }} catch (error) {{
        message.textContent = `复制失败：${{error.message}}`;
      }}
    }});
    async function copyText(value) {{
      const textValue = String(value || "");
      if (navigator.clipboard?.writeText && window.isSecureContext) {{
        try {{
          await navigator.clipboard.writeText(textValue);
          return;
        }} catch (_error) {{
        }}
      }}
      const textarea = document.createElement("textarea");
      textarea.value = textValue;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.top = "-1000px";
      textarea.style.left = "-1000px";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      try {{
        if (!document.execCommand("copy")) throw new Error("浏览器拒绝复制");
      }} finally {{
        textarea.remove();
      }}
    }}
    document.getElementById("runButton").addEventListener("click", async () => {{
      const button = document.getElementById("runButton");
      const message = document.getElementById("runMessage");
	      button.disabled = true;
	      message.textContent = "已发送";
	      try {{
	        const action = button.dataset.action || (currentJob?.status === "succeeded" ? "open-run-dir" : "run");
	        const actionUrl = action === "open-run-dir" ? openRunDirActionUrl : (action === "stop" ? stopActionUrl : runActionUrl);
	        await fetch(actionUrl, {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: "{{}}" }}).then(async r => {{
	          if (!r.ok) throw new Error((await r.json()).error || `HTTP ${{r.status}}`);
	          return r.json();
	        }});
	        message.textContent = action === "open-run-dir" ? "已打开资源目录" : (action === "stop" ? "已停止" : "运行中");
	        if (action === "run" || action === "stop") {{
	          await refresh();
	        }} else {{
	          button.disabled = false;
	        }}
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
            if path == "/api/mobile/audio-jobs":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["50"])[0])
                self.write_json(self.server_app.list_mobile_audio_jobs(limit))
                return
            match = re.fullmatch(r"/api/mobile/audio-jobs/([a-f0-9]{32})", path)
            if match:
                self.write_json(self.server_app.get_mobile_audio_job(match.group(1)))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})", path)
            if match:
                host = self.headers.get("Host", "").split(":", 1)[0] or None
                self.write_json(self.server_app.public_job(self.server_app.load_job(match.group(1)), public_host=host))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/logs/([a-z0-9-]+)", path)
            if match:
                query = parse_qs(parsed.query)
                limit = int(query.get("tail", ["80"])[0])
                full = parse_bool(query.get("full", ["false"])[0])
                self.write_json(self.server_app.stage_log(match.group(1), match.group(2), limit, full))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/resources/(.+)", path)
            if match:
                resource_path = unquote(match.group(2))
                file_path, content_type = self.server_app.resource_file(match.group(1), resource_path)
                self.write_file(file_path, content_type)
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/qa-index", path)
            if match:
                self.write_json(self.server_app.qa_index(match.group(1)))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/qa/history", path)
            if match:
                limit = int(query.get("limit", ["50"])[0])
                self.write_json(self.server_app.qa_history(match.group(1), limit=limit))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/web-evidence", path)
            if match:
                self.write_json(self.server_app.web_evidence(match.group(1)))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/skill-candidate", path)
            if match:
                self.write_json(self.server_app.skill_candidate(match.group(1)))
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
            if path == "/api/mobile/audio-jobs/from-transcript":
                payload, upload_path, filename = self.read_transcript_multipart()
                try:
                    self.write_json(
                        self.server_app.create_mobile_transcript_job(payload, upload_path, filename),
                        HTTPStatus.CREATED,
                    )
                finally:
                    upload_path.unlink(missing_ok=True)
                return
            payload = self.read_json_body()
            if path == "/api/video-link/jobs":
                self.write_json(self.server_app.create_job(payload), HTTPStatus.CREATED)
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/run", path)
            if match:
                self.write_json(
                    self.server_app.start_run(match.group(1), profile=payload.get("profile")),
                    HTTPStatus.ACCEPTED,
                )
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/stop", path)
            if match:
                self.write_json(self.server_app.stop_job(match.group(1)), HTTPStatus.ACCEPTED)
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/open-run-dir", path)
            if match:
                self.write_json(self.server_app.open_run_dir(match.group(1)))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/vscode-session", path)
            if match:
                restart = parse_bool(payload.get("restart", False))
                host = self.headers.get("Host", "").split(":", 1)[0] or None
                self.write_json(self.server_app.start_vscode_session(match.group(1), public_host=host, restart=restart))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/qa/ask", path)
            if match:
                self.write_json(self.server_app.ask_qa(match.group(1), payload))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/skill-candidate/generate", path)
            if match:
                self.write_json(self.server_app.generate_skill_candidate(match.group(1)))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/skill-candidate/enable", path)
            if match:
                self.write_json(self.server_app.enable_skill_candidate(match.group(1)))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/stages/([a-z0-9-]+)", path)
            if match:
                self.write_json(self.server_app.run_stage(match.group(1), match.group(2)))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/stages/([a-z0-9-]+)/rerun", path)
            if match:
                self.write_json(self.server_app.rerun_from_stage(match.group(1), match.group(2)), HTTPStatus.ACCEPTED)
                return
            raise BridgeError(HTTPStatus.NOT_FOUND, "not found")
        except BridgeError as exc:
            self.write_json({"error": exc.message}, exc.status)

    def do_DELETE(self) -> None:
        try:
            path = urlparse(self.path).path
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})", path)
            if match:
                self.write_json(self.server_app.delete_job(match.group(1)))
                return
            match = re.fullmatch(r"/api/video-link/jobs/([a-f0-9]{32})/vscode-session", path)
            if match:
                self.write_json(self.server_app.stop_vscode_session(match.group(1)))
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

    def read_transcript_multipart(self) -> tuple[dict[str, Any], Path, str]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type.lower():
            raise BridgeError(HTTPStatus.BAD_REQUEST, "expected multipart/form-data")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "multipart body is empty")
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + self.rfile.read(length)
        )
        payload: dict[str, Any] = {}
        transcript_part = None
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name in {"transcript", "transcript_json"}:
                transcript_part = part
            elif name:
                payload[name] = part.get_content().strip()
        if transcript_part is None:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "transcript JSON file is required")
        filename = transcript_part.get_filename() or "transcript.json"
        fd, temporary_name = tempfile.mkstemp(prefix="provided-transcript-", suffix=".json")
        with os.fdopen(fd, "wb") as handle:
            handle.write(transcript_part.get_payload(decode=True) or b"")
            handle.flush()
            os.fsync(handle.fileno())
        return payload, Path(temporary_name), filename

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

    def write_file(self, path: Path, content_type: str | None = None) -> None:
        body = path.read_bytes()
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f"inline; filename={quote(path.name)}")
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
