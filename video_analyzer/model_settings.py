from __future__ import annotations

import copy
import base64
import contextlib
import hashlib
import io
import ipaddress
import json
import logging
import math
import os
import re
import socket
import struct
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import requests
from PIL import Image, ImageDraw


MODEL_KIND_PROTOCOLS = {
    "asr": {
        "vibevoice_http",
        "qwen3_asr_http",
        "generic_http",
        "firered_asr2_http",
        "firered_3dspeaker_http",
        "openai_audio",
        "faster_whisper",
        "none",
    },
    "diarization": {
        "asr_embedded",
        "three_d_speaker",
        "pyannote_community",
        "wespeaker_diarization",
        "none",
    },
    "ocr": {
        "unlimited_ocr_openai",
        "dots_ocr_openai",
        "dots_mocr_openai",
        "openai_vision",
        "none",
    },
    "vision": {"openai_compatible", "none"},
    "text": {"openai_compatible", "none"},
    "selector": {"openai_compatible", "inherit_text", "none"},
    "review": {"openai_compatible", "inherit_text", "none"},
    "study": {"openai_compatible", "inherit_text", "none"},
    "triage": {"openai_compatible", "inherit_study", "inherit_text", "none"},
    "image": {"codex_imagegen", "none"},
}

ASR_CHUNK_MODES = {"model_default", "custom"}
ASR_SEGMENTATION_MODES = {"native", "fixed", "vad"}
ASR_CHUNK_OPTION_FIELDS = (
    "single_pass_max_duration_sec",
    "chunk_duration_sec",
    "chunk_overlap_sec",
)
ASR_CHUNK_DEFAULTS = {
    "vibevoice_http": {
        "segmentation_mode": "native",
        "single_pass_max_duration_sec": 900,
        "chunk_duration_sec": 370,
        "chunk_overlap_sec": 10,
    },
    "qwen3_asr_http": {
        "segmentation_mode": "fixed",
        "single_pass_max_duration_sec": 150,
        "chunk_duration_sec": 120,
        "chunk_overlap_sec": 10,
    },
    "firered_asr2_http": {
        "segmentation_mode": "vad",
        "vad_max_segment_sec": 50,
        "single_pass_max_duration_sec": 35,
        "chunk_duration_sec": 30,
        "chunk_overlap_sec": 3,
    },
}

PROFILE_MODEL_FIELDS = {
    "asr": "asr_model_id",
    "diarization": "diarization_model_id",
    "ocr": "ocr_model_id",
    "vision": "vision_model_id",
    "text": "text_model_id",
    "review": "review_model_id",
    "study": "study_card_model_id",
    "triage": "triage_model_id",
    "image": "image_model_id",
}

VIDEO_WORKFLOW_ID = "video_operation_manual"
AUDIO_WORKFLOW_ID = "audio_nx1"

VIDEO_PROFILE_FLOW = {
    "version": 4,
    "lanes": [
        {"id": "audio", "label": "音频"},
        {"id": "visual", "label": "视觉"},
        {"id": "main", "label": "主流程"},
        {"id": "fallback", "label": "故障回退"},
        {"id": "quality", "label": "质量"},
        {"id": "learning", "label": "学习"},
        {"id": "delivery", "label": "交付"},
    ],
    "nodes": [
        {"id": "input", "step": 1, "column": 1, "row": 2, "mobile_order": 1, "lane": "main", "title": "输入素材", "subtitle": "视频、字幕、评论与页面信息", "stage": "probe"},
        {"id": "prepare", "step": 2, "column": 2, "row": 2, "mobile_order": 2, "lane": "main", "title": "素材准备", "subtitle": "下载媒体并整理页面上下文", "stage": "prepare"},
        {"id": "audio_extract", "step": 3, "column": 3, "row": 1, "mobile_order": 3, "lane": "audio", "title": "音频提取", "subtitle": "生成可转写音轨", "stage": "analyze-core", "progress_step": "audio"},
        {"id": "asr", "step": 4, "column": 4, "row": 1, "mobile_order": 4, "lane": "audio", "title": "语音识别", "subtitle": "与说话人分离并行执行", "model_kind": "asr", "required": False, "stage": "analyze-core", "progress_step": "asr"},
        {"id": "diarization", "step": 4, "column": 4, "row": 2, "mobile_order": 5, "lane": "audio", "title": "说话人分离", "subtitle": "与语音识别并行生成声纹轨", "model_kind": "diarization", "required": False, "stage": "analyze-core"},
        {"id": "transcript_merge", "step": 5, "column": 5, "row": 1, "mobile_order": 6, "lane": "audio", "title": "转写与说话人合并", "subtitle": "对齐文字、时间戳与说话人", "stage": "analyze-core", "progress_step": "asr_done"},
        {"id": "frame_extract", "step": 3, "column": 3, "row": 3, "mobile_order": 7, "lane": "visual", "title": "候选帧提取", "subtitle": "采样并筛选可分析画面", "stage": "analyze-core", "progress_step": "frames"},
        {"id": "frame_audit", "step": 4, "column": 4, "row": 3, "mobile_order": 8, "lane": "visual", "title": "帧去重审计", "subtitle": "去重并补齐时间轴覆盖", "stage": "analyze-core", "progress_step": "ocr_audit"},
        {"id": "ocr", "step": 5, "column": 5, "row": 3, "mobile_order": 9, "lane": "visual", "title": "画面 OCR", "subtitle": "提取画面文字证据", "model_kind": "ocr", "required": False, "stage": "analyze-core", "progress_step": "ocr"},
        {"id": "vision", "step": 6, "column": 6, "row": 3, "mobile_order": 10, "lane": "visual", "title": "视觉理解", "subtitle": "在 OCR 之后理解动作、界面和场景", "model_kind": "vision", "required": False, "stage": "analyze-core", "progress_step": "vl"},
        {"id": "visual_evidence", "step": 7, "column": 7, "row": 3, "mobile_order": 11, "lane": "visual", "title": "视觉证据", "subtitle": "整理 OCR、VL 与帧引用", "stage": "analyze-core", "progress_step": "vl"},
        {"id": "evidence_merge", "step": 7, "column": 7, "row": 2, "mobile_order": 12, "lane": "main", "title": "证据汇合", "subtitle": "合并转写、视觉与页面上下文", "stage": "analyze-core", "progress_step": "manual"},
        {"id": "text", "step": 8, "column": 8, "row": 2, "mobile_order": 13, "lane": "main", "title": "核心分析", "subtitle": "结构化理解并生成手册草稿", "model_kind": "text", "required": True, "stage": "analyze-core", "progress_step": "manual"},
        {"id": "text_fallback", "step": 9, "column": 9, "row": 4, "mobile_order": 14, "lane": "fallback", "title": "核心文本兜底", "subtitle": "主文本模型失败时调用，可禁用", "model_slot": "text_fallback", "model_kind": "text", "required": False, "stage": "analyze-core", "progress_step": "manual"},
        {"id": "core_verify", "step": 10, "column": 10, "row": 2, "mobile_order": 15, "lane": "quality", "title": "核心校验", "subtitle": "检查核心产物与证据引用", "stage": "verify-core"},
        {"id": "study", "step": 11, "column": 11, "row": 3, "mobile_order": 16, "lane": "learning", "title": "学习账本", "subtitle": "提炼章节与学习卡片", "model_kind": "study", "required": False, "stage": "study-guide"},
        {"id": "triage", "step": 12, "column": 12, "row": 3, "mobile_order": 17, "lane": "learning", "title": "证据 Triage", "subtitle": "判断证据缺口与处理路由", "model_kind": "triage", "required": False, "stage": "study-guide"},
        {"id": "documents", "step": 13, "column": 12, "row": 2, "mobile_order": 18, "lane": "main", "title": "多文档章节分析", "subtitle": "生成知识笔记与初版报告", "model_kind": "text", "model_slot": "text", "required": True, "stage": "multidoc"},
        {"id": "deep_report", "step": 14, "column": 13, "row": 2, "mobile_order": 19, "lane": "main", "title": "章节深度报告", "subtitle": "逐章分析与最终综合", "model_kind": "text", "model_slot": "text", "required": True, "stage": "deep-v2", "stage_step": "chapters"},
        {"id": "deep_review", "step": 15, "column": 14, "row": 2, "mobile_order": 20, "lane": "quality", "title": "深度报告复核", "subtitle": "语义审核与格式校验", "model_kind": "review", "required": False, "stage": "deep-v2", "stage_step": "review"},
        {"id": "evidence_review", "step": 16, "column": 15, "row": 1, "mobile_order": 21, "lane": "quality", "title": "证据复核与发布门禁", "subtitle": "复核缺口对最终发布的影响", "model_kind": "text", "model_slot": "text", "required": True, "stage": "evidence-review"},
        {"id": "web_evidence", "step": 17, "column": 16, "row": 1, "mobile_order": 22, "lane": "quality", "title": "联网补证据", "subtitle": "仅补充可由外部来源验证的缺口", "model_kind": "text", "model_slot": "text", "required": True, "stage": "web-evidence"},
        {"id": "qa_index", "step": 18, "column": 17, "row": 2, "mobile_order": 23, "lane": "delivery", "title": "问答证据索引", "subtitle": "规则切片并建立可追溯索引", "stage": "qa-index"},
        {"id": "image_prompts", "step": 19, "column": 17, "row": 3, "mobile_order": 24, "lane": "delivery", "title": "配图提示词", "subtitle": "整理最终文档的配图需求", "stage": "image-prompts"},
        {"id": "image", "step": 20, "column": 18, "row": 3, "mobile_order": 25, "lane": "delivery", "title": "文档配图", "subtitle": "生成或复用最终文档插图", "model_kind": "image", "required": False, "stage": "final-publish", "stage_step": "images"},
        {"id": "final_publish", "step": 21, "column": 19, "row": 2, "mobile_order": 26, "lane": "delivery", "title": "最终定稿与发布", "subtitle": "补齐、插图并校验交付文档", "stage": "final-publish"},
        {"id": "operation_manual_doc", "step": 22, "column": 20, "row": 1, "mobile_order": 27, "lane": "delivery", "title": "操作手册", "subtitle": "operation_manual.md", "stage": "final-publish", "artifact_path": "operation_manual.md", "node_kind": "output"},
        {"id": "knowledge_notes_doc", "step": 22, "column": 20, "row": 2, "mobile_order": 28, "lane": "delivery", "title": "逐章知识笔记", "subtitle": "knowledge_notes_v2.md", "stage": "final-publish", "artifact_path": "docs_analysis_chapters/knowledge_notes_v2.md", "node_kind": "output"},
        {"id": "deep_report_doc", "step": 22, "column": 20, "row": 3, "mobile_order": 29, "lane": "delivery", "title": "深度报告", "subtitle": "deep_report_v2.md", "stage": "final-publish", "artifact_path": "docs_analysis_chapters/deep_report_v2.md", "node_kind": "output"},
        {"id": "manual_evidence_doc", "step": 22, "column": 20, "row": 4, "mobile_order": 30, "lane": "delivery", "title": "证据审计表", "subtitle": "manual_evidence.md", "stage": "final-publish", "artifact_path": "manual_evidence.md", "node_kind": "output"},
    ],
    "edges": [
        {"from": "input", "to": "prepare", "lane": "main", "label": "探测与下载"},
        {"from": "prepare", "to": "audio_extract", "lane": "audio", "label": "音轨"},
        {"from": "prepare", "to": "frame_extract", "lane": "visual", "label": "视频画面"},
        {"from": "prepare", "to": "evidence_merge", "lane": "main", "label": "页面上下文"},
        {"from": "audio_extract", "to": "asr", "lane": "audio", "label": "音频"},
        {"from": "audio_extract", "to": "diarization", "lane": "audio", "label": "并行音频"},
        {"from": "asr", "to": "transcript_merge", "lane": "audio", "label": "转写文本"},
        {"from": "diarization", "to": "transcript_merge", "lane": "audio", "label": "说话人片段"},
        {"from": "transcript_merge", "to": "evidence_merge", "lane": "audio", "label": "带说话人文字稿"},
        {"from": "frame_extract", "to": "frame_audit", "lane": "visual", "label": "候选帧"},
        {"from": "frame_audit", "to": "ocr", "lane": "visual", "label": "审计保留帧"},
        {"from": "ocr", "to": "vision", "lane": "visual", "label": "OCR 文本"},
        {"from": "vision", "to": "visual_evidence", "lane": "visual", "label": "画面解释"},
        {"from": "visual_evidence", "to": "evidence_merge", "lane": "visual", "label": "帧证据"},
        {"from": "evidence_merge", "to": "text", "lane": "main", "label": "证据包"},
        {"from": "text", "to": "core_verify", "lane": "main", "label": "主模型成功"},
        {"from": "text", "to": "text_fallback", "lane": "fallback", "label": "主模型失败"},
        {"from": "text_fallback", "to": "core_verify", "lane": "fallback", "label": "兜底结果"},
        {"from": "core_verify", "to": "study", "lane": "quality", "label": "已校验证据"},
        {"from": "study", "to": "triage", "lane": "learning", "label": "证据缺口"},
        {"from": "triage", "to": "documents", "lane": "learning", "label": "学习账本"},
        {"from": "documents", "to": "deep_report", "lane": "main", "label": "章节材料"},
        {"from": "deep_report", "to": "deep_review", "lane": "quality", "label": "报告草稿"},
        {"from": "deep_review", "to": "evidence_review", "lane": "quality", "label": "复核结果"},
        {"from": "evidence_review", "to": "web_evidence", "lane": "quality", "label": "可补证缺口"},
        {"from": "web_evidence", "to": "qa_index", "lane": "delivery", "label": "完整证据"},
        {"from": "qa_index", "to": "image_prompts", "lane": "delivery", "label": "最终材料"},
        {"from": "image_prompts", "to": "image", "lane": "delivery", "label": "配图需求"},
        {"from": "qa_index", "to": "final_publish", "lane": "delivery", "label": "问答索引"},
        {"from": "image", "to": "final_publish", "lane": "delivery", "label": "最终图片"},
        {"from": "final_publish", "to": "operation_manual_doc", "lane": "delivery", "label": "交付"},
        {"from": "final_publish", "to": "knowledge_notes_doc", "lane": "delivery", "label": "交付"},
        {"from": "final_publish", "to": "deep_report_doc", "lane": "delivery", "label": "交付"},
        {"from": "final_publish", "to": "manual_evidence_doc", "lane": "delivery", "label": "交付"},
    ],
}

AUDIO_PROFILE_FLOW = {
    "version": 1,
    "lanes": [
        {"id": "primary", "label": "本地音频"},
        {"id": "fallback", "label": "云端回退"},
        {"id": "analysis", "label": "内容分析"},
        {"id": "delivery", "label": "NX1 交付"},
    ],
    "nodes": [
        {"id": "audio_input", "step": 1, "column": 1, "row": 2, "mobile_order": 1, "lane": "primary", "title": "NX1 音频输入", "subtitle": "原始文件由 NX1 持久管理", "stage": "prepare"},
        {"id": "asr", "step": 2, "column": 2, "row": 1, "mobile_order": 2, "lane": "primary", "title": "本地语音识别", "subtitle": "与说话人分离并行执行", "model_slot": "asr", "model_kind": "asr", "required": True},
        {"id": "diarization", "step": 2, "column": 2, "row": 2, "mobile_order": 3, "lane": "primary", "title": "本地说话人分离", "subtitle": "并行生成声纹轨后再对齐", "model_slot": "diarization", "model_kind": "diarization", "required": False},
        {"id": "asr_fallback", "step": 2, "column": 2, "row": 3, "mobile_order": 4, "lane": "fallback", "title": "云端 ASR 回退", "subtitle": "仅在本地资源繁忙时启用", "model_slot": "asr_fallback", "model_kind": "asr", "required": False},
        {"id": "diarization_fallback", "step": 2, "column": 2, "row": 4, "mobile_order": 5, "lane": "fallback", "title": "云端分离回退", "subtitle": "与云端 ASR 并行执行", "model_slot": "diarization_fallback", "model_kind": "diarization", "required": False},
        {"id": "template_selector", "step": 3, "column": 3, "row": 2, "mobile_order": 6, "lane": "analysis", "title": "模板选择", "subtitle": "等待文字稿与声纹轨对齐完成", "model_slot": "selector", "model_kind": "selector", "required": True},
        {"id": "text", "step": 4, "column": 4, "row": 2, "mobile_order": 7, "lane": "analysis", "title": "总结与脑图", "subtitle": "按所选模板生成最终内容", "model_slot": "text", "model_kind": "text", "required": True},
        {"id": "artifact_package", "step": 5, "column": 5, "row": 2, "mobile_order": 8, "lane": "delivery", "title": "产物封装", "subtitle": "整理转写、总结和结构化结果", "stage": "analyze-core"},
        {"id": "nx1_sync", "step": 6, "column": 6, "row": 2, "mobile_order": 9, "lane": "delivery", "title": "回传 NX1", "subtitle": "镜像资源、发布结果并确认", "stage": "final-publish"},
    ],
    "edges": [
        {"from": "audio_input", "to": "asr", "lane": "primary"},
        {"from": "audio_input", "to": "diarization", "lane": "primary"},
        {"from": "audio_input", "to": "asr_fallback", "lane": "fallback"},
        {"from": "audio_input", "to": "diarization_fallback", "lane": "fallback"},
        {"from": "asr", "to": "template_selector", "lane": "primary"},
        {"from": "diarization", "to": "template_selector", "lane": "primary"},
        {"from": "asr_fallback", "to": "template_selector", "lane": "fallback"},
        {"from": "diarization_fallback", "to": "template_selector", "lane": "fallback"},
        {"from": "template_selector", "to": "text", "lane": "analysis"},
        {"from": "text", "to": "artifact_package", "lane": "analysis"},
        {"from": "artifact_package", "to": "nx1_sync", "lane": "delivery"},
    ],
}

WORKFLOW_MODEL_FIELDS = {
    VIDEO_WORKFLOW_ID: {
        kind: {"field": field, "kind": kind, "required": kind == "text"}
        for kind, field in PROFILE_MODEL_FIELDS.items()
    }
    | {
        "text_fallback": {
            "field": "text_fallback_model_id",
            "kind": "text",
            "required": False,
        }
    },
    AUDIO_WORKFLOW_ID: {
        "asr": {"field": "asr_model_id", "kind": "asr", "required": True},
        "diarization": {"field": "diarization_model_id", "kind": "diarization", "required": False},
        "asr_fallback": {"field": "asr_fallback_model_id", "kind": "asr", "required": False},
        "diarization_fallback": {"field": "diarization_fallback_model_id", "kind": "diarization", "required": False},
        "selector": {"field": "template_selector_model_id", "kind": "selector", "required": True},
        "text": {"field": "text_model_id", "kind": "text", "required": True},
    },
}

PROFILE_WORKFLOWS = {
    VIDEO_WORKFLOW_ID: {
        "id": VIDEO_WORKFLOW_ID,
        "label": "视频操作手册",
        "description": "视频、页面上下文、OCR、视觉理解与多文档发布",
        "flow": VIDEO_PROFILE_FLOW,
        "model_fields": WORKFLOW_MODEL_FIELDS[VIDEO_WORKFLOW_ID],
    },
    AUDIO_WORKFLOW_ID: {
        "id": AUDIO_WORKFLOW_ID,
        "label": "NX1 音频分析",
        "description": "ASR、说话人分离、模板选择、总结与结果回传",
        "flow": AUDIO_PROFILE_FLOW,
        "model_fields": WORKFLOW_MODEL_FIELDS[AUDIO_WORKFLOW_ID],
    },
}

ALL_PROFILE_MODEL_FIELDS = {
    slot: spec
    for workflow in WORKFLOW_MODEL_FIELDS.values()
    for slot, spec in workflow.items()
}

CONTROL_RESOURCES = (
    ("asr-disabled", "asr", "none", "禁用 ASR"),
    ("diarization-asr-embedded", "diarization", "asr_embedded", "使用 ASR 内置说话人"),
    ("diarization-disabled", "diarization", "none", "禁用说话人分离"),
    ("ocr-disabled", "ocr", "none", "禁用 OCR"),
    ("vision-disabled", "vision", "none", "禁用视觉理解"),
    ("text-disabled", "text", "none", "禁用文本模型兜底"),
    ("review-inherit-text", "review", "inherit_text", "继承文本模型"),
    ("review-disabled", "review", "none", "禁用独立审核模型"),
    ("study-inherit-text", "study", "inherit_text", "继承文本模型"),
    ("study-disabled", "study", "none", "禁用学习卡片"),
    ("triage-inherit-study", "triage", "inherit_study", "继承学习卡片模型"),
    ("triage-inherit-text", "triage", "inherit_text", "继承文本模型"),
    ("triage-disabled", "triage", "none", "禁用证据 Triage"),
    ("image-codex-imagegen", "image", "codex_imagegen", "Codex ImageGen"),
    ("image-disabled", "image", "none", "禁用文档配图"),
    ("selector-inherit-text", "selector", "inherit_text", "继承文本模型"),
    ("selector-disabled", "selector", "none", "禁用模板选择模型"),
)

MODEL_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SECRET_FIELD_NAMES = {"api_key", "token", "access_token", "password", "secret"}
MODEL_TEST_MODES = {"quick", "inference", "pathway"}
MODEL_TEST_CACHE_SECONDS = 300
logger = logging.getLogger(__name__)


class SettingsValidationError(ValueError):
    pass


def reject_inline_secrets(value: Any, path: str = "settings") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in SECRET_FIELD_NAMES:
                raise SettingsValidationError(f"{path}.{key} must use an environment variable reference")
            reject_inline_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_inline_secrets(item, f"{path}[{index}]")


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value)
        return merged
    return copy.deepcopy(override)


def apply_disabled_runtime_profiles(config: dict[str, Any]) -> dict[str, Any]:
    filtered = copy.deepcopy(config)
    disabled = set(normalize_string_list(filtered.get("disabled_runtime_profiles")))
    if disabled:
        profiles = filtered.get("runtime_profiles") or {}
        filtered["runtime_profiles"] = {
            name: profile
            for name, profile in profiles.items()
            if name not in disabled
        }
    return filtered


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        for part in str(item).split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
    return result


def stable_resource_id(kind: str, protocol: str, payload: dict[str, Any]) -> str:
    label = str(payload.get("model") or payload.get("name") or protocol)
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:28] or protocol.replace("_", "-")
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return f"{kind}-{slug}-{digest}"[:64]


def validate_url(value: str, field: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SettingsValidationError(f"{field} must be an http(s) URL")
    return value.rstrip("/")


def validate_model_resource(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not MODEL_ID_RE.fullmatch(model_id):
        raise SettingsValidationError("model id must use lowercase letters, numbers, dash, or underscore")
    kind = str(payload.get("kind") or "").strip()
    protocol = str(payload.get("protocol") or "").strip()
    if kind not in MODEL_KIND_PROTOCOLS:
        raise SettingsValidationError(f"unsupported model kind: {kind}")
    if protocol not in MODEL_KIND_PROTOCOLS[kind]:
        raise SettingsValidationError(f"unsupported {kind} protocol: {protocol}")

    cleaned = {
        "id": model_id,
        "name": str(payload.get("name") or model_id).strip()[:120],
        "kind": kind,
        "protocol": protocol,
    }
    model = str(payload.get("model") or "").strip()
    if model:
        cleaned["model"] = model[:240]
    endpoints = normalize_string_list(payload.get("endpoints") or payload.get("endpoint"))
    if endpoints:
        cleaned["endpoints"] = [validate_url(item, "endpoint") for item in endpoints]
    health_url = str(payload.get("health_url") or "").strip()
    if health_url:
        cleaned["health_url"] = validate_url(health_url, "health_url")
    api_key_env = str(payload.get("api_key_env") or "").strip()
    if api_key_env:
        if not ENV_NAME_RE.fullmatch(api_key_env):
            raise SettingsValidationError("api_key_env must be a valid environment variable name")
        cleaned["api_key_env"] = api_key_env
    options = payload.get("options") or {}
    if not isinstance(options, dict):
        raise SettingsValidationError("model options must be an object")
    reject_inline_secrets(options, "options")
    cleaned["options"] = copy.deepcopy(options)

    endpoint_required = protocol in {
        "vibevoice_http",
        "qwen3_asr_http",
        "generic_http",
        "firered_asr2_http",
        "firered_3dspeaker_http",
        "openai_audio",
        "unlimited_ocr_openai",
        "dots_ocr_openai",
        "dots_mocr_openai",
        "openai_vision",
        "openai_compatible",
    }
    if endpoint_required and not endpoints:
        raise SettingsValidationError(f"{protocol} requires at least one endpoint")
    model_required = protocol in {
        "openai_audio",
        "unlimited_ocr_openai",
        "dots_ocr_openai",
        "dots_mocr_openai",
        "openai_vision",
        "openai_compatible",
    }
    if model_required and not model:
        raise SettingsValidationError(f"{protocol} requires a model name")
    return cleaned


def _resource(
    kind: str,
    protocol: str,
    name: str,
    *,
    model: str = "",
    endpoints: Any = None,
    api_key_env: str = "",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "kind": kind,
        "protocol": protocol,
        "model": model,
        "endpoints": normalize_string_list(endpoints),
        "api_key_env": api_key_env,
        "options": options or {},
    }
    payload["id"] = stable_resource_id(kind, protocol, payload)
    return payload


def _add_resource(catalog: dict[str, dict[str, Any]], resource: dict[str, Any]) -> str:
    resource_id = str(resource["id"])
    catalog.setdefault(resource_id, resource)
    return resource_id


def _add_control_resources(catalog: dict[str, dict[str, Any]]) -> None:
    for resource_id, kind, protocol, name in CONTROL_RESOURCES:
        catalog.setdefault(
            resource_id,
            {
                "id": resource_id,
                "name": name,
                "kind": kind,
                "protocol": protocol,
                "options": {},
                "source": "control",
            },
        )


def _builtin_service_url(config: dict[str, Any], service: str, fallback: str) -> str:
    endpoints = config.get("endpoints") or {}
    value = str((endpoints.get("services") or {}).get(service) or fallback)
    for name, host in (endpoints.get("hosts") or {}).items():
        value = value.replace(f"{{{name}}}", str(host))
        value = value.replace(f"{{hosts.{name}}}", str(host))
    return value


def _add_builtin_model_resources(
    catalog: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    services = (config.get("endpoints") or {}).get("services") or {}
    resources = {
        "asr-vibevoice-local": {
            "name": "VibeVoice ASR（本地 P40）",
            "kind": "asr",
            "protocol": "vibevoice_http",
            "model": "microsoft/VibeVoice-ASR",
            "endpoints": normalize_string_list(
                services.get("vibevoice_local_url")
                or "http://127.0.0.1:18012/api/asr/transcribe"
            ),
            "options": {
                "deployment": "local",
                "worker_count": 5,
                **ASR_CHUNK_DEFAULTS["vibevoice_http"],
            },
        },
        "asr-qwen3-1_7b-local": {
            "name": "Qwen3-ASR-1.7B（本地 P40）",
            "kind": "asr",
            "protocol": "qwen3_asr_http",
            "model": "Qwen/Qwen3-ASR-1.7B",
            "endpoints": normalize_string_list(
                services.get("qwen3_asr_url")
                or "http://127.0.0.1:18013/api/asr/transcribe"
            ),
            "options": {
                "deployment": "local",
                "worker_count": 5,
                **ASR_CHUNK_DEFAULTS["qwen3_asr_http"],
            },
        },
        "asr-firered2-local": {
            "name": "FireRedASR2-AED（本地 P40 · Ray）",
            "kind": "asr",
            "protocol": "firered_asr2_http",
            "model": "firered_asr2_aed",
            "endpoints": normalize_string_list(
                services.get("firered_asr2_url")
                or "http://127.0.0.1:18014/api/asr/transcribe"
            ),
            "options": {
                "deployment": "local",
                "worker_count": 5,
                "concurrency": 5,
                "gpu_ids": [0, 1, 2, 4, 5],
                "dispatch_mode": "ray",
                **ASR_CHUNK_DEFAULTS["firered_asr2_http"],
            },
        },
        "diarization-3dspeaker-local": {
            "name": "3D-Speaker（中文本地）",
            "kind": "diarization",
            "protocol": "three_d_speaker",
            "model": "speech_campplus_sv_zh-cn_16k-common",
            "options": {"deployment": "local", "backend": "3dspeaker"},
        },
        "diarization-pyannote-community1-local": {
            "name": "Pyannote Community-1（本地）",
            "kind": "diarization",
            "protocol": "pyannote_community",
            "model": "pyannote/speaker-diarization-community-1",
            "options": {
                "deployment": "local",
                "backend": "pyannote_community",
                "external_python": "/home/ai/pyannote-community-venv/bin/python",
                "device": "cpu",
            },
        },
        "diarization-wespeaker-cn-local": {
            "name": "WeSpeaker ResNet34 CnCeleb（中文本地）",
            "kind": "diarization",
            "protocol": "wespeaker_diarization",
            "model": "ResNet34_LM",
            "options": {
                "deployment": "local",
                "backend": "wespeaker",
                "external_python": "/home/ai/diarization-ab-venv/bin/python",
                "model_id": "chinese",
                "device": "cuda",
            },
        },
        "ocr-unlimited-local": {
            "name": "Baidu Unlimited-OCR（本地）",
            "kind": "ocr",
            "protocol": "unlimited_ocr_openai",
            "model": "baidu/Unlimited-OCR",
            "endpoints": normalize_string_list(
                services.get("unlimited_ocr_base_urls")
                or "http://127.0.0.1:18088/v1"
            ),
            "options": {
                "deployment": "local",
                "engine": "unlimited",
                "worker_count": 5,
                "concurrency": 5,
                "gpu_ids": [0, 1, 2, 4, 5],
                "cache": "on",
                "max_tokens": 8192,
                "max_image_long_side": 0,
                "image_mode": "gundam",
                "skip_model_inventory_check": True,
            },
        },
        "ocr-dots-local": {
            "name": "rednote-hilab DotsOCR（本地 DotsMOCR 部署）",
            "kind": "ocr",
            "protocol": "dots_ocr_openai",
            "model": "rednote-hilab/dots.ocr",
            "endpoints": normalize_string_list(
                services.get("dots_ocr_base_urls")
                or "http://127.0.0.1:18088/v1"
            ),
            "options": {
                "deployment": "local",
                "engine": "dots",
                "worker_count": 5,
                "concurrency": 5,
                "cache": "on",
                "max_tokens": 1024,
                "max_image_long_side": 1280,
                "skip_model_inventory_check": True,
            },
        },
        "vision-minicpm-v45-local": {
            "name": "MiniCPM-V 4.5（本地 P40）",
            "kind": "vision",
            "protocol": "openai_compatible",
            "model": "minicpm-v-4.5-v100",
            "endpoints": normalize_string_list(
                services.get("minicpm_local_base_url")
                or "http://127.0.0.1:18082/v1"
            ),
            "options": {
                "deployment": "local",
                "engine": "minicpm_v45",
                "worker_count": 5,
                "concurrency": 5,
            },
        },
        "vision-qwen3-vl-4b-local": {
            "name": "Qwen3-VL-4B-Instruct Q4_K_M（本地 P40）",
            "kind": "vision",
            "protocol": "openai_compatible",
            "model": "qwen3-vl-4b-instruct",
            "endpoints": normalize_string_list(
                services.get("qwen3_vl_base_url")
                or "http://127.0.0.1:18082/v1"
            ),
            "options": {
                "deployment": "local",
                "engine": "qwen3_vl_4b",
                "worker_count": 5,
                "concurrency": 5,
                "model_path_env": "QWEN3_VL_MODEL_PATH",
                "mmproj_path_env": "QWEN3_VL_MMPROJ_PATH",
            },
        },
        "text-amd-lmstudio-bonsai-27b": {
            "name": "LM Studio · Bonsai 27B（AMD）",
            "kind": "text",
            "protocol": "openai_compatible",
            "model": "prism-ml/bonsai-27b",
            "endpoints": [
                _builtin_service_url(
                    config,
                    "amd_fast_base_url",
                    "http://100.90.114.26:18081/v1",
                )
            ],
            "options": {
                "deployment": "remote",
                "runtime": "lm_studio",
                "device": "AMD",
                "quantization": "Q1_0",
                "context_length": 65792,
                "reasoning_effort": "none",
                "text_temperature": 0.2,
                "text_timeout_seconds": 900,
            },
        },
        "text-local-bonsai-27b-6gpu": {
            "name": "BONSAI 27B（本地五张 P40 按需池）",
            "kind": "text",
            "protocol": "openai_compatible",
            "model": "prism-ml/bonsai-27b",
            "endpoints": ["http://127.0.0.1:18103/v1"],
            "options": {
                "deployment": "local",
                "runtime": "llama.cpp",
                "text_gpu_ids": [0, 1, 2, 4, 5],
                "text_worker_count": 5,
                "text_concurrency": 5,
                "worker_count": 5,
                "concurrency": 5,
                "quantization": "Q1_0",
                "context_length": 128405,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "text_temperature": 0.7,
                "top_k": 20,
                "top_p": 0.95,
                "enable_thinking": True,
                "preserve_thinking": False,
                "text_timeout_seconds": 1800,
            },
        },
        "text-deepseek-v4-pro": {
            "name": "DeepSeek V4 Pro（云端）",
            "kind": "text",
            "protocol": "openai_compatible",
            "model": "deepseek-v4-pro",
            "endpoints": ["https://api.deepseek.com"],
            "api_key_env": "DEEPSEEK_API_KEY",
            "options": {
                "deployment": "cloud",
                "text_temperature": 1.0,
                "deepseek_thinking": "disabled",
                "text_timeout_seconds": 900,
            },
        },
        "text-deepseek-v4-flash": {
            "name": "DeepSeek V4 Flash（云端）",
            "kind": "text",
            "protocol": "openai_compatible",
            "model": "deepseek-v4-flash",
            "endpoints": ["https://api.deepseek.com"],
            "api_key_env": "DEEPSEEK_API_KEY",
            "options": {
                "deployment": "cloud",
                "text_temperature": 1.0,
                "deepseek_thinking": "disabled",
                "text_timeout_seconds": 900,
            },
        },
    }
    for resource_id, resource in resources.items():
        catalog.setdefault(
            resource_id,
            {"id": resource_id, **resource, "source": "builtin"},
        )


def build_settings_document(config: dict[str, Any]) -> dict[str, Any]:
    catalog: dict[str, dict[str, Any]] = {}
    explicit_catalog = config.get("model_catalog") or {}
    if isinstance(explicit_catalog, dict):
        for model_id, payload in explicit_catalog.items():
            if isinstance(payload, dict):
                item = copy.deepcopy(payload)
                item["id"] = model_id
                item["source"] = "custom"
                catalog[model_id] = item
    _add_control_resources(catalog)
    _add_builtin_model_resources(catalog, config)

    services = (config.get("endpoints") or {}).get("services") or {}
    global_ocr = config.get("ocr") or {}
    global_speaker = config.get("speaker_diarization") or {}
    global_study = config.get("study_cards") or {}
    profiles: dict[str, dict[str, Any]] = {}

    for profile_name, raw_profile in (config.get("runtime_profiles") or {}).items():
        profile = copy.deepcopy(raw_profile or {})
        workflow_id = str(profile.get("workflow_id") or VIDEO_WORKFLOW_ID)
        if workflow_id not in PROFILE_WORKFLOWS:
            workflow_id = VIDEO_WORKFLOW_ID
        profile["workflow_id"] = workflow_id
        refs = {kind: str(profile.get(field) or "") for kind, field in PROFILE_MODEL_FIELDS.items()}

        if not refs["asr"]:
            provider = str(profile.get("asr_provider") or "vibevoice")
            if provider == "vibevoice":
                resource = _resource(
                    "asr",
                    "vibevoice_http",
                    f"{profile_name} VibeVoice",
                    endpoints=profile.get("vibevoice_urls") or profile.get("vibevoice_url") or services.get("vibevoice_urls"),
                    options={
                        key: profile[key]
                        for key in (
                            "use_native_chunking",
                            "single_pass_max_duration_sec",
                            "chunk_duration_sec",
                            "chunk_overlap_sec",
                            "chunk_parallel_workers",
                            "remote_max_attempts",
                            "remote_retry_delay_seconds",
                            "distributed_min_seconds",
                            "distributed_workers",
                            "speaker_upper_bound",
                        )
                        if key in profile
                    },
                )
            elif provider == "firered_3dspeaker":
                resource = _resource(
                    "asr",
                    "firered_3dspeaker_http",
                    f"{profile_name} FireRed2",
                    endpoints=profile.get("firered_3dspeaker_url") or services.get("firered_3dspeaker_url"),
                )
            elif provider == "faster_whisper":
                resource = _resource("asr", "faster_whisper", "Local Faster Whisper")
            elif provider == "none":
                resource = catalog["asr-disabled"]
            else:
                resource = _resource(
                    "asr",
                    "generic_http",
                    f"{profile_name} HTTP ASR",
                    endpoints=profile.get("remote_asr_urls") or profile.get("remote_asr_url"),
                )
            refs["asr"] = _add_resource(catalog, resource)

        if not refs["diarization"]:
            if not global_speaker.get("enabled", True):
                resource = catalog["diarization-disabled"]
            else:
                resource = _resource(
                    "diarization",
                    "three_d_speaker",
                    "3D-Speaker",
                    options=global_speaker,
                )
            refs["diarization"] = _add_resource(catalog, resource)

        if not refs["ocr"]:
            provider = str(profile.get("ocr_provider") or global_ocr.get("provider") or "auto")
            protocol = "none" if provider == "none" else (
                "openai_vision" if provider == "openai_vision" else "dots_mocr_openai"
            )
            resource = _resource(
                "ocr",
                protocol,
                f"{profile_name} OCR",
                model=str(profile.get("ocr_model") or global_ocr.get("model") or "model"),
                endpoints=profile.get("ocr_base_urls") or profile.get("ocr_base_url") or global_ocr.get("base_urls"),
                options={
                    "concurrency": profile.get("ocr_concurrency", global_ocr.get("concurrency", "auto")),
                    "cache": profile.get("ocr_cache", global_ocr.get("cache", "on")),
                    "timeout_seconds": profile.get("ocr_timeout_seconds", global_ocr.get("timeout_seconds")),
                },
            )
            if protocol == "none":
                resource = catalog["ocr-disabled"]
            refs["ocr"] = _add_resource(catalog, resource)

        if not refs["vision"]:
            resource = _resource(
                "vision",
                "openai_compatible",
                f"{profile_name} Vision",
                model=str(profile.get("vision_model") or ""),
                endpoints=profile.get("vision_base_url") or profile.get("llm_base_url"),
                api_key_env=str(profile.get("vision_api_key_env") or profile.get("api_key_env") or ""),
                options={"concurrency": profile.get("vl_concurrency", 3)},
            )
            refs["vision"] = _add_resource(catalog, resource)

        if not refs["text"]:
            resource = _resource(
                "text",
                "openai_compatible",
                f"{profile_name} Text",
                model=str(profile.get("text_model") or ""),
                endpoints=profile.get("text_base_url") or profile.get("llm_base_url"),
                api_key_env=str(profile.get("text_api_key_env") or profile.get("api_key_env") or ""),
                options={
                    key: profile[key]
                    for key in ("text_temperature", "text_timeout_seconds", "deepseek_thinking", "reasoning_effort")
                    if key in profile
                },
            )
            refs["text"] = _add_resource(catalog, resource)

        if not refs["review"]:
            review_model = str(profile.get("review_model") or profile.get("text_model") or "")
            if review_model == str(profile.get("text_model") or ""):
                resource = catalog["review-inherit-text"]
            else:
                resource = _resource(
                    "review",
                    "openai_compatible",
                    f"{profile_name} Review",
                    model=review_model,
                    endpoints=profile.get("review_base_url") or profile.get("text_base_url") or profile.get("llm_base_url"),
                    api_key_env=str(profile.get("review_api_key_env") or profile.get("text_api_key_env") or ""),
                    options={
                        key: profile[key]
                        for key in ("review_temperature", "review_deepseek_thinking", "review_reasoning_effort")
                        if key in profile
                    },
                )
            refs["review"] = _add_resource(catalog, resource)

        if not refs["study"]:
            study_url = profile.get("study_card_llm_base_url") or global_study.get("llm_base_url")
            study_model = profile.get("study_card_model") or global_study.get("model")
            resource = (
                _resource(
                    "study",
                    "openai_compatible",
                    f"{profile_name} Study",
                    model=str(study_model or ""),
                    endpoints=study_url,
                    api_key_env=str(profile.get("study_card_api_key_env") or global_study.get("api_key_env") or ""),
                    options={"temperature": profile.get("study_card_temperature", global_study.get("temperature", 0.1))},
                )
                if study_url and study_model
                else catalog["study-inherit-text"]
            )
            refs["study"] = _add_resource(catalog, resource)

        if not refs["triage"]:
            triage_url = profile.get("triage_llm_base_url")
            triage_model = profile.get("triage_model")
            resource = (
                _resource(
                    "triage",
                    "openai_compatible",
                    f"{profile_name} Triage",
                    model=str(triage_model),
                    endpoints=triage_url,
                    api_key_env=str(profile.get("triage_api_key_env") or ""),
                    options={"temperature": profile.get("triage_temperature", 0.0)},
                )
                if triage_url and triage_model
                else catalog["triage-inherit-study"]
            )
            refs["triage"] = _add_resource(catalog, resource)

        if not refs["image"]:
            refs["image"] = _add_resource(
                catalog,
                catalog["image-codex-imagegen"],
            )

        for kind, field in PROFILE_MODEL_FIELDS.items():
            profile[field] = refs[kind]
        selector_id = str(profile.get("template_selector_model_id") or "")
        if not selector_id:
            selector_url = (
                profile.get("template_selector_base_url")
                or profile.get("study_card_llm_base_url")
                or global_study.get("llm_base_url")
            )
            selector_model = (
                profile.get("template_selector_model")
                or profile.get("study_card_model")
                or global_study.get("model")
            )
            selector = (
                _resource(
                    "selector",
                    "openai_compatible",
                    f"{profile_name} Template Selector",
                    model=str(selector_model or ""),
                    endpoints=selector_url,
                    api_key_env=str(
                        profile.get("template_selector_api_key_env")
                        or profile.get("study_card_api_key_env")
                        or global_study.get("api_key_env")
                        or ""
                    ),
                    options={
                        "temperature": profile.get(
                            "template_selector_temperature",
                            profile.get(
                                "study_card_temperature",
                                global_study.get("temperature", 0.1),
                            ),
                        )
                    },
                )
                if selector_url and selector_model
                else catalog["selector-inherit-text"]
            )
            selector_id = _add_resource(catalog, selector)
        profile["template_selector_model_id"] = selector_id
        profile["asr_fallback_model_id"] = str(
            profile.get("asr_fallback_model_id") or "asr-disabled"
        )
        profile["diarization_fallback_model_id"] = str(
            profile.get("diarization_fallback_model_id")
            or "diarization-disabled"
        )
        profile["text_fallback_model_id"] = str(
            profile.get("text_fallback_model_id") or "text-disabled"
        )
        profiles[profile_name] = profile

    for item in catalog.values():
        item.setdefault("source", "derived")
    return {
        "active_runtime_profile": config.get("active_runtime_profile"),
        "models": catalog,
        "profiles": profiles,
        "schema": {
            "kinds": {key: sorted(value) for key, value in MODEL_KIND_PROTOCOLS.items()},
            "profile_model_fields": PROFILE_MODEL_FIELDS,
            "profile_flow": VIDEO_PROFILE_FLOW,
            "workflows": PROFILE_WORKFLOWS,
            "asr_chunk_modes": sorted(ASR_CHUNK_MODES),
            "asr_segmentation_modes": sorted(ASR_SEGMENTATION_MODES),
            "asr_chunk_defaults": copy.deepcopy(ASR_CHUNK_DEFAULTS),
        },
    }


def expand_runtime_profile(config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    expanded = copy.deepcopy(profile)
    catalog = build_settings_document(config)["models"]

    def model_for(kind: str) -> dict[str, Any] | None:
        model_id = str(expanded.get(PROFILE_MODEL_FIELDS[kind]) or "")
        item = catalog.get(model_id)
        return item if isinstance(item, dict) else None

    def model_for_field(field: str) -> dict[str, Any] | None:
        item = catalog.get(str(expanded.get(field) or ""))
        return item if isinstance(item, dict) else None

    expanded["workflow_id"] = str(
        expanded.get("workflow_id") or VIDEO_WORKFLOW_ID
    )

    asr = model_for("asr")
    if asr:
        protocol = asr.get("protocol")
        endpoints = normalize_string_list(asr.get("endpoints"))
        options = dict(asr.get("options") or {})
        if expanded.get("asr_chunk_mode") == "custom":
            options.update(
                {
                    key: expanded[key]
                    for key in ASR_CHUNK_OPTION_FIELDS
                    if expanded.get(key) is not None
                }
            )
        if expanded.get("asr_worker_count") is not None:
            options["worker_count"] = expanded["asr_worker_count"]
            options["concurrency"] = expanded["asr_worker_count"]
        if protocol == "firered_asr2_http":
            if expanded.get("asr_segmentation_mode"):
                options["segmentation_mode"] = expanded["asr_segmentation_mode"]
            if expanded.get("vad_max_segment_sec") is not None:
                options["vad_max_segment_sec"] = expanded["vad_max_segment_sec"]
        provider = {
            "vibevoice_http": "vibevoice",
            "qwen3_asr_http": "qwen3_asr",
            "generic_http": "remote_http",
            "firered_asr2_http": "firered_asr2",
            "firered_3dspeaker_http": "firered_3dspeaker",
            "openai_audio": "openai_audio",
            "faster_whisper": "faster_whisper",
            "none": "none",
        }[protocol]
        expanded["asr_provider"] = provider
        if protocol == "vibevoice_http":
            expanded["vibevoice_urls"] = endpoints
            expanded["vibevoice_url"] = endpoints[0] if endpoints else ""
            expanded.update(options)
        elif protocol == "qwen3_asr_http":
            expanded["qwen3_asr_url"] = endpoints[0] if endpoints else ""
            expanded["qwen3_asr_model"] = asr.get("model")
            expanded["qwen3_asr_options"] = options
        elif protocol == "generic_http":
            expanded["remote_asr_urls"] = endpoints
            expanded["remote_asr_url"] = endpoints[0] if endpoints else ""
        elif protocol == "firered_asr2_http":
            expanded["firered_asr2_url"] = endpoints[0] if endpoints else ""
            expanded["firered_asr2_options"] = options
        elif protocol == "firered_3dspeaker_http":
            expanded["firered_3dspeaker_url"] = endpoints[0] if endpoints else ""
        elif protocol == "openai_audio":
            expanded["openai_audio_url"] = endpoints[0] if endpoints else ""
            expanded["openai_audio_model"] = asr.get("model")
            expanded["asr_api_key_env"] = asr.get("api_key_env")

    diarization = model_for("diarization")
    if diarization:
        protocol = diarization.get("protocol")
        options = dict(diarization.get("options") or {})
        backend = {
            "three_d_speaker": "3dspeaker",
            "pyannote_community": "pyannote_community",
            "wespeaker_diarization": "wespeaker",
        }.get(protocol)
        options["enabled"] = backend is not None
        options["assignment_enabled"] = backend is not None
        if backend:
            options["backend"] = backend
        if protocol == "asr_embedded":
            options["enabled"] = False
            options["assignment_enabled"] = False
        expanded["speaker_diarization"] = options

    selector = model_for_field("template_selector_model_id")
    if selector:
        protocol = str(selector.get("protocol") or "")
        expanded["template_selector_enabled"] = protocol != "none"
        if protocol == "inherit_text":
            expanded["template_selector_inherit"] = "text"
        elif protocol == "openai_compatible":
            endpoints = normalize_string_list(selector.get("endpoints"))
            expanded["template_selector_base_url"] = endpoints[0] if endpoints else ""
            expanded["template_selector_model"] = selector.get("model")
            expanded["template_selector_api_key_env"] = selector.get("api_key_env")
            for key, value in (selector.get("options") or {}).items():
                expanded[f"template_selector_{key}"] = value

    fallback_asr = model_for_field("asr_fallback_model_id")
    fallback_diarization = model_for_field("diarization_fallback_model_id")
    fallback_enabled = bool(
        fallback_asr
        and fallback_diarization
        and fallback_asr.get("protocol") != "none"
        and fallback_diarization.get("protocol") == "asr_embedded"
    )
    expanded["audio_cloud_fallback"] = {
        "enabled": fallback_enabled,
        "asr": copy.deepcopy(fallback_asr or {}),
        "diarization": copy.deepcopy(fallback_diarization or {}),
        "trigger": "local_resource_busy",
    }

    text_fallback = model_for_field("text_fallback_model_id")
    text_fallback_enabled = bool(
        text_fallback and text_fallback.get("protocol") != "none"
    )
    text_fallback_options = dict((text_fallback or {}).get("options") or {})
    text_fallback_endpoints = normalize_string_list(
        (text_fallback or {}).get("endpoints")
    )
    expanded["text_fallback_enabled"] = text_fallback_enabled
    expanded["text_fallback_base_url"] = (
        text_fallback_endpoints[0] if text_fallback_enabled and text_fallback_endpoints else ""
    )
    expanded["text_fallback_model"] = (
        (text_fallback or {}).get("model") if text_fallback_enabled else ""
    )
    expanded["text_fallback_api_key_env"] = (
        (text_fallback or {}).get("api_key_env") if text_fallback_enabled else ""
    )
    for key, value in text_fallback_options.items():
        expanded[f"text_fallback_{key}"] = value

    ocr = model_for("ocr")
    if ocr:
        protocol = ocr.get("protocol")
        endpoints = normalize_string_list(ocr.get("endpoints"))
        options = dict(ocr.get("options") or {})
        expanded["ocr_provider"] = {
            "unlimited_ocr_openai": "unlimited_ocr",
            "dots_ocr_openai": "dots_ocr",
            "dots_mocr_openai": "dots_mocr_vllm",
            "openai_vision": "openai_vision",
            "none": "none",
        }[protocol]
        expanded["ocr_base_urls"] = endpoints
        expanded["ocr_base_url"] = endpoints[0] if endpoints else ""
        expanded["ocr_model"] = ocr.get("model")
        if "concurrency" in options:
            expanded["ocr_concurrency"] = options["concurrency"]
        if "cache" in options:
            expanded["ocr_cache"] = options["cache"]
        if options.get("timeout_seconds") is not None:
            expanded["ocr_timeout_seconds"] = options["timeout_seconds"]
        if options.get("max_tokens") is not None:
            expanded["ocr_max_tokens"] = options["max_tokens"]
        if options.get("max_image_long_side") is not None:
            expanded["ocr_max_image_long_side"] = options["max_image_long_side"]
        if options.get("image_mode") is not None:
            expanded["ocr_image_mode"] = options["image_mode"]
        expanded["ocr_engine"] = options.get("engine")
        expanded["ocr_worker_count"] = options.get("worker_count")

    vision = model_for("vision")
    if vision:
        if vision.get("protocol") == "none":
            expanded["vl_frame_policy"] = "none"
            expanded["vision_model"] = ""
        else:
            endpoints = normalize_string_list(vision.get("endpoints"))
            expanded["vision_base_url"] = endpoints[0] if endpoints else ""
            expanded["vision_model"] = vision.get("model")
            expanded["vision_api_key_env"] = vision.get("api_key_env")
            if (vision.get("options") or {}).get("concurrency") is not None:
                expanded["vl_concurrency"] = (vision.get("options") or {})["concurrency"]
            expanded["vision_runtime"] = copy.deepcopy(vision.get("options") or {})

    text = model_for("text")
    if text and text.get("protocol") != "none":
        endpoints = normalize_string_list(text.get("endpoints"))
        expanded["text_base_url"] = endpoints[0] if endpoints else ""
        expanded["llm_base_url"] = expanded["text_base_url"]
        expanded["text_model"] = text.get("model")
        expanded["text_api_key_env"] = text.get("api_key_env")
        for key, value in (text.get("options") or {}).items():
            expanded[key] = value

    for kind, prefix in (("review", "review"), ("study", "study_card"), ("triage", "triage")):
        resource = model_for(kind)
        if not resource:
            continue
        protocol = str(resource.get("protocol") or "")
        expanded[f"{prefix}_enabled"] = protocol != "none"
        if protocol.startswith("inherit_"):
            expanded[f"{prefix}_inherit"] = protocol.removeprefix("inherit_")
            continue
        if protocol == "none":
            expanded[f"{prefix}_model"] = ""
            continue
        endpoints = normalize_string_list(resource.get("endpoints"))
        expanded[f"{prefix}_base_url" if prefix == "review" else f"{prefix}_llm_base_url"] = (
            endpoints[0] if endpoints else ""
        )
        expanded[f"{prefix}_model"] = resource.get("model")
        expanded[f"{prefix}_api_key_env"] = resource.get("api_key_env")
        for key, value in (resource.get("options") or {}).items():
            option_key = key if key.startswith(f"{prefix}_") else f"{prefix}_{key}"
            expanded[option_key] = value

    image = model_for("image")
    if image:
        expanded["image_enabled"] = image.get("protocol") != "none"
        expanded["image_provider"] = image.get("protocol")
        expanded["image_model"] = image.get("model")
    return expanded


def validate_profile(
    profile_name: str,
    payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not PROFILE_ID_RE.fullmatch(profile_name):
        raise SettingsValidationError("profile id contains unsupported characters")
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    reject_inline_secrets(settings, "settings")
    cleaned = copy.deepcopy(settings)
    workflow_id = str(
        payload.get("workflow_id")
        or settings.get("workflow_id")
        or VIDEO_WORKFLOW_ID
    ).strip()
    workflow = PROFILE_WORKFLOWS.get(workflow_id)
    if not workflow:
        raise SettingsValidationError(f"unknown workflow: {workflow_id}")
    cleaned["workflow_id"] = workflow_id
    try:
        asr_worker_count = int(cleaned.get("asr_worker_count") or 5)
    except (TypeError, ValueError) as exc:
        raise SettingsValidationError("asr_worker_count must be an integer") from exc
    if not 1 <= asr_worker_count <= 5:
        raise SettingsValidationError("asr_worker_count must be between 1 and 5")
    cleaned["asr_worker_count"] = asr_worker_count
    asr_segmentation_mode = str(
        cleaned.get("asr_segmentation_mode") or ""
    ).strip().lower()
    if asr_segmentation_mode and asr_segmentation_mode not in ASR_SEGMENTATION_MODES:
        raise SettingsValidationError(
            f"unsupported ASR segmentation mode: {asr_segmentation_mode}"
        )
    if asr_segmentation_mode:
        cleaned["asr_segmentation_mode"] = asr_segmentation_mode
    try:
        vad_max_segment_sec = float(cleaned.get("vad_max_segment_sec") or 50)
    except (TypeError, ValueError) as exc:
        raise SettingsValidationError("vad_max_segment_sec must be a number") from exc
    if not 1 <= vad_max_segment_sec <= 55:
        raise SettingsValidationError(
            "vad_max_segment_sec must be between 1 and 55"
        )
    cleaned["vad_max_segment_sec"] = vad_max_segment_sec
    asr_chunk_mode = str(cleaned.get("asr_chunk_mode") or "model_default").strip()
    if asr_chunk_mode not in ASR_CHUNK_MODES:
        raise SettingsValidationError(f"unsupported ASR chunk mode: {asr_chunk_mode}")
    cleaned["asr_chunk_mode"] = asr_chunk_mode
    if asr_chunk_mode == "custom":
        normalized_chunk_options: dict[str, float] = {}
        for key in ASR_CHUNK_OPTION_FIELDS:
            try:
                value = float(cleaned.get(key))
            except (TypeError, ValueError) as exc:
                raise SettingsValidationError(f"{key} must be a positive number") from exc
            if value < 0 or (key != "chunk_overlap_sec" and value == 0):
                raise SettingsValidationError(
                    f"{key} must be a non-negative number"
                )
            normalized_chunk_options[key] = value
        if (
            normalized_chunk_options["chunk_overlap_sec"]
            >= normalized_chunk_options["chunk_duration_sec"]
        ):
            raise SettingsValidationError(
                "chunk_overlap_sec must be smaller than chunk_duration_sec"
            )
        cleaned.update(normalized_chunk_options)
    else:
        for key in ASR_CHUNK_OPTION_FIELDS:
            cleaned.pop(key, None)
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    catalog = build_settings_document(config)["models"]
    selected_asr: dict[str, Any] | None = None
    for slot, spec in workflow["model_fields"].items():
        field = spec["field"]
        kind = spec["kind"]
        model_id = str(models.get(slot) or payload.get(field) or "").strip()
        if not model_id and slot == "asr_fallback":
            model_id = "asr-disabled"
        if not model_id and slot == "diarization_fallback":
            model_id = "diarization-disabled"
        if not model_id and not spec.get("required"):
            disabled = f"{kind}-disabled"
            if disabled in catalog:
                model_id = disabled
        if not model_id:
            raise SettingsValidationError(f"profile requires a {slot} model")
        resource = catalog.get(model_id)
        if not resource or resource.get("kind") != kind:
            raise SettingsValidationError(f"unknown {slot} model: {model_id}")
        if (
            workflow_id == AUDIO_WORKFLOW_ID
            and slot == "diarization_fallback"
            and resource.get("protocol") not in {"asr_embedded", "none"}
        ):
            raise SettingsValidationError(
                "diarization_fallback must use ASR embedded speakers or be disabled"
            )
        if spec.get("required") and resource.get("protocol") == "none":
            raise SettingsValidationError(f"{slot} model cannot be disabled")
        if slot == "asr":
            selected_asr = resource
        cleaned[field] = model_id
    if selected_asr:
        protocol = str(selected_asr.get("protocol") or "")
        default_segmentation = str(
            (selected_asr.get("options") or {}).get("segmentation_mode")
            or ("fixed" if protocol == "qwen3_asr_http" else "native")
        )
        segmentation = str(
            cleaned.get("asr_segmentation_mode") or default_segmentation
        )
        allowed = {
            "firered_asr2_http": {"fixed", "vad"},
            "qwen3_asr_http": {"fixed"},
            "vibevoice_http": {"native"},
        }.get(protocol)
        if allowed and segmentation not in allowed:
            raise SettingsValidationError(
                f"{protocol} does not support ASR segmentation mode: {segmentation}"
            )
        cleaned["asr_segmentation_mode"] = segmentation
        if protocol == "firered_asr2_http" and asr_chunk_mode == "custom":
            if normalized_chunk_options["single_pass_max_duration_sec"] > 55:
                raise SettingsValidationError(
                    "FireRedASR2 single-pass duration must not exceed 55 seconds"
                )
            if normalized_chunk_options["chunk_duration_sec"] > 55:
                raise SettingsValidationError(
                    "FireRedASR2 chunk duration must not exceed 55 seconds"
                )
    cleaned["label"] = str(payload.get("label") or profile_name).strip()[:120]
    cleaned["description"] = str(payload.get("description") or "").strip()[:500]
    return expand_runtime_profile(config, cleaned)


def _test_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_local_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return host.endswith(".local") or host.endswith(".ts.net")


def _test_session(url: str) -> requests.Session:
    session = requests.Session()
    if _is_local_url(url):
        session.trust_env = False
    return session


def _health_url_for(resource: dict[str, Any]) -> str:
    configured = str(resource.get("health_url") or "").strip()
    if configured:
        return configured
    endpoints = normalize_string_list(resource.get("endpoints"))
    if not endpoints:
        return ""
    endpoint = endpoints[0]
    protocol = str(resource.get("protocol") or "")
    if protocol in {
        "vibevoice_http",
        "qwen3_asr_http",
        "generic_http",
        "firered_asr2_http",
        "firered_3dspeaker_http",
    }:
        parsed = urlparse(endpoint)
        return parsed._replace(path="/api/health", params="", query="", fragment="").geturl()
    if protocol in {
        "unlimited_ocr_openai",
        "dots_ocr_openai",
        "dots_mocr_openai",
        "openai_vision",
        "openai_compatible",
    }:
        return f"{endpoint.rstrip('/')}/models"
    return endpoint


def _auth_headers(resource: dict[str, Any]) -> tuple[dict[str, str], str]:
    env_name = str(resource.get("api_key_env") or "").strip()
    if env_name and not os.environ.get(env_name):
        from .config import load_default_deepseek_env

        load_default_deepseek_env()
    if env_name and not os.environ.get(env_name):
        return {}, f"缺少环境变量 {env_name}"
    token = os.environ.get(env_name, "0") if env_name else "0"
    return {"Authorization": f"Bearer {token}"}, ""


def _tiny_image_data_url() -> str:
    image = Image.new("RGB", (320, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 36, 302, 144), outline="black", width=3)
    draw.text((74, 78), "TEST 123", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _tiny_wav_bytes(duration_seconds: float = 1.5) -> bytes:
    sample_rate = 16000
    frame_count = int(sample_rate * duration_seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            envelope = min(1.0, index / 800, (frame_count - index) / 800)
            value = int(6000 * envelope * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


def _response_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        return str(message.get("content") or message.get("reasoning_content") or "").strip()
    return str(payload.get("text") or "").strip()


def _friendly_test_error(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.MissingSchema):
        return f"端点 URL 无效：{exc.request.url if exc.request else exc}"
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "连接端点超时"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return "端点响应超时，模型可能仍在冷启动"
    if isinstance(exc, requests.exceptions.ConnectionError):
        request = getattr(exc, "request", None)
        parsed = urlparse(request.url) if request and request.url else None
        if parsed and parsed.hostname:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            return f"无法连接 {parsed.hostname}:{port}"
        return "无法连接模型端点"
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return f"端点返回 HTTP {exc.response.status_code}"
    return str(exc)[:500]


class RuntimeSettingsStore:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.default_path = self.repo_root / "video_analyzer" / "config" / "default_config.json"
        self.user_path = self.repo_root / "config" / "config.json"
        self._test_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._test_cache_lock = threading.Lock()

    def _read(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SettingsValidationError(f"configuration root must be an object: {path}")
        return data

    def load(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        raw_defaults = self._read(self.default_path)
        user = self._read(self.user_path)
        from .config import resolve_endpoint_config

        defaults = resolve_endpoint_config(raw_defaults)
        merged = resolve_endpoint_config(
            apply_disabled_runtime_profiles(deep_merge(raw_defaults, user))
        )
        return defaults, user, merged

    def _profile_test_runtime_config(
        self,
        merged: dict[str, Any],
        profile_name: str,
        workflow: dict[str, Any],
        refs: dict[str, Any],
    ) -> dict[str, Any]:
        profiles = merged.get("runtime_profiles") or {}
        profile = copy.deepcopy(profiles.get(profile_name) or {})
        profile["workflow_id"] = workflow["id"]
        for slot, spec in workflow["model_fields"].items():
            profile[spec["field"]] = str(refs.get(slot) or "").strip()
        expanded = expand_runtime_profile(merged, profile)

        runtime_config = copy.deepcopy(merged)
        asr = runtime_config.setdefault("asr", {})
        vibevoice = asr.setdefault("vibevoice", {})
        asr["provider"] = expanded.get("asr_provider")
        for key in (
            "vibevoice_url",
            "vibevoice_urls",
            "worker_count",
            "use_native_chunking",
            "single_pass_max_duration_sec",
            "chunk_duration_sec",
            "chunk_overlap_sec",
            "chunk_parallel_workers",
            "qwen3_asr_url",
            "qwen3_asr_model",
            "qwen3_asr_options",
            "firered_asr2_url",
            "firered_asr2_options",
        ):
            if key in expanded:
                vibevoice[key] = copy.deepcopy(expanded[key])

        ocr = runtime_config.setdefault("ocr", {})
        ocr["provider"] = expanded.get("ocr_provider")
        ocr["base_url"] = expanded.get("ocr_base_url")
        ocr["base_urls"] = copy.deepcopy(expanded.get("ocr_base_urls") or [])
        ocr["engine"] = expanded.get("ocr_engine")
        ocr["worker_count"] = expanded.get("ocr_worker_count")
        for expanded_key, runtime_key in (
            ("ocr_concurrency", "concurrency"),
            ("ocr_cache", "cache"),
            ("ocr_timeout_seconds", "timeout_seconds"),
            ("ocr_max_tokens", "max_tokens"),
            ("ocr_max_image_long_side", "max_image_long_side"),
            ("ocr_image_mode", "image_mode"),
        ):
            if expanded.get(expanded_key) is not None:
                ocr[runtime_key] = copy.deepcopy(expanded[expanded_key])

        manual = runtime_config.setdefault("operation_manual", {})
        manual["vision_base_url"] = expanded.get("vision_base_url")
        manual["vision_model"] = expanded.get("vision_model")
        manual["vision_runtime"] = copy.deepcopy(expanded.get("vision_runtime") or {})
        return runtime_config

    def _model_test_runtime_config(
        self,
        merged: dict[str, Any],
        item: dict[str, Any],
    ) -> dict[str, Any]:
        runtime_config = copy.deepcopy(merged)
        kind = str(item.get("kind") or "")
        protocol = str(item.get("protocol") or "")
        endpoints = normalize_string_list(item.get("endpoints"))
        options = copy.deepcopy(item.get("options") or {})

        if kind == "asr":
            asr = runtime_config.setdefault("asr", {})
            vibevoice = asr.setdefault("vibevoice", {})
            asr["provider"] = {
                "vibevoice_http": "vibevoice",
                "qwen3_asr_http": "qwen3_asr",
                "firered_asr2_http": "firered_asr2",
                "generic_http": "remote_http",
            }.get(protocol, protocol)
            if protocol == "vibevoice_http":
                vibevoice["deep_remote_urls"] = endpoints
            elif protocol == "qwen3_asr_http":
                vibevoice["qwen3_asr_url"] = endpoints[0] if endpoints else ""
                vibevoice["qwen3_asr_model"] = item.get("model")
                vibevoice["qwen3_asr_options"] = options
            elif protocol == "firered_asr2_http":
                vibevoice["firered_asr2_url"] = endpoints[0] if endpoints else ""
                vibevoice["firered_asr2_options"] = options
            elif protocol == "generic_http":
                vibevoice["remote_urls"] = endpoints
        elif kind == "ocr":
            ocr = runtime_config.setdefault("ocr", {})
            ocr["provider"] = {
                "unlimited_ocr_openai": "unlimited_ocr",
                "dots_ocr_openai": "dots_ocr",
                "dots_mocr_openai": "dots_mocr_vllm",
                "openai_vision": "openai_vision",
            }.get(protocol, protocol)
            ocr["base_url"] = endpoints[0] if endpoints else ""
            ocr["base_urls"] = endpoints
            ocr["engine"] = options.get("engine")
            ocr["worker_count"] = options.get("worker_count")
            for key in (
                "concurrency",
                "cache",
                "timeout_seconds",
                "max_tokens",
                "max_image_long_side",
                "image_mode",
            ):
                if options.get(key) is not None:
                    ocr[key] = copy.deepcopy(options[key])
        elif kind == "vision":
            manual = runtime_config.setdefault("operation_manual", {})
            manual["vision_base_url"] = endpoints[0] if endpoints else ""
            manual["vision_model"] = item.get("model")
            manual["vision_runtime"] = options
        return runtime_config

    def public_settings(self) -> dict[str, Any]:
        defaults, user, merged = self.load()
        document = build_settings_document(merged)
        built_in_models = set(build_settings_document(defaults)["models"])
        built_in_profiles = set((defaults.get("runtime_profiles") or {}))
        for model_id, item in document["models"].items():
            item["built_in"] = model_id in built_in_models
            item["overridden"] = model_id in (user.get("model_catalog") or {})
        for profile_name, item in document["profiles"].items():
            item["built_in"] = profile_name in built_in_profiles
            item["overridden"] = profile_name in (user.get("runtime_profiles") or {})
            item["name"] = profile_name
        document["models"] = sorted(document["models"].values(), key=lambda item: (item["kind"], item["name"]))
        document["profiles"] = sorted(document["profiles"].values(), key=lambda item: item["name"])
        return document

    def save_model(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = validate_model_resource(model_id, payload)
        _defaults, user, _merged = self.load()
        user.setdefault("model_catalog", {})[model_id] = {key: value for key, value in cleaned.items() if key != "id"}
        self._write_user(user)
        return next(item for item in self.public_settings()["models"] if item["id"] == model_id)

    def delete_model(self, model_id: str) -> dict[str, Any]:
        _defaults, user, merged = self.load()
        document = build_settings_document(merged)
        references = [
            profile_name
            for profile_name, profile in document["profiles"].items()
            if model_id
            in [
                str(profile.get(spec["field"]) or "")
                for spec in ALL_PROFILE_MODEL_FIELDS.values()
            ]
        ]
        if references:
            raise SettingsValidationError(f"model is used by profiles: {', '.join(references)}")
        catalog = user.get("model_catalog") or {}
        if model_id not in catalog:
            raise FileNotFoundError(model_id)
        del catalog[model_id]
        if not catalog:
            user.pop("model_catalog", None)
        self._write_user(user)
        return {"status": "deleted", "id": model_id}

    def save_profile(self, profile_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        _defaults, user, merged = self.load()
        disabled = normalize_string_list(user.get("disabled_runtime_profiles"))
        if profile_name in disabled:
            disabled.remove(profile_name)
            if disabled:
                user["disabled_runtime_profiles"] = disabled
            else:
                user.pop("disabled_runtime_profiles", None)
            merged = apply_disabled_runtime_profiles(deep_merge(self._read(self.default_path), user))
        document = build_settings_document(merged)
        models_by_id = {item["id"]: item for item in document["models"].values()}
        requested = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        user_catalog = user.setdefault("model_catalog", {})
        for model_id in requested.values():
            if (
                model_id in models_by_id
                and model_id not in user_catalog
                and models_by_id[model_id].get("source") != "control"
            ):
                user_catalog[model_id] = {
                    key: copy.deepcopy(value)
                    for key, value in models_by_id[model_id].items()
                    if key not in {"id", "source", "built_in", "overridden"}
                }
        merged = deep_merge(self._read(self.default_path), user)
        cleaned = validate_profile(profile_name, payload, merged)
        user.setdefault("runtime_profiles", {})[profile_name] = cleaned
        self._write_user(user)
        return next(item for item in self.public_settings()["profiles"] if item["name"] == profile_name)

    def delete_profile(self, profile_name: str) -> dict[str, Any]:
        defaults, user, merged = self.load()
        if str(merged.get("active_runtime_profile") or "") == profile_name:
            raise SettingsValidationError("active profile cannot be deleted")
        profiles = user.get("runtime_profiles") or {}
        built_in = profile_name in (defaults.get("runtime_profiles") or {})
        if built_in:
            profiles.pop(profile_name, None)
            if not profiles:
                user.pop("runtime_profiles", None)
            disabled = normalize_string_list(user.get("disabled_runtime_profiles"))
            if profile_name not in disabled:
                disabled.append(profile_name)
            user["disabled_runtime_profiles"] = disabled
        else:
            if profile_name not in profiles:
                raise FileNotFoundError(profile_name)
            del profiles[profile_name]
            if not profiles:
                user.pop("runtime_profiles", None)
        self._write_user(user)
        return {
            "status": "disabled" if built_in else "deleted",
            "name": profile_name,
        }

    def activate_profile(self, profile_name: str) -> dict[str, Any]:
        _defaults, user, merged = self.load()
        if profile_name not in (merged.get("runtime_profiles") or {}):
            raise FileNotFoundError(profile_name)
        user["active_runtime_profile"] = profile_name
        self._write_user(user)
        return {"active_runtime_profile": profile_name}

    def test_model(
        self,
        model_id: str,
        mode: str = "quick",
        *,
        force: bool = False,
        context: str = "",
        _acquire_lock: bool = True,
    ) -> dict[str, Any]:
        if mode not in MODEL_TEST_MODES:
            raise SettingsValidationError(f"unsupported test mode: {mode}")
        document = self.public_settings()
        item = next((candidate for candidate in document["models"] if candidate["id"] == model_id), None)
        if not item:
            raise FileNotFoundError(model_id)
        cache_key = hashlib.sha256(
            json.dumps(
                {"mode": mode, "item": item, "context": context[:1000]},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if not force:
            with self._test_cache_lock:
                cached = self._test_cache.get(cache_key)
            if cached and time.monotonic() - cached[0] < MODEL_TEST_CACHE_SECONDS:
                result = copy.deepcopy(cached[1])
                result["cached"] = True
                return result

        _defaults, _user, merged = self.load()
        endpoints = normalize_string_list(item.get("endpoints"))
        needs_local_stage = (
            _acquire_lock
            and mode != "quick"
            and any(_is_local_url(endpoint) for endpoint in endpoints)
        )
        lock_context = contextlib.nullcontext()
        if needs_local_stage:
            stage = {"asr": "asr", "ocr": "ocr", "vision": "vl"}.get(
                str(item.get("kind") or "")
            )
            if stage:
                from .local_model_runtime import local_model_stage

                lock_context = local_model_stage(
                    stage,
                    self._model_test_runtime_config(merged, item),
                    logger,
                    f"settings-model-test:{model_id}",
                )
            else:
                from .local_model_runtime import local_model_runtime_lock

                lock_context = local_model_runtime_lock(
                    merged,
                    logger,
                    f"settings-model-test:{model_id}",
                    stage=str(item.get("kind") or "text"),
                )
        started = time.monotonic()
        try:
            with lock_context:
                result = (
                    self._quick_test_resource(item)
                    if mode == "quick"
                    else self._inference_test_resource(item, context=context)
                )
        except Exception as exc:
            result = {
                "ok": False,
                "status": "failed",
                "detail": f"本地模型阶段启动失败：{_friendly_test_error(exc)}",
            }
        result.update(
            {
                "model_id": model_id,
                "kind": item.get("kind"),
                "mode": mode,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "checked_at": _test_timestamp(),
                "cached": False,
            }
        )
        with self._test_cache_lock:
            self._test_cache[cache_key] = (time.monotonic(), copy.deepcopy(result))
        return result

    def _quick_test_resource(self, item: dict[str, Any]) -> dict[str, Any]:
        protocol = str(item.get("protocol") or "")
        if protocol in {"none", "inherit_text", "inherit_study", "asr_embedded"}:
            status = "disabled" if protocol == "none" else "inherited"
            return {"ok": True, "status": status, "detail": item.get("name") or protocol}
        if protocol == "codex_imagegen":
            import shutil

            executable = shutil.which("codex")
            return {
                "ok": bool(executable),
                "status": "reachable" if executable else "missing",
                "detail": executable or "未找到 codex 命令",
            }
        if protocol in {
            "three_d_speaker",
            "pyannote_community",
            "wespeaker_diarization",
            "faster_whisper",
        }:
            return {"ok": True, "status": "configured", "detail": "本地进程内能力，将在真实任务中加载"}
        target = _health_url_for(item)
        if not target:
            return {"ok": False, "status": "invalid", "detail": "未配置可探测端点"}
        if urlparse(target).scheme not in {"http", "https"}:
            return {"ok": False, "status": "invalid", "detail": f"端点 URL 无效：{target}"}
        headers, auth_error = _auth_headers(item)
        if auth_error:
            return {"ok": False, "status": "missing_credentials", "detail": auth_error}
        try:
            response = _test_session(target).get(target, headers=headers, timeout=8)
            response.raise_for_status()
            payload = response.json() if "json" in response.headers.get("Content-Type", "").lower() else {}
            service_status = str(payload.get("status") or "").lower() if isinstance(payload, dict) else ""
            ready = payload.get("ready") if isinstance(payload, dict) else None
            if ready is False or service_status in {"sleeping", "idle", "unloaded"}:
                return {"ok": True, "status": "sleeping", "detail": f"HTTP {response.status_code}，代理可用，模型休眠"}
            if (
                target.endswith("/models")
                and item.get("model")
                and isinstance(payload, dict)
                and not (item.get("options") or {}).get("skip_model_inventory_check")
            ):
                available = {
                    str(model.get("id") or model.get("name") or "")
                    for model in (payload.get("data") or [])
                    if isinstance(model, dict)
                }
                if available and str(item["model"]) not in available:
                    return {
                        "ok": False,
                        "status": "model_missing",
                        "detail": f"端点可达，但未发现模型 {item['model']}",
                    }
            return {"ok": True, "status": "reachable", "detail": f"HTTP {response.status_code}"}
        except Exception as exc:
            host = urlparse(target).hostname or ""
            if (
                isinstance(exc, requests.exceptions.ConnectionError)
                and host in {"127.0.0.1", "localhost", "::1"}
            ):
                return {
                    "ok": True,
                    "status": "sleeping",
                    "detail": "本地按需服务当前未启动；最小推理会自动冷启动",
                }
            return {"ok": False, "status": "unreachable", "detail": _friendly_test_error(exc)}

    def _inference_test_resource(self, item: dict[str, Any], *, context: str = "") -> dict[str, Any]:
        protocol = str(item.get("protocol") or "")
        kind = str(item.get("kind") or "")
        if protocol in {"none", "inherit_text", "inherit_study", "asr_embedded"}:
            return self._quick_test_resource(item)
        if protocol == "codex_imagegen":
            quick = self._quick_test_resource(item)
            if quick["ok"]:
                quick.update({"status": "auth_only", "detail": "已验证本机命令；为避免生成费用，未实际出图"})
            return quick
        if protocol in {
            "three_d_speaker",
            "pyannote_community",
            "wespeaker_diarization",
            "faster_whisper",
        }:
            return {
                "ok": True,
                "status": "configured",
                "detail": "本地进程内模型不在轻量测试中加载",
            }

        endpoints = normalize_string_list(item.get("endpoints"))
        if not endpoints:
            return {"ok": False, "status": "invalid", "detail": "未配置推理端点"}
        endpoint = endpoints[0]
        if urlparse(endpoint).scheme not in {"http", "https"}:
            return {"ok": False, "status": "invalid", "detail": f"端点 URL 无效：{endpoint}"}
        headers, auth_error = _auth_headers(item)
        if auth_error:
            return {"ok": False, "status": "missing_credentials", "detail": auth_error}
        timeout = 900 if _is_local_url(endpoint) else 60
        session = _test_session(endpoint)
        try:
            if protocol in {
                "vibevoice_http",
                "qwen3_asr_http",
                "generic_http",
                "firered_asr2_http",
                "firered_3dspeaker_http",
            }:
                response = session.post(
                    endpoint,
                    files={"audio": ("pathway-smoke.wav", _tiny_wav_bytes(), "audio/wav")},
                    data={"use_native_chunking": "false", "single_pass_max_duration_sec": "5"},
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("success") is False or payload.get("error"):
                    raise RuntimeError(str(payload.get("error") or payload))
                text = str(payload.get("text") or "").strip()
                return {
                    "ok": True,
                    "status": "passed",
                    "detail": "微型音频请求成功" + (f"，返回 {len(text)} 字" if text else "，测试音为非语音音调"),
                    "sample": text[:160],
                }
            if protocol == "openai_audio":
                response = session.post(
                    endpoint,
                    headers=headers,
                    files={"file": ("pathway-smoke.wav", _tiny_wav_bytes(), "audio/wav")},
                    data={"model": item.get("model"), "response_format": "verbose_json"},
                    timeout=timeout,
                )
                response.raise_for_status()
                text = str(response.json().get("text") or "").strip()
                return {"ok": True, "status": "passed", "detail": "微型音频请求成功", "sample": text[:160]}

            prompt = "只回复 OK。"
            content: Any = prompt
            if kind in {"ocr", "vision"}:
                prompt = "识别图片中的文字，只返回识别结果。"
                image_part = {"type": "image_url", "image_url": {"url": _tiny_image_data_url()}}
                content = (
                    [
                        image_part,
                        {
                            "type": "text",
                            "text": f"<|img|><|imgpad|><|endofimg|>{prompt}",
                        },
                    ]
                    if protocol in {
                        "unlimited_ocr_openai",
                        "dots_ocr_openai",
                        "dots_mocr_openai",
                    }
                    else [{"type": "text", "text": prompt}, image_part]
                )
            elif context:
                prompt = f"这是上游通路摘要：\n{context[:1800]}\n只回复 PIPELINE_OK。"
                content = prompt
            payload = {
                "model": item.get("model"),
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 16,
            }
            reasoning_effort = (item.get("options") or {}).get("reasoning_effort")
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
            request_headers = {**headers, "Content-Type": "application/json"}
            response = session.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                headers=request_headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            text = _response_text(response.json())
            if not text:
                raise RuntimeError("模型返回成功，但响应内容为空")
            return {
                "ok": True,
                "status": "passed",
                "detail": f"最小推理成功，返回 {len(text)} 字",
                "sample": text[:160],
            }
        except Exception as exc:
            return {"ok": False, "status": "failed", "detail": _friendly_test_error(exc)}

    def test_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode") or "quick")
        if mode not in MODEL_TEST_MODES:
            raise SettingsValidationError(f"unsupported test mode: {mode}")
        force = bool(payload.get("force"))
        workflow_id = str(payload.get("workflow_id") or VIDEO_WORKFLOW_ID)
        workflow = PROFILE_WORKFLOWS.get(workflow_id)
        if not workflow:
            raise SettingsValidationError(f"unknown workflow: {workflow_id}")
        refs = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        document = self.public_settings()
        models_by_id = {item["id"]: item for item in document["models"]}
        flow = workflow["flow"]
        selected: dict[str, dict[str, Any]] = {}
        for slot, spec in workflow["model_fields"].items():
            model_id = str(refs.get(slot) or "").strip()
            if not model_id and not spec.get("required"):
                model_id = f"{spec['kind']}-disabled"
            item = models_by_id.get(model_id)
            if not item or item.get("kind") != spec["kind"]:
                raise SettingsValidationError(f"unknown {slot} model: {model_id}")
            selected[slot] = item

        started = time.monotonic()
        results: dict[str, dict[str, Any]] = {}
        context_parts: list[str] = []
        _defaults, _user, merged = self.load()
        runtime_config = self._profile_test_runtime_config(
            merged,
            str(payload.get("profile_name") or ""),
            workflow,
            refs,
        )
        needs_lock = mode != "quick" and any(
            _is_local_url(endpoint)
            for item in selected.values()
            for endpoint in normalize_string_list(item.get("endpoints"))
        )
        lock_context = contextlib.nullcontext()
        if needs_lock:
            from .local_model_runtime import local_model_runtime_session

            lock_context = local_model_runtime_session(
                runtime_config,
                logger,
                f"settings-profile-test:{payload.get('profile_name') or 'draft'}",
        )
        with lock_context:
            tested_slots: dict[str, dict[str, Any]] = {}
            for node in sorted(flow["nodes"], key=lambda item: item["mobile_order"]):
                slot = node.get("model_slot") or node.get("model_kind")
                if not slot:
                    continue
                if slot in tested_slots:
                    results[node["id"]] = {
                        **tested_slots[slot],
                        "node_id": node["id"],
                        "reused_slot_result": slot,
                    }
                    continue
                item = selected[slot]
                stage = {"asr": "asr", "ocr": "ocr", "vision": "vl"}.get(
                    str(item.get("kind") or "")
                )
                stage_context = contextlib.nullcontext()
                if stage and mode != "quick":
                    from .local_model_runtime import local_model_stage

                    stage_context = local_model_stage(
                        stage,
                        runtime_config,
                        logger,
                        f"settings-profile-test:{payload.get('profile_name') or 'draft'}:{stage}",
                    )
                try:
                    with stage_context:
                        result = self.test_model(
                            item["id"],
                            "quick" if mode == "quick" else "inference",
                            force=force or mode == "pathway",
                            context="\n".join(context_parts) if mode == "pathway" else "",
                            _acquire_lock=False,
                        )
                except Exception as exc:
                    result = {
                        "ok": False,
                        "status": "failed",
                        "detail": f"本地模型阶段启动失败：{_friendly_test_error(exc)}",
                        "model_id": item["id"],
                        "kind": item.get("kind"),
                        "mode": "inference",
                        "elapsed_ms": 0,
                        "checked_at": _test_timestamp(),
                        "cached": False,
                    }
                result["node_id"] = node["id"]
                results[node["id"]] = result
                tested_slots[slot] = dict(result)
                sample = str(result.get("sample") or "").strip()
                if result.get("ok") and sample:
                    context_parts.append(f"{slot}: {sample}")

        model_ok = all(result.get("ok") for result in results.values())
        incoming: dict[str, list[str]] = {}
        for edge in flow["edges"]:
            incoming.setdefault(edge["to"], []).append(edge["from"])
        for flow_node in sorted(flow["nodes"], key=lambda item: item["mobile_order"]):
            if flow_node.get("model_kind"):
                continue
            dependencies_ok = all(results.get(node_id, {}).get("ok") for node_id in incoming.get(flow_node["id"], []))
            fixed_ok = dependencies_ok if mode == "pathway" else True
            results[flow_node["id"]] = {
                "node_id": flow_node["id"],
                "ok": fixed_ok,
                "status": "passed" if mode == "pathway" and fixed_ok else ("blocked" if not fixed_ok else "configured"),
                "detail": "流程衔接正常" if fixed_ok else "上游节点存在失败",
                "mode": mode,
            }
        elapsed_ms = round((time.monotonic() - started) * 1000)
        failed = [item for item in results.values() if not item.get("ok") and item.get("status") != "blocked"]
        blocked = [item for item in results.values() if item.get("status") == "blocked"]
        return {
            "ok": model_ok and not failed,
            "mode": mode,
            "profile_name": str(payload.get("profile_name") or ""),
            "workflow_id": workflow_id,
            "elapsed_ms": elapsed_ms,
            "checked_at": _test_timestamp(),
            "results": results,
            "summary": {
                "passed": sum(1 for item in results.values() if item.get("ok")),
                "failed": len(failed),
                "blocked": len(blocked),
                "total": len(results),
                "detail": "全链路通路正常" if not failed else str(failed[0].get("detail") or "通路测试失败"),
            },
        }

    def _write_user(self, data: dict[str, Any]) -> None:
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=self.user_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_name, self.user_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
