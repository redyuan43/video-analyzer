#!/usr/bin/env python3
"""Audio-first template analysis for uploaded recorder files."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_analyzer.artifacts import write_json, write_transcript_markdown  # noqa: E402
from video_analyzer.asr_providers import (  # noqa: E402
    ASRStrategyResult,
    extract_audio_to_wav,
    transcribe_with_provider_result,
    transcribe_with_strategy,
)
from video_analyzer.audio_processor import AudioProcessor, AudioTranscript  # noqa: E402
from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient  # noqa: E402
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature  # noqa: E402
from video_analyzer.local_model_runtime import local_model_runtime_session, local_model_stage  # noqa: E402
from video_analyzer.resource_locks import analyzer_resource_lock  # noqa: E402
from video_analyzer.speaker_diarization import refine_transcript_speakers  # noqa: E402


DEFAULT_TEMPLATE_CATALOG = REPO_ROOT / "video-analyzer-ui" / "video_analyzer_ui" / "static" / "data" / "audio_prompt_templates.json"
DEFAULT_TEMPLATE_ID = "auto"
MAX_TRANSCRIPT_CHARS_FOR_CLASSIFY = 9000
MAX_TRANSCRIPT_CHARS_FOR_SUMMARY = 28000
MAX_TRANSCRIPT_CHARS_FOR_GUIDE = 18000
CLASSIFICATION_CANDIDATE_LIMIT = 48
logger = logging.getLogger("audio_template_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe uploaded audio, choose a prompt template, and summarize it.")
    parser.add_argument("media_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile")
    parser.add_argument("--focus-prompt", default="")
    parser.add_argument("--template-catalog", default=str(DEFAULT_TEMPLATE_CATALOG))
    parser.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID)
    parser.add_argument("--language", default="zh-CN")
    parser.add_argument("--source-name", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    media_path = Path(args.media_path).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not media_path.is_file():
        raise FileNotFoundError(f"missing media file: {media_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = load_operation_config(args)
    templates = load_templates(Path(args.template_catalog))

    audio_path = extract_audio_to_wav(media_path, output_dir)
    if audio_path is None:
        raise RuntimeError(f"audio extraction produced no audio stream: {media_path}")

    transcript, asr_result = transcribe_audio(audio_path, output_dir, config)
    if transcript is None or not transcript.text.strip():
        raise RuntimeError("Required ASR transcript was not produced for uploaded audio")
    transcript, speaker_report = refine_audio_speakers(audio_path, transcript, output_dir, config)
    if asr_result:
        asr_result.transcript = transcript
    transcript_path = write_transcript_markdown(transcript, output_dir / "transcript.md")

    selector_client, selector_model, selector_base_url, _selector_temperature = build_template_selector_client(config)
    content_client, content_model, content_base_url, content_temperature = build_content_analysis_client(config, args.profile)
    selected, classification = choose_template(
        client=selector_client,
        model=selector_model,
        templates=templates,
        transcript_text=transcript.text,
        focus_prompt=args.focus_prompt,
        explicit_template_id=args.template_id,
    )
    summary = summarize_with_template(
        client=content_client,
        model=content_model,
        template=selected,
        transcript_text=transcript.text,
        focus_prompt=args.focus_prompt,
        language=args.language,
        temperature=content_temperature,
        source_name=args.source_name or media_path.name,
    )
    write_audio_only_manifest(output_dir, media_path, audio_path)
    build_light_study_guide(content_client, content_model, output_dir, transcript, summary, content_temperature)

    manual_path = write_operation_manual(output_dir, selected, classification, summary, args.focus_prompt)
    evidence_path = write_manual_evidence(output_dir, media_path, transcript_path, selected, classification, asr_result)
    analysis_path = write_analysis_json(
        output_dir=output_dir,
        media_path=media_path,
        transcript=transcript,
        asr_result=asr_result,
        speaker_report=speaker_report,
        selected_template=selected,
        classification=classification,
        summary=summary,
        selector_base_url=selector_base_url,
        selector_model=selector_model,
        content_base_url=content_base_url,
        content_model=content_model,
        manual_path=manual_path,
        evidence_path=evidence_path,
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )

    print(f"[audio-template] transcript: {transcript_path}")
    print(f"[audio-template] selected_template: {selected.get('title_zh') or selected.get('title')}")
    print(f"[done] manual: {manual_path}")
    print(f"[done] analysis: {analysis_path}")
    print(f"[done] run_dir: {output_dir}")
    return 0


def load_operation_config(args: argparse.Namespace) -> Config:
    config = Config(args.config)
    config.update_from_args(
        argparse.Namespace(
            task="operation_manual",
            profile=args.profile,
            client=None,
            asr_provider=None,
        )
    )
    return config


def load_templates(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"template catalog must be a list: {path}")
    templates = [
        item
        for item in data
        if isinstance(item, dict) and item.get("id") and item.get("prompt_original")
    ]
    if not templates:
        raise ValueError(f"template catalog is empty: {path}")
    return templates


def transcribe_audio(audio_path: Path, output_dir: Path, config: Config) -> tuple[AudioTranscript | None, ASRStrategyResult | None]:
    asr_config = config.get("asr", {})
    provider = asr_config.get("provider", "faster_whisper")
    asr_result: ASRStrategyResult | None = None
    transcript: AudioTranscript | None = None
    asr_lock = (
        contextlib.nullcontext()
        if provider == "none"
        else analyzer_resource_lock(config.config, "asr", str(output_dir), logger)
    )
    with local_model_runtime_session(config.config, logger, str(output_dir)):
        with asr_lock:
            with local_model_stage("asr", config.config, logger, str(output_dir)):
                if provider == "auto":
                    strategy = asr_config.get("strategy", "balanced")
                    speaker_config = config.get("speaker_diarization") or {}
                    if (
                        strategy == "balanced"
                        and truthy_config_value(speaker_config.get("enabled"), default=True)
                        and truthy_config_value(speaker_config.get("force_deep_asr"), default=True)
                    ):
                        strategy = "deep"
                    asr_result = transcribe_with_strategy(
                        strategy=strategy,
                        audio_path=audio_path,
                        language=config.get("audio", {}).get("language", ""),
                        whisper_model=config.get("audio", {}).get("whisper_model", "medium"),
                        device=config.get("audio", {}).get("device", "cpu"),
                        vibevoice_config=asr_config.get("vibevoice", {}),
                    )
                    transcript = asr_result.transcript
                elif provider == "faster_whisper":
                    processor = AudioProcessor(
                        language=config.get("audio", {}).get("language", ""),
                        model_size_or_path=config.get("audio", {}).get("whisper_model", "medium"),
                        device=config.get("audio", {}).get("device", "cpu"),
                    )
                    transcript = processor.transcribe(audio_path)
                else:
                    asr_result = transcribe_with_provider_result(
                        provider=provider,
                        audio_path=audio_path,
                        language=config.get("audio", {}).get("language", ""),
                        whisper_model=config.get("audio", {}).get("whisper_model", "medium"),
                        device=config.get("audio", {}).get("device", "cpu"),
                        vibevoice_config=asr_config.get("vibevoice", {}),
                    )
                    transcript = asr_result.transcript
    if asr_result is None:
        asr_result = ASRStrategyResult(
            strategy=f"provider:{provider}",
            transcript=transcript,
            fast_transcript=transcript,
            providers_run=[] if provider == "none" else [provider],
        )
    return transcript, asr_result


def refine_audio_speakers(
    audio_path: Path,
    transcript: AudioTranscript,
    output_dir: Path,
    config: Config,
) -> tuple[AudioTranscript, dict[str, Any]]:
    speaker_config = config.get("speaker_diarization") or {}
    try:
        refined, report = refine_transcript_speakers(audio_path, transcript, speaker_config)
    except Exception as exc:
        logger.warning("speaker diarization refinement failed: %s", exc)
        report = {"enabled": True, "error": str(exc)}
        refined = transcript
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    write_json(qa_dir / "speaker_diarization_report.json", report)
    return refined, report


def truthy_config_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def build_template_selector_client(config: Config) -> tuple[GenericOpenAIAPIClient, str, str, float]:
    study_config = config.get("study_cards") or {}
    base_url = study_config.get("llm_base_url") or "http://agx.taild500c8.ts.net:11434/v1"
    model = study_config.get("model") or "qwen3:4b-instruct"
    temperature = float(study_config.get("temperature", 0.1))
    client = GenericOpenAIAPIClient(
        api_key=resolve_api_key(study_config.get("api_key"), study_config.get("api_key_env"), base_url),
        api_url=base_url,
        timeout_seconds=int(study_config.get("timeout_seconds", 600)),
        extra_body=build_openai_extra_body(study_config, base_url, prefix=""),
    )
    return client, model, base_url, temperature


def build_content_analysis_client(config: Config, profile_name: str | None) -> tuple[GenericOpenAIAPIClient, str, str, float]:
    profile = config.get_runtime_profile(profile_name)
    manual_config = config.get("operation_manual") or {}
    base_url = (
        profile.get("text_base_url")
        or profile.get("llm_base_url")
        or manual_config.get("text_base_url")
        or manual_config.get("llm_base_url")
    )
    model = profile.get("text_model") or manual_config.get("text_model") or profile.get("vision_model") or manual_config.get("vision_model")
    if not base_url or not model:
        raise ValueError(f"runtime profile {profile_name or '(default)'} is missing text model configuration")
    temperature = resolve_temperature(profile, resolve_temperature(manual_config, 0.2))
    api_key = resolve_api_key(
        profile.get("api_key") or manual_config.get("text_api_key") or manual_config.get("api_key"),
        profile.get("text_api_key_env") or profile.get("api_key_env") or manual_config.get("text_api_key_env") or manual_config.get("api_key_env"),
        base_url,
    )
    client = GenericOpenAIAPIClient(
        api_key=api_key,
        api_url=base_url,
        timeout_seconds=int(profile.get("timeout_seconds") or manual_config.get("timeout_seconds") or 600),
        extra_body=build_openai_extra_body(profile, base_url, prefix=""),
    )
    return client, model, base_url, temperature


def choose_template(
    client: GenericOpenAIAPIClient,
    model: str,
    templates: list[dict[str, Any]],
    transcript_text: str,
    focus_prompt: str,
    explicit_template_id: str = DEFAULT_TEMPLATE_ID,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if explicit_template_id and explicit_template_id != DEFAULT_TEMPLATE_ID:
        selected = next((item for item in templates if item.get("id") == explicit_template_id), None)
        if selected:
            return selected, {"method": "explicit", "template_id": explicit_template_id, "confidence": 1.0}
    candidates = template_candidates(templates, transcript_text, focus_prompt)
    prompt = render_classification_prompt(candidates, transcript_text, focus_prompt)
    try:
        response = client.generate(prompt, model=model, temperature=0.0, num_predict=900)["response"]
        payload = parse_json_object(response)
        template_id = str(payload.get("template_id") or "").strip()
        selected = next((item for item in templates if item.get("id") == template_id), None)
        if selected:
            payload.setdefault("method", "qwen3-4b")
            payload["candidate_count"] = len(candidates)
            return selected, payload
        fallback = keyword_template(templates, transcript_text, focus_prompt)
        payload["fallback_reason"] = f"model returned unknown template_id: {template_id}"
        return fallback, payload
    except Exception as exc:
        fallback = keyword_template(templates, transcript_text, focus_prompt)
        return fallback, {
            "method": "keyword-fallback",
            "reason": str(exc),
            "template_id": fallback.get("id"),
            "confidence": 0.0,
        }


def render_classification_prompt(templates: list[dict[str, Any]], transcript_text: str, focus_prompt: str) -> str:
    catalog = "\n".join(
        f"- id={item.get('id')} | 标题={item.get('title_zh') or item.get('title')} | 分类={item.get('first_category_zh')}/{item.get('second_category_zh')}"
        for item in templates
    )
    return f"""你是音频内容场景分类器。请先阅读转写文本和用户关注点，再从模板目录中选择一个最合适的模板。

只输出 JSON，不要输出解释。JSON 字段：
- template_id: 必须是模板目录中的 id
- scene: 你判断的中文场景
- reason: 选择原因，一句话
- confidence: 0 到 1

用户关注点：
{focus_prompt or "无"}

模板目录：
{catalog}

转写文本节选：
{clip_text(transcript_text, MAX_TRANSCRIPT_CHARS_FOR_CLASSIFY)}
"""


def template_candidates(templates: list[dict[str, Any]], transcript_text: str, focus_prompt: str) -> list[dict[str, Any]]:
    haystack = f"{focus_prompt}\n{transcript_text[:9000]}".lower()
    category_weights = {
        "meeting": ("会议", "纪要", "讨论", "决策", "待办", "meeting", "minutes", "agenda"),
        "call": ("电话", "通话", "喂", "hello", "客户", "call", "client"),
        "it": ("gpu", "cpu", "diffusion", "模型", "ai", "技术", "部署", "代码", "system", "technical"),
        "interview": ("访谈", "采访", "interview"),
        "education": ("讲座", "课程", "课堂", "培训", "lecture", "training"),
        "speech": ("演讲", "主题演讲", "播客", "podcast", "speech"),
    }
    title_weights = (
        "summary",
        "minutes",
        "action items",
        "documentation",
        "technical",
        "500 word",
        "纪要",
        "摘要",
        "总结",
    )

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(templates):
        text = " ".join(
            str(item.get(key) or "")
            for key in (
                "id",
                "title",
                "title_zh",
                "first_category",
                "first_category_zh",
                "second_category",
                "second_category_zh",
                "tags",
            )
        ).lower()
        score = 0
        first_category = str(item.get("first_category") or "").lower()
        for category, words in category_weights.items():
            hits = sum(1 for word in words if word.lower() in haystack)
            if hits and first_category == category:
                score += hits * 10
        score += sum(4 for word in title_weights if word in text)
        if "默认总结" in text or "default summary" in text:
            score -= 8
        if score > 0:
            scored.append((score, -index, item))

    scored.sort(reverse=True, key=lambda value: (value[0], value[1]))
    candidates = [item for _, _, item in scored[:CLASSIFICATION_CANDIDATE_LIMIT]]
    if not candidates:
        candidates = [keyword_template(templates, transcript_text, focus_prompt)]
    fallback = next((item for item in templates if item.get("id") == "media2text-media2text-默认总结-a7c420093f"), None)
    if fallback and all(item.get("id") != fallback.get("id") for item in candidates):
        candidates.append(fallback)
    return candidates


def keyword_template(templates: list[dict[str, Any]], transcript_text: str, focus_prompt: str) -> dict[str, Any]:
    haystack = f"{focus_prompt}\n{transcript_text[:6000]}".lower()
    category = "meeting"
    if any(word in haystack for word in ("讲座", "课程", "课堂", "培训", "lecture", "class", "training")):
        category = "education"
    elif any(word in haystack for word in ("访谈", "采访", "interview")):
        category = "interview"
    elif any(word in haystack for word in ("电话", "通话", "call")):
        category = "call"
    elif any(word in haystack for word in ("销售", "客户", "sales")):
        category = "sales"
    elif any(word in haystack for word in ("法律", "案件", "合同", "court", "legal")):
        category = "law"
    elif any(word in haystack for word in ("医疗", "患者", "病历", "medical", "patient")):
        category = "medical"
    matches = [item for item in templates if item.get("first_category") == category]
    preferred = [
        item
        for item in matches
        if any(key in f"{item.get('title_zh')} {item.get('title')} {item.get('second_category')}".lower() for key in ("summary", "纪要", "摘要", "minutes"))
    ]
    return (preferred or matches or templates)[0]


def summarize_with_template(
    client: GenericOpenAIAPIClient,
    model: str,
    template: dict[str, Any],
    transcript_text: str,
    focus_prompt: str,
    language: str,
    temperature: float,
    source_name: str = "",
) -> str:
    transcript = clip_text(transcript_text, MAX_TRANSCRIPT_CHARS_FOR_SUMMARY)
    task_prompt = render_template_prompt(
        str(template.get("prompt_original") or "").strip(),
        transcript=transcript,
        focus_prompt=focus_prompt,
        recording_time=recording_time_from_source(source_name),
    )
    prompt = f"""请使用 {language} 输出。

你正在处理录音笔音频的转写结果。必须基于转写文本总结，不要编造未出现的信息。

录音文件名：{source_name or "未提供"}
录音文件时间：{recording_time_from_source(source_name) or "未从文件名识别"}

约束：
- 模板询问会议日期或会议时间时，优先使用录音文件时间。
- 如果转写文本明确提到另一个会议日期，以转写文本为准，并说明它来自转写。
- 不要根据当前日期、任务创建时间或常识推断会议日期。
- 会议日期/会议时间可以使用录音文件时间；行动项截止日期不能默认使用录音文件时间。
- 截止日期只有在转写明确出现具体日历日期时才写具体日期。
- 如果转写只出现“明天”“下周”“下周四”“月底”等相对时间，必须保留原文，并标注“相对时间，未换算”。
- 严禁把相对时间换算成 YYYY-MM-DD 或“某年某月某日”，即使可以根据录音时间推算。
- 可以总结相对顺序和依赖关系，但不要把相对时间改写成具体日历日期。

用户关注点：
{focus_prompt or "无"}

模板名称：{template.get('title_zh') or template.get('title')}

{task_prompt}
"""
    return client.generate(prompt, model=model, temperature=temperature, num_predict=4096)["response"].strip()


def render_template_prompt(template_prompt: str, transcript: str, focus_prompt: str, recording_time: str = "") -> str:
    values = {
        "TRANSCRIPT": transcript,
        "TEXT": transcript,
        "INPUT": transcript,
        "CONTENT": transcript,
        "FOCUS_PROMPT": focus_prompt or "无",
        "USER_FOCUS": focus_prompt or "无",
        "MEETING_DATE": recording_time or "未从录音中识别",
        "MEETING_TIME": recording_time or "未从录音中识别",
        "RECORDING_TIME": recording_time or "未从录音中识别",
        "DATE": recording_time or "未从录音中识别",
        "MEETING_TYPE": "自动识别",
        "PROJECT_NAME": "自动识别",
    }

    def replace_double_brace(match: re.Match[str]) -> str:
        key = match.group(1).strip().upper()
        return values.get(key, "未提供")

    rendered = re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", replace_double_brace, template_prompt)
    rendered = rendered.replace("{text}", transcript).replace("{transcript}", transcript)
    if transcript not in rendered:
        rendered = f"{rendered}\n\n转写文本：\n{transcript}"
    return rendered


def recording_time_from_source(source_name: str) -> str:
    match = re.search(r"(20\d{2})([01]\d)([0-3]\d)([0-2]\d)([0-5]\d)([0-5]\d)", source_name or "")
    if not match:
        return ""
    year, month, day, hour, minute, second = match.groups()
    return f"{int(year)}年{int(month)}月{int(day)}日 {hour}:{minute}:{second}"


def write_audio_only_manifest(output_dir: Path, media_path: Path, audio_path: Path) -> Path:
    path = output_dir / "frames_manifest.json"
    write_json(
        path,
        {
            "version": 1,
            "source": "audio_only",
            "media_path": str(media_path),
            "audio_path": str(audio_path),
            "frames": [],
        },
    )
    return path


def build_light_study_guide(
    client: GenericOpenAIAPIClient,
    model: str,
    output_dir: Path,
    transcript: AudioTranscript,
    summary: str,
    temperature: float,
) -> Path:
    segments = list(transcript.segments or [])
    prompt = f"""请基于录音转写生成用于手机脑图展示的 JSON。

只输出 JSON，不要输出解释。结构必须为：
{{"title":"...","summary":"...","chapters":[{{"index":1,"title":"...","start":"00:00","end":"03:20","summary":"...","key_points":["...","..."]}}]}}

要求：
- 章节 3 到 8 个。
- 每章 key_points 3 到 5 条。
- 标题和要点使用简体中文。
- start/end 使用 mm:ss 或 hh:mm:ss。
- 内容必须来自转写，不要编造。

已有摘要：
{clip_text(summary, 4000)}

转写文本：
{clip_text(transcript.text, MAX_TRANSCRIPT_CHARS_FOR_GUIDE)}
"""
    try:
        response = client.generate(prompt, model=model, temperature=temperature, num_predict=4096)["response"]
        guide = normalize_study_guide(parse_json_object(response), segments, summary)
    except Exception as exc:
        logger.warning("light study guide generation failed: %s", exc)
        guide = fallback_study_guide(segments, summary)
    path = output_dir / "study_guide.json"
    write_json(path, guide)
    (output_dir / "study_overview.md").write_text(render_study_overview(guide), encoding="utf-8")
    return path


def normalize_study_guide(guide: dict[str, Any], segments: list[dict[str, Any]], summary: str) -> dict[str, Any]:
    chapters = guide.get("chapters") if isinstance(guide.get("chapters"), list) else []
    normalized = []
    for index, item in enumerate(chapters[:8], 1):
        if not isinstance(item, dict):
            continue
        points = item.get("key_points") if isinstance(item.get("key_points"), list) else []
        normalized.append(
            {
                "index": int(item.get("index") or index),
                "title": str(item.get("title") or f"章节 {index}").strip(),
                "start": str(item.get("start") or "").strip(),
                "end": str(item.get("end") or "").strip(),
                "summary": str(item.get("summary") or "").strip(),
                "key_points": [str(point).strip() for point in points if str(point).strip()][:5],
            }
        )
    if not normalized:
        return fallback_study_guide(segments, summary)
    return {
        "title": str(guide.get("title") or "录音脑图").strip(),
        "summary": str(guide.get("summary") or summary[:500]).strip(),
        "chapters": normalized,
    }


def fallback_study_guide(segments: list[dict[str, Any]], summary: str) -> dict[str, Any]:
    if not segments:
        return {"title": "录音脑图", "summary": summary[:500], "chapters": []}
    bucket_count = min(6, max(1, len(segments) // 8 or 1))
    bucket_size = max(1, (len(segments) + bucket_count - 1) // bucket_count)
    chapters = []
    for index in range(bucket_count):
        bucket = segments[index * bucket_size : (index + 1) * bucket_size]
        if not bucket:
            continue
        text = " ".join(str(item.get("text") or item.get("content") or "").strip() for item in bucket).strip()
        points = [part.strip() for part in re.split(r"[。！？!?]\s*", text) if part.strip()][:4]
        chapters.append(
            {
                "index": index + 1,
                "title": points[0][:24] if points else f"章节 {index + 1}",
                "start": format_seconds(bucket[0].get("start") or 0),
                "end": format_seconds(bucket[-1].get("end") or bucket[-1].get("start") or 0),
                "summary": text[:180],
                "key_points": points or [text[:80]],
            }
        )
    return {"title": "录音脑图", "summary": summary[:500], "chapters": chapters}


def render_study_overview(guide: dict[str, Any]) -> str:
    lines = [f"# {guide.get('title') or '录音脑图'}", "", str(guide.get("summary") or "").strip(), ""]
    for chapter in guide.get("chapters") or []:
        lines.append(f"## {chapter.get('index')}. {chapter.get('title')}")
        lines.append(f"{chapter.get('start', '')} - {chapter.get('end', '')}".strip())
        if chapter.get("summary"):
            lines.append(str(chapter["summary"]))
        for point in chapter.get("key_points") or []:
            lines.append(f"- {point}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_seconds(value: Any) -> str:
    try:
        seconds = int(float(value or 0))
    except (TypeError, ValueError):
        seconds = 0
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def write_operation_manual(output_dir: Path, template: dict[str, Any], classification: dict[str, Any], summary: str, focus_prompt: str) -> Path:
    path = output_dir / "operation_manual.md"
    text = f"""# 音频总结

- 模板：{template.get('title_zh') or template.get('title')}
- 场景：{classification.get('scene') or template.get('first_category_zh') or template.get('first_category')}
- 选择原因：{classification.get('reason') or classification.get('fallback_reason') or classification.get('reason', '')}
- 用户关注点：{focus_prompt or '无'}

## 总结

{summary}
"""
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def write_manual_evidence(
    output_dir: Path,
    media_path: Path,
    transcript_path: Path | None,
    template: dict[str, Any],
    classification: dict[str, Any],
    asr_result: ASRStrategyResult | None,
) -> Path:
    path = output_dir / "manual_evidence.md"
    text = f"""# 音频分析证据

- 媒体文件：`{media_path}`
- 转写文件：`{transcript_path or ''}`
- ASR 策略：{asr_result.strategy if asr_result else 'unknown'}
- 模板 ID：{template.get('id')}
- 模板名称：{template.get('title_zh') or template.get('title')}
- 分类方法：{classification.get('method')}
- 置信度：{classification.get('confidence')}
"""
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def write_analysis_json(
    output_dir: Path,
    media_path: Path,
    transcript: AudioTranscript,
    asr_result: ASRStrategyResult | None,
    speaker_report: dict[str, Any],
    selected_template: dict[str, Any],
    classification: dict[str, Any],
    summary: str,
    selector_base_url: str,
    selector_model: str,
    content_base_url: str,
    content_model: str,
    manual_path: Path,
    evidence_path: Path,
    elapsed_seconds: float,
) -> Path:
    orin = output_dir / "orin"
    orin.mkdir(parents=True, exist_ok=True)
    write_json(orin / "transcript.json", {"text": transcript.text, "segments": transcript.segments, "metadata": transcript.metadata})
    write_transcript_markdown(transcript, orin / "transcript.md")
    if asr_result:
        write_json(orin / "asr.json", asr_result.to_metadata())
    write_json(orin / "frame_analyses.json", [])
    write_json(orin / "visual_events.json", [])
    write_json(orin / "ocr_events.json", [])
    write_json(output_dir / "audio_template_analysis.json", {
        "selected_template": selected_template,
        "classification": classification,
        "summary": summary,
        "template_selector_model": selector_model,
        "summary_model": content_model,
    })
    payload = {
        "metadata": {
            "task": "audio_template_summary",
            "source_type": "audio_upload",
            "media_path": str(media_path),
            "text_base_url": content_base_url,
            "text_model": content_model,
            "template_selector_base_url": selector_base_url,
            "template_selector_model": selector_model,
            "asr_strategy": asr_result.strategy if asr_result else None,
            "transcript_markdown": str(output_dir / "transcript.md"),
            "transcription_successful": True,
            "audio_language": transcript.language,
            "speaker_diarization": speaker_report,
            "frames_extracted": 0,
            "frames_processed": 0,
            "timings": {"total_seconds": elapsed_seconds},
        },
        "transcript": {
            "text": transcript.text,
            "segments": transcript.segments,
            "metadata": transcript.metadata,
        },
        "asr": asr_result.to_metadata() if asr_result else None,
        "speaker_diarization": speaker_report,
        "audio_template_analysis": {
            "selected_template": selected_template,
            "classification": classification,
            "summary": summary,
            "template_selector_model": selector_model,
            "summary_model": content_model,
        },
        "ocr_events": [],
        "ocr_text_events": [],
        "visual_events": [],
        "manual_steps": {
            "response": summary,
            "manual_path": str(manual_path),
            "evidence_path": str(evidence_path),
            "quality_gate_passed": True,
        },
        "frame_analyses": [],
        "operation_manual": {
            "response": summary,
            "manual_path": str(manual_path),
            "evidence_path": str(evidence_path),
            "quality_gate_passed": True,
        },
    }
    analysis_path = output_dir / "analysis.json"
    write_json(analysis_path, payload)
    return analysis_path


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("classification response must be a JSON object")
    return payload


def clip_text(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.65)].rstrip()
    tail = text[-int(limit * 0.30) :].lstrip()
    return f"{head}\n\n[中间内容已截断]\n\n{tail}"


if __name__ == "__main__":
    raise SystemExit(main())
