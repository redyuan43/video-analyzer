#!/usr/bin/env python3
"""Audio-first template analysis for uploaded recorder files."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
from video_analyzer.speaker_diarization import (  # noqa: E402
    prepare_speaker_assignment,
    process_transcript_speakers,
)
from video_analyzer.transcription_pipeline import (  # noqa: E402
    load_provided_transcript,
    speaker_diarization_can_run_parallel,
)


DEFAULT_TEMPLATE_CATALOG = REPO_ROOT / "video-analyzer-ui" / "video_analyzer_ui" / "static" / "data" / "audio_prompt_templates.json"
DEFAULT_TEMPLATE_ID = "auto"
AUDIO_PIPELINE_PROFILE = "audio_nx1"
AUDIO_PIPELINE_VERSION = 1
DOWAY_SOURCE_REPO = "Doway AI server"
DOWAY_SOURCE_PATH = "analysis/doway_prompts/server_prompts_zh.json"
DOWAY_GENERAL_TEMPLATE_ID = "2"
MAX_TRANSCRIPT_CHARS_FOR_CLASSIFY = 9000
MAX_TRANSCRIPT_CHARS_FOR_GUIDE = 18000
CLASSIFICATION_CANDIDATE_LIMIT = 48
SUMMARY_SINGLE_PASS_CHARS = 24000
SUMMARY_MAP_CHUNK_CHARS = 20000
SUMMARY_REDUCE_BATCH_CHARS = 48000
CLIENT_TEMPLATE_BLOCK_RE = re.compile(r"【模板指令开始】[\s\S]*?【模板指令结束】\s*")
CLIENT_USER_SUPPLEMENT_MARKER = "【用户补充】"
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
    parser.add_argument("--transcript-json")
    parser.add_argument(
        "--compute-route",
        choices=("local", "cloud_fallback"),
        default="local",
    )
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
    focus_prompt = client_focus_supplement(args.focus_prompt)

    if args.transcript_json:
        transcript_json = Path(args.transcript_json).expanduser().resolve()
        transcript, asr_result = load_provided_transcript(transcript_json)
        audio_path = media_path
        speaker_report = {
            "enabled": False,
            "skipped": True,
            "reason": "provided_transcript",
        }
    else:
        audio_path = extract_audio_to_wav(media_path, output_dir)
        if audio_path is None:
            raise RuntimeError(f"audio extraction produced no audio stream: {media_path}")
        prepared_assignment = None
        if speaker_diarization_can_run_parallel(config):
            speaker_config = config.get("speaker_diarization") or {}
            logger.info(
                "Starting ASR and speaker diarization in parallel (backend=%s)",
                speaker_config.get("backend") or "3dspeaker",
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                diarization_future = executor.submit(
                    prepare_speaker_assignment,
                    audio_path,
                    speaker_config,
                )
                transcript, asr_result = transcribe_audio(audio_path, output_dir, config)
                try:
                    prepared_assignment = diarization_future.result()
                except Exception as exc:
                    logger.warning("parallel speaker diarization failed: %s", exc)
                    prepared_assignment = (
                        [],
                        {
                            "enabled": True,
                            "mode": "assignment",
                            "backend": speaker_config.get("backend") or "3dspeaker",
                            "notes": ["parallel speaker diarization failed"],
                            "error": str(exc),
                        },
                    )
        else:
            transcript, asr_result = transcribe_audio(audio_path, output_dir, config)
        if transcript is not None:
            transcript, speaker_report = refine_audio_speakers(
                audio_path,
                transcript,
                output_dir,
                config,
                prepared_assignment=prepared_assignment,
            )
    if transcript is None or not transcript.text.strip():
        raise RuntimeError(
            "No recognizable speech was produced by ASR for uploaded audio"
        )
    if asr_result:
        asr_result.transcript = transcript
    transcript_path = write_transcript_markdown(transcript, output_dir / "transcript.md")

    selector_client, selector_model, selector_base_url, _selector_temperature = build_template_selector_client(config)
    content_client, content_model, content_base_url, content_temperature = build_content_analysis_client(config, args.profile)
    analysis_transcript = format_transcript_for_analysis(transcript)
    selected, classification = choose_template(
        client=selector_client,
        model=selector_model,
        templates=templates,
        transcript_text=analysis_transcript,
        focus_prompt=focus_prompt,
        explicit_template_id=args.template_id,
    )
    summary = summarize_with_template(
        client=content_client,
        model=content_model,
        template=selected,
        transcript_text=analysis_transcript,
        focus_prompt=focus_prompt,
        language=args.language,
        temperature=content_temperature,
        source_name=args.source_name or media_path.name,
    )
    write_audio_only_manifest(output_dir, media_path, audio_path)
    study_guide_path = build_light_study_guide(
        content_client,
        content_model,
        output_dir,
        transcript,
        summary,
        content_temperature,
    )

    manual_path = write_operation_manual(output_dir, selected, classification, summary, focus_prompt)
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
        study_guide_path=study_guide_path,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        compute_route=args.compute_route,
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
    if args.compute_route == "cloud_fallback":
        apply_cloud_fallback(config, args.profile)
    return config


def apply_cloud_fallback(config: Config, profile_name: str | None) -> None:
    profile = config.get_runtime_profile(profile_name)
    fallback = profile.get("audio_cloud_fallback") or {}
    if not fallback.get("enabled"):
        raise ValueError(
            f"runtime profile {profile_name or '(default)'} does not enable audio cloud fallback"
        )
    asr = fallback.get("asr") or {}
    diarization = fallback.get("diarization") or {}
    protocol = str(asr.get("protocol") or "")
    endpoints = [str(item) for item in (asr.get("endpoints") or []) if str(item)]
    provider = {
        "vibevoice_http": "vibevoice",
        "generic_http": "remote_http",
        "firered_3dspeaker_http": "firered_3dspeaker",
        "openai_audio": "openai_audio",
    }.get(protocol)
    if not provider or not endpoints:
        raise ValueError(f"unsupported audio cloud fallback ASR protocol: {protocol or '(missing)'}")
    if str(diarization.get("protocol") or "") != "asr_embedded":
        raise ValueError("audio cloud fallback requires ASR embedded speaker labels")

    asr_config = config.config.setdefault("asr", {})
    asr_config["provider"] = provider
    vibevoice = asr_config.setdefault("vibevoice", {})
    if protocol == "vibevoice_http":
        vibevoice["deep_remote_urls"] = endpoints
    elif protocol == "generic_http":
        vibevoice["remote_urls"] = endpoints
    elif protocol == "firered_3dspeaker_http":
        vibevoice["firered_3dspeaker_url"] = endpoints[0]
    elif protocol == "openai_audio":
        vibevoice["openai_audio_url"] = endpoints[0]
        vibevoice["openai_audio_model"] = asr.get("model")
        vibevoice["asr_api_key_env"] = asr.get("api_key_env")
    config.config["speaker_diarization"] = {
        "enabled": False,
        "assignment_enabled": False,
        "source": "asr_embedded",
    }


def client_focus_supplement(value: str) -> str:
    focus = CLIENT_TEMPLATE_BLOCK_RE.sub("", str(value or ""))
    return focus.replace(CLIENT_USER_SUPPLEMENT_MARKER, "").strip()


def load_templates(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"template catalog must be a list: {path}")
    if not data:
        raise ValueError(f"Doway template catalog must not be empty: {path}")

    ids: set[str] = set()
    templates: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Doway template catalog entry {index} must be an object")
        template_id = item.get("id")
        if not isinstance(template_id, str) or not template_id.isdigit():
            raise ValueError(f"Doway template catalog entry {index} has invalid numeric string id: {template_id!r}")
        if template_id in ids:
            raise ValueError(f"Doway template catalog contains duplicate id: {template_id}")
        ids.add(template_id)

        prompt = item.get("prompt_original")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Doway template {template_id} has an empty prompt_original")
        server = item.get("server")
        if not isinstance(server, dict):
            raise ValueError(f"Doway template {template_id} is missing server metadata")
        actual_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if server.get("prompt_sha256") != actual_sha256:
            raise ValueError(f"Doway template {template_id} prompt_sha256 mismatch")
        if str(server.get("template_id")) != template_id:
            raise ValueError(f"Doway template {template_id} server template_id mismatch")
        if any(server.get(key) != "zh" for key in ("requested_language", "source_language", "response_language")):
            raise ValueError(f"Doway template {template_id} is not from the Chinese server catalog")
        if item.get("source_repo") != DOWAY_SOURCE_REPO or item.get("source_path") != DOWAY_SOURCE_PATH:
            raise ValueError(f"Doway template {template_id} has an unexpected source")
        templates.append(item)
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
    *,
    prepared_assignment: tuple[list[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[AudioTranscript, dict[str, Any]]:
    speaker_config = config.get("speaker_diarization") or {}
    try:
        refined, report = process_transcript_speakers(
            audio_path,
            transcript,
            speaker_config,
            prepared_assignment=prepared_assignment,
        )
        if prepared_assignment is not None:
            report["parallel_with_asr"] = True
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
    profile = config.get_runtime_profile(None)
    study_config = config.get("study_cards") or {}
    inherit_text = profile.get("template_selector_inherit") == "text"
    base_url = (
        profile.get("text_base_url")
        if inherit_text
        else profile.get("template_selector_base_url")
    ) or study_config.get("llm_base_url") or "http://agx.taild500c8.ts.net:11434/v1"
    model = (
        profile.get("text_model")
        if inherit_text
        else profile.get("template_selector_model")
    ) or study_config.get("model") or "qwen3:4b-instruct"
    temperature = float(
        profile.get("template_selector_temperature")
        or study_config.get("temperature", 0.1)
    )
    api_key_env = (
        profile.get("text_api_key_env")
        if inherit_text
        else profile.get("template_selector_api_key_env")
    )
    if inherit_text:
        extra_body = build_openai_extra_body(profile, base_url)
    else:
        extra_body = build_openai_extra_body(
            profile,
            base_url,
            prefix="template_selector_",
        ) or build_openai_extra_body(study_config, base_url)
    client = GenericOpenAIAPIClient(
        api_key=resolve_api_key(
            study_config.get("api_key"),
            api_key_env or study_config.get("api_key_env"),
            base_url,
        ),
        api_url=base_url,
        timeout_seconds=int(study_config.get("timeout_seconds", 600)),
        extra_body=extra_body,
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
        requested_id = str(explicit_template_id)
        selected = next((item for item in templates if item.get("id") == requested_id), None)
        if selected is None:
            raise ValueError(f"unknown explicit Doway template_id: {requested_id}")
        return selected, {"method": "explicit", "template_id": requested_id, "confidence": 1.0}
    candidates = template_candidates(templates, transcript_text, focus_prompt)
    prompt = render_classification_prompt(candidates, transcript_text, focus_prompt)
    try:
        response = client.generate(prompt, model=model, temperature=0.0, num_predict=900)["response"]
        payload = parse_json_object(response)
        template_id = str(payload.get("template_id") or "").strip()
        selected = next((item for item in candidates if item.get("id") == template_id), None)
        if selected:
            payload["method"] = "small-model"
            payload["candidate_count"] = len(candidates)
            return selected, payload
        fallback = keyword_template(templates, transcript_text, focus_prompt)
        return fallback, {
            "method": "keyword-fallback",
            "reason": f"small model returned an id outside local candidates: {template_id or '(empty)'}",
            "model_template_id": template_id,
            "template_id": fallback.get("id"),
            "candidate_count": len(candidates),
            "confidence": 0.0,
        }
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
        "genera": ("总结", "摘要", "概述", "通用", "summary", "overview"),
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
    for item in templates:
        text = template_search_text(item)
        score = 0
        first_category = str(item.get("first_category") or "").lower()
        for category, words in category_weights.items():
            hits = sum(1 for word in words if word.lower() in haystack)
            if hits and first_category == category:
                score += hits * 10
        score += sum(4 for word in title_weights if word in text)
        if score > 0:
            scored.append((score, -numeric_template_id(item), item))

    scored.sort(reverse=True, key=lambda value: (value[0], value[1]))
    candidates = [item for _, _, item in scored[:CLASSIFICATION_CANDIDATE_LIMIT]]
    if not candidates:
        candidates = [keyword_template(templates, transcript_text, focus_prompt)]
    fallback = next((item for item in templates if item.get("id") == DOWAY_GENERAL_TEMPLATE_ID), None)
    if fallback and all(item.get("id") != fallback.get("id") for item in candidates):
        if len(candidates) >= CLASSIFICATION_CANDIDATE_LIMIT:
            candidates[-1] = fallback
        else:
            candidates.append(fallback)
    return candidates


def keyword_template(templates: list[dict[str, Any]], transcript_text: str, focus_prompt: str) -> dict[str, Any]:
    haystack = f"{focus_prompt}\n{transcript_text[:6000]}".lower()
    category = "genera"
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
    preference_terms = ("纪要", "摘要", "总结", "概述", "summary", "minutes", "autopilot")
    ranked = sorted(
        matches,
        key=lambda item: (
            -sum(term in template_search_text(item) for term in preference_terms),
            numeric_template_id(item),
        ),
    )
    if ranked:
        return ranked[0]
    fallback = next((item for item in templates if item.get("id") == DOWAY_GENERAL_TEMPLATE_ID), None)
    if fallback is None:
        raise ValueError(f"Doway general fallback template {DOWAY_GENERAL_TEMPLATE_ID} is missing")
    return fallback


def template_search_text(template: dict[str, Any]) -> str:
    return " ".join(
        str(template.get(key) or "")
        for key in (
            "id",
            "title",
            "title_zh",
            "description",
            "first_category",
            "first_category_zh",
            "second_category",
            "second_category_zh",
            "tags",
        )
    ).lower()


def numeric_template_id(template: dict[str, Any]) -> int:
    return int(str(template.get("id")))


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
    template_prompt = str(template.get("prompt_original") or "")
    if not template_prompt.strip():
        raise ValueError(f"Doway template {template.get('id')} has an empty prompt")
    recording_time = recording_time_from_source(source_name)
    if len(transcript_text) <= SUMMARY_SINGLE_PASS_CHARS:
        prompt = build_final_summary_prompt(
            template_prompt=template_prompt,
            transcript=transcript_text,
            focus_prompt=focus_prompt,
            language=language,
            source_name=source_name,
            recording_time=recording_time,
            template_title=str(template.get("title_zh") or template.get("title") or template.get("id")),
            transcript_label="完整转写文本",
        )
        return generate_required_text(
            client,
            prompt,
            model=model,
            temperature=temperature,
            num_predict=4096,
            stage="final summary",
        )

    chunks = split_transcript_chunks(transcript_text, SUMMARY_MAP_CHUNK_CHARS)
    map_summaries = []
    for index, chunk in enumerate(chunks, 1):
        map_prompt = f"""你正在为长录音总结做第 {index}/{len(chunks)} 个连续分块的事实提炼。

请使用 {language}，完整保留本分块中的说话人、时间线、具体数字、观点、决策、行动项、负责人、截止时间、风险和未决问题。不要套用最终总结模板，不要推断未出现的信息。输出紧凑但信息充分的结构化 Markdown，供后续 reduce 使用。

用户补充关注点（仅补充，不得取代原文信息）：
{focus_prompt or "无"}

连续分块 {index}/{len(chunks)}：
{chunk}
"""
        map_summaries.append(
            generate_required_text(
                client,
                map_prompt,
                model=model,
                temperature=temperature,
                num_predict=1800,
                stage=f"map chunk {index}/{len(chunks)}",
            )
        )

    reduce_source = reduce_map_summaries(
        client=client,
        model=model,
        summaries=map_summaries,
        language=language,
        temperature=temperature,
    )
    prompt = build_final_summary_prompt(
        template_prompt=template_prompt,
        transcript=reduce_source,
        focus_prompt=focus_prompt,
        language=language,
        source_name=source_name,
        recording_time=recording_time,
        template_title=str(template.get("title_zh") or template.get("title") or template.get("id")),
        transcript_label=f"按原始顺序生成的 {len(chunks)} 个分块提炼结果",
    )
    return generate_required_text(
        client,
        prompt,
        model=model,
        temperature=temperature,
        num_predict=4096,
        stage="final reduce summary",
    )


def build_final_summary_prompt(
    template_prompt: str,
    transcript: str,
    focus_prompt: str,
    language: str,
    source_name: str,
    recording_time: str,
    template_title: str,
    transcript_label: str,
) -> str:
    task_prompt = render_template_prompt(
        template_prompt,
        transcript=transcript,
        focus_prompt=focus_prompt,
        recording_time=recording_time,
    )
    return f"""请使用 {language} 输出。

你正在处理录音笔音频的转写结果。必须基于转写文本总结，不要编造未出现的信息。

录音文件名：{source_name or "未提供"}
录音文件时间：{recording_time or "未提供"}
输入形态：{transcript_label}

约束：
- 下方 Doway 模板是主要输出规范，必须完整遵守；用户补充关注点只能补充强调，不能替代、删减或改写模板要求。
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

Doway 模板名称：{template_title}

{task_prompt}
"""


def generate_required_text(
    client: GenericOpenAIAPIClient,
    prompt: str,
    model: str,
    temperature: float,
    num_predict: int,
    stage: str,
) -> str:
    payload = client.generate(prompt, model=model, temperature=temperature, num_predict=num_predict)
    response = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(f"content model returned an empty response during {stage}")
    return response.strip()


def split_transcript_chunks(text: str, max_chars: int = SUMMARY_MAP_CHUNK_CHARS) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = str(text or "")
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > max_chars:
            chunks.append(current)
            current = ""
        while len(line) > max_chars:
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        current += line
    if current:
        chunks.append(current)
    if "".join(chunks) != text:
        raise AssertionError("deterministic transcript chunking lost content")
    return chunks


def reduce_map_summaries(
    client: GenericOpenAIAPIClient,
    model: str,
    summaries: list[str],
    language: str,
    temperature: float,
) -> str:
    current = list(summaries)
    level = 1
    while len(render_summary_sections(current)) > SUMMARY_REDUCE_BATCH_CHARS:
        batches = pack_summary_batches(current, SUMMARY_REDUCE_BATCH_CHARS)
        reduced: list[str] = []
        for index, batch in enumerate(batches, 1):
            prompt = f"""请使用 {language} 合并以下连续分块提炼结果，严格保留时间线、说话人、事实、数字、决策、行动项、风险和未决问题。不得补写，不得套用最终模板。输出紧凑的结构化 Markdown。

层级 {level}，批次 {index}/{len(batches)}：
{render_summary_sections(batch)}
"""
            reduced.append(
                generate_required_text(
                    client,
                    prompt,
                    model=model,
                    temperature=temperature,
                    num_predict=1800,
                    stage=f"intermediate reduce {level}.{index}",
                )
            )
        if len(reduced) >= len(current):
            raise RuntimeError("intermediate reduce did not reduce long transcript context")
        current = reduced
        level += 1
    return render_summary_sections(current)


def pack_summary_batches(summaries: list[str], max_chars: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for summary in summaries:
        estimated = len(summary) + 64
        if current and current_chars + estimated > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        if estimated > max_chars:
            for chunk in split_transcript_chunks(summary, max_chars - 64):
                if current:
                    batches.append(current)
                    current = []
                    current_chars = 0
                batches.append([chunk])
            continue
        current.append(summary)
        current_chars += estimated
    if current:
        batches.append(current)
    return batches


def render_summary_sections(summaries: list[str]) -> str:
    return "\n\n".join(
        f"## 分块提炼 {index}/{len(summaries)}\n{summary}"
        for index, summary in enumerate(summaries, 1)
    )


def render_template_prompt(
    template_prompt: str,
    transcript: str,
    focus_prompt: str,
    recording_time: str = "",
    recording_end_time: str = "",
    duration: str = "",
    location: str = "",
) -> str:
    recording_date = recording_time.split(" ", 1)[0] if recording_time else ""
    values = {
        "TRANSCRIPT": transcript,
        "TEXT": transcript,
        "INPUT": transcript,
        "CONTENT": transcript,
        "FOCUSPROMPT": focus_prompt or "未提供",
        "USERFOCUS": focus_prompt or "未提供",
        "MEETINGDATE": recording_date or "未提供",
        "MEETINGTIME": recording_time or "未提供",
        "RECORDINGTIME": recording_time or "未提供",
        "DATE": recording_date or "未提供",
        "RECORDSTARTTIME": recording_time or "未提供",
        "RECORDENDTIME": recording_end_time or "未提供",
        "DURATION": duration or "未提供",
        "LOCATION": location or "未提供",
        "MEETINGTYPE": "自动识别",
        "PROJECTNAME": "自动识别",
    }

    def replace_placeholder(match: re.Match[str]) -> str:
        key = re.sub(r"[^A-Z0-9]", "", match.group(1).upper())
        return values.get(key, "未提供")

    rendered = re.sub(r"\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}", replace_placeholder, template_prompt)
    rendered = re.sub(r"(?<!\{)\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}(?!\})", replace_placeholder, rendered)
    if transcript not in rendered:
        rendered = f"{rendered}\n\n转写文本：\n{transcript}"
    return rendered


def format_transcript_for_analysis(transcript: AudioTranscript) -> str:
    lines = []
    for segment in transcript.segments or []:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or segment.get("content") or "").strip()
        if not text:
            continue
        speaker = str(segment.get("speaker") or segment.get("speaker_id") or "说话人未提供").strip()
        start = format_seconds(segment.get("start") or 0)
        end = format_seconds(segment.get("end") or segment.get("start") or 0)
        lines.append(f"[{start}-{end}] {speaker}: {text}")
    return "\n".join(lines) if lines else transcript.text


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
            "pipeline_profile": AUDIO_PIPELINE_PROFILE,
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
    prompt = f"""请基于录音转写和最终总结生成用于手机展示的结构化 JSON。

只输出 JSON，不要输出解释。结构必须为：
{{"title":"...","summary":"...","keywords":["..."],"action_items":[{{"task":"...","owner":"...","deadline":"..."}}],"chapters":[{{"index":1,"title":"...","start":"00:00","end":"03:20","summary":"...","key_points":["...","..."]}}]}}

要求：
- 章节 3 到 8 个。
- 每章 key_points 3 到 5 条。
- keywords 输出 3 到 10 个适合移动端标签展示的关键词。
- action_items 只提取原文明确出现的行动；负责人或截止时间未知时写“未提供”，没有行动项时输出空数组。
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
        guide = fallback_study_guide(segments, summary, generation_error=str(exc))
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
        return fallback_study_guide(segments, summary, generation_error="model returned no usable chapters")
    keywords = guide.get("keywords") if isinstance(guide.get("keywords"), list) else []
    action_items = normalize_action_items(guide.get("action_items"))
    result = {
        "title": str(guide.get("title") or "录音脑图").strip(),
        "summary": str(guide.get("summary") or summary[:500]).strip(),
        "keywords": [str(keyword).strip() for keyword in keywords if str(keyword).strip()][:10],
        "action_items": action_items,
        "chapters": normalized,
        "generation_status": "generated",
    }
    result["mindmap"] = build_mindmap(result["title"], normalized)
    return result


def normalize_action_items(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value[:20]:
        if isinstance(item, str):
            task = item.strip()
            owner = "未提供"
            deadline = "未提供"
        elif isinstance(item, dict):
            task = str(item.get("task") or item.get("action") or item.get("title") or "").strip()
            owner = str(item.get("owner") or item.get("assignee") or "未提供").strip()
            deadline = str(item.get("deadline") or item.get("due_date") or "未提供").strip()
        else:
            continue
        if task:
            normalized.append({"task": task, "owner": owner or "未提供", "deadline": deadline or "未提供"})
    return normalized


def build_mindmap(title: str, chapters: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "title": title,
        "nodes": [
            {
                "id": f"chapter-{chapter['index']}",
                "label": chapter["title"],
                "children": list(chapter.get("key_points") or []),
            }
            for chapter in chapters
        ],
    }


def fallback_study_guide(
    segments: list[dict[str, Any]],
    summary: str,
    generation_error: str = "",
) -> dict[str, Any]:
    if not segments:
        result = {
            "title": title_from_summary(summary),
            "summary": summary[:500],
            "keywords": extract_keywords(summary),
            "action_items": [],
            "chapters": [],
            "generation_status": "fallback",
            "generation_error": generation_error or "no transcript segments",
        }
        result["mindmap"] = build_mindmap(result["title"], [])
        return result
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
    result = {
        "title": title_from_summary(summary),
        "summary": summary[:500],
        "keywords": extract_keywords(summary),
        "action_items": [],
        "chapters": chapters,
        "generation_status": "fallback",
        "generation_error": generation_error or "structured content model output was unavailable",
    }
    result["mindmap"] = build_mindmap(result["title"], chapters)
    return result


def title_from_summary(summary: str) -> str:
    match = re.search(r"<title>\s*([^<]+?)\s*</title>", summary, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()[:40]
    for line in summary.splitlines():
        title = re.sub(r"^[#*\s]+", "", line).strip()
        if title:
            return title[:40]
    return "录音分析"


def extract_keywords(text: str) -> list[str]:
    stopwords = {"一个", "进行", "以及", "需要", "可以", "这个", "没有", "录音", "总结", "内容", "提供"}
    words = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9._+-]{2,}", text)
    counts: dict[str, int] = {}
    for word in words:
        normalized = word.lower() if word.isascii() else word
        if normalized in stopwords:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    return [word for word, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]]


def render_study_overview(guide: dict[str, Any]) -> str:
    lines = [f"# {guide.get('title') or '录音脑图'}", "", str(guide.get("summary") or "").strip(), ""]
    if guide.get("keywords"):
        lines.extend(["## 关键词", "", "、".join(str(item) for item in guide["keywords"]), ""])
    if guide.get("action_items"):
        lines.extend(["## 行动项", ""])
        for item in guide["action_items"]:
            lines.append(
                f"- {item.get('task')}（负责人：{item.get('owner') or '未提供'}；截止：{item.get('deadline') or '未提供'}）"
            )
        lines.append("")
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
    study_guide_path: Path,
    elapsed_seconds: float,
    compute_route: str = "local",
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
    study_guide = load_study_guide_payload(study_guide_path)
    structured_content = {
        "title": str(study_guide.get("title") or title_from_summary(summary)),
        "summary": summary,
        "keywords": list(study_guide.get("keywords") or []),
        "action_items": list(study_guide.get("action_items") or []),
        "study_guide": study_guide,
        "mindmap": study_guide.get("mindmap") or build_mindmap(
            str(study_guide.get("title") or title_from_summary(summary)),
            list(study_guide.get("chapters") or []),
        ),
    }
    audio_template_analysis = {
        "pipeline_profile": AUDIO_PIPELINE_PROFILE,
        "pipeline_version": AUDIO_PIPELINE_VERSION,
        "compute_route": compute_route,
        "selected_template": selected_template,
        "classification": classification,
        "summary": summary,
        "title": structured_content["title"],
        "keywords": structured_content["keywords"],
        "action_items": structured_content["action_items"],
        "study_guide": structured_content["study_guide"],
        "mindmap": structured_content["mindmap"],
        "template_selector_model": selector_model,
        "summary_model": content_model,
    }
    write_json(output_dir / "audio_template_analysis.json", audio_template_analysis)
    payload = {
        "pipeline_profile": AUDIO_PIPELINE_PROFILE,
        "pipeline_version": AUDIO_PIPELINE_VERSION,
        "compute_route": compute_route,
        "metadata": {
            "task": "audio_template_summary",
            "pipeline_profile": AUDIO_PIPELINE_PROFILE,
            "pipeline_version": AUDIO_PIPELINE_VERSION,
            "compute_route": compute_route,
            "source_type": "audio_upload",
            "media_path": str(media_path),
            "text_base_url": content_base_url,
            "text_model": content_model,
            "template_selector_base_url": selector_base_url,
            "template_selector_model": selector_model,
            "asr_strategy": asr_result.strategy if asr_result else None,
            "provided_transcript": bool(asr_result and asr_result.strategy == "provided_transcript"),
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
        "audio_template_analysis": audio_template_analysis,
        "structured_content": structured_content,
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


def load_study_guide_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"study guide output is missing or invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"study guide output must be a JSON object: {path}")
    return payload


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
