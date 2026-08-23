#!/usr/bin/env python3
"""Audio-first template analysis for uploaded recorder files."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import re
import requests
import subprocess
import sys
import time
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_analyzer.artifacts import write_json, write_transcript_markdown  # noqa: E402
from video_analyzer.analysis_progress import write_analysis_progress  # noqa: E402
from video_analyzer.asr_providers import (  # noqa: E402
    ASRStrategyResult,
    extract_audio_to_wav,
    transcribe_with_provider_result,
    transcribe_with_strategy,
)
from video_analyzer.audio_processor import AudioProcessor, AudioTranscript  # noqa: E402
from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient  # noqa: E402
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature  # noqa: E402
from video_analyzer.local_model_runtime import (  # noqa: E402
    local_model_runtime_session,
    local_model_stage,
    local_model_stage_needed,
    try_local_model_runtime_lock,
)
from video_analyzer.resource_locks import analyzer_resource_lock  # noqa: E402
from video_analyzer.transcription_pipeline import (  # noqa: E402
    load_provided_transcript,
    transcribe_and_diarize_configured_audio,
)

try:  # noqa: E402
    import ray
except ImportError:  # pragma: no cover - exercised only in incomplete runtimes
    ray = None


DEFAULT_TEMPLATE_CATALOG = REPO_ROOT / "video-analyzer-ui" / "video_analyzer_ui" / "static" / "data" / "audio_prompt_templates.json"
DEFAULT_TEMPLATE_ID = "auto"
AUDIO_PIPELINE_PROFILE = "audio_nx1"
AUDIO_PIPELINE_VERSION = 1
DOWAY_SOURCE_REPO = "Doway AI server"
DOWAY_SOURCE_PATH = "analysis/doway_prompts/server_prompts_zh.json"
DOWAY_GENERAL_TEMPLATE_ID = "2"
MAX_TRANSCRIPT_CHARS_FOR_CLASSIFY = 9000
MAX_TRANSCRIPT_CHARS_FOR_GUIDE = 18000
TEMPLATE_SELECTOR_SHARD_COUNT = 5
TEMPLATE_SELECTOR_TOP_K = 3
TEMPLATE_SELECTOR_MIN_CONFIDENCE = 0.75
TEMPLATE_SELECTOR_AUDIT_FILE = "template_selection.json"
TEMPLATE_SELECTOR_PROMPT_CHAR_LIMIT = 6000
TEMPLATE_SELECTOR_TRANSCRIPT_CHARS = 1200
TEMPLATE_FINAL_REQUIREMENT_CHARS = 120
TEMPLATE_CONTENT_FORMS = {
    "interview",
    "meeting",
    "speech",
    "call",
    "education",
    "personal_note",
    "general",
}
TEMPLATE_FORM_FALLBACK_IDS = {
    "interview": "400000",
    "meeting": "100005",
    "speech": "200008",
    "call": "2000078",
    "education": "2000162",
    "personal_note": "15",
    "general": DOWAY_GENERAL_TEMPLATE_ID,
}
SUMMARY_SINGLE_PASS_CHARS = 24000
SUMMARY_MAP_CHUNK_CHARS = 20000
SUMMARY_REDUCE_BATCH_CHARS = 48000
SUMMARY_MAP_MAX_TOKENS = 1800
SUMMARY_FINAL_MAX_TOKENS = 4096
SUMMARY_DUPLICATE_RATIO_LIMIT = 0.15
SUMMARY_MAX_STATEMENT_REPEATS = 2
ASR_PREFLIGHT_MIN_DURATION_SECONDS = 20 * 60
ASR_PREFLIGHT_SAMPLE_SECONDS = 30
ASR_PREFLIGHT_SAMPLE_COUNT = 5
CLIENT_TEMPLATE_BLOCK_RE = re.compile(r"【模板指令开始】[\s\S]*?【模板指令结束】\s*")
CLIENT_USER_SUPPLEMENT_MARKER = "【用户补充】"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SUMMARY_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
SUMMARY_TIME_RANGE_RE = re.compile(
    r"\[(\d{2}:\d{2}(?::\d{2})?)-(\d{2}:\d{2}(?::\d{2})?)\]"
)
SUMMARY_QUESTION_RE = re.compile(
    r"^(\s*(?:\*\*)?问题)(\d+)([:：].*?(?:\*\*)?)\s*$"
)
logger = logging.getLogger("audio_template_analysis")


class AudioNodeProgress:
    TIMING_KEYS = {
        "asr": "asr_seconds",
        "diarization": "diarization_seconds",
        "transcript_merge": "transcript_merge_seconds",
        "template_selector": "template_selector_seconds",
        "text": "manual_generation_seconds",
        "artifact_package": "artifact_package_seconds",
    }
    CORE_STEPS = {
        "asr": "asr",
        "diarization": "asr",
        "transcript_merge": "asr_done",
        "template_selector": "manual",
        "text": "manual",
        "artifact_package": "write",
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.started: dict[str, float] = {}
        self.timings: dict[str, float] = {}

    def update(
        self,
        node_id: str,
        status: str,
        message: str | None = None,
    ) -> None:
        update: dict[str, Any] = {"status": status, "message": message}
        if status == "running":
            self.started.setdefault(node_id, time.perf_counter())
        elif status in {"succeeded", "failed", "skipped", "stopped"}:
            started = self.started.get(node_id)
            if started is not None:
                duration = round(max(0.0, time.perf_counter() - started), 3)
                update["duration_seconds"] = duration
                timing_key = self.TIMING_KEYS.get(node_id)
                if timing_key:
                    self.timings[timing_key] = duration
        write_analysis_progress(
            self.output_dir,
            self.CORE_STEPS.get(node_id, "manual"),
            message=message,
            node_updates={node_id: update},
        )

    def finish(
        self,
        message: str,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        write_analysis_progress(
            self.output_dir,
            "write",
            status="succeeded",
            message=message,
            artifacts=artifacts,
        )


def selector_request_extra_body() -> dict[str, Any]:
    return {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
    }


def content_request_extra_body(model: str) -> dict[str, Any]:
    if "bonsai" in str(model or "").lower():
        return selector_request_extra_body()
    return {}


def selector_client_spec(client: GenericOpenAIAPIClient) -> dict[str, Any]:
    return {
        "api_key": client.api_key,
        "api_url": client.base_url,
        "max_retries": client.max_retries,
        "timeout_seconds": client.timeout_seconds,
        "extra_body": dict(client.extra_body or {}),
        "request_headers": dict(client.request_headers or {}),
    }


if ray is not None:
    @ray.remote(num_cpus=0.2)
    class TemplateSelectorShardActor:
        def __init__(self, client_spec: dict[str, Any], model: str):
            self.client = GenericOpenAIAPIClient(**client_spec)
            self.model = model

        def select(self, prompt: str) -> str:
            return self.client.generate(
                prompt,
                model=self.model,
                temperature=0.0,
                num_predict=1400,
                extra_body=selector_request_extra_body(),
            )["response"]


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
    node_progress = AudioNodeProgress(output_dir)
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
        node_progress.update(
            "asr",
            "succeeded",
            "reused provided transcript; ASR was not executed",
        )
        provided_speakers = {
            str(
                segment.get("speaker")
                or segment.get("speaker_id")
                or segment.get("Speaker")
                or ""
            ).strip()
            for segment in transcript.segments
            if isinstance(segment, dict)
        }
        provided_speakers.discard("")
        node_progress.update(
            "diarization",
            "succeeded" if provided_speakers else "skipped",
            (
                f"reused {len(provided_speakers)} provided speaker labels"
                if provided_speakers
                else "provided transcript has no speaker labels"
            ),
        )
        node_progress.update(
            "transcript_merge",
            "succeeded",
            "reused aligned transcript",
        )
    else:
        audio_path = extract_audio_to_wav(media_path, output_dir)
        if audio_path is None:
            raise RuntimeError(f"audio extraction produced no audio stream: {media_path}")
        transcription_runtime_lock_held = local_model_stage_needed(
            "asr",
            config.config,
        )
        transcription_runtime = (
            local_model_runtime_session(config.config, logger, str(output_dir))
            if transcription_runtime_lock_held
            else contextlib.nullcontext()
        )
        with transcription_runtime:
            transcript, asr_result, speaker_report = (
                transcribe_and_diarize_configured_audio(
                    audio_path,
                    output_dir,
                    config,
                    use_asr_strategy=False,
                    logger=logger,
                    progress_callback=node_progress.update,
                    runtime_lock_held=transcription_runtime_lock_held,
                )
            )
    if transcript is None or not has_meaningful_speech(transcript.text):
        raise RuntimeError(
            "NO_SPEECH: no recognizable human speech was produced by ASR"
        )
    if asr_result:
        asr_result.transcript = transcript
    transcript_path = write_transcript_markdown(transcript, output_dir / "transcript.md")

    analysis_transcript = format_transcript_for_analysis(transcript)
    selector_executed = args.template_id == DEFAULT_TEMPLATE_ID
    primary_profile = config.get_runtime_profile(args.profile)
    fallback_profile_value = primary_profile.get("llm_fallback_profile")
    fallback_profile_name = (
        fallback_profile_value.strip()
        if isinstance(fallback_profile_value, str)
        else ""
    )
    local_candidate = bool(
        fallback_profile_name and local_text_capacity_ready(primary_profile)
    )
    lock_context = (
        try_local_model_runtime_lock(
            config.config,
            logger,
            str(output_dir),
            stage="text",
        )
        if local_candidate
        else contextlib.nullcontext(False)
    )
    with lock_context as local_lock_acquired:
        text_route = (
            "local"
            if local_candidate and local_lock_acquired
            else ("cloud_fallback" if fallback_profile_name else "configured")
        )
        text_route_reason = (
            "local_ready"
            if text_route == "local"
            else (
                "local_model_lock_busy"
                if local_candidate
                else "local_model_unavailable_or_busy"
            )
        )
        selected_config = config
        selected_profile_name = args.profile
        if text_route == "cloud_fallback":
            selected_config = load_profile_config(args.config, fallback_profile_name)
            selected_profile_name = fallback_profile_name
        selector_client, selector_model, selector_base_url, _selector_temperature = (
            build_template_selector_client(
                selected_config,
                immediate_local=text_route == "local",
            )
        )
        content_client, content_model, content_base_url, content_temperature = (
            build_content_analysis_client(
                selected_config,
                selected_profile_name,
                immediate_local=text_route == "local",
            )
        )
        summary_settings = summary_generation_settings(
            selected_config.get_runtime_profile(selected_profile_name)
        )
        if selector_executed:
            node_progress.update(
                "template_selector",
                "running",
                (
                    "selecting summary template locally"
                    if text_route == "local"
                    else "local text capacity busy; selecting template with Trae"
                ),
            )
        try:
            selected, classification = choose_template(
                client=selector_client,
                model=selector_model,
                templates=templates,
                transcript_text=analysis_transcript,
                focus_prompt=focus_prompt,
                explicit_template_id=args.template_id,
                output_dir=output_dir,
            )
        except Exception as exc:
            if text_route != "local" or not fallback_profile_name or not local_text_busy_error(exc):
                if selector_executed:
                    node_progress.update("template_selector", "failed", str(exc))
                raise
            text_route = "cloud_fallback"
            text_route_reason = "local_worker_race_busy"
            selected_config = load_profile_config(args.config, fallback_profile_name)
            selected_profile_name = fallback_profile_name
            selector_client, selector_model, selector_base_url, _selector_temperature = (
                build_template_selector_client(selected_config)
            )
            content_client, content_model, content_base_url, content_temperature = (
                build_content_analysis_client(
                    selected_config,
                    selected_profile_name,
                )
            )
            summary_settings = summary_generation_settings(
                selected_config.get_runtime_profile(selected_profile_name)
            )
            node_progress.update(
                "template_selector",
                "running",
                "local text worker became busy; retrying immediately with Trae",
            )
            selected, classification = choose_template(
                client=selector_client,
                model=selector_model,
                templates=templates,
                transcript_text=analysis_transcript,
                focus_prompt=focus_prompt,
                explicit_template_id=args.template_id,
                output_dir=output_dir,
            )
        classification["execution_route"] = text_route
        classification["execution_route_reason"] = text_route_reason
        node_progress.update(
            "template_selector",
            "succeeded",
            f"template {selected.get('id')} · {selected.get('title_zh') or selected.get('title')} selected via {text_route}",
        )
        node_progress.update(
            "text",
            "running",
            (
                "generating summary and mind map locally"
                if text_route == "local"
                else "generating summary and mind map with Trae"
            ),
        )
        try:
            summary = summarize_with_template(
                client=content_client,
                model=content_model,
                template=selected,
                transcript_text=analysis_transcript,
                focus_prompt=focus_prompt,
                language=args.language,
                temperature=content_temperature,
                source_name=args.source_name or media_path.name,
                settings=summary_settings,
            )
            summary_quality = assess_summary_quality(summary, summary_settings)
            study_guide_path = build_light_study_guide(
                content_client,
                content_model,
                output_dir,
                transcript,
                summary,
                content_temperature,
            )
        except Exception as exc:
            if text_route != "local" or not fallback_profile_name or not local_text_busy_error(exc):
                node_progress.update("text", "failed", str(exc))
                raise
            text_route = "cloud_fallback"
            text_route_reason = "local_worker_race_busy"
            selected_config = load_profile_config(args.config, fallback_profile_name)
            selected_profile_name = fallback_profile_name
            content_client, content_model, content_base_url, content_temperature = (
                build_content_analysis_client(
                    selected_config,
                    selected_profile_name,
                )
            )
            summary_settings = summary_generation_settings(
                selected_config.get_runtime_profile(selected_profile_name)
            )
            node_progress.update(
                "text",
                "running",
                "local text worker became busy; retrying immediately with Trae",
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
                settings=summary_settings,
            )
            summary_quality = assess_summary_quality(summary, summary_settings)
            study_guide_path = build_light_study_guide(
                content_client,
                content_model,
                output_dir,
                transcript,
                summary,
                content_temperature,
            )
        node_progress.update(
            "text",
            "succeeded",
            f"summary and mind map ready via {text_route}",
        )
    node_progress.update("artifact_package", "running", "writing final audio artifacts")
    write_audio_only_manifest(output_dir, media_path, audio_path)

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
        timings=node_progress.timings,
        compute_route=args.compute_route,
        summary_quality=summary_quality,
        execution_routes={
            "template_selector": {
                "route": text_route,
                "provider": (
                    "trae_deepseek"
                    if text_route == "cloud_fallback"
                    else "local_qwen"
                ),
                "model": selector_model,
                "reason": text_route_reason,
                "local_wait_seconds": 0,
            },
            "summary": {
                "route": text_route,
                "provider": (
                    "trae_deepseek"
                    if text_route == "cloud_fallback"
                    else "local_qwen"
                ),
                "model": content_model,
                "reason": text_route_reason,
                "local_wait_seconds": 0,
            },
        },
    )
    node_progress.update("artifact_package", "succeeded", "final audio artifacts written")
    node_progress.finish(
        "core artifacts complete",
        artifacts={"analysis_json": str(analysis_path)},
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


def load_profile_config(config_dir: str, profile_name: str) -> Config:
    config = Config(config_dir)
    config.update_from_args(
        argparse.Namespace(
            task="operation_manual",
            profile=profile_name,
            client=None,
            asr_provider=None,
        )
    )
    return config


def local_text_capacity_ready(profile: dict[str, Any]) -> bool:
    base_url = str(
        profile.get("text_base_url")
        or profile.get("llm_base_url")
        or ""
    ).rstrip("/")
    if not base_url.startswith(("http://127.0.0.1", "http://localhost")):
        return False
    required = max(
        1,
        min(
            TEMPLATE_SELECTOR_SHARD_COUNT,
            int(profile.get("text_worker_count") or profile.get("worker_count") or 1),
        ),
    )
    try:
        response = requests.get(
            f"{base_url.removesuffix('/v1')}/api/health",
            timeout=(1, 2),
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return False
    return bool(payload.get("ok")) and int(
        payload.get("available_workers") or 0
    ) >= required


def local_text_busy_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status_code == 503 or any(
        marker in text
        for marker in (
            "worker pool is busy",
            "model-resource-busy",
            "local-model-lock",
            "503",
        )
    )


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
        "tencent_hy_asr_ws": "tencent_hy_asr",
    }.get(protocol)
    if not provider or not endpoints:
        raise ValueError(f"unsupported audio cloud fallback ASR protocol: {protocol or '(missing)'}")
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
    elif protocol == "tencent_hy_asr_ws":
        vibevoice["tencent_hy_asr_endpoint"] = endpoints[0]
        vibevoice["tencent_hy_asr_model"] = asr.get("model")
        vibevoice["tencent_hy_asr_options"] = dict(asr.get("options") or {})
        vibevoice["asr_api_key_env"] = asr.get("api_key_env")

    diarization_protocol = str(diarization.get("protocol") or "")
    diarization_deployment = str(
        (diarization.get("options") or {}).get("deployment") or ""
    )
    if diarization_protocol == "asr_embedded" or diarization_deployment == "local":
        config.config["speaker_diarization"] = {
            "enabled": False,
            "assignment_enabled": False,
            "source": "cloud_asr_without_local_wait",
        }
    else:
        backend = {
            "three_d_speaker": "3dspeaker",
            "three_d_speaker_http": "remote_3dspeaker_http",
            "pyannote_community": "pyannote_community",
            "wespeaker_diarization": "wespeaker",
        }.get(diarization_protocol)
        if not backend:
            raise ValueError(
                f"unsupported audio cloud fallback diarization protocol: "
                f"{diarization_protocol or '(missing)'}"
            )
        speaker_config = dict(diarization.get("options") or {})
        endpoints = [
            str(item)
            for item in (diarization.get("endpoints") or [])
            if str(item)
        ]
        if diarization_protocol == "three_d_speaker_http":
            speaker_config["endpoints"] = endpoints
            speaker_config["endpoint"] = endpoints[0] if endpoints else ""
        speaker_config.update(
            {
                "enabled": True,
                "assignment_enabled": True,
                "parallel_with_asr": True,
                "backend": backend,
            }
        )
        config.config["speaker_diarization"] = speaker_config


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


def transcribe_audio(
    audio_path: Path,
    output_dir: Path,
    config: Config,
    *,
    asr_stage_prepared: bool = False,
) -> tuple[AudioTranscript | None, ASRStrategyResult | None]:
    asr_config = config.get("asr", {})
    provider = asr_config.get("provider", "faster_whisper")
    asr_result: ASRStrategyResult | None = None
    transcript: AudioTranscript | None = None
    asr_lock = (
        contextlib.nullcontext()
        if provider == "none"
        else analyzer_resource_lock(config.config, "asr", str(output_dir), logger)
    )
    runtime_context = (
        contextlib.nullcontext()
        if asr_stage_prepared
        else local_model_runtime_session(config.config, logger, str(output_dir))
    )
    with runtime_context:
        with asr_lock:
            stage_context = (
                contextlib.nullcontext()
                if asr_stage_prepared
                else local_model_stage("asr", config.config, logger, str(output_dir))
            )
            with stage_context:
                preflight_long_audio(audio_path, output_dir, config)
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


def preflight_long_audio(audio_path: Path, output_dir: Path, config: Config) -> None:
    duration = wav_duration_seconds(audio_path)
    if duration < ASR_PREFLIGHT_MIN_DURATION_SECONDS:
        return
    provider = str((config.get("asr") or {}).get("provider") or "faster_whisper")
    if provider == "none":
        return

    sample_dir = output_dir / "asr-preflight"
    sample_dir.mkdir(parents=True, exist_ok=True)
    def sample_once(index: int, offset: float) -> dict[str, Any]:
        sample_path = sample_dir / f"sample-{index:02d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{offset:.3f}",
                "-t",
                str(ASR_PREFLIGHT_SAMPLE_SECONDS),
                "-i",
                str(audio_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(sample_path),
            ],
            check=True,
            timeout=120,
        )
        result = transcribe_with_provider_result(
            provider=provider,
            audio_path=sample_path,
            language=config.get("audio", {}).get("language", ""),
            whisper_model=config.get("audio", {}).get("whisper_model", "medium"),
            device=config.get("audio", {}).get("device", "cpu"),
            vibevoice_config=(config.get("asr") or {}).get("vibevoice", {}),
        )
        text = str((result.transcript.text if result.transcript else "") or "").strip()
        return {
            "offset_seconds": round(offset, 3),
            "text": text,
            "meaningful_speech": has_meaningful_speech(text),
        }

    offsets = preflight_offsets(duration)
    with ThreadPoolExecutor(max_workers=len(offsets)) as executor:
        records = list(
            executor.map(
                lambda item: sample_once(*item),
                enumerate(offsets, 1),
            )
        )

    report_path = output_dir / "asr-preflight.json"
    write_json(
        report_path,
        {
            "provider": provider,
            "audio_duration_seconds": round(duration, 3),
            "sample_duration_seconds": ASR_PREFLIGHT_SAMPLE_SECONDS,
            "samples": records,
        },
    )
    if not any(record["meaningful_speech"] for record in records):
        raise RuntimeError(
            f"ASR preflight found no recognizable speech; see {report_path}"
        )


def preflight_offsets(duration: float) -> list[float]:
    last_start = max(0.0, duration - ASR_PREFLIGHT_SAMPLE_SECONDS)
    if ASR_PREFLIGHT_SAMPLE_COUNT == 1:
        return [last_start / 2]
    return [
        last_start * index / (ASR_PREFLIGHT_SAMPLE_COUNT - 1)
        for index in range(ASR_PREFLIGHT_SAMPLE_COUNT)
    ]


def has_meaningful_speech(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized or re.fullmatch(r"\[[^\]]+\]", normalized):
        return False
    return any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in normalized)


def wav_duration_seconds(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            rate = wav_file.getframerate()
            return wav_file.getnframes() / rate if rate else 0.0
    except (FileNotFoundError, wave.Error, OSError):
        return 0.0


def refine_audio_speakers(
    audio_path: Path,
    transcript: AudioTranscript,
    output_dir: Path,
    config: Config,
    *,
    prepared_assignment: tuple[list[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[AudioTranscript, dict[str, Any]]:
    from video_analyzer.speaker_diarization import process_transcript_speakers

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


def build_template_selector_client(
    config: Config,
    *,
    immediate_local: bool = False,
) -> tuple[GenericOpenAIAPIClient, str, str, float]:
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
        max_retries=1 if immediate_local else 3,
        request_headers=(
            {"X-Bonsai-Acquire-Timeout": "0"}
            if immediate_local
            else None
        ),
    )
    return client, model, base_url, temperature


def build_content_analysis_client(
    config: Config,
    profile_name: str | None,
    *,
    immediate_local: bool = False,
) -> tuple[GenericOpenAIAPIClient, str, str, float]:
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
    profile_has_endpoint = bool(
        profile.get("text_base_url") or profile.get("llm_base_url")
    )
    api_key = resolve_api_key(
        (
            profile.get("api_key")
            if profile_has_endpoint
            else (
                manual_config.get("text_api_key")
                or manual_config.get("api_key")
            )
        ),
        (
            profile.get("text_api_key_env") or profile.get("api_key_env")
            if profile_has_endpoint
            else (
                manual_config.get("text_api_key_env")
                or manual_config.get("api_key_env")
            )
        ),
        base_url,
    )
    client = GenericOpenAIAPIClient(
        api_key=api_key,
        api_url=base_url,
        timeout_seconds=int(profile.get("timeout_seconds") or manual_config.get("timeout_seconds") or 600),
        extra_body=build_openai_extra_body(profile, base_url, prefix=""),
        max_retries=1 if immediate_local else 3,
        request_headers=(
            {"X-Bonsai-Acquire-Timeout": "0"}
            if immediate_local
            else None
        ),
    )
    return client, model, base_url, temperature


def choose_template(
    client: GenericOpenAIAPIClient,
    model: str,
    templates: list[dict[str, Any]],
    transcript_text: str,
    focus_prompt: str,
    explicit_template_id: str = DEFAULT_TEMPLATE_ID,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if explicit_template_id and explicit_template_id != DEFAULT_TEMPLATE_ID:
        requested_id = str(explicit_template_id)
        selected = next((item for item in templates if item.get("id") == requested_id), None)
        if selected is None:
            raise ValueError(f"unknown explicit Doway template_id: {requested_id}")
        audit = {
            "version": 2,
            "method": "explicit",
            "catalog_count": len(templates),
            "catalog_fingerprint": template_catalog_fingerprint(templates),
            "transcript_sha256": hashlib.sha256(
                transcript_text.encode("utf-8")
            ).hexdigest(),
            "selector_model": model,
            "selected_template_id": requested_id,
            "warnings": [],
        }
        audit_path = write_template_selection_audit(output_dir, audit)
        return selected, {
            "method": "explicit",
            "selection_method": "explicit",
            "template_id": requested_id,
            "confidence": 1.0,
            "content_form": infer_content_form(transcript_text),
            "domain": str(selected.get("first_category") or ""),
            "runner_up_id": "",
            "margin": 100.0,
            "warnings": [],
            "audit_path": audit_path,
        }

    audit: dict[str, Any] = {
        "version": 2,
        "method": "bonsai_parallel_tournament",
        "catalog_count": len(templates),
        "catalog_fingerprint": template_catalog_fingerprint(templates),
        "transcript_sha256": hashlib.sha256(
            transcript_text.encode("utf-8")
        ).hexdigest(),
        "transcript_excerpt_chars": min(
            len(transcript_text),
            MAX_TRANSCRIPT_CHARS_FOR_CLASSIFY,
        ),
        "selector_model": model,
        "shard_count": TEMPLATE_SELECTOR_SHARD_COUNT,
        "content_form_hint": infer_content_form(transcript_text),
        "warnings": [],
    }
    try:
        shards = build_template_shards(
            templates,
            TEMPLATE_SELECTOR_SHARD_COUNT,
        )
        shard_results = run_parallel_template_shards(
            client=client,
            model=model,
            shards=shards,
            transcript_text=transcript_text,
            focus_prompt=focus_prompt,
        )
        audit["shards"] = shard_results
        form_votes = Counter(
            str(result.get("content_form") or "")
            for result in shard_results
        )
        audit["form_votes"] = dict(form_votes)
        majority_form, majority_count = (
            form_votes.most_common(1)[0] if form_votes else ("", 0)
        )
        if majority_form not in TEMPLATE_CONTENT_FORMS or majority_count < 3:
            raise RuntimeError(
                "template selector did not reach a three-of-five content-form majority"
            )
        content_form_hint = str(audit.get("content_form_hint") or "general")
        if (
            content_form_hint != "general"
            and majority_form != content_form_hint
        ):
            raise RuntimeError(
                "template selector content-form majority "
                f"{majority_form} conflicts with transcript evidence "
                f"{content_form_hint}"
            )

        finalists = template_selection_finalists(templates, shard_results)
        audit["finalists"] = compact_template_finalists(finalists)
        if not finalists:
            raise RuntimeError("template selector produced no valid finalists")
        adjudication_finalists, excluded_finalists = (
            filter_finalists_for_output_intent(
                finalists,
                focus_prompt,
                transcript_text,
            )
        )
        audit["excluded_finalists"] = compact_template_finalists(
            excluded_finalists
        )
        if not adjudication_finalists:
            raise RuntimeError(
                "template selector produced no finalists compatible with "
                "the requested output intent"
            )
        final_decision = run_template_final_adjudication(
            client=client,
            model=model,
            finalists=adjudication_finalists,
            transcript_text=transcript_text,
            focus_prompt=focus_prompt,
            majority_form=majority_form,
        )
        audit["final_decision"] = final_decision
        template_id = str(final_decision.get("template_id") or "")
        selected = next(
            (item for item in templates if item.get("id") == template_id),
            None,
        )
        confidence = float(final_decision.get("confidence") or 0.0)
        final_form = str(final_decision.get("content_form") or "")
        if selected is None:
            raise RuntimeError(
                f"final template selector returned unknown id: {template_id or '(empty)'}"
            )
        if final_form != majority_form:
            raise RuntimeError(
                f"final content form {final_form or '(empty)'} disagrees with majority {majority_form}"
            )
        selected_form = template_declared_content_form(selected)
        if selected_form and selected_form != final_form:
            raise RuntimeError(
                f"selected template category {selected_form} conflicts with "
                f"content form {final_form}"
            )
        if confidence < TEMPLATE_SELECTOR_MIN_CONFIDENCE:
            raise RuntimeError(
                f"final template confidence {confidence:.3f} is below "
                f"{TEMPLATE_SELECTOR_MIN_CONFIDENCE:.2f}"
            )

        audit["selected_template_id"] = template_id
        audit_path = write_template_selection_audit(output_dir, audit)
        return selected, {
            "method": "bonsai_parallel_tournament",
            "selection_method": "bonsai_parallel_tournament",
            "template_id": template_id,
            "scene": str(
                final_decision.get("scene")
                or content_form_label(final_form)
            ),
            "reason": str(final_decision.get("reason") or ""),
            "confidence": confidence,
            "candidate_count": len(templates),
            "content_form": final_form,
            "domain": str(final_decision.get("domain") or ""),
            "runner_up_id": str(final_decision.get("runner_up_id") or ""),
            "margin": float(final_decision.get("margin") or 0.0),
            "warnings": [],
            "audit_path": audit_path,
        }
    except Exception as exc:
        content_form = infer_content_form(transcript_text)
        fallback = fallback_template_for_form(templates, content_form)
        warning = str(exc)
        public_reason = selector_fallback_public_reason(exc)
        audit["warnings"] = [warning]
        audit["fallback"] = {
            "content_form": content_form,
            "template_id": fallback.get("id"),
            "reason": warning,
        }
        audit["selected_template_id"] = fallback.get("id")
        audit_path = write_template_selection_audit(output_dir, audit)
        return fallback, {
            "method": "content-form-fallback",
            "selection_method": "content-form-fallback",
            "reason": public_reason,
            "template_id": fallback.get("id"),
            "scene": content_form_label(content_form),
            "confidence": 0.0,
            "candidate_count": len(templates),
            "content_form": content_form,
            "domain": str(fallback.get("first_category") or ""),
            "runner_up_id": "",
            "margin": 0.0,
            "warnings": [public_reason],
            "audit_path": audit_path,
        }


def build_template_shards(
    templates: list[dict[str, Any]],
    shard_count: int = TEMPLATE_SELECTOR_SHARD_COUNT,
) -> list[list[dict[str, Any]]]:
    if shard_count <= 0:
        raise ValueError("template selector shard_count must be positive")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    ordered = sorted(
        templates,
        key=lambda item: (
            str(item.get("first_category") or ""),
            numeric_template_id(item),
        ),
    )
    for index, item in enumerate(ordered):
        shards[index % shard_count].append(item)
    if {
        str(item.get("id"))
        for shard in shards
        for item in shard
    } != {str(item.get("id")) for item in templates}:
        raise AssertionError("template sharding lost or duplicated catalog entries")
    return shards


def render_template_shard_prompt(
    templates: list[dict[str, Any]],
    transcript_text: str,
    focus_prompt: str,
    shard_index: int,
    shard_count: int,
) -> str:
    catalog = "\n\n".join(
        render_compact_template_candidate(item)
        for item in templates
    )
    content_form_hint = infer_content_form(transcript_text)
    prompt_template = f"""你是音频模板选择初审智能体，正在评审目录分片 {shard_index}/{shard_count}。

选择原则按以下顺序执行：
1. 内容形态优先：访谈、会议、演讲/播客独白、电话、教育讲座、个人口述或通用。
2. 专业领域其次：IT、金融、医疗、法律等不能覆盖内容形态。
3. 最后比较模板要求与用户关注点、期望输出结构是否匹配。

内容形态提示为 {content_form_hint}。有主持人提问和嘉宾回答时应判为 interview；
单人持续陈述才是 speech。若不同意提示，必须给出对话结构证据。

只输出一个 JSON 对象：
{{"content_form":"interview|meeting|speech|call|education|personal_note|general",
  "domain":"简短领域名称",
  "ranked":[
    {{"template_id":"目录中的ID","form_fit":95,"domain_fit":80,"instruction_fit":85,"reason":"不超过60字"}}
  ]}}

ranked 必须恰好包含本分片中最合适的 {TEMPLATE_SELECTOR_TOP_K} 个不同模板，按综合适配度降序。
三个评分必须是 0 到 100 的数字；内容形态不兼容的模板 form_fit 不得超过 30。

用户关注点：
{focus_prompt or "无"}

转写文本节选：
__TRANSCRIPT_EXCERPT__

模板目录分片：
{catalog}
"""
    return fit_selector_prompt(prompt_template, transcript_text)


def render_template_candidate(template: dict[str, Any]) -> str:
    return (
        f"id={template.get('id')}\n"
        f"标题={template.get('title_zh') or template.get('title')}\n"
        f"目录分类={template.get('first_category_zh') or template.get('first_category')}/"
        f"{template.get('second_category_zh') or template.get('second_category')}\n"
        f"完整要求：\n{template.get('prompt_original') or ''}"
    )


def render_compact_template_candidate(
    template: dict[str, Any],
    *,
    include_requirement: bool = False,
) -> str:
    title = clip_text(
        str(template.get("title_zh") or template.get("title") or ""),
        18,
    )
    category = clip_text(
        str(
            template.get("first_category_zh")
            or template.get("first_category")
            or ""
        ),
        10,
    )
    lines = [
        f"id={template.get('id')}",
        f"标题={title}",
        f"分类={category}",
    ]
    if include_requirement:
        requirement = re.sub(
            r"\s+",
            " ",
            str(template.get("prompt_original") or ""),
        ).strip()
        lines.append(
            f"要求摘要={clip_text(requirement, TEMPLATE_FINAL_REQUIREMENT_CHARS)}"
        )
    return "\n".join(lines)


def fit_selector_prompt(prompt_template: str, transcript_text: str) -> str:
    marker = "__TRANSCRIPT_EXCERPT__"
    if marker not in prompt_template:
        raise ValueError("selector prompt template is missing transcript marker")
    base_length = len(prompt_template) - len(marker)
    available = min(
        TEMPLATE_SELECTOR_TRANSCRIPT_CHARS,
        TEMPLATE_SELECTOR_PROMPT_CHAR_LIMIT - base_length,
    )
    if available < 0:
        raise ValueError(
            "template selector catalog exceeds the prompt character budget"
        )
    prompt = prompt_template.replace(
        marker,
        clip_text(transcript_text, available),
    )
    if len(prompt) > TEMPLATE_SELECTOR_PROMPT_CHAR_LIMIT:
        raise AssertionError("template selector prompt exceeded its hard budget")
    return prompt


def selector_fallback_public_reason(exc: Exception) -> str:
    diagnostic = ANSI_ESCAPE_RE.sub("", str(exc or ""))
    lowered = diagnostic.lower()
    if "exceed" in lowered and "context" in lowered:
        return "自动模板选择输入超出模型上下文，已按内容形态使用默认模板。"
    if "timeout" in lowered or "timed out" in lowered:
        return "自动模板选择暂时超时，已按内容形态使用默认模板。"
    return "自动模板选择暂不可用，已按内容形态使用默认模板。"


def run_parallel_template_shards(
    *,
    client: GenericOpenAIAPIClient,
    model: str,
    shards: list[list[dict[str, Any]]],
    transcript_text: str,
    focus_prompt: str,
) -> list[dict[str, Any]]:
    prompts = [
        render_template_shard_prompt(
            shard,
            transcript_text,
            focus_prompt,
            index,
            len(shards),
        )
        for index, shard in enumerate(shards, 1)
    ]
    if not isinstance(client, GenericOpenAIAPIClient):
        return [
            parse_template_shard_result(
                client.generate(
                    prompt,
                    model=model,
                    temperature=0.0,
                    num_predict=800,
                    extra_body=selector_request_extra_body(),
                )["response"],
                shard,
                index,
            )
            for index, (prompt, shard) in enumerate(zip(prompts, shards), 1)
        ]
    if ray is None:
        raise RuntimeError("Ray is required for parallel template selection")

    started_here = False
    actors: list[Any] = []
    results: list[dict[str, Any] | None] = [None] * len(shards)
    pending_indexes = list(range(len(shards)))
    last_errors: dict[int, str] = {}
    try:
        if not ray.is_initialized():
            ray.init(
                namespace="video-analyzer-template-selector",
                ignore_reinit_error=True,
                include_dashboard=False,
                num_cpus=max(1, len(shards)),
            )
            started_here = True
        client_spec = selector_client_spec(client)
        for attempt in range(2):
            if not pending_indexes:
                break
            round_actors = [
                TemplateSelectorShardActor.remote(client_spec, model)
                for _ in pending_indexes
            ]
            actors.extend(round_actors)
            refs = [
                actor.select.remote(prompts[index])
                for actor, index in zip(round_actors, pending_indexes)
            ]
            failed: list[int] = []
            for actor, ref, index in zip(round_actors, refs, pending_indexes):
                try:
                    results[index] = parse_template_shard_result(
                        ray.get(ref),
                        shards[index],
                        index + 1,
                    )
                except Exception as exc:
                    last_errors[index] = str(exc)
                    failed.append(index)
                finally:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
            pending_indexes = failed
        if pending_indexes:
            details = "; ".join(
                f"shard {index + 1}: {last_errors.get(index, 'unknown error')}"
                for index in pending_indexes
            )
            raise RuntimeError(f"template selector shard failed after retry: {details}")
        return [result for result in results if result is not None]
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        if started_here:
            ray.shutdown()


def parse_template_shard_result(
    response: str,
    shard: list[dict[str, Any]],
    shard_index: int,
) -> dict[str, Any]:
    payload = parse_json_object(response)
    content_form = str(payload.get("content_form") or "")
    if content_form not in TEMPLATE_CONTENT_FORMS:
        raise ValueError(
            f"shard {shard_index} returned unsupported content_form: {content_form}"
        )
    allowed_ids = {str(item.get("id")) for item in shard}
    ranked = payload.get("ranked")
    if not isinstance(ranked, list):
        raise ValueError(f"shard {shard_index} did not return ranked templates")
    normalized = []
    seen = set()
    for item in ranked:
        if not isinstance(item, dict):
            continue
        template_id = str(item.get("template_id") or "")
        if template_id not in allowed_ids or template_id in seen:
            continue
        seen.add(template_id)
        normalized.append(
            {
                "template_id": template_id,
                "form_fit": bounded_score(item.get("form_fit")),
                "domain_fit": bounded_score(item.get("domain_fit")),
                "instruction_fit": bounded_score(item.get("instruction_fit")),
                "reason": str(item.get("reason") or "")[:240],
            }
        )
    expected_count = min(TEMPLATE_SELECTOR_TOP_K, len(shard))
    if not normalized:
        raise ValueError(
            f"shard {shard_index} returned no valid finalists"
        )
    warnings = []
    if len(normalized) < expected_count:
        missing_count = expected_count - len(normalized)
        warnings.append(
            f"model returned {len(normalized)}/{expected_count} valid finalists; "
            f"completed {missing_count} deterministically"
        )
        remaining = [
            item
            for item in shard
            if str(item.get("id")) not in seen
        ]
        matching = [
            item
            for item in remaining
            if template_declared_content_form(item) == content_form
        ]
        for item in (matching + remaining):
            template_id = str(item.get("id") or "")
            if template_id in seen:
                continue
            seen.add(template_id)
            normalized.append(
                {
                    "template_id": template_id,
                    "form_fit": 0.0,
                    "domain_fit": 0.0,
                    "instruction_fit": 0.0,
                    "reason": "模型未返回足够候选，由同分片稳定补位。",
                }
            )
            if len(normalized) == expected_count:
                break
    return {
        "shard": shard_index,
        "template_count": len(shard),
        "content_form": content_form,
        "domain": str(payload.get("domain") or "")[:120],
        "ranked": normalized,
        "warnings": warnings,
    }


def bounded_score(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def template_selection_finalists(
    templates: list[dict[str, Any]],
    shard_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    catalog = {str(item.get("id")): item for item in templates}
    finalists = []
    seen = set()
    for result in shard_results:
        for rank, score in enumerate(result.get("ranked") or [], 1):
            template_id = str(score.get("template_id") or "")
            if template_id in seen or template_id not in catalog:
                continue
            seen.add(template_id)
            finalists.append(
                {
                    "shard": result.get("shard"),
                    "shard_rank": rank,
                    "shard_content_form": result.get("content_form"),
                    "shard_domain": result.get("domain"),
                    "scores": score,
                    "template": catalog[template_id],
                }
            )
    return finalists


def compact_template_finalists(
    finalists: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "shard": item.get("shard"),
            "shard_rank": item.get("shard_rank"),
            "shard_content_form": item.get("shard_content_form"),
            "shard_domain": item.get("shard_domain"),
            "scores": item.get("scores"),
            "template_id": str((item.get("template") or {}).get("id") or ""),
            "title": (
                (item.get("template") or {}).get("title_zh")
                or (item.get("template") or {}).get("title")
            ),
            "category": "/".join(
                str(value)
                for value in (
                    (item.get("template") or {}).get("first_category_zh")
                    or (item.get("template") or {}).get("first_category"),
                    (item.get("template") or {}).get("second_category_zh")
                    or (item.get("template") or {}).get("second_category"),
                )
                if value
            ),
        }
        for item in finalists
    ]


def filter_finalists_for_output_intent(
    finalists: list[dict[str, Any]],
    focus_prompt: str,
    transcript_text: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allow_transformation = focus_requests_transformation(focus_prompt)
    interview_scenario = infer_interview_scenario(transcript_text)
    accepted = []
    excluded = []
    for item in finalists:
        template = item.get("template") or {}
        if (
            not allow_transformation
            and template_output_intent(template) == "transform"
        ):
            excluded.append(item)
            continue
        template_scenario = template_interview_scenario(template)
        if (
            interview_scenario
            and template_scenario not in {"", interview_scenario}
        ):
            excluded.append(item)
            continue
        accepted.append(item)
    if interview_scenario:
        exact_scenario = [
            item
            for item in accepted
            if template_interview_scenario(item.get("template") or {})
            == interview_scenario
        ]
        if exact_scenario:
            excluded.extend(
                item
                for item in accepted
                if item not in exact_scenario
            )
            accepted = exact_scenario
    return accepted, excluded


def focus_requests_transformation(focus_prompt: str) -> bool:
    haystack = str(focus_prompt or "").lower()
    return any(
        term in haystack
        for term in (
            "改写",
            "重写",
            "翻译",
            "营销",
            "脚本",
            "生成播客",
            "创建播客",
            "生成文章",
            "生成书",
            "rewrite",
            "translate",
            "marketing",
            "script",
        )
    )


def template_output_intent(template: dict[str, Any]) -> str:
    haystack = " ".join(
        str(template.get(key) or "")
        for key in ("title", "title_zh", "prompt_original")
    ).lower()
    transformative_terms = (
        "创建播客",
        "生成播客",
        "音频播报",
        "营销内容",
        "营销脚本",
        "播客脚本",
        "内容再利用",
        "社交媒体帖子",
        "生成书籍",
        "改写",
        "重写",
        "翻译成",
        "rewrite",
        "translate",
    )
    return (
        "transform"
        if any(term in haystack for term in transformative_terms)
        else "summarize"
    )


def infer_interview_scenario(transcript_text: str) -> str:
    haystack = str(transcript_text or "")[:12000].lower()
    if (
        ("主持人" in haystack and "嘉宾" in haystack)
        or "播客" in haystack
        or "podcast" in haystack
    ):
        return "podcast"
    if any(
        term in haystack
        for term in ("求职", "招聘", "候选人", "面试官", "应聘", "职位面试")
    ):
        return "job"
    if any(
        term in haystack
        for term in ("用户访谈", "用户调研", "产品调研", "用户需求访谈")
    ):
        return "user_research"
    if any(term in haystack for term in ("教练会议", "教练辅导", "coaching")):
        return "coaching"
    if any(term in haystack for term in ("一对一", "1on1", "kpt")):
        return "one_on_one"
    return ""


def template_interview_scenario(template: dict[str, Any]) -> str:
    haystack = " ".join(
        str(template.get(key) or "")
        for key in ("title", "title_zh")
    ).lower()
    if any(term in haystack for term in ("工作面试", "求职", "招聘")):
        return "job"
    if "用户访谈" in haystack:
        return "user_research"
    if "教练" in haystack:
        return "coaching"
    if any(term in haystack for term in ("一对一", "1on1", "kpt")):
        return "one_on_one"
    if "播客" in haystack:
        return "podcast"
    return ""


def run_template_final_adjudication(
    *,
    client: GenericOpenAIAPIClient,
    model: str,
    finalists: list[dict[str, Any]],
    transcript_text: str,
    focus_prompt: str,
    majority_form: str,
) -> dict[str, Any]:
    interview_scenario = infer_interview_scenario(transcript_text)
    runner_up_instruction = (
        "runner_up_id 必须留空，因为决赛只有一个合法候选。"
        if len(finalists) == 1
        else "runner_up_id 必须填写决赛中的第二名模板ID。"
    )
    finalist_text = "\n\n".join(
        (
            f"初选分片={item.get('shard')} 排名={item.get('shard_rank')} "
            f"分片形态={item.get('shard_content_form')} "
            f"分片领域={item.get('shard_domain')} "
            f"初选评分={json.dumps(item.get('scores'), ensure_ascii=False)}\n"
            f"{render_compact_template_candidate(item['template'], include_requirement=True)}"
        )
        for item in finalists
    )
    prompt_template = f"""你是音频模板选择终审智能体。五路初审对内容形态的多数判断是：
{majority_form}

必须首先服从内容形态，再比较专业领域和输出要求。不得仅因主题包含 AI、模型、医疗或金融，就选择与访谈/会议等形态不兼容的模板。
访谈包括主持人/采访者与嘉宾/受访者持续问答的播客和深度对谈；演讲仅指一位主讲人持续陈述。最终模板若明确属于另一种内容形态，必须淘汰。
当前默认输出目标是忠实总结、记录和分析已有录音，不是把内容改写成新播客、营销稿、脚本、书籍或翻译稿。除非用户关注点明确要求再创作，否则优先选择摘要、采访记录、笔记或分析模板。
当前访谈子场景判断为 {interview_scenario or "通用"}。播客访谈不得选择工作面试、教练会议、用户调研或一对一绩效沟通模板；其他专用访谈场景也必须与转写证据一致。

只输出一个 JSON 对象：
{{"template_id":"决赛模板ID",
  "runner_up_id":"第二名模板ID",
  "content_form":"{majority_form}",
  "domain":"简短领域名称",
  "scene":"中文场景",
  "confidence":0.0,
  "margin":0,
  "reason":"不超过100字"}}

confidence 范围 0 到 1，margin 范围 0 到 100。
{runner_up_instruction}

用户关注点：
{focus_prompt or "无"}

转写文本节选：
__TRANSCRIPT_EXCERPT__

决赛模板：
{finalist_text}
"""
    prompt = fit_selector_prompt(prompt_template, transcript_text)
    response = client.generate(
        prompt,
        model=model,
        temperature=0.0,
        num_predict=600,
        extra_body=selector_request_extra_body(),
    )["response"]
    payload = parse_json_object(response)
    allowed_ids = {
        str(item["template"].get("id"))
        for item in finalists
    }
    template_id = str(payload.get("template_id") or "")
    runner_up_id = str(payload.get("runner_up_id") or "")
    if template_id not in allowed_ids:
        raise ValueError(
            f"final adjudicator returned id outside finalists: {template_id or '(empty)'}"
        )
    if len(finalists) == 1:
        runner_up_id = ""
    elif runner_up_id and runner_up_id not in allowed_ids:
        raise ValueError(
            f"final adjudicator returned invalid runner_up_id: {runner_up_id}"
        )
    content_form = str(payload.get("content_form") or "")
    if content_form not in TEMPLATE_CONTENT_FORMS:
        raise ValueError(
            f"final adjudicator returned unsupported content_form: {content_form}"
        )
    payload["template_id"] = template_id
    payload["runner_up_id"] = runner_up_id
    payload["content_form"] = content_form
    payload["confidence"] = max(
        0.0,
        min(1.0, float(payload.get("confidence") or 0.0)),
    )
    payload["margin"] = bounded_score(payload.get("margin"))
    return payload


def infer_content_form(transcript_text: str) -> str:
    haystack = str(transcript_text or "")[:9000].lower()
    rules = (
        (
            "interview",
            (
                "主持人",
                "嘉宾",
                "受访者",
                "采访",
                "访谈",
                "深度对谈",
                "podcast",
                "interview",
            ),
        ),
        ("meeting", ("会议", "参会", "议程", "决策", "行动项", "待办")),
        ("call", ("电话", "通话", "来电", "call")),
        ("education", ("课程", "课堂", "讲座", "培训", "教授", "老师")),
        ("personal_note", ("口述", "日记", "备忘", "提醒自己", "我的想法")),
        ("speech", ("演讲", "主题演讲", "podcast", "播客")),
    )
    for content_form, terms in rules:
        if any(term in haystack for term in terms):
            return content_form
    return "general"


def template_declared_content_form(
    template: dict[str, Any],
) -> str:
    category = str(template.get("first_category") or "").strip().lower()
    return {
        "interview": "interview",
        "meeting": "meeting",
        "speech": "speech",
        "call": "call",
        "education": "education",
    }.get(category, "")


def fallback_template_for_form(
    templates: list[dict[str, Any]],
    content_form: str,
) -> dict[str, Any]:
    template_id = TEMPLATE_FORM_FALLBACK_IDS.get(
        content_form,
        DOWAY_GENERAL_TEMPLATE_ID,
    )
    selected = next(
        (item for item in templates if str(item.get("id")) == template_id),
        None,
    )
    if selected is None and template_id != DOWAY_GENERAL_TEMPLATE_ID:
        selected = next(
            (
                item
                for item in templates
                if str(item.get("id")) == DOWAY_GENERAL_TEMPLATE_ID
            ),
            None,
        )
    if selected is None:
        raise ValueError(
            f"template catalog is missing fallback template {template_id}"
        )
    return selected


def content_form_label(content_form: str) -> str:
    return {
        "interview": "访谈",
        "meeting": "会议",
        "speech": "演讲或播客独白",
        "call": "电话通话",
        "education": "教育讲座",
        "personal_note": "个人口述",
        "general": "通用音频",
    }.get(content_form, "通用音频")


def template_catalog_fingerprint(templates: list[dict[str, Any]]) -> str:
    payload = [
        {
            "id": str(item.get("id") or ""),
            "prompt_sha256": (
                (item.get("server") or {}).get("prompt_sha256")
                or hashlib.sha256(
                    str(item.get("prompt_original") or "").encode("utf-8")
                ).hexdigest()
            ),
        }
        for item in sorted(templates, key=numeric_template_id)
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_template_selection_audit(
    output_dir: Path | None,
    payload: dict[str, Any],
) -> str:
    if output_dir is None:
        return ""
    path = output_dir / TEMPLATE_SELECTOR_AUDIT_FILE
    write_json(path, payload)
    return path.name


def numeric_template_id(template: dict[str, Any]) -> int:
    return int(str(template.get("id")))


def parse_bool_setting(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def summary_generation_settings(profile: dict[str, Any] | None) -> dict[str, Any]:
    profile = profile or {}
    return {
        "single_pass_chars": max(
            1000,
            int(profile.get("summary_single_pass_chars") or SUMMARY_SINGLE_PASS_CHARS),
        ),
        "map_chunk_chars": max(
            1000,
            int(profile.get("summary_map_chunk_chars") or SUMMARY_MAP_CHUNK_CHARS),
        ),
        "reduce_batch_chars": max(
            2000,
            int(
                profile.get("summary_reduce_batch_chars")
                or SUMMARY_REDUCE_BATCH_CHARS
            ),
        ),
        "map_max_tokens": max(
            256,
            int(profile.get("summary_map_max_tokens") or SUMMARY_MAP_MAX_TOKENS),
        ),
        "final_max_tokens": max(
            512,
            int(
                profile.get("summary_final_max_tokens")
                or SUMMARY_FINAL_MAX_TOKENS
            ),
        ),
        "duplicate_ratio_limit": min(
            1.0,
            max(
                0.0,
                float(
                    profile.get("summary_duplicate_ratio_limit")
                    or SUMMARY_DUPLICATE_RATIO_LIMIT
                ),
            ),
        ),
        "max_statement_repeats": max(
            1,
            int(
                profile.get("summary_max_statement_repeats")
                or SUMMARY_MAX_STATEMENT_REPEATS
            ),
        ),
        "corrective_retry": parse_bool_setting(
            profile.get("summary_corrective_retry"),
            True,
        ),
    }


def summarize_with_template(
    client: GenericOpenAIAPIClient,
    model: str,
    template: dict[str, Any],
    transcript_text: str,
    focus_prompt: str,
    language: str,
    temperature: float,
    source_name: str = "",
    settings: dict[str, Any] | None = None,
) -> str:
    settings = {
        **summary_generation_settings({}),
        **dict(settings or {}),
    }
    template_prompt = str(template.get("prompt_original") or "")
    if not template_prompt.strip():
        raise ValueError(f"Doway template {template.get('id')} has an empty prompt")
    recording_time = recording_time_from_source(source_name)
    if len(transcript_text) <= settings["single_pass_chars"]:
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
        summary = generate_required_text(
            client,
            prompt,
            model=model,
            temperature=temperature,
            num_predict=settings["final_max_tokens"],
            stage="final summary",
        )
        return ensure_summary_quality(
            client=client,
            model=model,
            summary=summary,
            correction_prompt=prompt,
            settings=settings,
            stage="final summary",
        )

    chunks = split_transcript_chunks(transcript_text, settings["map_chunk_chars"])
    map_summaries = []
    for index, chunk in enumerate(chunks, 1):
        map_prompt = f"""你正在为长录音总结做第 {index}/{len(chunks)} 个连续分块的事实提炼。

请使用 {language}，完整保留本分块中的说话人、时间线、具体数字、观点、决策、行动项、负责人、截止时间、风险和未决问题。不要套用最终总结模板，不要推断未出现的信息。相邻时间段表达同一观点时合并为一条；每条只写一个独立事实，不得复制同一句话覆盖不同时间段。输出紧凑但信息充分的结构化 Markdown，供后续 reduce 使用。

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
                num_predict=settings["map_max_tokens"],
                stage=f"map chunk {index}/{len(chunks)}",
            )
        )

    reduce_source = reduce_map_summaries(
        client=client,
        model=model,
        summaries=map_summaries,
        language=language,
        temperature=temperature,
        max_chars=settings["reduce_batch_chars"],
        max_tokens=settings["map_max_tokens"],
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
    summary = generate_required_text(
        client,
        prompt,
        model=model,
        temperature=temperature,
        num_predict=settings["final_max_tokens"],
        stage="final reduce summary",
    )
    return ensure_summary_quality(
        client=client,
        model=model,
        summary=summary,
        correction_prompt=prompt,
        settings=settings,
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
- 直接输出最终结果，不要输出思考过程、分析步骤、草稿说明或英文元叙述。
- 下方 Doway 模板是主要输出规范，必须完整遵守；用户补充关注点只能补充强调，不能替代、删减或改写模板要求。
- 模板询问会议日期或会议时间时，优先使用录音文件时间。
- 如果转写文本明确提到另一个会议日期，以转写文本为准，并说明它来自转写。
- 不要根据当前日期、任务创建时间或常识推断会议日期。
- 会议日期/会议时间可以使用录音文件时间；行动项截止日期不能默认使用录音文件时间。
- 截止日期只有在转写明确出现具体日历日期时才写具体日期。
- 如果转写只出现“明天”“下周”“下周四”“月底”等相对时间，必须保留原文，并标注“相对时间，未换算”。
- 严禁把相对时间换算成 YYYY-MM-DD 或“某年某月某日”，即使可以根据录音时间推算。
- 可以总结相对顺序和依赖关系，但不要把相对时间改写成具体日历日期。
- 采访过程必须按真实问题组织，不要为每一个转写分段机械生成一条记录。
- 相邻时间段表达同一观点时应合并；禁止用同一句话覆盖多个不同时间段。
- 每个问题和回答必须包含该部分独有的信息，禁止重复问题、重复回答或重复时间范围。
- 如果某个时间范围没有足够信息，宁可省略，也不要复制上一条内容补齐。

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
    payload = client.generate(
        prompt,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        extra_body=content_request_extra_body(model),
    )
    response = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(f"content model returned an empty response during {stage}")
    if payload.get("response_source") == "reasoning_content":
        raise RuntimeError(
            f"content model returned reasoning_content instead of final output during {stage}"
        )
    if looks_like_reasoning_trace(response):
        raise RuntimeError(
            f"content model leaked a reasoning trace during {stage}"
        )
    return response.strip()


def looks_like_reasoning_trace(text: str) -> bool:
    head = str(text or "").lstrip()[:600].lower()
    return any(
        marker in head
        for marker in (
            "here's a thinking process",
            "here is a thinking process",
            "**analyze user input:**",
            "analyze user input:",
            "mental mapping",
            "draft generation",
        )
    )


def normalize_summary_statement(value: str) -> str:
    text = SUMMARY_TIME_RANGE_RE.sub("", str(value or ""))
    text = re.sub(r"[*_`#>]", "", text)
    text = re.sub(r"[\s，。；：、,.!?！？;:\"'“”‘’（）()\[\]【】]+", "", text)
    return text.lower()


def assess_summary_quality(
    summary: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = {
        **summary_generation_settings({}),
        **dict(settings or {}),
    }
    statements = []
    time_ranges = []
    for line in str(summary or "").splitlines():
        match = SUMMARY_BULLET_RE.match(line)
        if not match:
            continue
        statement = normalize_summary_statement(match.group(1))
        if statement:
            statements.append(statement)
        time_match = SUMMARY_TIME_RANGE_RE.search(line)
        if time_match:
            time_ranges.append(time_match.group(0))

    counts = Counter(statements)
    repeated_statement_count = sum(
        count - 1 for count in counts.values() if count > 1
    )
    max_statement_repeat = max(counts.values(), default=0)
    duplicate_ratio = (
        repeated_statement_count / len(statements) if statements else 0.0
    )
    time_counts = Counter(time_ranges)
    duplicate_time_range_count = sum(
        count - 1 for count in time_counts.values() if count > 1
    )
    issues = []
    if (
        len(statements) >= 6
        and duplicate_ratio > settings["duplicate_ratio_limit"]
    ):
        issues.append(
            "duplicate statement ratio "
            f"{duplicate_ratio:.3f} exceeds "
            f"{settings['duplicate_ratio_limit']:.3f}"
        )
    if max_statement_repeat > settings["max_statement_repeats"]:
        issues.append(
            "one statement repeats "
            f"{max_statement_repeat} times; maximum is "
            f"{settings['max_statement_repeats']}"
        )
    if duplicate_time_range_count:
        issues.append(
            f"{duplicate_time_range_count} duplicate timestamp ranges detected"
        )
    return {
        "passed": not issues,
        "statement_count": len(statements),
        "unique_statement_count": len(counts),
        "repeated_statement_count": repeated_statement_count,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "max_statement_repeat": max_statement_repeat,
        "timestamp_range_count": len(time_ranges),
        "duplicate_time_range_count": duplicate_time_range_count,
        "issues": issues,
    }


def deduplicate_summary_lines(summary: str) -> tuple[str, int]:
    lines = str(summary or "").splitlines()
    seen = set()
    kept = []
    removed = 0
    for line in lines:
        match = SUMMARY_BULLET_RE.match(line)
        if match:
            statement = normalize_summary_statement(match.group(1))
            if statement and statement in seen:
                removed += 1
                continue
            if statement:
                seen.add(statement)
        kept.append(line)

    blocks = []
    current = []
    for line in kept:
        if SUMMARY_QUESTION_RE.match(line) and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)

    filtered = []
    question_index = 0
    for block in blocks:
        heading_match = SUMMARY_QUESTION_RE.match(block[0]) if block else None
        if heading_match:
            body = [line for line in block[1:] if line.strip()]
            if not body:
                removed += 1
                continue
            question_index += 1
            block[0] = (
                f"{heading_match.group(1)}{question_index}"
                f"{heading_match.group(3)}"
            )
        filtered.extend(block)

    return "\n".join(filtered).strip(), removed


def ensure_summary_quality(
    *,
    client: GenericOpenAIAPIClient,
    model: str,
    summary: str,
    correction_prompt: str,
    settings: dict[str, Any],
    stage: str,
) -> str:
    quality = assess_summary_quality(summary, settings)
    if quality["passed"]:
        return summary
    if not settings["corrective_retry"]:
        raise RuntimeError(
            f"summary quality gate failed during {stage}: "
            + "; ".join(quality["issues"])
        )

    correction = f"""前一次输出未通过质量检查：
{chr(10).join(f"- {issue}" for issue in quality["issues"])}

请重新生成完整最终结果。必须保留原始事实覆盖，但要合并相邻重复观点，删除重复句子和重复时间范围。每个问题只保留独有内容，不得通过复制上一条来补齐。直接输出修正后的完整结果。

原始任务：
{correction_prompt}
"""
    corrected = generate_required_text(
        client,
        correction,
        model=model,
        temperature=0.0,
        num_predict=settings["final_max_tokens"],
        stage=f"{stage} corrective retry",
    )
    corrected_quality = assess_summary_quality(corrected, settings)
    if corrected_quality["passed"]:
        return corrected

    deduplicated, removed = deduplicate_summary_lines(corrected)
    deduplicated_quality = assess_summary_quality(deduplicated, settings)
    if removed and deduplicated_quality["passed"]:
        logger.warning(
            "Summary corrective retry required deterministic deduplication: "
            "removed %s repeated lines",
            removed,
        )
        return deduplicated
    raise RuntimeError(
        f"summary quality gate failed after corrective retry during {stage}: "
        + "; ".join(deduplicated_quality["issues"])
    )


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
    max_chars: int = SUMMARY_REDUCE_BATCH_CHARS,
    max_tokens: int = SUMMARY_MAP_MAX_TOKENS,
) -> str:
    current = list(summaries)
    level = 1
    while len(render_summary_sections(current)) > max_chars:
        batches = pack_summary_batches(current, max_chars)
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
                    num_predict=max_tokens,
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
        text = str(
            segment.get("text")
            or segment.get("content")
            or segment.get("Text")
            or segment.get("Content")
            or ""
        ).strip()
        if not text:
            continue
        speaker = str(
            segment.get("speaker")
            or segment.get("speaker_id")
            or segment.get("Speaker")
            or "说话人未提供"
        ).strip()
        start_value = segment.get("start")
        if start_value is None:
            start_value = segment.get("Start")
        end_value = segment.get("end")
        if end_value is None:
            end_value = segment.get("End")
        start = format_seconds(start_value or 0)
        end = format_seconds(end_value if end_value is not None else start_value or 0)
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
        response = generate_required_text(
            client,
            prompt,
            model=model,
            temperature=temperature,
            num_predict=4096,
            stage="study guide",
        )
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


def mermaid_mindmap_label(value: Any, max_chars: int) -> str:
    text = re.sub(r"[\r\n\t]+", " ", str(value or ""))
    text = re.sub(r"[\[\]{}()\"`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return clip_text(text, max_chars) or "未命名"


def render_mindmap_mermaid(guide: dict[str, Any]) -> str:
    mindmap = guide.get("mindmap") if isinstance(guide.get("mindmap"), dict) else {}
    nodes = mindmap.get("nodes") if isinstance(mindmap.get("nodes"), list) else []
    lines = [
        "mindmap",
        f"  root(({mermaid_mindmap_label(mindmap.get('title') or guide.get('title'), 36)}))",
    ]
    for node in nodes[:8]:
        if not isinstance(node, dict):
            continue
        lines.append(f"    {mermaid_mindmap_label(node.get('label'), 28)}")
        children = node.get("children") if isinstance(node.get("children"), list) else []
        for child in children[:5]:
            lines.append(f"      {mermaid_mindmap_label(child, 42)}")
    return "\n".join(lines)


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
    lines.extend(
        [
            "## 内容脑图",
            "",
            "```mermaid",
            render_mindmap_mermaid(guide),
            "```",
            "",
        ]
    )
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
    reason = public_document_reason(
        classification.get("reason")
        or classification.get("fallback_reason")
        or ""
    )
    text = f"""# 音频总结

- 模板：{template.get('title_zh') or template.get('title')}
- 场景：{classification.get('scene') or template.get('first_category_zh') or template.get('first_category')}
- 选择原因：{reason}
- 用户关注点：{focus_prompt or '无'}

## 总结

{summary}
"""
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def public_document_reason(value: Any) -> str:
    text = ANSI_ESCAPE_RE.sub("", str(value or "")).strip()
    internal_markers = (
        "traceback",
        "ray::",
        "raytaskerror",
        "original exception",
        'file "/home/',
        "openaiapierror",
        "failed to deserialize exception",
    )
    lowered = text.lower()
    if any(marker in lowered for marker in internal_markers):
        return "自动模板选择暂不可用，已按内容形态使用默认模板。"
    text = re.sub(r"\s+", " ", text)
    return clip_text(text, 180) or "根据内容形态和模板要求选择。"


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
    timings: dict[str, float] | None = None,
    compute_route: str = "local",
    summary_quality: dict[str, Any] | None = None,
    execution_routes: dict[str, Any] | None = None,
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
    quality = dict(summary_quality or assess_summary_quality(summary))
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
        "quality": quality,
        "execution_routes": dict(execution_routes or {}),
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
            "timings": {
                **dict(timings or {}),
                "total_seconds": elapsed_seconds,
            },
            "execution_routes": dict(execution_routes or {}),
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
            "quality_gate_passed": bool(quality.get("passed")),
            "quality_review": quality,
        },
        "frame_analyses": [],
        "operation_manual": {
            "response": summary,
            "manual_path": str(manual_path),
            "evidence_path": str(evidence_path),
            "quality_gate_passed": bool(quality.get("passed")),
            "quality_review": quality,
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
