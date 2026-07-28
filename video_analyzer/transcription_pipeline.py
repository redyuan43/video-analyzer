"""Shared ASR and speaker-diarization stage for Video Analyzer."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .artifacts import write_json
from .asr_providers import ASRStrategyResult, transcribe_with_provider_result, transcribe_with_strategy
from .audio_processor import AudioProcessor, AudioTranscript
from .local_model_runtime import local_model_stage, local_model_stage_needed
from .resource_locks import analyzer_resource_lock
from .speaker_diarization import process_transcript_speakers


def load_provided_transcript(path: Path) -> tuple[AudioTranscript, ASRStrategyResult]:
    """Load a structured transcript without invoking any audio/model stage."""
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("provided transcript JSON must be an object")
    transcript_payload = payload.get("transcript")
    if isinstance(transcript_payload, dict):
        payload = transcript_payload
    segments = payload.get("segments") or []
    metadata = payload.get("metadata") or {}
    if not isinstance(segments, list):
        raise ValueError("provided transcript segments must be a list")
    if not isinstance(metadata, dict):
        raise ValueError("provided transcript metadata must be an object")
    source_hash = hashlib.sha256(raw).hexdigest()
    metadata = dict(metadata)
    metadata["provided_transcript"] = {
        "source_path": str(path.resolve()),
        "source_sha256": source_hash,
    }
    transcript = AudioTranscript(
        text=str(payload.get("text") or ""),
        segments=segments,
        language=str(payload.get("language") or ""),
        metadata=metadata,
    )
    result = ASRStrategyResult(
        strategy="provided_transcript",
        transcript=transcript,
        fast_transcript=transcript,
        providers_run=[],
    )
    return transcript, result


def transcribe_configured_audio(
    audio_path: Path | None,
    output_dir: Path,
    config: Any,
    *,
    use_asr_strategy: bool,
    logger: logging.Logger,
    provided_transcript_path: Path | None = None,
) -> tuple[AudioTranscript | None, ASRStrategyResult]:
    """Run the configured ASR provider with the same locking used by the CLI."""
    if provided_transcript_path is not None:
        return load_provided_transcript(provided_transcript_path)
    if audio_path is None:
        raise ValueError("audio_path is required when no transcript JSON is provided")
    asr_config = config.get("asr", {})
    provider = asr_config.get("provider", "faster_whisper")
    transcript: AudioTranscript | None = None
    asr_result: ASRStrategyResult | None = None
    asr_lock = (
        contextlib.nullcontext()
        if provider == "none" or not local_model_stage_needed("asr", config.config)
        else analyzer_resource_lock(config.config, "asr", str(output_dir), logger)
    )
    with asr_lock:
        with local_model_stage("asr", config.config, logger, str(output_dir)):
            if use_asr_strategy:
                asr_result = transcribe_with_strategy(
                    strategy=asr_config.get("strategy", "balanced"),
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


def apply_speaker_diarization(
    audio_path: Path,
    transcript: AudioTranscript,
    output_dir: Path,
    config: Any,
    *,
    logger: logging.Logger,
) -> tuple[AudioTranscript, dict[str, Any]]:
    """Run the standard Video Analyzer speaker assignment/refinement step."""
    existing_report = (
        dict(transcript.metadata.get("speaker_diarization") or {})
        if isinstance(transcript.metadata, dict)
        else {}
    )
    transcript_metadata = (
        transcript.metadata if isinstance(transcript.metadata, dict) else {}
    )
    if transcript_metadata.get("speaker_diarization_applied"):
        report = {
            **existing_report,
            "enabled": True,
            "skipped": True,
            "reason": "asr_provider_already_applied_diarization",
        }
        qa_dir = output_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        write_json(qa_dir / "speaker_diarization_report.json", report)
        return transcript, report
    speaker_config = config.get("speaker_diarization") or {}
    try:
        refined, report = process_transcript_speakers(
            audio_path,
            transcript,
            speaker_config,
        )
    except Exception as exc:
        logger.warning("speaker diarization failed: %s", exc)
        refined = transcript
        report = {"enabled": True, "error": str(exc)}
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    write_json(qa_dir / "speaker_diarization_report.json", report)
    return refined, report
