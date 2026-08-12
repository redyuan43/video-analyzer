"""Shared ASR and speaker-diarization stage for Video Analyzer."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .artifacts import write_json
from .asr_providers import ASRStrategyResult, transcribe_with_provider_result, transcribe_with_strategy
from .audio_processor import AudioProcessor, AudioTranscript
from .local_model_runtime import local_model_runtime_lock, local_model_stage, local_model_stage_needed
from .resource_locks import analyzer_resource_lock
from .speaker_diarization import prepare_speaker_assignment, process_transcript_speakers


ASR_PROVIDERS_WITH_INTERNAL_DIARIZATION = {
    "auto",
    "firered_3dspeaker",
    "vibevoice",
    "vibevoice_remote",
}


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
    prepared_assignment: tuple[list[dict[str, Any]], dict[str, Any]] | None = None,
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
        with local_model_runtime_lock(
            config.config,
            logger,
            f"speaker-diarization:{output_dir}",
            stage="diarization",
        ):
            refined, report = process_transcript_speakers(
                audio_path,
                transcript,
                speaker_config,
                prepared_assignment=prepared_assignment,
            )
    except Exception as exc:
        logger.warning("speaker diarization failed: %s", exc)
        refined = transcript
        report = {"enabled": True, "error": str(exc)}
    if prepared_assignment is not None:
        report["parallel_with_asr"] = True
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    write_json(qa_dir / "speaker_diarization_report.json", report)
    return refined, report


def transcribe_and_diarize_configured_audio(
    audio_path: Path,
    output_dir: Path,
    config: Any,
    *,
    use_asr_strategy: bool,
    logger: logging.Logger,
    progress_callback: Callable[[str, str, str | None], None] | None = None,
) -> tuple[AudioTranscript | None, ASRStrategyResult, dict[str, Any] | None]:
    """Run independent speaker-turn detection in parallel with configured ASR."""
    speaker_config = config.get("speaker_diarization") or {}
    asr_provider = str((config.get("asr") or {}).get("provider") or "").strip().lower()
    should_prepare_in_parallel = speaker_diarization_can_run_parallel(config)

    def report_progress(node_id: str, status: str, message: str | None = None) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(node_id, status, message)
        except Exception:
            logger.debug("Could not report transcription node progress", exc_info=True)

    if not should_prepare_in_parallel:
        report_progress("asr", "running", f"running {asr_provider or 'configured'} ASR")
        transcript, asr_result = transcribe_configured_audio(
            audio_path,
            output_dir,
            config,
            use_asr_strategy=use_asr_strategy,
            logger=logger,
        )
        report_progress("asr", "succeeded" if transcript is not None else "failed", "ASR finished")
        if transcript is None:
            return transcript, asr_result, None
        report_progress("diarization", "running", "aligning speaker turns")
        transcript, report = apply_speaker_diarization(
            audio_path,
            transcript,
            output_dir,
            config,
            logger=logger,
        )
        report_status = "failed" if report.get("error") else "succeeded"
        progress_message = report.get("error") or "speaker diarization finished"
        report_progress("diarization", report_status, progress_message)
        asr_result.transcript = transcript
        return transcript, asr_result, report

    logger.info(
        "Starting ASR and speaker diarization in parallel (provider=%s, backend=%s)",
        asr_provider,
        speaker_config.get("backend") or "3dspeaker",
    )
    report_progress("asr", "running", f"running {asr_provider or 'configured'} ASR")
    report_progress("diarization", "running", "running in parallel with ASR")

    def prepare_assignment():
        try:
            prepared = prepare_speaker_assignment(
                audio_path,
                speaker_config,
            )
        except Exception as exc:
            report_progress("diarization", "failed", str(exc))
            raise
        report_progress("diarization", "succeeded", "speaker turns prepared")
        return prepared

    with ThreadPoolExecutor(max_workers=1) as executor:
        diarization_future = executor.submit(
            prepare_assignment,
        )
        try:
            transcript, asr_result = transcribe_configured_audio(
                audio_path,
                output_dir,
                config,
                use_asr_strategy=use_asr_strategy,
                logger=logger,
            )
        except Exception as exc:
            report_progress("asr", "failed", str(exc))
            raise
        report_progress("asr", "succeeded" if transcript is not None else "failed", "ASR finished")
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

    if transcript is None:
        return transcript, asr_result, None
    transcript, report = apply_speaker_diarization(
        audio_path,
        transcript,
        output_dir,
        config,
        logger=logger,
        prepared_assignment=prepared_assignment,
    )
    report_progress(
        "diarization",
        "failed" if report.get("error") else "succeeded",
        report.get("error") or "speaker diarization aligned",
    )
    asr_result.transcript = transcript
    return transcript, asr_result, report


def speaker_diarization_can_run_parallel(config: Any) -> bool:
    speaker_config = config.get("speaker_diarization") or {}
    asr_provider = str((config.get("asr") or {}).get("provider") or "").strip().lower()
    return (
        _truthy(speaker_config.get("parallel_with_asr"), default=True)
        and _truthy(speaker_config.get("enabled"), default=True)
        and _truthy(speaker_config.get("assignment_enabled"), default=True)
        and asr_provider not in ASR_PROVIDERS_WITH_INTERNAL_DIARIZATION
    )


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)
