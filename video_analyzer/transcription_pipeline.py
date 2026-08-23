"""Shared ASR and speaker-diarization stage for Video Analyzer."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

try:
    import ray
except ImportError:  # pragma: no cover - installation validation covers this.
    ray = None

from .artifacts import write_json
from .asr_providers import ASRStrategyResult, transcribe_with_provider_result, transcribe_with_strategy
from .audio_processor import AudioProcessor, AudioTranscript
from .local_model_runtime import local_model_runtime_lock, local_model_stage, local_model_stage_needed
from .resource_locks import analyzer_resource_lock
from .speaker_diarization import prepare_speaker_assignment, process_transcript_speakers


RAY_TRANSCRIPTION_LOCK = threading.Lock()


class ParallelBranchError(RuntimeError):
    def __init__(self, branch: str, error: Exception) -> None:
        super().__init__(f"{branch} branch failed: {error}")
        self.branch = branch
        self.original_error = error


class RuntimeConfigView:
    """Minimal serializable Config-compatible view for Ray actors."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


if ray is not None:

    @ray.remote(num_cpus=1)
    class AsrBranchActor:
        def transcribe(
            self,
            audio_path: str,
            output_dir: str,
            config_payload: dict[str, Any],
            use_asr_strategy: bool,
        ) -> tuple[AudioTranscript | None, ASRStrategyResult]:
            actor_logger = logging.getLogger("video_analyzer.ray.asr")
            transcript, result = transcribe_configured_audio(
                Path(audio_path),
                Path(output_dir),
                RuntimeConfigView(config_payload),
                use_asr_strategy=use_asr_strategy,
                logger=actor_logger,
                asr_stage_prepared=True,
            )
            if transcript is not None:
                transcript.metadata = dict(transcript.metadata or {})
                transcript.metadata["asr_dispatch"] = "ray_actor"
            return transcript, result


    @ray.remote(num_cpus=1)
    class DiarizationBranchActor:
        def diarize(
            self,
            audio_path: str,
            config_payload: dict[str, Any],
            speaker_config: dict[str, Any],
            runtime_lock_held: bool,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            lock_context = (
                contextlib.nullcontext()
                if runtime_lock_held or local_model_stage_needed("asr", config_payload)
                else local_model_runtime_lock(
                    config_payload,
                    logging.getLogger("video_analyzer.ray.diarization"),
                    f"speaker-diarization:{audio_path}",
                    stage="diarization",
                )
            )
            with lock_context:
                turns, report = prepare_speaker_assignment(
                    Path(audio_path),
                    speaker_config,
                )
            report = dict(report or {})
            report["dispatch_mode"] = "ray_actor"
            return turns, report


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
    asr_stage_prepared: bool = False,
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
        stage_context = (
            contextlib.nullcontext()
            if asr_stage_prepared
            else local_model_stage("asr", config.config, logger, str(output_dir))
        )
        with stage_context:
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
    if transcript_metadata.get("speaker_diarization_applied") and prepared_assignment is None:
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
    diarization_required = _truthy(speaker_config.get("required"), default=False) and (
        _truthy(speaker_config.get("enabled"), default=True)
        and _truthy(speaker_config.get("assignment_enabled"), default=True)
        and str(speaker_config.get("backend") or "").strip()
        not in {"", "none", "asr_embedded"}
    )
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
        if diarization_required:
            raise RuntimeError(f"selected speaker diarization model failed: {exc}") from exc
        refined = transcript
        report = {"enabled": True, "error": str(exc)}
    if diarization_required and report.get("error"):
        raise RuntimeError(
            f"selected speaker diarization model failed: {report['error']}"
        )
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
    runtime_lock_held: bool = False,
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
        report_status = "skipped" if report.get("error") else "succeeded"
        progress_message = report.get("error") or "speaker diarization finished"
        report_progress("diarization", report_status, progress_message)
        report_progress("transcript_merge", "running", "finalizing aligned transcript")
        asr_result.transcript = transcript
        report_progress("transcript_merge", "succeeded", "aligned transcript ready")
        return transcript, asr_result, report

    logger.info(
        "Starting ASR and speaker diarization in parallel (provider=%s, backend=%s)",
        asr_provider,
        speaker_config.get("backend") or "3dspeaker",
    )
    report_progress("asr", "running", f"running {asr_provider or 'configured'} ASR")
    report_progress("diarization", "running", "running in parallel with ASR")

    with local_model_stage("asr", config.config, logger, str(output_dir)):
        try:
            transcript, asr_result, prepared_assignment = (
                run_parallel_transcription_branches(
                    audio_path,
                    output_dir,
                    config.config,
                    speaker_config,
                    use_asr_strategy=use_asr_strategy,
                    runtime_lock_held=runtime_lock_held,
                )
            )
        except ParallelBranchError as exc:
            failed_node = "diarization" if exc.branch == "diarization" else "asr"
            cancelled_node = "asr" if failed_node == "diarization" else "diarization"
            report_progress(failed_node, "failed", str(exc.original_error))
            report_progress(
                cancelled_node,
                "failed",
                f"cancelled because {failed_node} failed",
            )
            raise
        report_progress("asr", "succeeded" if transcript is not None else "failed", "ASR finished")
        prepared_turns, prepared_report = prepared_assignment
        if prepared_report.get("error"):
            report_progress("diarization", "skipped", str(prepared_report["error"]))
        elif not prepared_turns:
            report_progress("diarization", "skipped", "speaker diarization produced no turns")
        else:
            report_progress("diarization", "succeeded", "speaker turns prepared")

    if transcript is None:
        return transcript, asr_result, None
    report_progress("transcript_merge", "running", "aligning speaker turns")
    try:
        transcript, report = apply_speaker_diarization(
            audio_path,
            transcript,
            output_dir,
            config,
            logger=logger,
            prepared_assignment=prepared_assignment,
        )
    except Exception as exc:
        report_progress("transcript_merge", "failed", str(exc))
        raise
    report_progress(
        "transcript_merge",
        "skipped" if report.get("error") else "succeeded",
        report.get("error") or "aligned transcript ready",
    )
    asr_result.transcript = transcript
    return transcript, asr_result, report


def run_parallel_transcription_branches(
    audio_path: Path,
    output_dir: Path,
    config_payload: dict[str, Any],
    speaker_config: dict[str, Any],
    *,
    use_asr_strategy: bool,
    runtime_lock_held: bool = False,
) -> tuple[
    AudioTranscript | None,
    ASRStrategyResult,
    tuple[list[dict[str, Any]], dict[str, Any]],
]:
    """Run independent ASR and diarization branches as Ray actors."""
    if ray is None:
        raise RuntimeError("Ray is required for parallel ASR and speaker diarization")

    with RAY_TRANSCRIPTION_LOCK:
        started_here = False
        actors: list[Any] = []
        remote_diarization = (
            str(speaker_config.get("backend") or "")
            == "remote_3dspeaker_http"
        )
        try:
            if not ray.is_initialized():
                previous_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    speaker_config.get("gpu_id") or 0
                )
                try:
                    ray_options = {
                        "namespace": "video-analyzer-transcription",
                        "ignore_reinit_error": True,
                        "include_dashboard": False,
                        "num_cpus": 2,
                        "num_gpus": 0 if remote_diarization else 1,
                    }
                    if not remote_diarization:
                        ray_options["resources"] = {"p40_diarization": 1}
                    ray.init(
                        **ray_options,
                    )
                finally:
                    if previous_cuda_devices is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_devices
                started_here = True
            asr_actor = AsrBranchActor.remote()
            diarization_options = {"num_gpus": 0 if remote_diarization else 1}
            if not remote_diarization:
                diarization_options["resources"] = {"p40_diarization": 1}
            diarization_actor = DiarizationBranchActor.options(
                **diarization_options
            ).remote()
            actors.extend((asr_actor, diarization_actor))
            asr_ref = asr_actor.transcribe.remote(
                str(audio_path),
                str(output_dir),
                config_payload,
                use_asr_strategy,
            )
            diarization_ref = diarization_actor.diarize.remote(
                str(audio_path),
                config_payload,
                speaker_config,
                runtime_lock_held,
            )
            pending = {asr_ref: "asr", diarization_ref: "diarization"}
            results: dict[str, Any] = {}
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                ref = ready[0]
                branch = pending.pop(ref)
                try:
                    results[branch] = ray.get(ref)
                except Exception as exc:
                    raise ParallelBranchError(branch, exc) from exc
            transcript, asr_result = results["asr"]
            prepared_assignment = results["diarization"]
            return transcript, asr_result, prepared_assignment
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass
            if started_here:
                ray.shutdown()


def speaker_diarization_can_run_parallel(config: Any) -> bool:
    speaker_config = config.get("speaker_diarization") or {}
    return (
        _truthy(speaker_config.get("parallel_with_asr"), default=True)
        and _truthy(speaker_config.get("enabled"), default=True)
        and _truthy(speaker_config.get("assignment_enabled"), default=True)
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
