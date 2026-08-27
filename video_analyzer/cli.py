import argparse
import contextlib
import json
import logging
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

from .analysis_progress import write_analysis_progress
from .analyzer import VideoAnalyzer
from .artifacts import write_orin_artifacts, write_transcript_markdown
from .asr_providers import ASRStrategyResult, extract_audio_to_wav
from .audio_processor import AudioTranscript
from .candidate_frame_strategies import parse_candidate_frame_strategy
from .clients.generic_openai_api import GenericOpenAIAPIClient
from .clients.ollama import OllamaClient
from .config import Config, build_openai_extra_body, get_client, get_model, resolve_api_key, resolve_temperature
from .frame import VideoProcessor
from .frame_dedup_audit import select_audited_frames, write_frame_dedup_audit
from .frame_manifest import MANIFEST_NAME, read_frames_from_manifest, write_frame_manifest
from .frame_selection import (
    AUTO,
    FrameDecision,
    FrameSelectionOptions,
    build_frame_context_window,
    make_skipped_visual_event,
    parse_auto_float,
    parse_auto_int,
    resolve_candidate_frame_budget,
    resolve_vl_context_gap_seconds,
    select_vl_frames,
)
from .jetson_frames import (
    extract_frames_with_jetson_workers,
    extract_frames_with_local_gpu_workers,
    extract_local_screen_keyframes,
)
from .local_model_runtime import local_model_runtime_session, local_model_stage
from .manual import (
    embed_step_images,
    generate_operation_manual,
    prepare_frame_assets,
    read_context_file,
    resolve_manual_prompt_char_budget,
    review_operation_manual_markdown,
    write_frame_evidence_index,
)
from .ocr import OCREvent, run_ocr
from .ocr_keyframes import (
    AUTO as OCR_AUTO,
)
from .ocr_keyframes import (
    build_ocr_text_events,
    resolve_ocr_scan_sample_fps,
    select_ocr_keyframes,
)
from .prompt import PromptLoader
from .resource_locks import analyzer_resource_lock
from .review_artifacts import write_run_manifest, write_visual_review
from .transcription_pipeline import (
    speaker_diarization_can_run_parallel,
    transcribe_and_diarize_configured_audio,
)
from .vl_checkpoint import (
    analysis_signature,
    frame_sha256,
    load_vl_checkpoint,
    write_vl_checkpoint,
)

# Initialize logger at module level
logger = logging.getLogger(__name__)

from .cli_parser import build_arg_parser  # noqa: E402

from .cli_helpers import (  # noqa: E402
    DEFAULT_VL_TARGET_SECONDS,
    analyze_frames_for_vl,
    append_evidence_boundary_section,
    cleanup_files,
    create_client,
    create_operation_manual_fallback_client,
    create_operation_manual_text_client,
    get_log_level,
    load_ocr_checkpoint,
    media_has_video_stream,
    ocr_signature_payload,
    parse_jetson_frame_weights,
    read_page_context_metadata,
    read_transcript_markdown,
    recent_vl_seconds_per_frame,
    reusable_frames_from_manifest,
    scan_frames_count_from_metadata,
    vl_signature_payload,
    write_ocr_checkpoint,
)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Set up logging with specified level
    log_level = get_log_level(args.log_level)
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Force reconfiguration of the root logger
    )
    # Ensure our module logger has the correct level
    logger.setLevel(log_level)

    output_arg = Path(args.output or "output")
    if args.resume_existing and not args.transcript_file:
        existing_transcript = output_arg / "transcript.md"
        if existing_transcript.is_file() and existing_transcript.stat().st_size > 0:
            args.transcript_file = str(existing_transcript)
            args.asr_provider = "none"
            logger.info("Resume mode found existing transcript: %s", existing_transcript)

    # Load and update configuration
    config = Config(args.config)
    config.update_from_args(args)

    # Initialize components
    video_path = Path(args.video_path)
    has_video_stream = media_has_video_stream(video_path)
    output_dir = Path(config.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    client = create_client(config)
    model = get_model(config)
    prompt_loader = PromptLoader(config.get("prompt_dir"), config.get("prompts", []))
    local_runtime_stack = contextlib.ExitStack()
    local_runtime_active = False

    def enter_local_runtime() -> None:
        nonlocal local_runtime_active
        if not local_runtime_active:
            local_runtime_stack.enter_context(local_model_runtime_session(config.config, logger, str(output_dir)))
            local_runtime_active = True

    def release_local_runtime() -> None:
        nonlocal local_runtime_active
        if local_runtime_active:
            local_runtime_stack.close()
            local_runtime_active = False
    
    try:
        transcript = None
        asr_result = None
        frames = []
        frame_analyses = []
        video_description = None
        operation_manual = None
        ocr_events = []
        task = config.get("task", "describe")
        page_context = ""
        page_context_metadata = {"context_file": "", "text_length": 0}
        transcript_markdown_path = None
        speaker_diarization_report = None
        timings = {}
        frame_selection_metadata = {}
        frame_extraction_metadata = {}
        frame_dedup_audit_metadata = {}
        frame_dedup_audit_path = None
        visual_review_metadata = {}
        visual_review_path = None
        run_manifest_metadata = {}
        run_manifest_path = None
        ocr_keyframe_metadata = {}
        ocr_text_events = []
        ocr_metadata = {}
        selected_frame_numbers = set()
        frame_decisions = []
        total_started = time.perf_counter()
        current_progress_step = "audio"
        write_analysis_progress(output_dir, current_progress_step, message="analysis started")

        def report_transcription_progress(
            node_id: str,
            node_status: str,
            node_message: str | None,
        ) -> None:
            write_analysis_progress(
                output_dir,
                current_progress_step,
                message=node_message,
                node_updates={
                    node_id: {
                        "status": node_status,
                        "message": node_message,
                    }
                },
            )
        
        # Stage 1: Frame and Audio Processing
        if args.start_stage <= 2:
            enter_local_runtime()

        if args.start_stage <= 1:
            stage_started = time.perf_counter()
            if args.transcript_file:
                current_progress_step = "asr"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="using existing transcript file",
                    node_updates={
                        "asr": {"status": "running", "message": "using existing transcript file"},
                        "diarization": {"status": "skipped", "message": "provided transcript"},
                    },
                )
                transcript_path = Path(args.transcript_file)
                logger.info("Using existing transcript file: %s", transcript_path)
                transcript = read_transcript_markdown(transcript_path)
                asr_result = ASRStrategyResult(
                    strategy="external_transcript_file",
                    transcript=transcript,
                    fast_transcript=transcript,
                    providers_run=["external_transcript_file"],
                )
                transcript_markdown_path = write_transcript_markdown(transcript, output_dir / "transcript.md")
                current_progress_step = "asr_done"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="transcript ready",
                    artifacts={"transcript": str(transcript_markdown_path)} if transcript_markdown_path else {},
                    node_updates={
                        "asr": {"status": "succeeded", "message": "existing transcript loaded"},
                        "transcript_merge": {"status": "succeeded", "message": "transcript ready"},
                    },
                )
            else:
                current_progress_step = "audio"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="extracting audio",
                    node_updates={
                        "audio_extract": {"status": "running", "message": "extracting audio"},
                    },
                )
                logger.info("Extracting audio from video...")
                try:
                    audio_path = extract_audio_to_wav(video_path, output_dir)
                except Exception as e:
                    logger.error(f"Error extracting audio: {e}")
                    audio_path = None

                if audio_path is None:
                    logger.debug("No audio found in video - skipping transcription")
                    transcript = None
                    write_analysis_progress(
                        output_dir,
                        current_progress_step,
                        message="no audio stream found",
                        node_updates={
                            "audio_extract": {"status": "skipped", "message": "no audio stream"},
                            "asr": {"status": "skipped", "message": "no audio stream"},
                            "diarization": {"status": "skipped", "message": "no audio stream"},
                            "transcript_merge": {"status": "skipped", "message": "no transcript"},
                        },
                    )
                else:
                    write_analysis_progress(
                        output_dir,
                        current_progress_step,
                        message="audio track ready",
                        artifacts={"audio": str(audio_path)},
                        node_updates={
                            "audio_extract": {"status": "succeeded", "message": "audio track ready"},
                        },
                    )
                    current_progress_step = "asr"
                    parallel_diarization = speaker_diarization_can_run_parallel(config)
                    progress_message = (
                        "transcribing audio and diarizing speakers in parallel"
                        if parallel_diarization
                        else "transcribing audio"
                    )
                    write_analysis_progress(
                        output_dir,
                        current_progress_step,
                        message=progress_message,
                        node_updates={
                            "asr": {"status": "running", "message": "transcribing audio"},
                            "diarization": {
                                "status": "running" if parallel_diarization else "pending",
                                "message": (
                                    "running in parallel with ASR"
                                    if parallel_diarization
                                    else "waiting for ASR"
                                ),
                            },
                        },
                    )
                    logger.info("%s...", progress_message.capitalize())
                    asr_config = config.get("asr", {})
                    provider = asr_config.get("provider", "faster_whisper")
                    use_asr_strategy = task == "operation_manual" and args.asr_provider is None and provider == "auto"
                    transcript, asr_result, speaker_diarization_report = (
                        transcribe_and_diarize_configured_audio(
                            audio_path,
                            output_dir,
                            config,
                            use_asr_strategy=use_asr_strategy,
                            logger=logger,
                            progress_callback=report_transcription_progress,
                            runtime_lock_held=local_runtime_active,
                        )
                    )
                    if transcript is None:
                        require_transcript = bool(asr_config.get("require_transcript", task == "operation_manual"))
                        if require_transcript and provider != "none":
                            failures = "; ".join(asr_result.failures or asr_result.merge_notes) if asr_result else ""
                            raise RuntimeError(
                                "Required ASR transcript was not produced. Check the configured Spark ASR/VibeVoice "
                                f"endpoint health instead of falling back to another device. {failures}".strip()
                            )
                        logger.warning("Could not generate reliable transcript. Proceeding with video analysis only.")
                    else:
                        transcript_markdown_path = write_transcript_markdown(transcript, output_dir / "transcript.md")
                        current_progress_step = "asr_done"
                        write_analysis_progress(
                            output_dir,
                            current_progress_step,
                            message="transcript ready",
                            artifacts={"transcript": str(transcript_markdown_path)} if transcript_markdown_path else {},
                            node_updates={
                                "asr": {"status": "succeeded", "message": "transcript ready"},
                                "diarization": {
                                    "status": (
                                        "failed"
                                        if (speaker_diarization_report or {}).get("error")
                                        else "succeeded"
                                        if speaker_diarization_report
                                        else "skipped"
                                    ),
                                    "message": (
                                        (speaker_diarization_report or {}).get("error")
                                        or "speaker turns aligned"
                                        if speaker_diarization_report
                                        else "speaker diarization not enabled"
                                    ),
                                },
                                "transcript_merge": {
                                    "status": "succeeded",
                                    "message": "transcript and speaker turns ready",
                                },
                            },
                        )
            timings["asr_seconds"] = round(time.perf_counter() - stage_started, 3)
            
            current_progress_step = "frames"
            write_analysis_progress(
                output_dir,
                current_progress_step,
                message="extracting candidate frames",
                node_updates={
                    "frame_extract": {"status": "running", "message": "extracting candidate frames"},
                },
            )
            logger.info(f"Extracting frames from video using model {model}...")
            stage_started = time.perf_counter()
            processor = VideoProcessor(
                video_path, 
                output_dir / "frames", 
                model
            )
            video_duration = processor._probe_duration(config.get("duration"))
            reused_frames = []
            reuse_metadata = {}
            if args.resume_existing and task == "operation_manual":
                reused_frames, reuse_metadata = reusable_frames_from_manifest(output_dir, args.frame_extractor)
                if reused_frames:
                    logger.info(
                        "Resume mode reusing %s candidate frames from %s",
                        len(reused_frames),
                        reuse_metadata.get("path"),
                    )
                elif reuse_metadata.get("reuse_rejected_reason"):
                    logger.info("Resume mode will re-extract frames: %s", reuse_metadata["reuse_rejected_reason"])
            if reused_frames:
                frames = reused_frames
                frame_extraction_metadata = reuse_metadata
                frame_manifest_path = output_dir / MANIFEST_NAME
                ocr_scan_sample_fps = resolve_ocr_scan_sample_fps(
                    args.ocr_scan_sample_fps,
                    args.pipeline_mode,
                    video_duration,
                )
                candidate_budget = resolve_candidate_frame_budget(
                    video_duration_seconds=video_duration,
                    pipeline_mode=args.pipeline_mode,
                    candidate_frames=args.candidate_frames,
                    explicit_max_frames=args.max_frames,
                )
                frame_extraction_metadata["frame_manifest"] = str(frame_manifest_path)
                frame_extraction_metadata["ocr_scan_sample_fps"] = ocr_scan_sample_fps
                current_progress_step = "frames_done"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="candidate frames reused",
                    artifacts={"frame_manifest": str(frame_manifest_path)},
                    node_updates={
                        "frame_extract": {"status": "succeeded", "message": "candidate frames reused"},
                    },
                )
                frame_selection_metadata = {
                    "pipeline_mode": args.pipeline_mode,
                    "video_duration_seconds": video_duration,
                    "candidate_frames": args.candidate_frames,
                    "candidate_frame_budget": candidate_budget,
                    "explicit_max_frames": args.max_frames,
                }
            elif not has_video_stream:
                logger.info("Input has no video stream; skipping frame extraction.")
                frame_manifest_path = write_frame_manifest(frames, output_dir, source="audio_only")
                frame_extraction_metadata = {
                    "backend": "audio_only",
                    "has_video_stream": False,
                    "frame_manifest": str(frame_manifest_path),
                    "video_duration_seconds": video_duration,
                }
                if task == "operation_manual":
                    frame_selection_metadata = {
                        "pipeline_mode": args.pipeline_mode,
                        "video_duration_seconds": video_duration,
                        "candidate_frames": args.candidate_frames,
                        "candidate_frame_budget": 0,
                        "explicit_max_frames": args.max_frames,
                    }
                current_progress_step = "frames_done"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="audio-only input; skipped frame extraction",
                    artifacts={"frame_manifest": str(frame_manifest_path)},
                    node_updates={
                        "frame_extract": {"status": "skipped", "message": "audio-only input"},
                        "ocr": {"status": "skipped", "message": "audio-only input"},
                        "vision": {"status": "skipped", "message": "audio-only input"},
                        "visual_evidence": {"status": "skipped", "message": "audio-only input"},
                    },
                )
            elif task == "operation_manual":
                ocr_scan_sample_fps = resolve_ocr_scan_sample_fps(
                    args.ocr_scan_sample_fps,
                    args.pipeline_mode,
                    video_duration,
                )
                candidate_budget = resolve_candidate_frame_budget(
                    video_duration_seconds=video_duration,
                    pipeline_mode=args.pipeline_mode,
                    candidate_frames=args.candidate_frames,
                    explicit_max_frames=args.max_frames,
                )
                if args.frame_extractor == "local_gpu":
                    extraction = extract_frames_with_local_gpu_workers(
                        video_path=video_path,
                        output_dir=output_dir / "frames",
                        video_duration_seconds=video_duration,
                        pipeline_mode=args.pipeline_mode,
                        candidate_budget=candidate_budget,
                        candidate_strategy=args.candidate_frame_strategy,
                        transcript=transcript,
                        sample_fps=args.jetson_sample_fps,
                        overlap_seconds=args.jetson_chunk_overlap_seconds,
                        gpu_indices=args.local_frame_gpus,
                    )
                    frames = extraction.frames
                    frame_extraction_metadata = extraction.metadata
                elif args.frame_extractor in {"jetson", "auto"}:
                    hosts = [host.strip() for host in args.jetson_frame_hosts.split(",") if host.strip()]
                    jetson_sample_fps = args.jetson_sample_fps
                    if str(jetson_sample_fps).strip().lower() == OCR_AUTO and str(args.ocr_scan_sample_fps).strip().lower() != OCR_AUTO:
                        jetson_sample_fps = ocr_scan_sample_fps
                    try:
                        extraction = extract_frames_with_jetson_workers(
                            video_path=video_path,
                            output_dir=output_dir / "frames",
                            hosts=hosts,
                            video_duration_seconds=video_duration,
                            pipeline_mode=args.pipeline_mode,
                            candidate_budget=candidate_budget,
                            candidate_strategy=args.candidate_frame_strategy,
                            transcript=transcript,
                            sample_fps=jetson_sample_fps,
                            backend=args.jetson_frame_backend,
                            overlap_seconds=args.jetson_chunk_overlap_seconds,
                            host_weights=parse_jetson_frame_weights(args.jetson_frame_weights),
                            require_hardware_decode=args.jetson_require_hwdec,
                            strict=args.frame_extractor == "jetson",
                        )
                    except Exception:
                        if args.frame_extractor == "jetson":
                            raise
                        logger.exception("Jetson frame extraction failed; falling back to local extraction")
                        extraction = extract_local_screen_keyframes(
                            processor=processor,
                            frames_per_minute=max(1, int(round(ocr_scan_sample_fps * 60))),
                            duration=config.get("duration"),
                            max_frames=candidate_budget,
                            transcript=transcript,
                        )
                    frames = extraction.frames
                    frame_extraction_metadata = extraction.metadata
                else:
                    extraction = extract_local_screen_keyframes(
                        processor=processor,
                        frames_per_minute=max(1, int(round(ocr_scan_sample_fps * 60))),
                        duration=config.get("duration"),
                        max_frames=candidate_budget,
                        transcript=transcript,
                    )
                    frames = extraction.frames
                    frame_extraction_metadata = extraction.metadata
                frame_manifest_path = write_frame_manifest(
                    frames,
                    output_dir,
                    source=str(frame_extraction_metadata.get("backend", "unknown")),
                )
                frame_extraction_metadata["frame_manifest"] = str(frame_manifest_path)
                frame_extraction_metadata["ocr_scan_sample_fps"] = ocr_scan_sample_fps
                current_progress_step = "frames_done"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="candidate frames ready",
                    artifacts={"frame_manifest": str(frame_manifest_path)},
                    node_updates={
                        "frame_extract": {"status": "succeeded", "message": "candidate frames ready"},
                    },
                )
                frame_selection_metadata = {
                    "pipeline_mode": args.pipeline_mode,
                    "video_duration_seconds": video_duration,
                    "candidate_frames": args.candidate_frames,
                    "candidate_frame_budget": candidate_budget,
                    "explicit_max_frames": args.max_frames,
                }
            else:
                frames = processor.extract_keyframes(
                    frames_per_minute=config.get("frames", {}).get("per_minute", 60),
                    duration=config.get("duration"),
                    max_frames=args.max_frames
                )
                frame_manifest_path = write_frame_manifest(frames, output_dir, source="local_keyframes")
                frame_extraction_metadata["frame_manifest"] = str(frame_manifest_path)
                current_progress_step = "frames_done"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="candidate frames ready",
                    artifacts={"frame_manifest": str(frame_manifest_path)},
                    node_updates={
                        "frame_extract": {"status": "succeeded", "message": "candidate frames ready"},
                    },
                )
            timings["candidate_frame_extraction_seconds"] = round(time.perf_counter() - stage_started, 3)
            audit_started = time.perf_counter()
            frame_dedup_audit_path, frame_dedup_audit = write_frame_dedup_audit(frames, output_dir)
            audited_ocr_frames = select_audited_frames(frames, frame_dedup_audit)
            audit_summary = frame_dedup_audit.get("summary") or {}
            timings["ocr_frame_audit_seconds"] = round(time.perf_counter() - audit_started, 3)
            frame_dedup_audit_metadata = {
                key: value
                for key, value in frame_dedup_audit.items()
                if key != "records"
            }

            if task == "operation_manual":
                audit_message = (
                    f"OCR frame audit retained {len(audited_ocr_frames)} of {len(frames)} "
                    f"candidate frames; max gap "
                    f"{float(audit_summary.get('max_kept_gap_seconds') or 0):.1f}s"
                )
                logger.info(audit_message)
                write_analysis_progress(
                    output_dir,
                    "ocr_audit",
                    message=audit_message,
                    artifacts={"frame_dedup_audit": str(frame_dedup_audit_path)},
                    node_updates={
                        "frame_audit": {
                            "status": "succeeded",
                            "message": audit_message,
                            "duration_seconds": timings["ocr_frame_audit_seconds"],
                        },
                    },
                )
                current_progress_step = "ocr"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="running OCR",
                    node_updates={
                        "ocr": {"status": "running", "message": "running OCR"},
                    },
                )
                logger.info("Running OCR on extracted frames...")
                stage_started = time.perf_counter()
                ocr_config = config.get("ocr", {})
                ocr_base_urls = ocr_config.get("base_urls")
                selected_ocr_frames, ocr_keyframe_decisions, ocr_keyframe_metadata = select_ocr_keyframes(
                    frames=audited_ocr_frames,
                    transcript=transcript,
                    video_duration_seconds=frame_selection_metadata.get("video_duration_seconds", config.get("duration") or 0.0),
                    pipeline_mode=args.pipeline_mode,
                    strategy=args.ocr_keyframe_strategy,
                    budget=args.ocr_keyframe_budget,
                    scan_frames_count=scan_frames_count_from_metadata(frame_extraction_metadata),
                )
                ocr_keyframe_metadata["source_candidate_frames_count"] = len(frames)
                ocr_keyframe_metadata["audited_candidate_frames_count"] = len(audited_ocr_frames)
                ocr_keyframe_metadata["audit_max_gap_seconds"] = audit_summary.get("max_kept_gap_seconds")
                logger.info(
                    "Selected %s OCR keyframes from %s audited frames (%s original) "
                    "after scanning %s preview frames",
                    len(selected_ocr_frames),
                    len(audited_ocr_frames),
                    len(frames),
                    ocr_keyframe_metadata.get("scan_frames_count"),
                )
                ocr_checkpoint_path = output_dir / "orin" / "ocr_events.partial.json"
                ocr_signature_data = ocr_signature_payload(ocr_config)
                ocr_signature = analysis_signature(ocr_signature_data)
                selected_ocr_by_number = {int(frame.number): frame for frame in selected_ocr_frames}
                ocr_events_by_frame = load_ocr_checkpoint(
                    ocr_checkpoint_path,
                    selected_ocr_frames,
                    ocr_signature,
                )
                missing_ocr_frames = [
                    frame
                    for frame in selected_ocr_frames
                    if int(frame.number) not in ocr_events_by_frame
                ]
                if ocr_events_by_frame:
                    logger.info(
                        "Resume mode reusing %s OCR results; %s frame(s) remain",
                        len(ocr_events_by_frame),
                        len(missing_ocr_frames),
                    )

                def save_ocr_event(event: OCREvent) -> None:
                    ocr_events_by_frame[int(event.frame_number)] = event
                    write_ocr_checkpoint(
                        ocr_checkpoint_path,
                        ocr_events_by_frame,
                        selected_ocr_by_number,
                        ocr_signature,
                        ocr_signature_data,
                    )

                if missing_ocr_frames:
                    with analyzer_resource_lock(config.config, "ocr", str(output_dir), logger):
                        with local_model_stage("ocr", config.config, logger, str(output_dir)):
                            run_ocr(
                                frames=missing_ocr_frames,
                                provider=ocr_config.get("provider", "auto"),
                                base_url=ocr_config.get("base_url", "auto"),
                                model=ocr_config.get("model", "model"),
                                prompt_mode=ocr_config.get("prompt_mode", "prompt_scene_spotting"),
                                base_urls=ocr_base_urls,
                                ocr_concurrency=ocr_config.get("concurrency", "auto"),
                                fallback_base_url=ocr_config.get(
                                    "fallback_base_url",
                                    config.get("operation_manual", {}).get("llm_base_url"),
                                ),
                                fallback_model=ocr_config.get(
                                    "fallback_model",
                                    config.get("operation_manual", {}).get("vision_model"),
                                ),
                                fallback_api_key=ocr_config.get(
                                    "fallback_api_key",
                                    config.get("clients", {}).get("openai_api", {}).get("api_key", "0"),
                                ),
                                request_timeout_seconds=ocr_config.get("timeout_seconds", 120),
                                max_tokens=ocr_config.get("max_tokens", 1024),
                                max_image_long_side=ocr_config.get("max_image_long_side", 1280),
                                retry_endpoints=bool(ocr_config.get("retry_endpoints", True)),
                                probe_timeout_seconds=ocr_config.get("probe_timeout_seconds", 5),
                                warmup_timeout_seconds=ocr_config.get("warmup_timeout_seconds", 180),
                                warmup_retry_interval_seconds=ocr_config.get("warmup_retry_interval_seconds", 5),
                                cache_mode=ocr_config.get("cache", "on"),
                                cache_dir=ocr_config.get("cache_dir", ".cache/video-analyzer/ocr"),
                                image_mode=ocr_config.get("image_mode", "gundam"),
                                progress_callback=save_ocr_event,
                            )
                else:
                    logger.info("No missing OCR frames; skipping OCR provider calls.")
                ocr_events = [
                    ocr_events_by_frame[int(frame.number)]
                    for frame in selected_ocr_frames
                    if int(frame.number) in ocr_events_by_frame
                ]
                ocr_text_events = build_ocr_text_events(ocr_events)
                ocr_keyframe_metadata["ocr_text_events_count"] = len(ocr_text_events)
                ocr_keyframe_metadata["text_events"] = ocr_text_events
                ocr_requested_endpoints = ocr_base_urls or [ocr_config.get("base_url", "auto")]
                ocr_provider_endpoints = sorted(
                    {
                        event.provider.split(":", 1)[1]
                        for event in ocr_events
                        if event.provider.startswith("dots_mocr_vllm:")
                    }
                )
                ocr_metadata = {
                    "requested_endpoints": ocr_requested_endpoints,
                    "effective_endpoints": ocr_provider_endpoints,
                    "effective_worker_count": len(ocr_provider_endpoints),
                    "concurrency": ocr_config.get("concurrency", "auto"),
                    "prompt_mode": ocr_config.get("prompt_mode", "prompt_scene_spotting"),
                    "max_tokens": ocr_config.get("max_tokens", 1024),
                    "max_image_long_side": ocr_config.get("max_image_long_side", 1280),
                    "retry_endpoints": bool(ocr_config.get("retry_endpoints", True)),
                    "cache_mode": ocr_config.get("cache", "on"),
                    "cache_dir": ocr_config.get("cache_dir", ".cache/video-analyzer/ocr"),
                    "cache_hits": sum(1 for event in ocr_events if event.cache_status == "hit"),
                    "cache_misses": sum(1 for event in ocr_events if event.cache_status == "miss"),
                    "cache_refreshes": sum(1 for event in ocr_events if event.cache_status == "refresh"),
                    "cache_disabled": sum(1 for event in ocr_events if event.cache_status == "disabled"),
                }
                timings["ocr_seconds"] = round(time.perf_counter() - stage_started, 3)
                current_progress_step = "ocr_ready"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="OCR results ready",
                    node_updates={
                        "ocr": {
                            "status": "succeeded",
                            "message": f"{len(ocr_events)} OCR frame results ready",
                        },
                    },
                )
            
        # Stage 2: Frame Analysis
        if args.start_stage <= 2:
            current_progress_step = "vl"
            write_analysis_progress(
                output_dir,
                current_progress_step,
                message="selecting and analyzing VL frames",
                node_updates={
                    "vision": {"status": "running", "message": "selecting and analyzing VL frames"},
                    "visual_evidence": {"status": "pending", "message": "waiting for VL results"},
                },
            )
            logger.info("Selecting and analyzing VL frames...")
            stage_started = time.perf_counter()
            analyzer = VideoAnalyzer(
                client, 
                model, 
                prompt_loader,
                config.get("clients", {}).get("temperature", 0.2),
                config.get("prompt", ""),
                frame_num_predict=config.get("response_length", {}).get("frame", 300),
                frame_no_think=bool(config.get("operation_manual", {}).get("frame_no_think", False)),
            )
            if task == "operation_manual":
                context_before = max(args.vl_context_before, 0)
                context_after = max(args.vl_context_after, 0)
                signature_payload = vl_signature_payload(
                    analyzer,
                    model=model,
                    context_before=context_before,
                    context_after=context_after,
                    context_max_gap=args.vl_context_max_gap,
                )
                checkpoint_signature = analysis_signature(signature_payload)
                checkpoint_path = output_dir / "orin" / "frame_analyses.partial.json"
                checkpoint_by_frame, checkpoint_metadata = load_vl_checkpoint(
                    checkpoint_path,
                    frames,
                    checkpoint_signature,
                    allow_legacy_ordered=max(args.vl_concurrency, 1) == 1,
                )
                checkpoint_durations = [
                    float(item.get("duration_seconds"))
                    for item in checkpoint_by_frame.values()
                    if isinstance(item.get("duration_seconds"), (int, float))
                    and float(item.get("duration_seconds")) > 0
                ]
                seconds_per_frame = (
                    float(median(checkpoint_durations))
                    if checkpoint_durations
                    else recent_vl_seconds_per_frame(output_dir, model)
                )
                options = FrameSelectionOptions(
                    pipeline_mode=args.pipeline_mode,
                    candidate_frames=args.candidate_frames,
                    min_vl_frames=args.min_vl_frames,
                    max_vl_frames=args.max_vl_frames,
                    vl_frame_policy=args.vl_frame_policy,
                    explicit_max_frames=args.max_frames,
                    vl_target_seconds=DEFAULT_VL_TARGET_SECONDS,
                    vl_seconds_per_frame=seconds_per_frame,
                )
                selected_frame_numbers, frame_decisions, selection_metadata = select_vl_frames(
                    frames=frames,
                    ocr_events=ocr_events,
                    transcript=transcript,
                    video_duration_seconds=frame_selection_metadata.get("video_duration_seconds", config.get("duration") or 0.0),
                    options=options,
                )
                budget_selected_frame_numbers = set(selected_frame_numbers)
                reusable_frame_numbers = set(checkpoint_by_frame)
                reused_outside_budget = reusable_frame_numbers - budget_selected_frame_numbers
                selected_frame_numbers |= reusable_frame_numbers
                if reused_outside_budget:
                    frame_decisions = [
                        replace(
                            decision,
                            selected_for_vl=True,
                            reason="reused_checkpoint",
                            skip_reason="",
                        )
                        if decision.frame_number in reused_outside_budget
                        else decision
                        for decision in frame_decisions
                    ]
                new_requests = budget_selected_frame_numbers - reusable_frame_numbers
                selection_metadata["vl_budget_selected_count"] = len(budget_selected_frame_numbers)
                selection_metadata["vl_checkpoint_reused_outside_budget"] = len(reused_outside_budget)
                selection_metadata["vl_frames_count"] = len(selected_frame_numbers)
                selection_metadata["vl_new_requests_count"] = len(new_requests)
                selection_metadata["vl_projected_remaining_seconds"] = round(
                    len(new_requests) * seconds_per_frame,
                    3,
                )
                selection_metadata["checkpoint"] = checkpoint_metadata
                if selection_metadata.get("vl_time_target_bypassed"):
                    logger.warning(
                        "Explicit VL policy=all bypasses the %.0f second target; projected VL time is %.1f seconds",
                        selection_metadata.get("vl_time_target_seconds") or DEFAULT_VL_TARGET_SECONDS,
                        selection_metadata.get("vl_projected_seconds") or 0.0,
                    )
                if checkpoint_metadata.get("legacy_migrated"):
                    write_vl_checkpoint(
                        checkpoint_path,
                        checkpoint_by_frame.values(),
                        signature=checkpoint_signature,
                        signature_payload=signature_payload,
                    )
                frame_selection_metadata.update(selection_metadata)
                timings["frame_selection_seconds"] = round(time.perf_counter() - stage_started, 3)
                vl_started = time.perf_counter()
                static_vl_progress = {
                    "policy": selection_metadata.get("vl_frame_policy_resolved"),
                    "quality_budget": selection_metadata.get("vl_quality_budget"),
                    "time_capacity": selection_metadata.get("vl_time_capacity"),
                    "target_seconds": selection_metadata.get("vl_time_target_seconds"),
                    "projected_seconds": selection_metadata.get("vl_projected_remaining_seconds"),
                    "seconds_per_frame_estimate": selection_metadata.get("vl_seconds_per_frame_estimate"),
                    "time_target_bypassed": selection_metadata.get("vl_time_target_bypassed"),
                    "budget_selected": selection_metadata.get("vl_budget_selected_count"),
                    "reused_outside_budget": selection_metadata.get("vl_checkpoint_reused_outside_budget"),
                }

                def update_vl_progress(snapshot: dict[str, Any]) -> None:
                    vl_progress = {**static_vl_progress, **snapshot}
                    total = max(int(vl_progress.get("total_selected") or 0), 0)
                    completed_count = max(int(vl_progress.get("completed") or 0), 0)
                    vl_progress["percent"] = int(round((completed_count / total) * 100)) if total else 100
                    write_analysis_progress(
                        output_dir,
                        "vl",
                        message=f"VL frames {completed_count}/{total}",
                        details={"vl": vl_progress},
                        node_updates={
                            "vision": {
                                "status": "running",
                                "message": f"VL frames {completed_count}/{total}",
                                "progress": vl_progress.get("percent"),
                            }
                        },
                    )
                if frames:
                    with analyzer_resource_lock(config.config, "vl", str(output_dir), logger):
                        with local_model_stage("vl", config.config, logger, str(output_dir)):
                            frame_analyses = analyze_frames_for_vl(
                                analyzer=analyzer,
                                frames=frames,
                                ocr_events=ocr_events,
                                selected_frame_numbers=selected_frame_numbers,
                                decisions=frame_decisions,
                                concurrency=max(args.vl_concurrency, 1),
                                context_before=context_before,
                                context_after=context_after,
                                context_max_gap=args.vl_context_max_gap,
                                checkpoint_path=checkpoint_path,
                                checkpoint_by_frame=checkpoint_by_frame,
                                checkpoint_signature=checkpoint_signature,
                                checkpoint_signature_payload=signature_payload,
                                progress_callback=update_vl_progress,
                            )
                else:
                    logger.info("No video frames available; skipping VL provider calls.")
                    frame_analyses = []
                timings["vl_seconds"] = round(time.perf_counter() - vl_started, 3)
            else:
                frame_analyses = []
                ocr_by_frame = {event.frame_number: event for event in ocr_events}
                for frame in frames:
                    ocr_text = ocr_by_frame.get(frame.number).text if frame.number in ocr_by_frame else ""
                    analysis = analyzer.analyze_frame(frame, ocr_text=ocr_text)
                    frame_analyses.append(analysis)
                timings["vl_seconds"] = round(time.perf_counter() - stage_started, 3)
            write_analysis_progress(
                output_dir,
                current_progress_step,
                message="visual evidence ready",
                node_updates={
                    "vision": {
                        "status": "succeeded",
                        "message": f"{len(frame_analyses)} frame analyses ready",
                        "progress": 100,
                    },
                    "visual_evidence": {
                        "status": "succeeded",
                        "message": "OCR and VL evidence ready",
                    },
                },
            )

        release_local_runtime()
                
        # Stage 3: Video Reconstruction
        if args.start_stage <= 3:
            if task == "operation_manual":
                current_progress_step = "manual"
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message="generating operation manual",
                    node_updates={
                        "evidence_merge": {
                            "status": "succeeded",
                            "message": "audio, visual and page evidence merged",
                        },
                        "text": {"status": "running", "message": "generating operation manual"},
                    },
                )
                logger.info("Generating operation manual...")
                stage_started = time.perf_counter()
                manual_config = config.get("operation_manual", {})
                runtime_profile = config.get_runtime_profile(
                    getattr(args, "profile", None)
                )
                text_client = create_operation_manual_text_client(config, client)
                (
                    fallback_text_client,
                    fallback_text_model,
                    fallback_text_temperature,
                ) = create_operation_manual_fallback_client(config)
                fallback_configured = bool(
                    fallback_text_client and fallback_text_model
                )
                page_context = read_context_file(config.get("context_file", ""))
                page_context_metadata = read_page_context_metadata(config.get("context_file", ""), page_context)
                text_model = manual_config.get("text_model") or model
                frame_assets = prepare_frame_assets(frames, output_dir)

                def report_text_fallback(status: str, message: str) -> None:
                    write_analysis_progress(
                        output_dir,
                        current_progress_step,
                        message=message,
                        node_updates={
                            "text": {
                                "status": "failed",
                                "message": message,
                            },
                            "text_fallback": {
                                "status": status,
                                "message": message,
                            },
                        },
                    )

                with local_model_stage("text", config.config, logger, str(output_dir)):
                    operation_manual = generate_operation_manual(
                        client=text_client,
                        text_model=text_model,
                        frame_analyses=frame_analyses,
                        frames=frames,
                        transcript=transcript,
                        asr_metadata=asr_result.to_metadata() if asr_result else {},
                        ocr_events=ocr_events,
                        page_context=page_context,
                        language=config.get("manual_language", "zh-CN"),
                        temperature=resolve_temperature(manual_config, config.get("clients", {}).get("temperature", 0.2)),
                        frame_assets=frame_assets,
                        no_think=bool(manual_config.get("manual_no_think", manual_config.get("frame_no_think", False))),
                        max_prompt_chars=resolve_manual_prompt_char_budget(
                            manual_config.get("context_length")
                            or runtime_profile.get("context_length"),
                            manual_config.get("max_prompt_chars"),
                        ),
                        fallback_client=fallback_text_client,
                        fallback_model=fallback_text_model,
                        fallback_temperature=fallback_text_temperature,
                        fallback_status_callback=(
                            report_text_fallback
                            if fallback_configured
                            else None
                        ),
                    )
                operation_manual["response"] = embed_step_images(
                    operation_manual.get("response", ""),
                    frames,
                    frame_assets,
                    frame_analyses=frame_analyses,
                    ocr_events=ocr_events,
                )
                operation_manual["response"] = append_evidence_boundary_section(
                    operation_manual.get("response", ""),
                    frame_selection_metadata,
                    ocr_keyframe_metadata,
                    ocr_events,
                )
                operation_manual["quality_review"] = review_operation_manual_markdown(
                    operation_manual.get("response", "")
                )
                operation_manual["quality_gate_passed"] = not any(
                    issue.get("severity") == "error" for issue in operation_manual["quality_review"]
                )
                manual_filename = "operation_manual.md" if operation_manual["quality_gate_passed"] else "operation_manual.quality_failed.md"
                operation_manual["manual_path"] = str(output_dir / manual_filename)
                for issue in operation_manual["quality_review"]:
                    level = logging.ERROR if issue.get("severity") == "error" else logging.WARNING
                    logger.log(level, "Operation manual quality issue [%s]: %s", issue.get("code"), issue.get("message"))
                evidence_path = write_frame_evidence_index(
                    frames=frames,
                    output_dir=output_dir,
                    ocr_events=ocr_events,
                    frame_analyses=frame_analyses,
                    frame_assets=frame_assets,
                )
                operation_manual["evidence_path"] = str(evidence_path)
                timings["manual_generation_seconds"] = round(time.perf_counter() - stage_started, 3)
                fallback_used = bool(operation_manual.get("fallback_used"))
                quality_passed = bool(operation_manual["quality_gate_passed"])
                if fallback_used:
                    text_node = {
                        "status": "failed",
                        "message": operation_manual.get("primary_error")
                        or "primary text model failed",
                    }
                    fallback_node = {
                        "status": "succeeded" if quality_passed else "failed",
                        "message": (
                            f"{fallback_text_model} completed"
                            if quality_passed
                            else operation_manual.get("fallback_error")
                            or "fallback output failed quality gate"
                        ),
                    }
                else:
                    text_node = {
                        "status": "succeeded" if quality_passed else "failed",
                        "message": (
                            "operation manual draft ready"
                            if quality_passed
                            else "operation manual generation failed quality gate"
                        ),
                    }
                    fallback_node = {
                        "status": "skipped",
                        "message": (
                            "primary model succeeded"
                            if fallback_configured
                            else "text fallback disabled"
                        ),
                    }
                write_analysis_progress(
                    output_dir,
                    current_progress_step,
                    message=(
                        "operation manual draft ready"
                        if quality_passed
                        else "operation manual draft failed quality gate"
                    ),
                    node_updates={
                        "text": text_node,
                        "text_fallback": fallback_node,
                    },
                )
            else:
                logger.info("Reconstructing video description...")
                video_description = analyzer.reconstruct_video(
                    frame_analyses, frames, transcript
                )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        current_progress_step = "write"
        write_analysis_progress(
            output_dir,
            current_progress_step,
            message="writing analysis outputs",
            node_updates=(
                None
                if operation_manual
                else {
                    "text": {
                        "status": "succeeded",
                        "message": "writing analysis outputs",
                    }
                }
            ),
        )
        timings["total_seconds"] = round(time.perf_counter() - total_started, 3)
        results = {
            "metadata": {
                "task": task,
                "client": config.get("clients", {}).get("default"),
                "model": model,
                "vision_base_url": config.get("operation_manual", {}).get("vision_base_url")
                or config.get("operation_manual", {}).get("llm_base_url"),
                "text_model": config.get("operation_manual", {}).get("text_model"),
                "text_base_url": config.get("operation_manual", {}).get("text_base_url")
                or config.get("operation_manual", {}).get("llm_base_url"),
                "text_temperature": resolve_temperature(
                    config.get("operation_manual", {}),
                    config.get("clients", {}).get("temperature", 0.2),
                ),
                "ocr_provider": config.get("ocr", {}).get("provider"),
                "ocr": ocr_metadata,
                "ocr_keyframes": ocr_keyframe_metadata,
                "asr_provider": config.get("asr", {}).get("provider"),
                "asr_strategy": config.get("asr", {}).get("strategy"),
                "context_file": config.get("context_file"),
                "page_description": page_context,
                "page_context": page_context_metadata,
                "whisper_model": config.get("audio", {}).get("whisper_model"),
                "frames_per_minute": config.get("frames", {}).get("per_minute"),
                "duration_processed": config.get("duration"),
                "frames_extracted": len(frames),
                "frames_processed": len(frame_analyses),
                "vl_frames_processed": len(selected_frame_numbers) if task == "operation_manual" else len(frame_analyses),
                "frame_selection": frame_selection_metadata,
                "frame_extraction": frame_extraction_metadata,
                "frame_dedup_audit": frame_dedup_audit_metadata,
                "visual_review": visual_review_metadata,
                "run_manifest": run_manifest_metadata,
                "vl_context": {
                    "before": max(args.vl_context_before, 0),
                    "after": max(args.vl_context_after, 0),
                    "max_gap_seconds": args.vl_context_max_gap,
                    "resolved_max_gap_seconds": resolve_vl_context_gap_seconds(frames, args.vl_context_max_gap) if frames else 0.0,
                },
                "timings": timings,
                "start_stage": args.start_stage,
                "audio_language": transcript.language if transcript else None,
                "transcription_successful": transcript is not None,
                "transcript_markdown": str(transcript_markdown_path) if transcript_markdown_path else None,
            },
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None,
                "metadata": transcript.metadata if transcript else None,
            } if transcript else None,
            "asr": asr_result.to_metadata() if asr_result else None,
            "ocr_events": [event.to_dict() for event in ocr_events],
            "ocr_text_events": ocr_text_events,
            "visual_events": frame_analyses,
            "manual_steps": operation_manual,
            "uncertainties": [
                event.to_dict() for event in ocr_events if event.status != "ok"
            ],
            "frame_analyses": frame_analyses,
            "video_description": video_description,
            "operation_manual": operation_manual
        }

        visual_review_path, visual_review_metadata = write_visual_review(
            output_dir=output_dir,
            video_path=video_path if has_video_stream else None,
            frames=frames,
            transcript=transcript,
            ocr_events=ocr_events,
            frame_analyses=frame_analyses,
            metadata=results["metadata"],
        )
        results["metadata"]["visual_review"] = visual_review_metadata
        run_manifest_path, run_manifest_metadata = write_run_manifest(
            output_dir=output_dir,
            results=results,
            visual_review_path=visual_review_path,
            dedup_audit_path=frame_dedup_audit_path,
        )
        results["metadata"]["run_manifest"] = run_manifest_metadata
        
        with open(output_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        orin_dir = write_orin_artifacts(output_dir, results, page_context)
        results["metadata"]["orin_dir"] = str(orin_dir)
        with open(output_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info("\nTranscript:")
        if transcript:
            logger.info(transcript.text)
        else:
            logger.info("No reliable transcript available")
            
        if video_description:
            logger.info("\nVideo Description:")
            logger.info(video_description.get("response", "No description generated"))

        if operation_manual:
            quality_passed = operation_manual.get("quality_gate_passed", True)
            manual_path = Path(operation_manual.get("manual_path", output_dir / ("operation_manual.md" if quality_passed else "operation_manual.quality_failed.md")))
            manual_path.write_text(operation_manual.get("response", ""), encoding="utf-8")
            if quality_passed:
                logger.info("Operation manual saved to %s", manual_path)
            else:
                logger.error("Operation manual failed quality gate; saved review artifact to %s", manual_path)
        
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        
        logger.info(f"Analysis complete. Results saved to {output_dir / 'analysis.json'}")
        write_analysis_progress(
            output_dir,
            "write",
            status="succeeded",
            message="analysis complete",
            artifacts={"analysis_json": str(output_dir / "analysis.json")},
        )
            
    except Exception as e:
        logger.error(f"Error during video analysis: {e}")
        failure_node = {
            "audio": "audio_extract",
            "asr": "asr",
            "asr_done": "transcript_merge",
            "frames": "frame_extract",
            "frames_done": "frame_extract",
            "ocr": "ocr",
            "ocr_ready": "ocr",
            "vl": "vision",
            "manual": "text",
            "write": "text",
        }.get(current_progress_step)
        write_analysis_progress(
            output_dir,
            current_progress_step,
            status="failed",
            message=str(e),
            node_updates={
                failure_node: {"status": "failed", "message": str(e)}
            }
            if failure_node
            else None,
        )
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        raise
    finally:
        release_local_runtime()

if __name__ == "__main__":
    main()
