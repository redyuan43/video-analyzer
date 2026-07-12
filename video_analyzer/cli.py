import argparse
import contextlib
from pathlib import Path
import json
import logging
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .artifacts import write_json, write_orin_artifacts, write_transcript_markdown
from .candidate_frame_strategies import parse_candidate_frame_strategy
from .config import Config, build_openai_extra_body, get_client, get_model, resolve_api_key, resolve_temperature
from .frame import VideoProcessor
from .frame_dedup_audit import write_frame_dedup_audit
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
from .frame_manifest import MANIFEST_NAME, read_frames_from_manifest, write_frame_manifest
from .jetson_frames import extract_frames_with_jetson_workers, extract_local_screen_keyframes
from .prompt import PromptLoader
from .analyzer import VideoAnalyzer
from .audio_processor import AudioProcessor, AudioTranscript
from .asr_providers import ASRStrategyResult, extract_audio_to_wav, transcribe_with_provider_result, transcribe_with_strategy
from .clients.ollama import OllamaClient
from .clients.generic_openai_api import GenericOpenAIAPIClient
from .manual import (
    embed_step_images,
    generate_operation_manual,
    prepare_frame_assets,
    read_context_file,
    review_operation_manual_markdown,
    write_frame_evidence_index,
)
from .ocr import run_ocr
from .ocr_keyframes import (
    AUTO as OCR_AUTO,
    build_ocr_text_events,
    resolve_ocr_scan_sample_fps,
    select_ocr_keyframes,
)
from .local_model_runtime import local_model_runtime_session, local_model_stage
from .resource_locks import analyzer_resource_lock
from .review_artifacts import write_run_manifest, write_visual_review
from .speaker_diarization import process_transcript_speakers

# Initialize logger at module level
logger = logging.getLogger(__name__)
TRANSCRIPT_LINE_RE = re.compile(
    r"^-\s+\[(?P<start>\d\d:\d\d:\d\d)\s+-\s+(?P<end>\d\d:\d\d:\d\d)\]\s+(?P<text>.*)$"
)
PROGRESS_FILENAME = "progress.json"


def media_has_video_stream(media_path: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=codec_type:stream_disposition=attached_pic",
                "-of",
                "json",
                str(media_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logger.warning("Could not probe video stream for %s; assuming video input: %s", media_path, exc)
        return True
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return bool(result.stdout.strip())
    for stream in payload.get("streams") or []:
        disposition = stream.get("disposition") or {}
        if stream.get("codec_type") == "video" and not disposition.get("attached_pic"):
            return True
    return False

def get_log_level(level_str: str) -> int:
    """Convert string log level to logging constant."""
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)


def seconds_from_timestamp(value: str) -> float:
    hours, minutes, seconds = [int(part) for part in value.split(":")]
    return float(hours * 3600 + minutes * 60 + seconds)


def read_transcript_markdown(path: Path) -> AudioTranscript:
    text = path.read_text(encoding="utf-8")
    language = ""
    segments = []
    full_text = []
    for line in text.splitlines():
        if line.startswith("- Language:"):
            language = line.split(":", 1)[1].strip()
            continue
        match = TRANSCRIPT_LINE_RE.match(line)
        if not match:
            continue
        segment_text = match.group("text").strip()
        if not segment_text:
            continue
        segments.append(
            {
                "start": seconds_from_timestamp(match.group("start")),
                "end": seconds_from_timestamp(match.group("end")),
                "text": segment_text,
            }
        )
        full_text.append(segment_text)
    if not segments:
        full_text = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and not line.startswith("#")
            and not line.startswith("- Language:")
            and not line.startswith("- Segments:")
        ]
    return AudioTranscript(text="\n".join(full_text).strip(), segments=segments, language=language)


def parse_jetson_frame_weights(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    weights: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid Jetson frame weight entry: {item}")
        host, weight = item.split("=", 1)
        weights[host.strip()] = max(float(weight), 0.1)
    return weights

def cleanup_files(output_dir: Path):
    """Clean up temporary files and directories."""
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.debug(f"Cleaned up frames directory: {frames_dir}")
            
        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.debug(f"Cleaned up audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def write_analysis_progress(
    output_dir: Path,
    current_step: str,
    status: str = "running",
    message: str | None = None,
    artifacts: dict[str, str] | None = None,
) -> None:
    """Best-effort durable progress for status UIs; never fail analysis work."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "stage": "analyze-core",
            "current_step": current_step,
            "status": status,
            "message": message,
            "artifacts": artifacts or {},
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        }
        progress_path = output_dir / PROGRESS_FILENAME
        tmp_path = output_dir / f".{PROGRESS_FILENAME}.tmp"
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(progress_path)
    except Exception:
        logger.debug("Could not write analysis progress", exc_info=True)


def parse_auto_int_arg(value: str) -> int | str:
    try:
        parsed = parse_auto_int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed is None:
        raise argparse.ArgumentTypeError("value must be auto or a non-negative integer")
    return parsed


def parse_auto_float_arg(value: str) -> float | str:
    try:
        parsed = parse_auto_float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed is None:
        raise argparse.ArgumentTypeError("value must be auto or a non-negative number")
    return parsed


def append_evidence_boundary_section(
    markdown: str,
    frame_selection_metadata: dict,
    ocr_keyframe_metadata: dict,
    ocr_events: list,
) -> str:
    if not markdown or "## 证据边界与需复核" in markdown:
        return markdown
    notes: list[str] = []
    vl_policy = str(frame_selection_metadata.get("vl_frame_policy_resolved") or "").lower()
    vl_frame_count = frame_selection_metadata.get("vl_frames_processed")
    if vl_frame_count is None:
        vl_frame_count = frame_selection_metadata.get("vl_frames_count")
    if vl_frame_count is None:
        vl_frame_count = len(frame_selection_metadata.get("frames") or frame_selection_metadata.get("selected_vl_frames") or [])
    if vl_policy == "none" or int(vl_frame_count or 0) == 0:
        notes.append("本次未运行或未选中 VL 视觉理解帧，界面细节主要依赖 OCR、ASR 与截图证据，关键操作建议人工复核。")
    if ocr_keyframe_metadata.get("ocr_text_events_count") == 0:
        notes.append("OCR 没有形成稳定文本事件，涉及按钮文案、菜单项和页面状态的结论需结合截图复核。")
    failed_ocr = [
        event for event in ocr_events
        if getattr(event, "status", "ok") not in {"ok", "skipped"}
    ]
    if failed_ocr:
        notes.append(f"有 {len(failed_ocr)} 个 OCR 帧未成功解析，对应时间点的文字证据置信度较低。")
    if not notes:
        return markdown
    section = "\n\n## 证据边界与需复核\n\n" + "\n".join(f"- {note}" for note in notes)
    return markdown.rstrip() + section + "\n"


def analyze_frames_for_vl(
    analyzer: VideoAnalyzer,
    frames,
    ocr_events,
    selected_frame_numbers: set[int],
    decisions: list[FrameDecision],
    concurrency: int,
    context_before: int,
    context_after: int,
    context_max_gap: float | str,
    checkpoint_path: Path | None = None,
):
    ocr_by_frame = {event.frame_number: event for event in ocr_events}
    context_ocr_texts = {event.frame_number: event.text for event in ocr_events if event.text}
    decisions_by_frame = {decision.frame_number: decision for decision in decisions}
    checkpoint_by_frame = load_frame_analysis_checkpoint(checkpoint_path)
    frame_analyses = [checkpoint_by_frame.get(frame.number) for frame in frames]

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        write_frame_analysis_checkpoint(checkpoint_path, [item for item in frame_analyses if item is not None])

    def analyze_one(index_frame):
        index, frame = index_frame
        ocr_text = ocr_by_frame.get(frame.number).text if frame.number in ocr_by_frame else ""
        context_window = build_frame_context_window(
            frames=frames,
            current_frame=frame,
            before=context_before,
            after=context_after,
            max_gap_seconds=context_max_gap,
        )
        return index, analyzer.analyze_frame(
            frame,
            ocr_text=ocr_text,
            context_window=context_window,
            context_ocr_texts=context_ocr_texts,
        )

    selected = [(index, frame) for index, frame in enumerate(frames) if frame.number in selected_frame_numbers]
    skipped = [(index, frame) for index, frame in enumerate(frames) if frame.number not in selected_frame_numbers]
    for index, frame in skipped:
        if frame_analyses[index] is None:
            frame_analyses[index] = make_skipped_visual_event(frame, decisions_by_frame[frame.number])

    selected = [(index, frame) for index, frame in selected if frame_analyses[index] is None]

    if not selected:
        save_checkpoint()
        return frame_analyses

    if concurrency <= 1:
        for index_frame in selected:
            index, analysis = analyze_one(index_frame)
            frame_analyses[index] = analysis
            save_checkpoint()
        return frame_analyses

    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as executor:
        futures = [executor.submit(analyze_one, item) for item in selected]
        for future in as_completed(futures):
            index, analysis = future.result()
            frame_analyses[index] = analysis
            save_checkpoint()
    return frame_analyses


def load_frame_analysis_checkpoint(path: Path | None) -> dict[int, dict]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    loaded = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        frame_number = item.get("frame_number", item.get("number"))
        if frame_number is None:
            continue
        loaded[int(frame_number)] = item
    return loaded


def write_frame_analysis_checkpoint(path: Path, analyses: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(analyses, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def scan_frames_count_from_metadata(metadata: dict) -> int | None:
    if not metadata:
        return None
    per_host = metadata.get("per_host") or []
    preview_total = sum(int(item.get("preview_frames") or 0) for item in per_host if isinstance(item, dict))
    if preview_total:
        return preview_total
    for key in ("scan_frames_count", "preview_frames", "raw_candidate_frames"):
        value = metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def frame_manifest_matches_request(metadata: dict, frame_extractor: str) -> bool:
    manifest_source = str(metadata.get("manifest_source") or "").strip().lower()
    requested = str(frame_extractor or "local").strip().lower()
    if requested == "jetson":
        return manifest_source == "jetson"
    if requested == "local":
        return manifest_source in {"local", "local_keyframes", "audio_only"}
    return bool(manifest_source)


def reusable_frames_from_manifest(output_dir: Path, frame_extractor: str):
    frames, metadata = read_frames_from_manifest(output_dir / MANIFEST_NAME, output_dir)
    if not frames:
        return [], metadata
    if not frame_manifest_matches_request(metadata, frame_extractor):
        metadata["reuse_rejected_reason"] = (
            f"manifest source {metadata.get('manifest_source')!r} does not match requested extractor {frame_extractor!r}"
        )
        return [], metadata
    metadata = dict(metadata)
    metadata["backend"] = f"reused_{metadata.get('manifest_source') or 'frames_manifest'}"
    metadata["resumed_existing"] = True
    return frames, metadata

def create_client(config: Config):
    """Create the appropriate client based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = get_client(config)
    
    if client_type == "ollama":
        return OllamaClient(client_config["url"])
    elif client_type == "openai_api":
        return GenericOpenAIAPIClient(
            client_config["api_key"],
            client_config["api_url"],
            timeout_seconds=int(client_config.get("timeout_seconds", 600)),
            extra_body=build_openai_extra_body(config.get("clients", {}).get("openai_api", {}), client_config["api_url"]),
        )
    else:
        raise ValueError(f"Unknown client type: {client_type}")


def create_operation_manual_text_client(config: Config, fallback_client):
    """Create the text-generation client for operation manuals.

    Operation-manual runs can route visual frame analysis to one endpoint and
    final Markdown generation to another. Non-OpenAI clients keep the legacy
    single-client behavior.
    """
    if config.get("clients", {}).get("default") != "openai_api":
        return fallback_client

    manual_config = config.get("operation_manual", {})
    openai_config = config.get("clients", {}).get("openai_api", {})
    text_base_url = (
        manual_config.get("text_base_url")
        or manual_config.get("llm_base_url")
        or openai_config.get("api_url")
    )
    if not text_base_url:
        return fallback_client
    return GenericOpenAIAPIClient(
        resolve_api_key(
            openai_config.get("api_key"),
            manual_config.get("text_api_key_env") or openai_config.get("api_key_env"),
            text_base_url,
        ),
        text_base_url,
        timeout_seconds=int(manual_config.get("text_timeout_seconds") or openai_config.get("timeout_seconds", 600)),
        extra_body=build_openai_extra_body(manual_config, text_base_url),
    )


def read_page_context_metadata(context_file: str, page_context: str) -> dict:
    metadata = {
        "context_file": context_file,
        "text_length": len(page_context or ""),
    }
    if not context_file:
        return metadata
    sidecar = Path(context_file).with_name("page_context.json")
    if not sidecar.exists():
        return metadata
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        metadata["diagnostics"] = [f"failed to read page_context.json: {exc}"]
        return metadata
    metadata.update(payload)
    metadata["text_length"] = len(page_context or "")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Analyze video using Vision models")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--config", type=str, default="config",
                        help="Path to configuration directory")
    parser.add_argument("--output", type=str, help="Output directory for analysis results")
    parser.add_argument("--client", type=str, help="Client to use (ollama or openrouter)")
    parser.add_argument("--ollama-url", type=str, help="URL for the Ollama service")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI-compatible service")
    parser.add_argument("--api-url", type=str, help="API URL for OpenAI-compatible API")
    parser.add_argument("--model", type=str, help="Name of the vision model to use")
    parser.add_argument("--duration", type=float, help="Duration in seconds to process")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames after analysis")
    parser.add_argument("--whisper-model", type=str, help="Whisper model size (tiny, base, small, medium, large), or path to local Whisper model snapshot")
    parser.add_argument("--start-stage", type=int, default=1, help="Stage to start processing from (1-3)")
    parser.add_argument("--resume-existing", action="store_true", help="Reuse completed core artifacts in the output directory")
    parser.add_argument("--max-frames", type=int, help="Explicit upper limit for the operation-manual candidate frame pool")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Question to ask about the video")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, help="Temperature for LLM generation")
    parser.add_argument("--task", choices=["describe", "operation_manual"], help="Analysis task")
    parser.add_argument("--manual-language", type=str, help="Language for operation manual output")
    parser.add_argument("--llm-base-url", type=str, help="OpenAI-compatible base URL for local LLMs")
    parser.add_argument("--vision-base-url", type=str, help="OpenAI-compatible base URL for frame vision analysis")
    parser.add_argument("--text-base-url", type=str, help="OpenAI-compatible base URL for final manual generation")
    parser.add_argument("--vision-model", type=str, help="Vision model used for frame analysis")
    parser.add_argument("--text-model", type=str, help="Text model used for manual generation")
    parser.add_argument("--ocr-provider", choices=["auto", "dots_mocr_vllm", "openai_vision", "none"], help="OCR provider")
    parser.add_argument("--ocr-base-url", action="append", help="OCR OpenAI-compatible base URL; can be provided multiple times")
    parser.add_argument("--ocr-concurrency", default=None, help="OCR concurrency per endpoint, or auto")
    parser.add_argument("--ocr-cache", choices=["on", "off", "refresh"], default=None, help="OCR cache mode")
    parser.add_argument("--ocr-cache-dir", default=None, help="OCR cache directory")
    parser.add_argument("--ocr-keyframe-strategy", choices=["auto", "scan-text", "legacy"], default="scan-text", help="OCR frame selection strategy")
    parser.add_argument("--ocr-keyframe-budget", type=parse_auto_int_arg, default=OCR_AUTO, help="auto or explicit OCR keyframe count")
    parser.add_argument("--ocr-scan-sample-fps", type=parse_auto_float_arg, default=OCR_AUTO, help="auto or low-cost preview scan FPS for OCR keyframe discovery")
    parser.add_argument("--ocr-timeout-seconds", type=float, default=None, help="Per-frame OCR request timeout")
    parser.add_argument(
        "--ocr-prompt-mode",
        choices=["prompt_scene_spotting", "prompt_layout_json", "prompt_ocr"],
        default=None,
        help="DotsMOCR prompt preset",
    )
    parser.add_argument("--ocr-max-tokens", type=int, default=None, help="DotsMOCR max output tokens per frame")
    parser.add_argument(
        "--ocr-max-image-long-side",
        type=int,
        default=None,
        help="Resize OCR images to this longest side before upload; <=0 disables resizing",
    )
    parser.add_argument(
        "--ocr-retry-endpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retry the same OCR frame on another healthy endpoint after failure",
    )
    parser.add_argument("--asr-provider", choices=["auto", "remote_http", "capswriter_http", "vibevoice", "faster_whisper", "none"], help="ASR provider")
    parser.add_argument("--asr-strategy", choices=["fast", "balanced", "deep"], help="Dual-ASR strategy for operation manuals")
    parser.add_argument("--remote-asr-url", action="append", help="Remote fast ASR endpoint; can be provided multiple times")
    parser.add_argument("--vibevoice-url", action="append", help="Remote GPU VibeVoice ASR endpoint; can be provided multiple times")
    parser.add_argument("--transcript-file", type=str, help="Use an existing transcript markdown file and skip audio ASR")
    parser.add_argument("--context-file", type=str, help="Extra page/video context file")
    parser.add_argument("--pipeline-mode", choices=["fast", "balanced", "deep"], default="balanced", help="Operation manual pipeline depth")
    parser.add_argument("--candidate-frames", type=parse_auto_int_arg, default=AUTO, help="auto or explicit candidate frame pool size")
    parser.add_argument(
        "--candidate-frame-strategy",
        type=parse_candidate_frame_strategy,
        default="auto",
        help="Internal candidate frame strategy: auto, legacy, generic, lecture, or operation",
    )
    parser.add_argument("--frame-extractor", choices=["local", "jetson", "auto"], default="local", help="Candidate frame extraction backend")
    parser.add_argument("--jetson-frame-hosts", default="nx2,nx3", help="Comma-separated Jetson SSH hosts for frame extraction")
    parser.add_argument("--jetson-frame-backend", choices=["auto", "ssh", "ray"], default="auto", help="Jetson frame worker transport")
    parser.add_argument("--jetson-sample-fps", default="auto", help="auto or preview sample fps used by Jetson frame workers")
    parser.add_argument("--jetson-chunk-overlap-seconds", type=float, default=2.0, help="Overlap seconds between Jetson frame chunks")
    parser.add_argument("--jetson-frame-weights", help="Comma-separated Jetson frame worker weights, e.g. nx1=1,nx2=1,agx=2")
    parser.add_argument(
        "--jetson-require-hwdec",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require Jetson workers to use hardware video decode instead of software ffmpeg",
    )
    parser.add_argument("--min-vl-frames", type=parse_auto_int_arg, default=AUTO, help="auto or minimum frames sent to VL")
    parser.add_argument("--max-vl-frames", type=parse_auto_int_arg, default=AUTO, help="auto or maximum frames sent to VL")
    parser.add_argument("--vl-frame-policy", choices=["auto", "all", "none"], default="auto", help="VL frame execution policy")
    parser.add_argument("--vl-concurrency", type=int, default=3, help="Concurrent VL frame analysis requests")
    parser.add_argument("--vl-context-before", type=int, default=0, help="Previous candidate frames to include as VL image context")
    parser.add_argument("--vl-context-after", type=int, default=0, help="Next candidate frames to include as VL image context")
    parser.add_argument("--vl-context-max-gap", type=parse_auto_float_arg, default=AUTO, help="auto or max adjacent seconds for VL context frames")
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
        
        # Stage 1: Frame and Audio Processing
        if args.start_stage <= 2:
            enter_local_runtime()

        if args.start_stage <= 1:
            stage_started = time.perf_counter()
            if args.transcript_file:
                current_progress_step = "asr"
                write_analysis_progress(output_dir, current_progress_step, message="using existing transcript file")
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
                )
            else:
                current_progress_step = "audio"
                write_analysis_progress(output_dir, current_progress_step, message="extracting audio")
                logger.info("Extracting audio from video...")
                try:
                    audio_path = extract_audio_to_wav(video_path, output_dir)
                except Exception as e:
                    logger.error(f"Error extracting audio: {e}")
                    audio_path = None

                if audio_path is None:
                    logger.debug("No audio found in video - skipping transcription")
                    transcript = None
                else:
                    current_progress_step = "asr"
                    write_analysis_progress(output_dir, current_progress_step, message="transcribing audio")
                    logger.info("Transcribing audio...")
                    asr_config = config.get("asr", {})
                    provider = asr_config.get("provider", "faster_whisper")
                    use_asr_strategy = task == "operation_manual" and args.asr_provider is None and provider == "auto"
                    asr_lock = (
                        contextlib.nullcontext()
                        if provider == "none"
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
                                audio_processor = AudioProcessor(
                                    language=config.get("audio", {}).get("language", ""),
                                    model_size_or_path=config.get("audio", {}).get("whisper_model", "medium"),
                                    device=config.get("audio", {}).get("device", "cpu"),
                                )
                                transcript = audio_processor.transcribe(audio_path)
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
                        speaker_config = config.get("speaker_diarization") or {}
                        try:
                            transcript, speaker_diarization_report = process_transcript_speakers(
                                audio_path,
                                transcript,
                                speaker_config,
                            )
                        except Exception as exc:
                            logger.warning("speaker diarization failed: %s", exc)
                            speaker_diarization_report = {"enabled": True, "error": str(exc)}
                        qa_dir = output_dir / "qa"
                        qa_dir.mkdir(parents=True, exist_ok=True)
                        write_json(qa_dir / "speaker_diarization_report.json", speaker_diarization_report)
                        if asr_result:
                            asr_result.transcript = transcript
                        transcript_markdown_path = write_transcript_markdown(transcript, output_dir / "transcript.md")
                        current_progress_step = "asr_done"
                        write_analysis_progress(
                            output_dir,
                            current_progress_step,
                            message="transcript ready",
                            artifacts={"transcript": str(transcript_markdown_path)} if transcript_markdown_path else {},
                        )
            timings["asr_seconds"] = round(time.perf_counter() - stage_started, 3)
            
            current_progress_step = "frames"
            write_analysis_progress(output_dir, current_progress_step, message="extracting candidate frames")
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
                if args.frame_extractor in {"jetson", "auto"}:
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
                )
            timings["candidate_frame_extraction_seconds"] = round(time.perf_counter() - stage_started, 3)
            frame_dedup_audit_path, frame_dedup_audit = write_frame_dedup_audit(frames, output_dir)
            frame_dedup_audit_metadata = {
                key: value
                for key, value in frame_dedup_audit.items()
                if key != "records"
            }

            if task == "operation_manual":
                current_progress_step = "ocr"
                write_analysis_progress(output_dir, current_progress_step, message="running OCR")
                logger.info("Running OCR on extracted frames...")
                stage_started = time.perf_counter()
                ocr_config = config.get("ocr", {})
                ocr_base_urls = ocr_config.get("base_urls")
                selected_ocr_frames, ocr_keyframe_decisions, ocr_keyframe_metadata = select_ocr_keyframes(
                    frames=frames,
                    transcript=transcript,
                    video_duration_seconds=frame_selection_metadata.get("video_duration_seconds", config.get("duration") or 0.0),
                    pipeline_mode=args.pipeline_mode,
                    strategy=args.ocr_keyframe_strategy,
                    budget=args.ocr_keyframe_budget,
                    scan_frames_count=scan_frames_count_from_metadata(frame_extraction_metadata),
                )
                logger.info(
                    "Selected %s OCR keyframes from %s candidate frames after scanning %s preview frames",
                    len(selected_ocr_frames),
                    len(frames),
                    ocr_keyframe_metadata.get("scan_frames_count"),
                )
                if selected_ocr_frames:
                    with analyzer_resource_lock(config.config, "ocr", str(output_dir), logger):
                        with local_model_stage("ocr", config.config, logger, str(output_dir)):
                            ocr_events = run_ocr(
                                frames=selected_ocr_frames,
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
                            )
                else:
                    logger.info("No OCR frames selected; skipping OCR provider calls.")
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
                write_analysis_progress(output_dir, current_progress_step, message="OCR results ready")
            
        # Stage 2: Frame Analysis
        if args.start_stage <= 2:
            current_progress_step = "vl"
            write_analysis_progress(output_dir, current_progress_step, message="selecting and analyzing VL frames")
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
                options = FrameSelectionOptions(
                    pipeline_mode=args.pipeline_mode,
                    candidate_frames=args.candidate_frames,
                    min_vl_frames=args.min_vl_frames,
                    max_vl_frames=args.max_vl_frames,
                    vl_frame_policy=args.vl_frame_policy,
                    explicit_max_frames=args.max_frames,
                )
                selected_frame_numbers, frame_decisions, selection_metadata = select_vl_frames(
                    frames=frames,
                    ocr_events=ocr_events,
                    transcript=transcript,
                    video_duration_seconds=frame_selection_metadata.get("video_duration_seconds", config.get("duration") or 0.0),
                    options=options,
                )
                frame_selection_metadata.update(selection_metadata)
                timings["frame_selection_seconds"] = round(time.perf_counter() - stage_started, 3)
                vl_started = time.perf_counter()
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
                                context_before=max(args.vl_context_before, 0),
                                context_after=max(args.vl_context_after, 0),
                                context_max_gap=args.vl_context_max_gap,
                                checkpoint_path=output_dir / "orin" / "frame_analyses.partial.json",
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

        release_local_runtime()
                
        # Stage 3: Video Reconstruction
        if args.start_stage <= 3:
            if task == "operation_manual":
                current_progress_step = "manual"
                write_analysis_progress(output_dir, current_progress_step, message="generating operation manual")
                logger.info("Generating operation manual...")
                stage_started = time.perf_counter()
                manual_config = config.get("operation_manual", {})
                text_client = create_operation_manual_text_client(config, client)
                page_context = read_context_file(config.get("context_file", ""))
                page_context_metadata = read_page_context_metadata(config.get("context_file", ""), page_context)
                text_model = manual_config.get("text_model") or model
                frame_assets = prepare_frame_assets(frames, output_dir)
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
            else:
                logger.info("Reconstructing video description...")
                video_description = analyzer.reconstruct_video(
                    frame_analyses, frames, transcript
                )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        current_progress_step = "write"
        write_analysis_progress(output_dir, current_progress_step, message="writing analysis outputs")
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
        write_analysis_progress(output_dir, current_progress_step, status="failed", message=str(e))
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        raise
    finally:
        release_local_runtime()

if __name__ == "__main__":
    main()
