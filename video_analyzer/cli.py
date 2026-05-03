import argparse
from pathlib import Path
import json
import logging
import shutil
import sys

from .artifacts import write_json, write_orin_artifacts, write_transcript_markdown
from .config import Config, get_client, get_model
from .frame import VideoProcessor
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

# Initialize logger at module level
logger = logging.getLogger(__name__)

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

def create_client(config: Config):
    """Create the appropriate client based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = get_client(config)
    
    if client_type == "ollama":
        return OllamaClient(client_config["url"])
    elif client_type == "openai_api":
        return GenericOpenAIAPIClient(client_config["api_key"], client_config["api_url"])
    else:
        raise ValueError(f"Unknown client type: {client_type}")


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
    parser.add_argument("--max-frames", type=int, default=sys.maxsize, help="Maximum number of frames to process")
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
    parser.add_argument("--vision-model", type=str, help="Vision model used for frame analysis")
    parser.add_argument("--text-model", type=str, help="Text model used for manual generation")
    parser.add_argument("--ocr-provider", choices=["auto", "dots_mocr_vllm", "openai_vision", "none"], help="OCR provider")
    parser.add_argument("--ocr-base-url", type=str, help="OCR OpenAI-compatible base URL or auto")
    parser.add_argument("--asr-provider", choices=["auto", "remote_http", "capswriter_http", "vibevoice", "faster_whisper", "none"], help="ASR provider")
    parser.add_argument("--asr-strategy", choices=["fast", "balanced", "deep"], help="Dual-ASR strategy for operation manuals")
    parser.add_argument("--remote-asr-url", action="append", help="Remote fast ASR endpoint; can be provided multiple times")
    parser.add_argument("--vibevoice-url", action="append", help="Remote GPU VibeVoice ASR endpoint; can be provided multiple times")
    parser.add_argument("--context-file", type=str, help="Extra page/video context file")
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

    # Load and update configuration
    config = Config(args.config)
    config.update_from_args(args)

    # Initialize components
    video_path = Path(args.video_path)
    output_dir = Path(config.get("output_dir"))
    client = create_client(config)
    model = get_model(config)
    prompt_loader = PromptLoader(config.get("prompt_dir"), config.get("prompts", []))
    
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
        
        # Stage 1: Frame and Audio Processing
        if args.start_stage <= 1:
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
                logger.info("Transcribing audio...")
                asr_config = config.get("asr", {})
                provider = asr_config.get("provider", "faster_whisper")
                use_asr_strategy = task == "operation_manual" and args.asr_provider is None and provider == "auto"
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
                    logger.warning("Could not generate reliable transcript. Proceeding with video analysis only.")
                else:
                    transcript_markdown_path = write_transcript_markdown(transcript, output_dir / "transcript.md")
            
            logger.info(f"Extracting frames from video using model {model}...")
            processor = VideoProcessor(
                video_path, 
                output_dir / "frames", 
                model
            )
            if task == "operation_manual":
                frames = processor.extract_screen_keyframes(
                    frames_per_minute=config.get("frames", {}).get("per_minute", 60),
                    duration=config.get("duration"),
                    max_frames=args.max_frames,
                )
            else:
                frames = processor.extract_keyframes(
                    frames_per_minute=config.get("frames", {}).get("per_minute", 60),
                    duration=config.get("duration"),
                    max_frames=args.max_frames
                )

            if task == "operation_manual":
                logger.info("Running OCR on extracted frames...")
                ocr_config = config.get("ocr", {})
                ocr_events = run_ocr(
                    frames=frames,
                    provider=ocr_config.get("provider", "auto"),
                    base_url=ocr_config.get("base_url", "auto"),
                    model=ocr_config.get("model", "model"),
                    prompt_mode=ocr_config.get("prompt_mode", "prompt_scene_spotting"),
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
                    probe_timeout_seconds=ocr_config.get("probe_timeout_seconds", 5),
                    warmup_timeout_seconds=ocr_config.get("warmup_timeout_seconds", 180),
                    warmup_retry_interval_seconds=ocr_config.get("warmup_retry_interval_seconds", 5),
                )
            
        # Stage 2: Frame Analysis
        if args.start_stage <= 2:
            logger.info("Analyzing frames...")
            analyzer = VideoAnalyzer(
                client, 
                model, 
                prompt_loader,
                config.get("clients", {}).get("temperature", 0.2),
                config.get("prompt", "")
            )
            frame_analyses = []
            ocr_by_frame = {event.frame_number: event for event in ocr_events}
            for frame in frames:
                ocr_text = ocr_by_frame.get(frame.number).text if frame.number in ocr_by_frame else ""
                analysis = analyzer.analyze_frame(frame, ocr_text=ocr_text)
                frame_analyses.append(analysis)
                
        # Stage 3: Video Reconstruction
        if args.start_stage <= 3:
            if task == "operation_manual":
                logger.info("Generating operation manual...")
                manual_config = config.get("operation_manual", {})
                page_context = read_context_file(config.get("context_file", ""))
                page_context_metadata = read_page_context_metadata(config.get("context_file", ""), page_context)
                text_model = manual_config.get("text_model") or model
                frame_assets = prepare_frame_assets(frames, output_dir)
                operation_manual = generate_operation_manual(
                    client=client,
                    text_model=text_model,
                    frame_analyses=frame_analyses,
                    frames=frames,
                    transcript=transcript,
                    asr_metadata=asr_result.to_metadata() if asr_result else {},
                    ocr_events=ocr_events,
                    page_context=page_context,
                    language=config.get("manual_language", "zh-CN"),
                    temperature=config.get("clients", {}).get("temperature", 0.2),
                    frame_assets=frame_assets,
                )
                operation_manual["response"] = embed_step_images(
                    operation_manual.get("response", ""),
                    frames,
                    frame_assets,
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
            else:
                logger.info("Reconstructing video description...")
                video_description = analyzer.reconstruct_video(
                    frame_analyses, frames, transcript
                )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "metadata": {
                "task": task,
                "client": config.get("clients", {}).get("default"),
                "model": model,
                "text_model": config.get("operation_manual", {}).get("text_model"),
                "ocr_provider": config.get("ocr", {}).get("provider"),
                "asr_provider": config.get("asr", {}).get("provider"),
                "asr_strategy": config.get("asr", {}).get("strategy"),
                "context_file": config.get("context_file"),
                "page_description": page_context,
                "page_context": page_context_metadata,
                "whisper_model": config.get("audio", {}).get("whisper_model"),
                "frames_per_minute": config.get("frames", {}).get("per_minute"),
                "duration_processed": config.get("duration"),
                "frames_extracted": len(frames),
                "frames_processed": min(len(frames), args.max_frames),
                "start_stage": args.start_stage,
                "audio_language": transcript.language if transcript else None,
                "transcription_successful": transcript is not None,
                "transcript_markdown": str(transcript_markdown_path) if transcript_markdown_path else None,
            },
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None
            } if transcript else None,
            "asr": asr_result.to_metadata() if asr_result else None,
            "ocr_events": [event.to_dict() for event in ocr_events],
            "visual_events": frame_analyses,
            "manual_steps": operation_manual,
            "uncertainties": [
                event.to_dict() for event in ocr_events if event.status != "ok"
            ],
            "frame_analyses": frame_analyses,
            "video_description": video_description,
            "operation_manual": operation_manual
        }
        
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
            
    except Exception as e:
        logger.error(f"Error during video analysis: {e}")
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        raise

if __name__ == "__main__":
    main()
