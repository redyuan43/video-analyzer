"""video-analyzer CLI 参数面：argparse 解析器构建。

由 video_analyzer.cli 调用 build_arg_parser()；参数集保持不变。
"""

from __future__ import annotations

import argparse

from .cli_helpers import parse_auto_float_arg, parse_auto_int_arg
from .candidate_frame_strategies import parse_candidate_frame_strategy
from .frame_selection import AUTO
from .ocr_keyframes import AUTO as OCR_AUTO


def build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--ocr-provider",
        choices=["auto", "unlimited_ocr", "dots_ocr", "dots_mocr_vllm", "openai_vision", "none"],
        help="OCR provider",
    )
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
        "--ocr-image-mode",
        choices=["gundam", "base"],
        default=None,
        help="Unlimited-OCR image preprocessing mode",
    )
    parser.add_argument(
        "--ocr-retry-endpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retry the same OCR frame on another healthy endpoint after failure",
    )
    parser.add_argument(
        "--asr-provider",
        choices=[
            "auto",
            "remote_http",
            "capswriter_http",
            "qwen3_asr",
            "firered_asr2",
            "firered_3dspeaker",
            "openai_audio",
            "tencent_hy_asr",
            "vibevoice",
            "faster_whisper",
            "none",
        ],
        help="ASR provider",
    )
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
    parser.add_argument(
        "--frame-extractor",
        choices=["local", "local_gpu", "jetson", "auto"],
        default="local",
        help="Candidate frame extraction backend",
    )
    parser.add_argument(
        "--local-frame-gpus",
        default="auto",
        help="Comma-separated local P40 GPU indices for local_gpu extraction, or auto",
    )
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
    return parser
