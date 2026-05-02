#!/usr/bin/env python3
"""HTTP wrapper for VibeVoice ASR using the native meeting workflow.

Deploy this file inside a VibeVoice-bench checkout and run it with that
project's virtualenv. Long audio is segmented by VibeVoice's own
chunk-reconcile path, not by video-analyzer.
"""

from __future__ import annotations

import os
import logging
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger(__name__)

ROOT = Path(os.getenv("VIBEVOICE_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()
for candidate in (ROOT, ROOT / "src", ROOT / "demo", ROOT / "src" / "demo"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from vibevoice_asr_meeting_workflow import (  # noqa: E402
    choose_attention_impl,
    load_audio_array,
    load_model,
    patch_transformers_for_rocm,
    run_chunk_reconcile_transcription,
    run_transcription,
)

MODEL_PATH = os.getenv(
    "VIBEVOICE_MODEL_PATH",
    str(Path.home() / ".cache/huggingface/hub/models--microsoft--VibeVoice-ASR/snapshots/d0c9efdb8d614685062c04425d91e01b6f37d944"),
)
TOKENIZER_PATH = os.getenv("VIBEVOICE_TOKENIZER_PATH", "")
DEVICE = os.getenv("VIBEVOICE_DEVICE", "cuda")
ATTN_IMPL_REQUESTED = os.getenv("VIBEVOICE_ATTN_IMPL", "sdpa")
DTYPE = os.getenv("VIBEVOICE_DTYPE", "auto")
MAX_NEW_TOKENS = int(os.getenv("VIBEVOICE_MAX_NEW_TOKENS", "32768"))
SPEAKER_EMBEDDING_MODEL = os.getenv("VIBEVOICE_SPEAKER_EMBEDDING_MODEL", "pyannote/embedding")

app = FastAPI(title="VibeVoice ASR HTTP API")
_lock = threading.Lock()
_model = None
_processor = None
_status = "idle"
_last_error: Optional[str] = None
_loaded_at: Optional[float] = None
_attention_impl: Optional[str] = None


def get_model_pair():
    global _model, _processor, _status, _last_error, _loaded_at, _attention_impl
    with _lock:
        if _model is not None and _processor is not None:
            return _model, _processor
        _status = "loading"
        _last_error = None
        try:
            patch_transformers_for_rocm()
            _attention_impl = choose_attention_impl(DEVICE, ATTN_IMPL_REQUESTED)
            _model, _processor = load_model(
                MODEL_PATH,
                DEVICE,
                _attention_impl,
                DTYPE,
                TOKENIZER_PATH,
            )
            _status = "ready"
            _loaded_at = time.time()
            return _model, _processor
        except Exception as exc:
            _status = "error"
            _last_error = repr(exc)
            raise


def segment_text(segments, raw_text: str) -> str:
    if segments:
        parts = []
        for seg in segments:
            text = seg.get("text") or seg.get("transcript") or seg.get("Content") or ""
            if text:
                parts.append(str(text).strip())
        if parts:
            return "\n".join(parts).strip()
    return (raw_text or "").strip()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


@app.get("/api/health")
def health():
    return {
        "service": "VibeVoice ASR HTTP API",
        "status": _status,
        "ready": _model is not None,
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "attn_implementation": _attention_impl or ATTN_IMPL_REQUESTED,
        "loaded_at": _loaded_at,
        "last_error": _last_error,
        "native_chunking": True,
    }


@app.post("/api/asr/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    hotword: str = Form(default=""),
    use_native_chunking: bool = Form(default=True),
    single_pass_max_duration_sec: Optional[float] = Form(default=None),
    chunk_duration_sec: Optional[float] = Form(default=None),
    chunk_overlap_sec: Optional[float] = Form(default=None),
    chunk_parallel_workers: Optional[int] = Form(default=None),
    speaker_upper_bound: Optional[int] = Form(default=None),
):
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(prefix="vibevoice_asr_", suffix=suffix, delete=False) as temp:
        temp_path = Path(temp.name)
        temp.write(await audio.read())
    started = time.time()
    try:
        audio_array, sample_rate = load_audio_array(str(temp_path))
        duration_sec = len(audio_array) / sample_rate if sample_rate else 0.0
        attention_impl = choose_attention_impl(DEVICE, ATTN_IMPL_REQUESTED)
        single_pass_threshold = float(single_pass_max_duration_sec or os.getenv("VIBEVOICE_SINGLE_PASS_MAX_DURATION_SEC", "420"))
        chunk_duration = float(chunk_duration_sec or os.getenv("VIBEVOICE_CHUNK_DURATION_SEC", "300"))
        chunk_overlap = float(chunk_overlap_sec or os.getenv("VIBEVOICE_CHUNK_OVERLAP_SEC", "25"))
        chunk_workers = int(chunk_parallel_workers or os.getenv("VIBEVOICE_CHUNK_PARALLEL_WORKERS", "1"))
        speaker_bound = int(speaker_upper_bound or os.getenv("VIBEVOICE_SPEAKER_UPPER_BOUND", "4"))
        use_single_pass = (not as_bool(use_native_chunking)) or duration_sec <= single_pass_threshold
        context_info = hotword.strip() or None

        if use_single_pass or chunk_workers <= 1 or DEVICE != "cuda":
            model, processor = get_model_pair()
            result = run_transcription(
                model=model,
                processor=processor,
                audio_file=audio_array,
                device=DEVICE,
                context_info=context_info,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )
            result["mode"] = "single_pass"
        else:
            result = run_chunk_reconcile_transcription(
                model=None,
                processor=None,
                model_path=MODEL_PATH,
                tokenizer_path=TOKENIZER_PATH,
                audio_file=str(temp_path),
                audio_array=audio_array,
                sample_rate=sample_rate,
                device=DEVICE,
                context_info=context_info,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                chunk_duration_sec=chunk_duration,
                chunk_overlap_sec=chunk_overlap,
                speaker_upper_bound=speaker_bound,
                chunk_parallel_workers=chunk_workers,
                attn_implementation=attention_impl,
                dtype_name=DTYPE,
                speaker_embedding_model=SPEAKER_EMBEDDING_MODEL,
            )

        text = segment_text(result.get("segments") or [], result.get("raw_text") or "")
        return {
            "success": bool(text),
            "text": text,
            "segments": result.get("segments") or [],
            "language": "unknown",
            "provider": "vibevoice_remote",
            "elapsed_seconds": round(time.time() - started, 3),
            "generation_time": result.get("generation_time_sec"),
            "raw_text": result.get("raw_text", ""),
            "mode": result.get("mode", "single_pass"),
            "chunk_parallel_workers": result.get("chunk_parallel_workers"),
            "quality_report": result.get("quality_report"),
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": repr(exc)})
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("Failed to clean up temporary audio file %s: %s", temp_path, exc)


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("VIBEVOICE_HOST", "0.0.0.0"), port=int(os.getenv("VIBEVOICE_PORT", "8002")))
