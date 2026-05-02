import logging
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    AudioSegment = None

from .audio_processor import AudioProcessor, AudioTranscript

logger = logging.getLogger(__name__)

VIBEVOICE_PYTHONS = [
    Path("/home/ivan/github/VibeVoice/.venv-vvasr4bit/bin/python"),
    Path("/home/ivan/github/VibeVoice/.venv-rocm71/bin/python"),
]
VIBEVOICE_SCRIPT = Path("/home/ivan/github/VibeVoice/demo/vibevoice_asr_inference_from_file.py")
VIBEVOICE_MODEL = "microsoft/VibeVoice-ASR"
CAPSWRITER_URL = "http://127.0.0.1:8001/api/asr/transcribe"
REMOTE_VIBEVOICE_URLS = [
    "http://spark-31d6.taild500c8.ts.net:8002/api/asr/transcribe",
    "http://edge.taild500c8.ts.net:8002/api/asr/transcribe",
]
REMOTE_ASR_URLS = [
    "http://edge.taild500c8.ts.net:8001/api/asr/transcribe",
    "http://spark-31d6.taild500c8.ts.net:8001/api/asr/transcribe",
]

DEEP_ASR_MIN_SECONDS = 180.0


@dataclass
class ASRStrategyResult:
    strategy: str
    transcript: Optional[AudioTranscript]
    fast_transcript: Optional[AudioTranscript] = None
    deep_transcript: Optional[AudioTranscript] = None
    providers_run: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    elapsed_seconds: Dict[str, float] = field(default_factory=dict)
    merge_notes: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "providers_run": self.providers_run,
            "failures": self.failures,
            "elapsed_seconds": self.elapsed_seconds,
            "merge_notes": self.merge_notes,
            "fast_transcript": _transcript_summary(self.fast_transcript),
            "deep_transcript": _transcript_summary(self.deep_transcript),
            "merged_transcript": _transcript_summary(self.transcript),
        }


def extract_audio_to_wav(video_path: Path, output_dir: Path) -> Optional[Path]:
    audio_path = output_dir / "audio.wav"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
        return audio_path
    except subprocess.CalledProcessError as exc:
        error_output = exc.stderr.decode(errors="replace")
        if "Output file does not contain any stream" in error_output:
            return None
        if AudioSegment is None:
            raise RuntimeError("ffmpeg failed and pydub is not installed") from exc
        logger.info("Falling back to pydub for audio extraction")
        video = AudioSegment.from_file(str(video_path))
        audio = video.set_channels(1).set_frame_rate(16000)
        audio.export(str(audio_path), format="wav")
        return audio_path


def transcribe_with_http_asr(audio_path: Path, url: str, hotword: str = "") -> Optional[AudioTranscript]:
    try:
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                url,
                files={"audio": (audio_path.name, audio_file, "audio/wav")},
                data={"hotword": hotword},
                timeout=900,
            )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            logger.warning("HTTP ASR failed from %s: %s", url, payload)
            return None
        return AudioTranscript(
            text=payload.get("text", ""),
            segments=payload.get("segments") or [],
            language=payload.get("language") or "unknown",
        )
    except Exception as exc:
        logger.warning("HTTP ASR unavailable from %s: %s", url, exc)
        return None


def transcribe_with_capswriter(audio_path: Path, hotword: str = "") -> Optional[AudioTranscript]:
    return transcribe_with_http_asr(audio_path, CAPSWRITER_URL, hotword=hotword)


def transcribe_with_remote_http(audio_path: Path, urls: Optional[list[str]] = None) -> Optional[AudioTranscript]:
    urls = REMOTE_ASR_URLS if urls is None else urls
    for url in urls:
        transcript = transcribe_with_http_asr(audio_path, url)
        if transcript and transcript.text.strip():
            transcript.segments = transcript.segments or [{"provider_url": url}]
            return transcript
    return None


def transcribe_with_vibevoice_remote(audio_path: Path, urls: Optional[list[str]] = None) -> Optional[AudioTranscript]:
    urls = [url for url in (REMOTE_VIBEVOICE_URLS if urls is None else urls) if url]
    if not urls:
        logger.warning("No remote GPU VibeVoice endpoint is configured")
        return None
    transcript = transcribe_with_remote_http(audio_path, urls)
    if transcript:
        transcript.segments = transcript.segments or []
        transcript.segments.append({"provider": "vibevoice_remote"})
    return transcript


def _vibevoice_python_candidates(config: Dict[str, str]) -> list[Path]:
    configured_python = config.get("python")
    if configured_python:
        return [Path(configured_python).expanduser()]
    return VIBEVOICE_PYTHONS


def transcribe_with_vibevoice(audio_path: Path, config: Optional[Dict[str, str]] = None) -> Optional[AudioTranscript]:
    config = config or {}
    remote_urls = config["deep_remote_urls"] if "deep_remote_urls" in config else None
    remote_transcript = transcribe_with_vibevoice_remote(audio_path, remote_urls)
    if remote_transcript and remote_transcript.text.strip():
        return remote_transcript

    if not config.get("allow_local", False):
        logger.warning("Local VibeVoice is disabled; configure a remote GPU VibeVoice endpoint or pass --allow-local-vibevoice")
        return None

    python_candidates = [path for path in _vibevoice_python_candidates(config) if path.exists()]
    if not python_candidates or not VIBEVOICE_SCRIPT.exists():
        logger.warning("VibeVoice ASR environment is not available")
        return None

    failures = []
    for python_path in python_candidates:
        command = [
            str(python_path),
            str(VIBEVOICE_SCRIPT),
            "--model_path",
            config.get("model_path", VIBEVOICE_MODEL),
            "--audio_files",
            str(audio_path),
            "--batch_size",
            "1",
            "--temperature",
            "0",
            "--num_beams",
            "1",
            "--attn_implementation",
            config.get("attn_implementation", "auto"),
        ]
        if config.get("device"):
            command.extend(["--device", config["device"]])

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            failures.append(f"{python_path}: exit {exc.returncode}; stderr={stderr[-2000:]}; stdout={stdout[-1000:]}")
            logger.warning("VibeVoice ASR failed with %s: %s", python_path, failures[-1])
            continue
        except Exception as exc:
            failures.append(f"{python_path}: {exc}")
            logger.warning("VibeVoice ASR failed with %s: %s", python_path, exc)
            continue

        raw = result.stdout.strip()
        text = _extract_vibevoice_text(raw)
        if not text:
            text = raw
        return AudioTranscript(text=text, segments=[{"raw_output": raw}], language="unknown")

    logger.warning("All VibeVoice ASR candidates failed: %s", failures)
    return None


def _extract_vibevoice_text(raw: str) -> str:
    lines = []
    capture = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "--- Raw Output ---":
            capture = True
            continue
        if capture and stripped.startswith("--- Structured Output"):
            break
        if capture and stripped and not stripped.startswith("="):
            lines.append(stripped)
    return "\n".join(lines).strip()


def transcribe_with_provider(
    provider: str,
    audio_path: Path,
    language: str,
    whisper_model: str,
    device: str,
    vibevoice_config: Optional[Dict[str, str]] = None,
) -> Optional[AudioTranscript]:
    return transcribe_with_provider_result(
        provider=provider,
        audio_path=audio_path,
        language=language,
        whisper_model=whisper_model,
        device=device,
        vibevoice_config=vibevoice_config,
    ).transcript


def transcribe_with_provider_result(
    provider: str,
    audio_path: Path,
    language: str,
    whisper_model: str,
    device: str,
    vibevoice_config: Optional[Dict[str, str]] = None,
) -> ASRStrategyResult:
    vibevoice_config = vibevoice_config or {}
    result = ASRStrategyResult(strategy=f"provider:{provider}", transcript=None)
    providers = ["remote_http", "vibevoice", "capswriter_http", "faster_whisper"] if provider == "auto" else [provider]
    for candidate in providers:
        if candidate == "none":
            result.merge_notes.append("ASR disabled by provider:none")
            return result
        if candidate == "remote_http":
            transcript = _timed_transcribe(
                result,
                candidate,
                lambda: transcribe_with_remote_http(audio_path, vibevoice_config.get("remote_urls")),
            )
        elif candidate == "capswriter_http":
            transcript = _timed_transcribe(result, candidate, lambda: transcribe_with_capswriter(audio_path))
        elif candidate == "vibevoice":
            transcript = _timed_transcribe(result, candidate, lambda: transcribe_with_vibevoice(audio_path, vibevoice_config))
        elif candidate == "faster_whisper":
            transcript = _timed_transcribe(
                result,
                candidate,
                lambda: AudioProcessor(language=language, model_size_or_path=whisper_model, device=device).transcribe(audio_path),
            )
        else:
            raise ValueError(f"Unknown ASR provider: {candidate}")

        if transcript and transcript.text.strip():
            logger.info("ASR succeeded with provider: %s", candidate)
            result.transcript = transcript
            if candidate == "vibevoice":
                result.deep_transcript = transcript
            else:
                result.fast_transcript = transcript
            result.merge_notes.append(f"used explicit ASR provider: {candidate}")
            return result
        logger.info("ASR provider did not produce transcript: %s", candidate)
    result.failures.append("no explicit ASR provider produced transcript text")
    return result


def transcribe_with_strategy(
    strategy: str,
    audio_path: Path,
    language: str,
    whisper_model: str,
    device: str,
    vibevoice_config: Optional[Dict[str, str]] = None,
) -> ASRStrategyResult:
    """Run the dual-ASR strategy used by operation_manual.

    remote_http is the fast timestamp anchor. VibeVoice is the slower long-audio
    semantic pass. The merged transcript keeps fast timestamps and uses the deep
    transcript for terminology and chapter-level text when available.
    """
    vibevoice_config = vibevoice_config or {}
    result = ASRStrategyResult(strategy=strategy, transcript=None)
    normalized = strategy or "balanced"

    if normalized == "fast":
        result.fast_transcript = _timed_transcribe(
            result,
            "remote_http",
            lambda: transcribe_with_remote_http(audio_path, vibevoice_config.get("remote_urls")),
        )
    elif normalized == "deep":
        result.fast_transcript = _timed_transcribe(
            result,
            "remote_http",
            lambda: transcribe_with_remote_http(audio_path, vibevoice_config.get("remote_urls")),
        )
        result.deep_transcript = _timed_transcribe(
            result,
            "vibevoice",
            lambda: transcribe_with_vibevoice(audio_path, vibevoice_config),
        )
        if not _has_transcript_text(result.fast_transcript) and not _has_transcript_text(result.deep_transcript):
            result.fast_transcript = _timed_transcribe(
                result,
                "faster_whisper",
                lambda: AudioProcessor(language=language, model_size_or_path=whisper_model, device=device).transcribe(audio_path),
            )
    elif normalized == "balanced":
        result.fast_transcript = _timed_transcribe(
            result,
            "remote_http",
            lambda: transcribe_with_remote_http(audio_path, vibevoice_config.get("remote_urls")),
        )
        fast_is_weak = _is_weak_fast_transcript(result.fast_transcript, _wav_duration(audio_path))
        if _should_run_deep_asr(audio_path, result.fast_transcript):
            result.deep_transcript = _timed_transcribe(
                result,
                "vibevoice",
                lambda: transcribe_with_vibevoice(audio_path, vibevoice_config),
            )
        else:
            result.merge_notes.append("balanced skipped VibeVoice: fast transcript was sufficient for this audio length")
        needs_fallback = (
            not _has_transcript_text(result.fast_transcript)
            or (fast_is_weak and not _has_transcript_text(result.deep_transcript))
        )
        if needs_fallback:
            result.fast_transcript = _timed_transcribe(
                result,
                "capswriter_http",
                lambda: transcribe_with_capswriter(audio_path),
            )
            fast_is_weak = _is_weak_fast_transcript(result.fast_transcript, _wav_duration(audio_path))
        needs_fallback = (
            not _has_transcript_text(result.fast_transcript)
            or (fast_is_weak and not _has_transcript_text(result.deep_transcript))
        )
        if needs_fallback:
            result.fast_transcript = _timed_transcribe(
                result,
                "faster_whisper",
                lambda: AudioProcessor(language=language, model_size_or_path=whisper_model, device=device).transcribe(audio_path),
            )
    else:
        raise ValueError(f"Unknown ASR strategy: {strategy}")

    result.transcript = merge_asr_transcripts(result.fast_transcript, result.deep_transcript)
    if result.deep_transcript and result.fast_transcript:
        result.merge_notes.append("merged VibeVoice long-context text with remoteHTTP timestamp anchors")
    elif result.deep_transcript:
        result.merge_notes.append("used VibeVoice transcript without timestamp anchors")
    elif result.fast_transcript:
        result.merge_notes.append("used fast ASR transcript")
    else:
        result.failures.append("no ASR provider produced transcript text")
    return result


def merge_asr_transcripts(
    fast_transcript: Optional[AudioTranscript],
    deep_transcript: Optional[AudioTranscript],
) -> Optional[AudioTranscript]:
    if not _has_transcript_text(fast_transcript) and not _has_transcript_text(deep_transcript):
        return None
    if _has_transcript_text(fast_transcript) and not _has_transcript_text(deep_transcript):
        return fast_transcript
    if _has_transcript_text(deep_transcript) and not _has_transcript_text(fast_transcript):
        return deep_transcript

    assert fast_transcript is not None
    assert deep_transcript is not None
    merged_segments = _align_deep_text_to_fast_segments(fast_transcript, deep_transcript.text)
    return AudioTranscript(
        text=deep_transcript.text.strip() or fast_transcript.text,
        segments=merged_segments,
        language=deep_transcript.language if deep_transcript.language != "unknown" else fast_transcript.language,
    )


def _align_deep_text_to_fast_segments(
    fast_transcript: AudioTranscript,
    deep_text: str,
) -> List[Dict[str, object]]:
    fast_segments = fast_transcript.segments or []
    if not fast_segments:
        return [{"text": deep_text, "source": "vibevoice"}]

    deep_sentences = _split_deep_text(deep_text)
    if not deep_sentences:
        deep_sentences = [deep_text]

    aligned = []
    sentence_count = len(deep_sentences)
    segment_count = len(fast_segments)
    sentence_assignments = _assign_deep_sentences(sentence_count, segment_count)
    for index, segment in enumerate(fast_segments):
        assigned_indexes = sentence_assignments.get(index, [])
        deep_slice = " ".join(deep_sentences[sentence_index] for sentence_index in assigned_indexes).strip()
        merged_segment = dict(segment)
        merged_segment["fast_text"] = segment.get("text", "")
        merged_segment["deep_text"] = deep_slice
        merged_segment["text"] = deep_slice or segment.get("text", "")
        merged_segment["source"] = "merged_remote_http_vibevoice"
        aligned.append(merged_segment)
    return aligned


def _assign_deep_sentences(sentence_count: int, segment_count: int) -> Dict[int, List[int]]:
    assignments: Dict[int, List[int]] = {index: [] for index in range(segment_count)}
    if sentence_count >= segment_count:
        for segment_index in range(segment_count):
            start = int(segment_index * sentence_count / segment_count)
            end = int((segment_index + 1) * sentence_count / segment_count)
            if segment_index == segment_count - 1:
                end = sentence_count
            assignments[segment_index] = list(range(start, max(end, start + 1)))
        return assignments

    if sentence_count == 1:
        assignments[0] = [0]
        return assignments

    for sentence_index in range(sentence_count):
        segment_index = round(sentence_index * (segment_count - 1) / (sentence_count - 1))
        assignments[segment_index].append(sentence_index)
    return assignments


def _split_deep_text(text: str) -> List[str]:
    import re

    pattern = r"[^。！？.!?\n]+[。！？.!?]?|[^\n]+"
    return [part.strip() for part in re.findall(pattern, text.strip()) if part.strip()]


def _should_run_deep_asr(audio_path: Path, fast_transcript: Optional[AudioTranscript]) -> bool:
    if not _has_transcript_text(fast_transcript):
        return True
    duration = _wav_duration(audio_path)
    if duration and duration >= DEEP_ASR_MIN_SECONDS:
        return True
    if _is_weak_fast_transcript(fast_transcript, duration):
        return True
    segments = fast_transcript.segments or []
    if duration and duration >= 90 and segments and len(segments) < max(2, int(duration / 90)):
        return True
    return False


def _is_weak_fast_transcript(transcript: Optional[AudioTranscript], duration: float) -> bool:
    if not _has_transcript_text(transcript):
        return True
    text = (transcript.text or "").strip()
    meaningful_chars = [char for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff"]
    if len(meaningful_chars) < 8:
        return True
    if duration and duration >= 30 and len(meaningful_chars) < duration * 0.25:
        return True
    normalized = "".join(meaningful_chars)
    if normalized and len(set(normalized)) <= 2 and len(normalized) >= 8:
        return True
    filler_tokens = {"嗯", "啊", "呃", "哦", "um", "uh", "嗯嗯"}
    compact = text.replace(" ", "").lower()
    if compact in filler_tokens:
        return True
    return False


def _wav_duration(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / rate if rate else 0.0
    except Exception:
        return 0.0


def _timed_transcribe(result: ASRStrategyResult, provider: str, callback) -> Optional[AudioTranscript]:
    start = time.monotonic()
    result.providers_run.append(provider)
    transcript = callback()
    result.elapsed_seconds[provider] = round(time.monotonic() - start, 3)
    if not _has_transcript_text(transcript):
        result.failures.append(f"{provider} produced no transcript")
    return transcript


def _transcript_summary(transcript: Optional[AudioTranscript]) -> Optional[Dict[str, object]]:
    if not transcript:
        return None
    text = transcript.text or ""
    return {
        "language": transcript.language,
        "text_preview": text[:500],
        "text_length": len(text),
        "segment_count": len(transcript.segments or []),
    }


def _has_transcript_text(transcript: Optional[AudioTranscript]) -> bool:
    return bool(transcript and transcript.text and transcript.text.strip())
