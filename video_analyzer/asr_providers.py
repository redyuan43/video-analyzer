import logging
import subprocess
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    AudioSegment = None

from .audio_processor import AudioProcessor, AudioTranscript

logger = logging.getLogger(__name__)

CAPSWRITER_URL = "http://spark-31d6.taild500c8.ts.net:8001/api/asr/transcribe"
REMOTE_VIBEVOICE_URLS = [
    "http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe",
]
REMOTE_ASR_URLS = [
    "http://spark-31d6.taild500c8.ts.net:8001/api/asr/transcribe",
]

DEEP_ASR_MIN_SECONDS = 180.0
VIBEVOICE_DISTRIBUTED_MIN_SECONDS = 420.0


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


def transcribe_with_http_asr(
    audio_path: Path,
    url: str,
    hotword: str = "",
    extra_data: Optional[Dict[str, object]] = None,
) -> Optional[AudioTranscript]:
    try:
        form_data = {"hotword": hotword}
        if extra_data:
            form_data.update({key: str(value) for key, value in extra_data.items() if value is not None})
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                url,
                files={"audio": (audio_path.name, audio_file, "audio/wav")},
                data=form_data,
                timeout=(30, 900),
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


def transcribe_with_vibevoice_remote(
    audio_path: Path,
    urls: Optional[list[str]] = None,
    options: Optional[Dict[str, object]] = None,
) -> Optional[AudioTranscript]:
    urls = [url for url in (REMOTE_VIBEVOICE_URLS if urls is None else urls) if url]
    if not urls:
        logger.warning("No remote GPU VibeVoice endpoint is configured")
        return None
    options = options or {}
    duration = _wav_duration(audio_path)
    distributed_min_seconds = float(options.get("distributed_min_seconds") or VIBEVOICE_DISTRIBUTED_MIN_SECONDS)
    if len(urls) > 1 and duration >= distributed_min_seconds:
        transcript = transcribe_with_vibevoice_distributed(audio_path, urls, options)
        if transcript and transcript.text.strip():
            return transcript
    for url in urls:
        transcript = transcribe_with_http_asr(audio_path, url, extra_data=options)
        if transcript and transcript.text.strip():
            transcript.segments = transcript.segments or []
            transcript.segments.append({"provider": "vibevoice_remote", "provider_url": url})
            return transcript
    return None


def transcribe_with_vibevoice_distributed(
    audio_path: Path,
    urls: list[str],
    options: Optional[Dict[str, object]] = None,
) -> Optional[AudioTranscript]:
    """Split long audio across remote VibeVoice workers on different machines."""
    options = options or {}
    duration = _wav_duration(audio_path)
    if not duration:
        return None
    worker_count = min(len(urls), max(1, int(options.get("distributed_workers") or len(urls))))
    active_urls = urls[:worker_count]
    logger.info("Running distributed VibeVoice ASR across %d endpoints", len(active_urls))
    with tempfile.TemporaryDirectory(prefix="vibevoice_distributed_") as temp_dir:
        chunks = _split_audio_evenly(audio_path, Path(temp_dir), worker_count, duration)
        if not chunks:
            return None
        results: list[Tuple[int, float, Optional[AudioTranscript], str]] = []
        request_options = dict(options)
        request_options["use_native_chunking"] = False
        request_options.pop("distributed_min_seconds", None)
        request_options.pop("distributed_workers", None)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _transcribe_vibevoice_worker,
                    chunk_path,
                    active_urls[index],
                    start_seconds,
                    request_options,
                ): (index, start_seconds, active_urls[index])
                for index, (chunk_path, start_seconds) in enumerate(chunks)
            }
            for future in as_completed(futures):
                index, start_seconds, url = futures[future]
                try:
                    results.append((index, start_seconds, future.result(), url))
                except Exception as exc:
                    logger.warning("Distributed VibeVoice worker failed for %s: %s", url, exc)
                    results.append((index, start_seconds, None, url))
        return _merge_distributed_vibevoice_results(results)


def transcribe_with_vibevoice(audio_path: Path, config: Optional[Dict[str, str]] = None) -> Optional[AudioTranscript]:
    config = config or {}
    remote_urls = config["deep_remote_urls"] if "deep_remote_urls" in config else None
    options = {
        "use_native_chunking": config.get("use_native_chunking", True),
        "distributed_min_seconds": config.get("distributed_min_seconds", VIBEVOICE_DISTRIBUTED_MIN_SECONDS),
        "distributed_workers": config.get("distributed_workers"),
    }
    return transcribe_with_vibevoice_remote(audio_path, remote_urls, options=options)


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
    providers = ["vibevoice", "remote_http", "capswriter_http", "faster_whisper"] if provider == "auto" else [provider]
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
        if not _has_transcript_text(result.deep_transcript) and fast_is_weak:
            result.merge_notes.append("balanced did not fallback outside Spark ASR; investigate configured Spark endpoints")
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


def _split_audio_evenly(audio_path: Path, output_dir: Path, parts: int, duration: float) -> List[Tuple[Path, float]]:
    chunks: List[Tuple[Path, float]] = []
    for index in range(parts):
        start = duration * index / parts
        end = duration * (index + 1) / parts
        chunk_path = output_dir / f"chunk_{index:03d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-to",
                f"{end:.3f}",
                "-i",
                str(audio_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(chunk_path),
            ],
            check=True,
            capture_output=True,
        )
        chunks.append((chunk_path, start))
    return chunks


def _transcribe_vibevoice_worker(
    chunk_path: Path,
    url: str,
    start_seconds: float,
    options: Dict[str, object],
) -> Optional[AudioTranscript]:
    transcript = transcribe_with_http_asr(chunk_path, url, extra_data=options)
    if not transcript:
        return None
    shifted_segments = []
    for segment in transcript.segments or []:
        shifted = dict(segment)
        _shift_segment_time(shifted, "start", start_seconds)
        _shift_segment_time(shifted, "end", start_seconds)
        _shift_segment_time(shifted, "start_time", start_seconds)
        _shift_segment_time(shifted, "end_time", start_seconds)
        shifted["provider_url"] = url
        shifted["chunk_offset_seconds"] = start_seconds
        shifted_segments.append(shifted)
    if not shifted_segments:
        shifted_segments = [{"provider_url": url, "chunk_offset_seconds": start_seconds, "text": transcript.text}]
    return AudioTranscript(text=transcript.text, segments=shifted_segments, language=transcript.language)


def _merge_distributed_vibevoice_results(
    results: list[Tuple[int, float, Optional[AudioTranscript], str]],
) -> Optional[AudioTranscript]:
    successful = [(index, start, transcript, url) for index, start, transcript, url in results if _has_transcript_text(transcript)]
    if not successful:
        return None
    successful.sort(key=lambda item: item[0])
    text = "\n".join(transcript.text.strip() for _index, _start, transcript, _url in successful if transcript and transcript.text).strip()
    segments: List[Dict[str, object]] = []
    for index, start, transcript, url in successful:
        for segment in transcript.segments or []:
            item = dict(segment)
            item.setdefault("provider", "vibevoice_remote_distributed")
            item.setdefault("provider_url", url)
            item.setdefault("chunk_index", index)
            item.setdefault("chunk_offset_seconds", start)
            segments.append(item)
    segments.append(
        {
            "provider": "vibevoice_remote_distributed",
            "provider_urls": [url for _index, _start, _transcript, url in successful],
            "chunk_count": len(successful),
        }
    )
    language = next((transcript.language for _index, _start, transcript, _url in successful if transcript and transcript.language), "unknown")
    return AudioTranscript(text=text, segments=segments, language=language)


def _shift_segment_time(segment: Dict[str, object], key: str, offset: float) -> None:
    if key not in segment or segment[key] is None:
        return
    try:
        segment[key] = float(segment[key]) + offset
    except (TypeError, ValueError):
        return


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
