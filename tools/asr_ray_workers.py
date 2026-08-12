"""Shared audio planning, Ray dispatch, and result merging for local ASR pools."""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    import ray
except ImportError:  # pragma: no cover - service startup validates the runtime.
    ray = None

RAY_RUNTIME_LOCK = threading.Lock()


@dataclass(frozen=True)
class AsrChunk:
    index: int
    path: Path
    start: float
    length: float


@dataclass
class ChunkAttempt:
    chunk: AsrChunk
    attempt: int = 1
    avoid_endpoint: str = ""
    first_error: str = ""


@dataclass(frozen=True)
class ChunkResult:
    chunk: AsrChunk
    payload: dict[str, Any]
    endpoint: str
    elapsed_seconds: float
    attempt: int
    first_error: str = ""


class AsrChunkError(RuntimeError):
    def __init__(self, attempt: ChunkAttempt, error: Exception) -> None:
        chunk = attempt.chunk
        first_error = attempt.first_error or str(error)
        super().__init__(
            "ASR chunk failed after retry "
            f"index={chunk.index} range={chunk.start:.3f}-{chunk.start + chunk.length:.3f}s "
            f"first_error={first_error} last_error={error}"
        )
        self.chunk = chunk
        self.first_error = first_error
        self.last_error = str(error)


def _require_ray() -> Any:
    if ray is None:
        raise RuntimeError("Ray is required for multi-worker ASR dispatch")
    return ray


if ray is not None:

    @ray.remote(num_cpus=1)
    class HttpAsrWorker:
        def __init__(self, endpoint: str, request_timeout: float) -> None:
            self.endpoint = endpoint
            self.request_timeout = request_timeout

        def ready(self) -> str:
            return self.endpoint

        def transcribe(
            self,
            item: tuple[int, str, float, float],
            form: dict[str, str],
        ) -> tuple[int, float, float, dict[str, Any], str, float]:
            index, path_text, start, length = item
            started = time.monotonic()
            payload = post_audio(
                self.endpoint,
                Path(path_text),
                form,
                request_timeout=self.request_timeout,
            )
            return (
                index,
                start,
                length,
                payload,
                self.endpoint,
                time.monotonic() - started,
            )


def audio_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(completed.stdout.strip())


def normalize_audio(path: Path, output: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output),
        ],
        check=True,
        timeout=300,
    )
    return output


def request_float(
    form: dict[str, str],
    key: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> float:
    raw = form.pop(key, "")
    if raw == "":
        return default
    value = float(raw)
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def request_choice(
    form: dict[str, str],
    key: str,
    default: str,
    choices: set[str],
) -> str:
    value = str(form.pop(key, "") or default).strip().lower()
    if value not in choices:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(choices))}")
    return value


def materialize_fixed_chunks(
    path: Path,
    directory: Path,
    duration: float,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[AsrChunk]:
    if overlap_seconds >= chunk_seconds:
        raise ValueError("chunk_overlap_sec must be smaller than chunk_duration_sec")
    step = chunk_seconds - overlap_seconds
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        length = min(chunk_seconds, duration - start)
        if spans and length <= overlap_seconds:
            break
        spans.append((start, start + length))
        start += step
    return materialize_segment_chunks(path, directory, spans, hard_limit_seconds=chunk_seconds)


def materialize_segment_chunks(
    path: Path,
    directory: Path,
    segments: Iterable[tuple[float, float]],
    *,
    hard_limit_seconds: float,
) -> list[AsrChunk]:
    chunks: list[AsrChunk] = []
    for raw_start, raw_end in segments:
        start = max(0.0, float(raw_start))
        end = max(start, float(raw_end))
        while end - start > 0.001:
            chunk_end = min(end, start + hard_limit_seconds)
            output = directory / f"chunk-{len(chunks):04d}.wav"
            _extract_audio_segment(path, output, start, chunk_end - start)
            chunks.append(
                AsrChunk(
                    index=len(chunks),
                    path=output,
                    start=start,
                    length=chunk_end - start,
                )
            )
            start = chunk_end
    return chunks


def _extract_audio_segment(path: Path, output: Path, start: float, length: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{length:.3f}",
            "-i",
            str(path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output),
        ],
        check=True,
        timeout=180,
    )


def post_audio(
    url: str,
    path: Path,
    form: dict[str, str],
    *,
    request_timeout: float,
) -> dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    try:
        with path.open("rb") as audio:
            response = session.post(
                url,
                files={"audio": (path.name, audio, "audio/wav")},
                data=form,
                timeout=(30, request_timeout),
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("success") is False or payload.get("error"):
            raise RuntimeError(str(payload.get("error") or payload))
        return payload
    finally:
        session.close()


def dispatch_asr_chunks(
    endpoints: list[str],
    chunks: list[AsrChunk],
    form: dict[str, str],
    *,
    request_timeout: float,
    max_attempts: int = 2,
) -> list[ChunkResult]:
    """Dynamically schedule chunks, retrying a failed chunk on another endpoint."""
    if not endpoints:
        raise ValueError("at least one ASR endpoint is required")
    if not chunks:
        return []
    attempts = max(1, int(max_attempts))
    if len(chunks) == 1:
        first_error = ""
        for attempt_index in range(attempts):
            endpoint = endpoints[attempt_index % len(endpoints)]
            started = time.monotonic()
            try:
                payload = post_audio(
                    endpoint,
                    chunks[0].path,
                    form,
                    request_timeout=request_timeout,
                )
            except Exception as exc:
                first_error = first_error or str(exc)
                if attempt_index + 1 >= attempts:
                    raise AsrChunkError(
                        ChunkAttempt(
                            chunk=chunks[0],
                            attempt=attempt_index + 1,
                            first_error=first_error,
                        ),
                        exc,
                    ) from exc
                continue
            return [
                ChunkResult(
                    chunk=chunks[0],
                    payload=payload,
                    endpoint=endpoint,
                    elapsed_seconds=time.monotonic() - started,
                    attempt=attempt_index + 1,
                    first_error=first_error,
                )
            ]
        raise RuntimeError("single ASR chunk exhausted retries")
    runtime = _require_ray()
    with RAY_RUNTIME_LOCK:
        runtime_started_here = False
        actors: list[tuple[Any, str]] = []
        try:
            if not runtime.is_initialized():
                runtime.init(
                    namespace="video-analyzer-asr",
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    num_cpus=len(endpoints),
                )
                runtime_started_here = True

            actors = [
                (HttpAsrWorker.remote(endpoint, request_timeout), endpoint)
                for endpoint in endpoints
            ]
            pending = deque(ChunkAttempt(chunk=chunk) for chunk in chunks)
            available = deque(actors)
            in_flight: dict[Any, tuple[Any, str, ChunkAttempt]] = {}
            results: list[ChunkResult] = []

            def take_for(endpoint: str) -> ChunkAttempt | None:
                if not pending:
                    return None
                if len(endpoints) == 1:
                    return pending.popleft()
                for _ in range(len(pending)):
                    candidate = pending.popleft()
                    if candidate.avoid_endpoint != endpoint:
                        return candidate
                    pending.append(candidate)
                return None

            def schedule() -> None:
                waiting = deque()
                while available:
                    actor, endpoint = available.popleft()
                    attempt = take_for(endpoint)
                    if attempt is None:
                        waiting.append((actor, endpoint))
                        continue
                    chunk = attempt.chunk
                    ref = actor.transcribe.remote(
                        (chunk.index, str(chunk.path), chunk.start, chunk.length),
                        form,
                    )
                    in_flight[ref] = (actor, endpoint, attempt)
                available.extend(waiting)

            runtime.get(
                [actor.ready.remote() for actor, _endpoint in actors],
                timeout=request_timeout,
            )
            schedule()
            while in_flight or pending:
                if not in_flight:
                    raise RuntimeError("ASR Ray scheduler could not assign pending chunks")
                ready, _ = runtime.wait(
                    list(in_flight),
                    num_returns=1,
                    timeout=request_timeout,
                )
                if not ready:
                    raise TimeoutError("ASR Ray worker timed out")
                for task_ref in ready:
                    actor, endpoint, attempt = in_flight.pop(task_ref)
                    available.append((actor, endpoint))
                    try:
                        index, start, length, payload, used_endpoint, elapsed = runtime.get(
                            task_ref
                        )
                        results.append(
                            ChunkResult(
                                chunk=AsrChunk(index, attempt.chunk.path, start, length),
                                payload=payload,
                                endpoint=used_endpoint,
                                elapsed_seconds=float(elapsed),
                                attempt=attempt.attempt,
                                first_error=attempt.first_error,
                            )
                        )
                    except Exception as exc:
                        if attempt.attempt >= attempts:
                            raise AsrChunkError(attempt, exc) from exc
                        pending.appendleft(
                            ChunkAttempt(
                                chunk=attempt.chunk,
                                attempt=attempt.attempt + 1,
                                avoid_endpoint=endpoint,
                                first_error=attempt.first_error or str(exc),
                            )
                        )
                schedule()
            return sorted(results, key=lambda item: item.chunk.index)
        finally:
            for actor, _endpoint in actors:
                with suppress(Exception):
                    runtime.kill(actor, no_restart=True)
            if runtime_started_here:
                with suppress(Exception):
                    runtime.shutdown()


def merge_asr_results(
    provider: str,
    results: list[ChunkResult],
    *,
    segmentation_mode: str,
    audio_duration_seconds: float,
    worker_count: int,
    segmentation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = ""
    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []
    language = "unknown"
    for result in sorted(results, key=lambda item: item.chunk.index):
        chunk = result.chunk
        payload = result.payload
        chunk_text = str(payload.get("text") or "").strip()
        text = (
            dedupe_join(text, chunk_text)
            if segmentation_mode == "fixed"
            else join_non_overlapping(text, chunk_text)
        )
        segments.extend(offset_segments(payload, chunk.start, chunk.length))
        words.extend(offset_words(payload, chunk.start))
        language = str(payload.get("language") or language)
        chunk_results.append(
            {
                "index": chunk.index,
                "start": round(chunk.start, 3),
                "end": round(chunk.start + chunk.length, 3),
                "length": round(chunk.length, 3),
                "endpoint": result.endpoint,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "attempt": result.attempt,
                **(
                    {"retry_error": result.first_error}
                    if result.first_error
                    else {}
                ),
                "confidence": _payload_confidence(payload),
                "text_length": len(chunk_text),
            }
        )
    speech_duration = sum(item.chunk.length for item in results)
    metadata = dict(segmentation_metadata or {})
    success = bool(text.strip())
    segments = _clamp_and_sort_timestamps(segments, audio_duration_seconds)
    words = _clamp_and_sort_timestamps(words, audio_duration_seconds)
    return {
        "success": success,
        **({"error": "no_transcript_text"} if not success else {}),
        "provider": provider,
        "text": text,
        "segments": segments,
        "words": words,
        "language": language,
        "dispatch_mode": "ray" if len(results) > 1 else "direct",
        "segmentation_mode": segmentation_mode,
        "audio_duration_seconds": round(audio_duration_seconds, 3),
        "speech_duration_seconds": round(speech_duration, 3),
        "worker_count": worker_count,
        "chunk_count": len(results),
        "chunk_results": chunk_results,
        **metadata,
    }


def dedupe_join(left: str, right: str) -> str:
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    limit = min(120, len(left), len(right))
    for size in range(limit, 3, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return f"{left}\n{right}"


def join_non_overlapping(left: str, right: str) -> str:
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n{right}"


def offset_segments(payload: dict[str, Any], start: float, duration: float) -> list[dict[str, Any]]:
    segments = []
    for source in payload.get("segments") or []:
        if not isinstance(source, dict):
            continue
        item = dict(source)
        if item.get("start") is not None:
            item["start"] = round(float(item["start"]) + start, 3)
        if item.get("end") is not None:
            item["end"] = round(float(item["end"]) + start, 3)
        segments.append(item)
    if not segments and str(payload.get("text") or "").strip():
        segments.append(
            {
                "start": round(start, 3),
                "end": round(start + duration, 3),
                "text": str(payload.get("text") or "").strip(),
            }
        )
    return segments


def offset_words(payload: dict[str, Any], start: float) -> list[dict[str, Any]]:
    words = []
    for source in payload.get("words") or []:
        if not isinstance(source, dict):
            continue
        item = dict(source)
        item["start"] = round(float(item.get("start") or 0) + start, 3)
        item["end"] = round(float(item.get("end") or 0) + start, 3)
        words.append(item)
    return words


def _payload_confidence(payload: dict[str, Any]) -> float | None:
    for source in payload.get("segments") or []:
        if isinstance(source, dict) and source.get("confidence") is not None:
            try:
                return float(source["confidence"])
            except (TypeError, ValueError):
                return None
    if payload.get("confidence") is not None:
        try:
            return float(payload["confidence"])
        except (TypeError, ValueError):
            return None
    return None


def _clamp_and_sort_timestamps(
    items: list[dict[str, Any]],
    duration: float,
) -> list[dict[str, Any]]:
    bounded = []
    for source in items:
        item = dict(source)
        start = max(0.0, min(float(item.get("start") or 0), duration))
        end = max(start, min(float(item.get("end") or start), duration))
        item["start"] = round(start, 3)
        item["end"] = round(end, 3)
        bounded.append(item)
    return sorted(bounded, key=lambda item: (item["start"], item["end"]))
