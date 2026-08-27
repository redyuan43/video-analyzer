"""Tencent Hy-ASR 3.0 Preview WebSocket client."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import threading
import time
import uuid
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from websockets.sync.client import connect

from .audio_processor import AudioTranscript

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "wss://asr.cloud.tencent.com/asr/v2"
DEFAULT_ENGINE = "Hy-ASR-3.0-preview"
DEFAULT_ENV_FILE = Path("~/.config/video-analyzer/tencentcloud.env").expanduser()
DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_PARALLEL_CHUNKS = 6
MAX_PREVIEW_PARALLEL_CHUNKS = 6
PCM_SAMPLE_RATE = 16000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1
PCM_PACKET_BYTES = 6400


def transcribe_with_tencent_hy_asr(
    audio_path: Path,
    endpoint: str = DEFAULT_ENDPOINT,
    options: dict[str, object] | None = None,
) -> AudioTranscript:
    options = dict(options or {})
    credentials = resolve_tencent_credentials(options)
    chunks = read_pcm_chunks(
        audio_path,
        float(options.get("chunk_duration_sec") or DEFAULT_CHUNK_SECONDS),
    )
    if not chunks:
        return AudioTranscript(
            text="",
            segments=[],
            language="unknown",
            metadata={"provider": "tencent_hy_asr_3_preview", "chunk_count": 0},
        )

    parallel_chunks = min(
        len(chunks),
        MAX_PREVIEW_PARALLEL_CHUNKS,
        max(1, int(options.get("parallel_chunks") or DEFAULT_PARALLEL_CHUNKS)),
    )
    results: list[tuple[int, float, list[dict[str, Any]], list[dict[str, Any]]]] = []
    executor = ThreadPoolExecutor(max_workers=parallel_chunks)
    failed = False
    try:
        futures = {
            executor.submit(
                transcribe_pcm_chunk,
                pcm_data,
                endpoint,
                credentials,
                options,
            ): (index, offset)
            for index, offset, pcm_data in chunks
        }
        for future in as_completed(futures):
            index, offset = futures[future]
            try:
                sentences, responses = future.result()
            except Exception as exc:
                failed = True
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(
                    f"Tencent Hy-ASR chunk {index} at {offset:.3f}s failed: {exc}"
                ) from exc
            results.append((index, offset, sentences, responses))
    finally:
        if not failed:
            executor.shutdown(wait=True)

    results.sort(key=lambda item: item[0])
    segments: list[dict[str, Any]] = []
    chunk_metadata: list[dict[str, Any]] = []
    for index, offset, sentences, responses in results:
        for sentence in sentences:
            segment = dict(sentence)
            segment["start"] = offset + float(segment.get("start") or 0.0)
            segment["end"] = offset + float(segment.get("end") or 0.0)
            segment["chunk_index"] = index
            segment["chunk_offset_seconds"] = offset
            segment["provider"] = "tencent_hy_asr_3_preview"
            segments.append(segment)
        chunk_metadata.append(
            {
                "chunk_index": index,
                "chunk_offset_seconds": offset,
                "response_count": len(responses),
                "sentence_count": len(sentences),
            }
        )

    text = "\n".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if str(segment.get("text") or "").strip()
    )
    return AudioTranscript(
        text=text,
        segments=segments,
        language="zh",
        metadata={
            "provider": "tencent_hy_asr_3_preview",
            "engine_model_type": str(options.get("engine_model_type") or DEFAULT_ENGINE),
            "endpoint": endpoint,
            "chunk_count": len(chunks),
            "parallel_chunks": parallel_chunks,
            "chunks": chunk_metadata,
        },
    )


def resolve_tencent_credentials(options: dict[str, object]) -> dict[str, str]:
    load_tencent_env_file(options)
    env_names = tencent_credential_env_names(options)
    missing = [env_name for env_name in env_names.values() if not os.environ.get(env_name)]
    if missing:
        raise RuntimeError(
            "Tencent Hy-ASR credentials are missing: "
            + ", ".join(missing)
            + f"; configure {tencent_env_path(options)}"
        )
    return {name: os.environ[env_name] for name, env_name in env_names.items()}


def tencent_credential_env_names(options: dict[str, object]) -> dict[str, str]:
    return {
        "app_id": str(options.get("app_id_env") or "TENCENTCLOUD_APP_ID"),
        "secret_id": str(options.get("secret_id_env") or "TENCENTCLOUD_SECRET_ID"),
        "secret_key": str(options.get("secret_key_env") or "TENCENTCLOUD_SECRET_KEY"),
    }


def missing_tencent_credentials(options: dict[str, object]) -> list[str]:
    load_tencent_env_file(options)
    return [
        env_name
        for env_name in tencent_credential_env_names(options).values()
        if not os.environ.get(env_name)
    ]


def load_tencent_env_file(options: dict[str, object]) -> bool:
    path = tencent_env_path(options)
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.split(None, 1)[1].strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")
    return True


def tencent_env_path(options: dict[str, object]) -> Path:
    return Path(
        str(
            os.environ.get("VIDEO_ANALYZER_TENCENTCLOUD_ENV")
            or options.get("env_file")
            or DEFAULT_ENV_FILE
        )
    ).expanduser()


def read_pcm_chunks(
    audio_path: Path,
    chunk_duration_seconds: float,
) -> list[tuple[int, float, bytes]]:
    chunk_duration_seconds = min(max(chunk_duration_seconds, 1.0), 59.0)
    with wave.open(str(audio_path), "rb") as wav_file:
        if (
            wav_file.getframerate() != PCM_SAMPLE_RATE
            or wav_file.getsampwidth() != PCM_SAMPLE_WIDTH
            or wav_file.getnchannels() != PCM_CHANNELS
            or wav_file.getcomptype() != "NONE"
        ):
            raise ValueError("Tencent Hy-ASR requires 16kHz mono 16-bit PCM WAV input")
        frames_per_chunk = max(1, int(PCM_SAMPLE_RATE * chunk_duration_seconds))
        chunks: list[tuple[int, float, bytes]] = []
        index = 0
        while True:
            pcm_data = wav_file.readframes(frames_per_chunk)
            if not pcm_data:
                break
            chunks.append((index, index * chunk_duration_seconds, pcm_data))
            index += 1
        return chunks


def transcribe_pcm_chunk(
    pcm_data: bytes,
    endpoint: str,
    credentials: dict[str, str],
    options: dict[str, object],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts = max(1, int(options.get("max_attempts") or 2))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return transcribe_pcm_chunk_once(pcm_data, endpoint, credentials, options)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(float(options.get("retry_delay_seconds") or 2.0))

    gain = float(options.get("decode_error_gain_fallback") or 0.7)
    if (
        last_error is not None
        and "Tencent Hy-ASR error 4007" in str(last_error)
        and 0.0 < gain < 1.0
    ):
        logger.warning(
            "Tencent Hy-ASR returned 4007 after %d attempt(s); retrying this chunk with PCM gain %.3f",
            attempts,
            gain,
        )
        return transcribe_pcm_chunk_once(
            scale_pcm_s16le(pcm_data, gain),
            endpoint,
            credentials,
            options,
        )

    if last_error is not None:
        raise last_error
    raise RuntimeError("Tencent Hy-ASR chunk exhausted retries")


def scale_pcm_s16le(pcm_data: bytes, gain: float) -> bytes:
    if not 0.0 < gain <= 1.0:
        raise ValueError("PCM gain must be greater than 0 and at most 1")
    samples = array("h")
    samples.frombytes(pcm_data)
    if sys.byteorder != "little":
        samples.byteswap()
    for index, sample in enumerate(samples):
        samples[index] = int(sample * gain)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def transcribe_pcm_chunk_once(
    pcm_data: bytes,
    endpoint: str,
    credentials: dict[str, str],
    options: dict[str, object],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    request_url = build_signed_url(endpoint, credentials, options)
    responses: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    receiver_error: list[Exception] = []
    completed = threading.Event()
    handshake_completed = threading.Event()
    receive_timeout = float(options.get("receive_timeout_seconds") or 90.0)
    handshake_timeout = float(options.get("handshake_timeout_seconds") or 15.0)

    with connect(
        request_url,
        open_timeout=float(options.get("connect_timeout_seconds") or 15.0),
        close_timeout=10,
        ping_interval=20,
        ping_timeout=20,
        max_size=4 * 1024 * 1024,
    ) as websocket:
        def receive_messages() -> None:
            try:
                while not completed.is_set():
                    message = websocket.recv(timeout=receive_timeout)
                    payload = json.loads(message)
                    responses.append(payload)
                    if int(payload.get("code") or 0) != 0:
                        raise RuntimeError(
                            f"Tencent Hy-ASR error {payload.get('code')}: {payload.get('message')}"
                        )
                    handshake_completed.set()
                    result = payload.get("result") or {}
                    if int(result.get("slice_type", -1)) == 2:
                        sentences.append(
                            {
                                "start": float(result.get("start_time") or 0) / 1000.0,
                                "end": float(result.get("end_time") or 0) / 1000.0,
                                "text": str(result.get("voice_text_str") or "").strip(),
                                "provider_index": result.get("index"),
                            }
                        )
                    if int(payload.get("final") or 0) == 1:
                        completed.set()
            except Exception as exc:
                receiver_error.append(exc)
                handshake_completed.set()
                completed.set()

        receiver = threading.Thread(target=receive_messages, daemon=True)
        receiver.start()
        if not handshake_completed.wait(timeout=handshake_timeout):
            raise TimeoutError("Tencent Hy-ASR handshake response timed out")
        if receiver_error:
            raise receiver_error[0]
        packet_interval = 0.2 / max(
            0.1,
            float(options.get("send_realtime_factor") or 1.0),
        )
        for offset in range(0, len(pcm_data), PCM_PACKET_BYTES):
            if receiver_error:
                break
            websocket.send(pcm_data[offset : offset + PCM_PACKET_BYTES])
            time.sleep(packet_interval)
        if not receiver_error:
            websocket.send(json.dumps({"type": "end"}))
        receiver.join(timeout=receive_timeout)
        if receiver.is_alive():
            raise TimeoutError("Tencent Hy-ASR final response timed out")
        if receiver_error:
            raise receiver_error[0]
    return sentences, responses


def build_signed_url(
    endpoint: str,
    credentials: dict[str, str],
    options: dict[str, object],
) -> str:
    parsed = urlparse(endpoint or DEFAULT_ENDPOINT)
    if parsed.scheme != "wss" or not parsed.hostname:
        raise ValueError("Tencent Hy-ASR endpoint must be a wss URL")
    timestamp = int(time.time())
    app_id = credentials["app_id"]
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/{app_id}"
    params = {
        "convert_num_mode": int(options.get("convert_num_mode") or 1),
        "engine_model_type": str(options.get("engine_model_type") or DEFAULT_ENGINE),
        "expired": timestamp + 24 * 60 * 60,
        "filter_dirty": int(options.get("filter_dirty") or 0),
        "filter_modal": int(options.get("filter_modal") or 0),
        "filter_punc": int(options.get("filter_punc") or 0),
        "needvad": 0,
        "nonce": int(uuid.uuid4().int % 10_000_000_000),
        "secretid": credentials["secret_id"],
        "sub_service_type": 1,
        "timestamp": timestamp,
        "voice_format": 1,
        "voice_id": str(uuid.uuid4()),
        "word_info": 0,
    }
    sorted_params = sorted(params.items())
    sign_query = "&".join(f"{key}={value}" for key, value in sorted_params)
    sign_source = f"{parsed.hostname}{path}?{sign_query}"
    signature = base64.b64encode(
        hmac.new(
            credentials["secret_key"].encode("utf-8"),
            sign_source.encode("utf-8"),
            hashlib.sha1,
        ).digest()
    ).decode("ascii")
    request_query = "&".join(
        f"{key}={quote(str(value), safe='')}" for key, value in sorted_params
    )
    return f"wss://{parsed.hostname}{path}?{request_query}&signature={quote(signature, safe='')}"
