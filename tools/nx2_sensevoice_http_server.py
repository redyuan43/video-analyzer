#!/usr/bin/env python3
"""Serve SenseVoice/FunASR through the analyzer's multipart ASR contract."""

from __future__ import annotations

import argparse
import cgi
import gc
import json
import logging
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18013)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vad-model", type=Path, required=True)
    parser.add_argument("--punc-model", type=Path, required=True)
    parser.add_argument("--idle-unload-seconds", type=int, default=300)
    return parser.parse_args()


def clean_text(value: str) -> str:
    return rich_transcription_postprocess(value).replace("🎼", "").replace("😊", "").strip()


def segments_from_result(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for item in result:
        timestamps = item.get("timestamp") or []
        words = item.get("words") or []
        if not timestamps or len(timestamps) != len(words):
            text = clean_text(str(item.get("text") or ""))
            if text:
                segments.append({"start": 0.0, "end": 0.0, "text": text})
            continue

        current_words: list[str] = []
        start_ms = int(timestamps[0][0])
        end_ms = start_ms
        for word, timestamp in zip(words, timestamps):
            current_words.append(str(word))
            end_ms = int(timestamp[1])
            if str(word).endswith(("。", "！", "？", ".", "!", "?")) or len(current_words) >= 42:
                text = clean_text("".join(current_words))
                if text:
                    segments.append({"start": round(start_ms / 1000, 3), "end": round(end_ms / 1000, 3), "text": text})
                current_words = []
                start_ms = end_ms
        if current_words:
            text = clean_text("".join(current_words))
            if text:
                segments.append({"start": round(start_ms / 1000, 3), "end": round(end_ms / 1000, 3), "text": text})
    return segments


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model: AutoModel | None = None
        self.last_activity = 0.0
        self.active_requests = 0
        self.lock = threading.RLock()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ready": self.model is not None,
                "active_requests": self.active_requests,
                "idle_unload_seconds": self.args.idle_unload_seconds,
            }

    def transcribe(self, audio: Path, hotword: str) -> dict[str, Any]:
        with self.lock:
            self.active_requests += 1
            try:
                if self.model is None:
                    self.model = AutoModel(
                        model=str(self.args.model),
                        vad_model=str(self.args.vad_model),
                        vad_kwargs={"max_single_segment_time": 30000},
                        punc_model=str(self.args.punc_model),
                        device="cuda:0",
                        disable_update=True,
                    )
                result = self.model.generate(
                    input=str(audio),
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                    hotword=hotword or None,
                    output_timestamp=True,
                )
                text = clean_text("\n".join(str(item.get("text") or "") for item in result))
                return {
                    "success": True,
                    "provider": "sensevoice_funasr",
                    "text": text,
                    "segments": segments_from_result(result),
                    "language": "zh",
                }
            finally:
                self.active_requests = max(0, self.active_requests - 1)
                self.last_activity = time.monotonic()

    def unload_if_idle(self) -> None:
        with self.lock:
            if self.model is None or self.active_requests:
                return
            if time.monotonic() - self.last_activity < self.args.idle_unload_seconds:
                return
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            logging.info("unloaded idle SenseVoice model")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    runtime: Runtime

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        if urlparse(self.path).path in {"/health", "/api/health"}:
            self.send_json(200, self.runtime.status())
            return
        self.send_plain(404, "not found\n")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/asr/transcribe":
            self.send_plain(404, "not found\n")
            return
        try:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                raise ValueError("expected multipart/form-data")
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            audio_item = form["audio"]
            if not getattr(audio_item, "file", None):
                raise ValueError("missing audio file")
            suffix = Path(getattr(audio_item, "filename", "") or "audio.wav").suffix or ".wav"
            with tempfile.TemporaryDirectory(prefix="sensevoice_") as directory:
                audio_path = Path(directory) / f"audio{suffix}"
                audio_path.write_bytes(audio_item.file.read())
                payload = self.runtime.transcribe(audio_path, str(form.getfirst("hotword", "")))
            self.send_json(200, payload)
        except Exception as exc:
            logging.exception("SenseVoice request failed")
            self.send_json(500, {"success": False, "error": str(exc)})

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def send_plain(self, status: int, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    for path in (args.model, args.vad_model, args.punc_model):
        if not path.is_dir():
            raise SystemExit(f"missing model directory: {path}")
    runtime = Runtime(args)
    Handler.runtime = runtime
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    def idle_reaper() -> None:
        while True:
            time.sleep(1)
            runtime.unload_if_idle()

    threading.Thread(target=idle_reaper, daemon=True).start()
    logging.info("SenseVoice server listening on %s:%s", args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
