#!/usr/bin/env python3
"""Expose a CUDA EasyOCR reader through the analyzer's OpenAI OCR contract."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import easyocr


READER: easyocr.Reader | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--model-storage-directory", type=Path, required=True)
    parser.add_argument("--languages", default="ch_sim,en")
    parser.add_argument("--served-model-name", default="easyocr-ch-sim-en")
    return parser.parse_args()


def load_reader(args: argparse.Namespace) -> easyocr.Reader:
    args.model_storage_directory.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(
        args.languages.split(","),
        gpu=True,
        model_storage_directory=str(args.model_storage_directory),
        verbose=False,
    )


def request_image(payload: dict[str, Any], directory: Path) -> Path:
    for message in payload.get("messages") or []:
        for item in message.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            url = str((item.get("image_url") or {}).get("url") or "")
            if not url.startswith("data:") or "," not in url:
                continue
            header, encoded = url.split(",", 1)
            mime = header.split(";", 1)[0].removeprefix("data:")
            suffix = ".png" if mime == "image/png" else ".jpg"
            path = directory / f"input{suffix}"
            path.write_bytes(base64.b64decode(encoded))
            return path
    raise ValueError("request does not contain a data: image_url")


def run_ocr(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if READER is None:
        raise RuntimeError("EasyOCR reader is not loaded")
    with tempfile.TemporaryDirectory(prefix="easyocr_") as temp:
        image = request_image(payload, Path(temp))
        results = READER.readtext(str(image), detail=1, paragraph=False)
    return [
        {
            "category": "text",
            "text": text,
            "confidence": round(float(confidence), 4),
            "box": [[round(float(value), 2) for value in point] for point in box],
        }
        for box, text, confidence in results
    ]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    settings: argparse.Namespace

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/health", "/api/health"}:
            self.send_json(
                200,
                {
                    "ready": READER is not None,
                    "model": self.settings.served_model_name,
                    "languages": self.settings.languages.split(","),
                },
            )
            return
        if path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.settings.served_model_name, "object": "model", "owned_by": "easyocr"}],
                },
            )
            return
        self.send_plain(404, "not found\n")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/v1/chat/completions":
            self.send_plain(404, "not found\n")
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            started = time.monotonic()
            items = run_ocr(payload)
            logging.info("OCR request completed in %.3fs with %s text items", time.monotonic() - started, len(items))
            self.send_json(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.settings.served_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(items, ensure_ascii=False)},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        except Exception as exc:
            logging.exception("EasyOCR request failed")
            self.send_json(500, {"error": {"message": str(exc), "type": "easyocr_error"}})

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
    global READER
    READER = load_reader(args)
    Handler.settings = args
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logging.info("EasyOCR server listening on %s:%s", args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
