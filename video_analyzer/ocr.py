import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .frame import Frame

logger = logging.getLogger(__name__)

DOTS_MOCR_ENDPOINTS = [
    "http://192.168.100.169:8000/v1",
    "http://192.168.100.131:8000/v1",
    "http://127.0.0.1:8000/v1",
]

PROMPTS = {
    "prompt_scene_spotting": (
        "Please detect and recognize all meaningful visible text in this UI or scene image. "
        "Return JSON only as an array of objects. Each object must include bbox "
        "[x1,y1,x2,y2], category, and text. Prefer exact UI labels, command text, "
        "filenames, button labels, code, and small captions."
    ),
    "prompt_ocr": (
        "Please recognize all visible text in the image. Return plain text only, preserving "
        "line breaks when useful."
    ),
}


@dataclass
class OCREvent:
    frame_number: int
    timestamp: float
    provider: str
    status: str
    text: str
    items: List[Dict[str, Any]]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "status": self.status,
            "text": self.text,
            "items": self.items,
            "error": self.error,
        }


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("items", "elements", "results", "text_instances"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    return None


class DotsMOCRVLLMProvider:
    def __init__(
        self,
        base_url: str = "auto",
        model: str = "model",
        prompt_mode: str = "prompt_scene_spotting",
        timeout: int = 120,
        probe_timeout_seconds: float = 5,
        warmup_timeout_seconds: float = 180,
        warmup_retry_interval_seconds: float = 5,
    ):
        self.base_url = base_url
        self.model = model
        self.prompt_mode = prompt_mode
        self.timeout = timeout
        self.probe_timeout_seconds = probe_timeout_seconds
        self.warmup_timeout_seconds = warmup_timeout_seconds
        self.warmup_retry_interval_seconds = warmup_retry_interval_seconds
        self.selected_base_url: Optional[str] = None
        self.diagnostics: List[Dict[str, str]] = []

    def probe(self) -> Optional[str]:
        endpoints = DOTS_MOCR_ENDPOINTS if self.base_url == "auto" else [self.base_url]
        self.diagnostics = []
        started = time.monotonic()
        deadline = started + max(0, self.warmup_timeout_seconds)
        attempt = 0
        while True:
            attempt += 1
            for endpoint in endpoints:
                normalized = endpoint.rstrip("/")
                try:
                    response = requests.get(f"{normalized}/models", timeout=self.probe_timeout_seconds)
                    response.raise_for_status()
                    elapsed = time.monotonic() - started
                    self.selected_base_url = normalized
                    if elapsed > self.probe_timeout_seconds:
                        logger.info("DotsMOCR endpoint ready after %.1fs: %s", elapsed, normalized)
                    return normalized
                except Exception as exc:
                    self.diagnostics.append({"endpoint": normalized, "attempt": str(attempt), "error": str(exc)})

            now = time.monotonic()
            if now >= deadline:
                break
            sleep_seconds = min(self.warmup_retry_interval_seconds, max(0, deadline - now))
            logger.info(
                "DotsMOCR endpoint not ready after %.1fs; waiting %.1fs for cold start before retry %s",
                now - started,
                sleep_seconds,
                attempt + 1,
            )
            time.sleep(sleep_seconds)
        return None

    def analyze_frame(self, frame: Frame) -> OCREvent:
        base_url = self.selected_base_url or self.probe()
        if not base_url:
            return OCREvent(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                provider="dots_mocr_vllm",
                status="unavailable",
                text="",
                items=[],
                error=f"No DotsMOCR vLLM endpoint is reachable: {self.diagnostics}",
            )

        prompt = PROMPTS.get(self.prompt_mode, PROMPTS["prompt_scene_spotting"])
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{_encode_image(frame.path)}"
                            },
                        },
                        {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"},
                    ],
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 10000,
        }

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": "Bearer 0", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            items = _extract_json_array(content) or []
            text = "\n".join(str(item.get("text", "")).strip() for item in items if item.get("text"))
            if not text:
                text = content.strip()
            return OCREvent(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                provider=f"dots_mocr_vllm:{base_url}",
                status="ok",
                text=text,
                items=items,
            )
        except Exception as exc:
            logger.warning("OCR failed for frame %s: %s", frame.number, exc)
            return OCREvent(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                provider=f"dots_mocr_vllm:{base_url}",
                status="error",
                text="",
                items=[],
                error=str(exc),
            )


class OpenAICompatibleVisionOCRProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "0",
        timeout: int = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "0"
        self.timeout = timeout

    def probe(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("OpenAI-compatible vision OCR endpoint unavailable: %s", exc)
            return False

    def analyze_frame(self, frame: Frame) -> OCREvent:
        prompt = (
            "Recognize all visible text in this screenshot or video frame. Return JSON only "
            "as an array of objects with fields category and text. Include exact UI labels, "
            "commands, filenames, URLs, code snippets, captions, and small on-screen text. "
            "Do not describe non-text visual content unless it helps identify a UI label."
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{_encode_image(frame.path)}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4000,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} {response.text[:2000]}")
            content = response.json()["choices"][0]["message"]["content"]
            items = _extract_json_array(content) or []
            text = "\n".join(str(item.get("text", "")).strip() for item in items if item.get("text"))
            if not text:
                text = content.strip()
            return OCREvent(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                provider=f"openai_vision:{self.base_url}:{self.model}",
                status="ok",
                text=text,
                items=items,
            )
        except Exception as exc:
            logger.warning("OpenAI-compatible vision OCR failed for frame %s: %s", frame.number, exc)
            return OCREvent(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                provider=f"openai_vision:{self.base_url}:{self.model}",
                status="error",
                text="",
                items=[],
                error=str(exc),
            )


def _unavailable_events(frames: List[Frame], error: str) -> List[OCREvent]:
    return [
        OCREvent(
            frame_number=frame.number,
            timestamp=frame.timestamp,
            provider="dots_mocr_vllm",
            status="unavailable",
            text="",
            items=[],
            error=error,
        )
        for frame in frames
    ]


def run_ocr(
    frames: List[Frame],
    provider: str,
    base_url: str,
    model: str,
    prompt_mode: str,
    fallback_base_url: Optional[str] = None,
    fallback_model: Optional[str] = None,
    fallback_api_key: str = "0",
    probe_timeout_seconds: float = 5,
    warmup_timeout_seconds: float = 180,
    warmup_retry_interval_seconds: float = 5,
) -> List[OCREvent]:
    if provider == "none":
        return []
    if provider not in {"auto", "dots_mocr_vllm", "openai_vision"}:
        raise ValueError(f"Unknown OCR provider: {provider}")

    if provider == "openai_vision":
        if not fallback_base_url or not fallback_model:
            raise ValueError("openai_vision OCR requires fallback_base_url and fallback_model")
        fallback = OpenAICompatibleVisionOCRProvider(
            base_url=fallback_base_url,
            model=fallback_model,
            api_key=fallback_api_key,
        )
        if not fallback.probe():
            return _unavailable_events(frames, f"OpenAI-compatible OCR endpoint is not reachable: {fallback_base_url}")
        return [fallback.analyze_frame(frame) for frame in frames]

    dots = DotsMOCRVLLMProvider(
        base_url=base_url,
        model=model,
        prompt_mode=prompt_mode,
        probe_timeout_seconds=probe_timeout_seconds,
        warmup_timeout_seconds=warmup_timeout_seconds,
        warmup_retry_interval_seconds=warmup_retry_interval_seconds,
    )
    if not dots.probe():
        error = f"DotsMOCR vLLM endpoint was not ready after {warmup_timeout_seconds}s: {dots.diagnostics}"
        if provider == "auto" and fallback_base_url and fallback_model:
            logger.warning("%s Falling back to OpenAI-compatible vision OCR.", error)
            fallback = OpenAICompatibleVisionOCRProvider(
                base_url=fallback_base_url,
                model=fallback_model,
                api_key=fallback_api_key,
            )
            if fallback.probe():
                events = [fallback.analyze_frame(frame) for frame in frames]
                for event in events:
                    if event.error:
                        event.error = f"DotsMOCR unavailable first: {error}; fallback error: {event.error}"
                return events
        return _unavailable_events(frames, error)
    return [dots.analyze_frame(frame) for frame in frames]
