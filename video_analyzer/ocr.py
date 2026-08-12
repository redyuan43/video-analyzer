import base64
import hashlib
import html
import ipaddress
import io
import json
import logging
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests
from PIL import Image

from .config import Config, normalize_string_list
from .frame import Frame

logger = logging.getLogger(__name__)

FALLBACK_DOTS_MOCR_ENDPOINTS = [
    "http://spark-31d6.taild500c8.ts.net:8000/v1",
    "http://edge.taild500c8.ts.net:8000/v1",
]


def default_ocr_endpoints(config: Optional[Dict[str, Any]] = None) -> list[str]:
    if config is not None:
        values = normalize_string_list(((config.get("endpoints") or {}).get("services") or {}).get("ocr_base_urls"))
        return values or FALLBACK_DOTS_MOCR_ENDPOINTS
    try:
        endpoints = (Config().get("endpoints") or {}).get("services", {}).get("ocr_base_urls")
        values = normalize_string_list(endpoints)
        return values or FALLBACK_DOTS_MOCR_ENDPOINTS
    except Exception as exc:
        logger.debug("Could not load OCR endpoints from config: %s", exc)
        return FALLBACK_DOTS_MOCR_ENDPOINTS


DOTS_MOCR_ENDPOINTS = default_ocr_endpoints()

PROMPTS = {
    "prompt_scene_spotting": (
        "Detect and recognize the text in the image."
    ),
    "prompt_layout_json": (
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

UNLIMITED_OCR_PROMPT = "<image>\nFree OCR."
UNLIMITED_OCR_NORMALIZATION_VERSION = 2
UNLIMITED_OCR_NGRAM_SIZES = [3, 35]
UNLIMITED_OCR_NGRAM_WINDOW = 128
UNLIMITED_OCR_REPETITION_PENALTY = 1.1
UNLIMITED_OCR_MAX_CROP_BLOCKS = 8
UNLIMITED_OCR_REPEAT_STOP_MIN_TOKENS = 256
UNLIMITED_OCR_REPEAT_BLOCK_TOKENS = 32
UNLIMITED_OCR_REPEAT_COUNT = 3
UNLIMITED_OCR_QUALITY_GATE_MODE = "observe"
_UNLIMITED_DETECTION_RE = re.compile(
    r"<\|det\|>\s*([^\[\n<]+?)\s*\[([^\]<]+)\]\s*<\|/det\|>",
    re.IGNORECASE,
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_NON_TEXT_CATEGORIES = {"image", "figure", "picture", "non-text"}


@dataclass
class OCREvent:
    frame_number: int
    timestamp: float
    provider: str
    status: str
    text: str
    items: List[Dict[str, Any]]
    error: Optional[str] = None
    cache_status: Optional[str] = None
    raw_text: Optional[str] = None
    quality_status: Optional[str] = None
    generation_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "status": self.status,
            "text": self.text,
            "items": self.items,
            "error": self.error,
            "cache_status": self.cache_status,
            "raw_text": self.raw_text,
            "quality_status": self.quality_status,
            "generation_metadata": self.generation_metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OCREvent":
        return cls(
            frame_number=int(payload.get("frame_number", 0)),
            timestamp=float(payload.get("timestamp", 0.0)),
            provider=str(payload.get("provider", "")),
            status=str(payload.get("status", "")),
            text=str(payload.get("text", "")),
            items=list(payload.get("items") or []),
            error=payload.get("error"),
            cache_status=payload.get("cache_status"),
            raw_text=payload.get("raw_text"),
            quality_status=payload.get("quality_status"),
            generation_metadata=dict(payload.get("generation_metadata") or {}) or None,
        )


def _encode_image(path: Path, max_long_side: int = 1280) -> str:
    if max_long_side <= 0:
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_long_side:
            scale = max_long_side / longest
            resized = (max(1, int(width * scale)), max(1, int(height * scale)))
            resampling = getattr(Image, "Resampling", Image)
            image = image.resize(resized, resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_image_path(frame: Frame) -> Optional[Path]:
    path = getattr(frame, "path", None)
    if isinstance(path, Path):
        return path
    if isinstance(path, str):
        return Path(path)
    return None


def _ocr_cache_key(
    frame: Frame,
    provider: str,
    model: str,
    prompt_mode: str,
    endpoint_family: str,
    max_tokens: int,
    max_image_long_side: int,
    image_mode: Optional[str] = None,
) -> Optional[str]:
    image_path = _frame_image_path(frame)
    if not image_path or not image_path.exists():
        return None
    payload = {
        "version": 3,
        "image_sha256": _hash_file(image_path),
        "provider": provider,
        "model": model,
        "prompt_mode": prompt_mode,
        "prompt": PROMPTS.get(prompt_mode, PROMPTS["prompt_scene_spotting"]),
        "endpoint_family": endpoint_family,
        "max_tokens": int(max_tokens),
        "max_image_long_side": int(max_image_long_side),
        "image_mode": image_mode if provider == "unlimited_ocr" else None,
        "normalization_version": (
            UNLIMITED_OCR_NORMALIZATION_VERSION if provider == "unlimited_ocr" else 0
        ),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_cached_event(cache_dir: Optional[Path], key: str, frame: Frame) -> Optional[OCREvent]:
    if not cache_dir or not key:
        return None
    cache_path = cache_dir / f"{key}.json"
    if not cache_path.exists():
        return None
    try:
        event = OCREvent.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
        event.frame_number = frame.number
        event.timestamp = frame.timestamp
        event.cache_status = "hit"
        return event
    except Exception as exc:
        logger.warning("Ignoring unreadable OCR cache entry %s: %s", cache_path, exc)
        return None


def _write_cached_event(cache_dir: Optional[Path], key: str, event: OCREvent) -> None:
    if not cache_dir or not key or event.status != "ok":
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    cache_path.write_text(json.dumps(event.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_mode_enabled(cache_mode: str) -> bool:
    return cache_mode in {"on", "refresh"}


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


def _parse_unlimited_bbox(value: str) -> Optional[List[int]]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        return None
    try:
        return [int(round(float(item))) for item in parts]
    except ValueError:
        return None


def _clean_unlimited_text(value: str) -> str:
    cleaned = _MARKDOWN_IMAGE_RE.sub("", value)
    cleaned = cleaned.replace("<|endoftext|>", "").replace("<｜end▁of▁sentence｜>", "")
    cleaned = re.sub(r"<\|det\|>\s*[A-Za-z_][\w-]*\s*\[?", "", cleaned)
    cleaned = cleaned.replace("<|/det|>", "")
    cleaned = re.sub(r"</(?:td|th)>", "\t", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</tr>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = html.unescape(cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    ignored_lines = {"[non-text]", "[no text]", "(no text to output)"}
    retained: List[str] = []
    for line in lines:
        normalized = line.lower().rstrip(".")
        if not line or normalized in ignored_lines:
            continue
        if normalized.startswith("the image contains no text"):
            continue
        if normalized.startswith("the image is too blurry to recognize any text"):
            continue
        retained.append(line)
    return "\n".join(retained)


def _trim_incomplete_unlimited_detection(raw_text: str) -> str:
    last_open = raw_text.rfind("<|det|>")
    last_close = raw_text.rfind("<|/det|>")
    if last_open > last_close:
        return raw_text[:last_open].rstrip()
    return raw_text


def _unlimited_quality_failures(text: str, items: List[Dict[str, Any]], raw_text: str) -> List[str]:
    failures: List[str] = []
    image_marker_count = len(_MARKDOWN_IMAGE_RE.findall(raw_text))
    if image_marker_count and not text:
        failures.append("image_only_output")
    if raw_text.count("<|det|>") != raw_text.count("<|/det|>"):
        failures.append("unbalanced_detection_markers")

    lines: List[str] = []
    for item in items:
        item_text = str(item.get("text") or "")
        lines.extend(line.strip() for line in item_text.splitlines() if len(line.strip()) >= 4)
    if not lines:
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 4]

    counts: Dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    if counts:
        max_count = max(counts.values())
        repeated_count = sum(count - 1 for count in counts.values() if count > 1)
        if max_count >= 8 or (len(lines) >= 12 and repeated_count / len(lines) >= 0.35):
            failures.append("repetitive_output")

    formula_lines = [
        line
        for line in lines
        if "\\frac" in line or "\\therefore" in line or "\\overrightarrow" in line
    ]
    if len(formula_lines) >= 5 and len(set(formula_lines)) <= max(2, len(formula_lines) // 4):
        failures.append("repetitive_formula_output")

    compact_text = re.sub(r"\s+", "", text)
    if len(compact_text) >= 200 and len(set(compact_text)) / len(compact_text) < 0.08:
        failures.append("low_character_diversity")
    alphanumeric_text = re.sub(r"[^\w\u3400-\u9fff]+", "", text, flags=re.UNICODE).replace("_", "").lower()
    if re.search(r"_{3,}", text) and alphanumeric_text in {"text", "recognizedtext", "ocrtext"}:
        failures.append("placeholder_output")

    if len(raw_text) >= 8000:
        failures.append("abnormally_long_output")
    if not text and len(_clean_unlimited_text(raw_text)) >= 80:
        failures.append("unparsed_nonempty_output")
    meaningful_chars = len(re.findall(r"[\w\u3400-\u9fff]", text))
    if len(raw_text) >= 200 and meaningful_chars < 20:
        failures.append("low_information_output")
    return failures


def normalize_unlimited_ocr_output(
    raw_text: str,
) -> tuple[str, List[Dict[str, Any]], str, List[str]]:
    parse_text = _trim_incomplete_unlimited_detection(raw_text)
    matches = list(_UNLIMITED_DETECTION_RE.finditer(parse_text))
    items: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(parse_text)
        category = match.group(1).strip().lower()
        item_text = _clean_unlimited_text(parse_text[match.end():next_start])
        bbox = _parse_unlimited_bbox(match.group(2))
        item: Dict[str, Any] = {"category": category, "text": item_text}
        if bbox is not None:
            item["bbox"] = bbox
        items.append(item)
        if item_text and category not in _NON_TEXT_CATEGORIES:
            text_parts.append(item_text)

    if matches:
        normalized_text = "\n".join(text_parts).strip()
    else:
        normalized_text = _clean_unlimited_text(parse_text)

    failures = _unlimited_quality_failures(normalized_text, items, parse_text)
    if failures:
        quality_status = "quality_failed"
    elif normalized_text:
        quality_status = "passed"
    else:
        quality_status = "empty"
    return normalized_text, items, quality_status, failures


class DotsMOCRVLLMProvider:
    def __init__(
        self,
        base_url: str = "auto",
        model: str = "model",
        prompt_mode: str = "prompt_scene_spotting",
        timeout: int = 120,
        max_tokens: int = 1024,
        max_image_long_side: int = 1280,
        probe_timeout_seconds: float = 5,
        warmup_timeout_seconds: float = 180,
        warmup_retry_interval_seconds: float = 5,
        provider_name: str = "dots_mocr_vllm",
        image_mode: str = "gundam",
    ):
        self.base_url = base_url
        self.model = model
        self.prompt_mode = prompt_mode
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_image_long_side = max_image_long_side
        self.probe_timeout_seconds = probe_timeout_seconds
        self.warmup_timeout_seconds = warmup_timeout_seconds
        self.warmup_retry_interval_seconds = warmup_retry_interval_seconds
        self.provider_name = provider_name
        self.image_mode = image_mode if image_mode in {"gundam", "base"} else "gundam"
        self.selected_base_url: Optional[str] = None
        self.diagnostics: List[Dict[str, str]] = []
        self.session = _make_session(self.base_url)

    def probe(self) -> Optional[str]:
        endpoints = default_ocr_endpoints() if self.base_url == "auto" else [self.base_url]
        self.diagnostics = []
        started = time.monotonic()
        deadline = started + max(0, self.warmup_timeout_seconds)
        attempt = 0
        while True:
            attempt += 1
            for endpoint in endpoints:
                normalized = endpoint.rstrip("/")
                try:
                    response = self.session.get(f"{normalized}/models", timeout=self.probe_timeout_seconds)
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

    def _request_frame(self, frame: Frame, image_mode: Optional[str] = None) -> OCREvent:
        base_url = self.selected_base_url or self.probe()
        if not base_url:
            return OCREvent(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                provider=self.provider_name,
                status="unavailable",
                text="",
                items=[],
                error=f"No DotsMOCR vLLM endpoint is reachable: {self.diagnostics}",
            )

        is_unlimited = self.provider_name == "unlimited_ocr"
        prompt = (
            UNLIMITED_OCR_PROMPT
            if is_unlimited
            else PROMPTS.get(self.prompt_mode, PROMPTS["prompt_scene_spotting"])
        )
        mode = image_mode or self.image_mode
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{_encode_image(frame.path, self.max_image_long_side)}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                prompt
                                if is_unlimited
                                else f"<|img|><|imgpad|><|endofimg|>{prompt}"
                            ),
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
        }
        if is_unlimited:
            payload["images_config"] = {
                "image_mode": mode,
                "max_crop_blocks": UNLIMITED_OCR_MAX_CROP_BLOCKS,
            }
            payload["custom_params"] = {
                "ngram_sizes": UNLIMITED_OCR_NGRAM_SIZES,
                "window_size": UNLIMITED_OCR_NGRAM_WINDOW,
                "repetition_penalty": UNLIMITED_OCR_REPETITION_PENALTY,
                "repeat_stop_min_tokens": UNLIMITED_OCR_REPEAT_STOP_MIN_TOKENS,
                "repeat_block_tokens": UNLIMITED_OCR_REPEAT_BLOCK_TOKENS,
                "repeat_count": UNLIMITED_OCR_REPEAT_COUNT,
            }

        try:
            started = time.monotonic()
            response = self.session.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": "Bearer 0", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            generation_metadata = dict(data.get("ocr_metadata") or {})
            generation_metadata.setdefault("elapsed_seconds", round(time.monotonic() - started, 3))
            generation_metadata.setdefault("image_mode", mode if is_unlimited else None)
            quality_status = None
            raw_text = None
            failures: List[str] = []
            if is_unlimited:
                raw_text = content
                text, items, quality_status, failures = normalize_unlimited_ocr_output(content)
                generation_metadata["quality_gate_mode"] = UNLIMITED_OCR_QUALITY_GATE_MODE
                generation_metadata["quality_warnings"] = failures
            else:
                items = _extract_json_array(content) or []
                text = "\n".join(str(item.get("text", "")).strip() for item in items if item.get("text"))
                if not text:
                    text = content.strip()
            observe_quality = (
                is_unlimited
                and UNLIMITED_OCR_QUALITY_GATE_MODE == "observe"
                and bool(text)
            )
            return OCREvent(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                provider=f"{self.provider_name}:{base_url}",
                status="ok" if observe_quality or not failures else "quality_failed",
                text=text if observe_quality or not failures else "",
                items=items,
                error=None if observe_quality else ", ".join(failures) if failures else None,
                raw_text=raw_text,
                quality_status=quality_status,
                generation_metadata=generation_metadata,
            )
        except Exception as exc:
            logger.warning("OCR failed for frame %s: %s", frame.number, exc)
            return OCREvent(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                provider=f"{self.provider_name}:{base_url}",
                status="error",
                text="",
                items=[],
                error=str(exc),
            )

    def analyze_frame(self, frame: Frame) -> OCREvent:
        return self._request_frame(frame)


class OpenAICompatibleVisionOCRProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "0",
        timeout: int = 180,
        max_image_long_side: int = 1280,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "0"
        self.timeout = timeout
        self.max_image_long_side = max_image_long_side
        self.session = _make_session(self.base_url)

    def probe(self) -> bool:
        try:
            response = self.session.get(
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
                                "url": f"data:image/jpeg;base64,{_encode_image(frame.path, self.max_image_long_side)}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4000,
        }
        try:
            response = self.session.post(
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

def _unavailable_events(
    frames: List[Frame],
    error: str,
    provider_name: str = "dots_mocr_vllm",
) -> List[OCREvent]:
    return [
        OCREvent(
            frame_number=frame.number,
            timestamp=frame.timestamp,
            provider=provider_name,
            status="unavailable",
            text="",
            items=[],
            error=error,
        )
        for frame in frames
    ]


def _normalize_endpoint(endpoint: str) -> str:
    return _resolve_tailscale_endpoint(endpoint.strip().rstrip("/"))


def _hostname_resolves(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, None)
        return True
    except OSError:
        return False


def _resolve_tailscale_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    hostname = parsed.hostname
    if not hostname or _hostname_resolves(hostname):
        return endpoint

    fallback_ip = _tailscale_ip_for_hostname(hostname)
    if not fallback_ip:
        return endpoint

    netloc = fallback_ip
    if parsed.port:
        netloc = f"{fallback_ip}:{parsed.port}"
    logger.warning("MagicDNS failed for %s; using Tailscale IP fallback %s", hostname, fallback_ip)
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _tailscale_ip_for_hostname(hostname: str) -> Optional[str]:
    short_name = hostname.split(".", 1)[0]
    try:
        completed = subprocess.run(
            ["tailscale", "status", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = json.loads(completed.stdout)
    except Exception as exc:
        logger.debug("Unable to read tailscale status for MagicDNS fallback: %s", exc)
        return None

    peers = status.get("Peer") or {}
    for peer in peers.values():
        peer_names = {
            str(peer.get("HostName") or ""),
            str(peer.get("DNSName") or "").rstrip("."),
        }
        if hostname not in peer_names and short_name not in peer_names:
            continue
        ips = peer.get("TailscaleIPs") or []
        return str(ips[0]) if ips else None
    return None


def _should_bypass_env_proxy(endpoint: str) -> bool:
    parsed = urlsplit(endpoint)
    host = parsed.hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_private or address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return host.endswith(".local") or host.endswith(".lan") or host.endswith(".taild500c8.ts.net")


def _make_session(endpoint: str) -> requests.Session:
    session = requests.Session()
    if _should_bypass_env_proxy(endpoint):
        session.trust_env = False
    return session


def _resolve_dots_endpoints(base_url: str, base_urls: Optional[List[str]] = None) -> List[str]:
    if base_urls:
        endpoints = base_urls
    elif base_url == "auto":
        endpoints = default_ocr_endpoints()
    else:
        endpoints = [item.strip() for item in base_url.split(",") if item.strip()]
    normalized = []
    for endpoint in endpoints:
        value = _normalize_endpoint(endpoint)
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _resolve_ocr_concurrency(value: int | str) -> int:
    if value == "auto":
        return 1
    return max(1, int(value))


def _probe_dots_providers(
    endpoints: List[str],
    model: str,
    prompt_mode: str,
    request_timeout_seconds: float,
    max_tokens: int,
    max_image_long_side: int,
    probe_timeout_seconds: float,
    warmup_timeout_seconds: float,
    warmup_retry_interval_seconds: float,
    provider_name: str,
    image_mode: str,
) -> List[DotsMOCRVLLMProvider]:
    providers = [
        DotsMOCRVLLMProvider(
            base_url=endpoint,
            model=model,
            prompt_mode=prompt_mode,
            timeout=int(request_timeout_seconds),
            max_tokens=max_tokens,
            max_image_long_side=max_image_long_side,
            probe_timeout_seconds=probe_timeout_seconds,
            warmup_timeout_seconds=warmup_timeout_seconds,
            warmup_retry_interval_seconds=warmup_retry_interval_seconds,
            provider_name=provider_name,
            image_mode=image_mode,
        )
        for endpoint in endpoints
    ]
    if not providers:
        return []
    healthy: List[DotsMOCRVLLMProvider] = []
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {executor.submit(provider.probe): provider for provider in providers}
        for future in as_completed(futures):
            provider = futures[future]
            try:
                if future.result():
                    healthy.append(provider)
            except Exception as exc:
                provider.diagnostics.append({"endpoint": provider.base_url, "attempt": "probe", "error": str(exc)})
    healthy.sort(key=lambda provider: endpoints.index(provider.base_url))
    return healthy


def run_ocr(
    frames: List[Frame],
    provider: str,
    base_url: str,
    model: str,
    prompt_mode: str,
    base_urls: Optional[List[str]] = None,
    ocr_concurrency: int | str = "auto",
    fallback_base_url: Optional[str] = None,
    fallback_model: Optional[str] = None,
    fallback_api_key: str = "0",
    request_timeout_seconds: float = 120,
    max_tokens: int = 1024,
    max_image_long_side: int = 1280,
    retry_endpoints: bool = True,
    probe_timeout_seconds: float = 5,
    warmup_timeout_seconds: float = 180,
    warmup_retry_interval_seconds: float = 5,
    cache_mode: str = "on",
    cache_dir: Optional[str] = ".cache/video-analyzer/ocr",
    image_mode: str = "gundam",
    progress_callback=None,
) -> List[OCREvent]:
    if provider == "none":
        return []
    if provider not in {
        "auto",
        "unlimited_ocr",
        "dots_ocr",
        "dots_mocr_vllm",
        "openai_vision",
    }:
        raise ValueError(f"Unknown OCR provider: {provider}")
    requested_provider = provider
    if provider in {"unlimited_ocr", "dots_ocr"}:
        provider = "dots_mocr_vllm"
    if cache_mode not in {"on", "off", "refresh"}:
        raise ValueError(f"Unknown OCR cache mode: {cache_mode}")
    cache_path = Path(cache_dir) if cache_dir and _cache_mode_enabled(cache_mode) else None

    def cached_or_analyze(frame: Frame, provider_name: str, endpoint_family: str, analyze) -> OCREvent:
        key = _ocr_cache_key(
            frame,
            provider_name,
            model,
            prompt_mode,
            endpoint_family,
            max_tokens,
            max_image_long_side,
            image_mode,
        )
        if key is None:
            event = analyze(frame)
            event.cache_status = "disabled"
            return event
        if cache_mode == "on":
            cached = _read_cached_event(cache_path, key, frame)
            if cached:
                return cached
        event = analyze(frame)
        event.cache_status = "refresh" if cache_mode == "refresh" else "miss" if cache_mode == "on" else "disabled"
        _write_cached_event(cache_path, key, event)
        return event

    def read_all_cached(provider_name: str, endpoint_family: str) -> Optional[List[OCREvent]]:
        if cache_mode != "on" or not frames:
            return None
        events = []
        for frame in frames:
            key = _ocr_cache_key(
                frame,
                provider_name,
                model,
                prompt_mode,
                endpoint_family,
                max_tokens,
                max_image_long_side,
                image_mode,
            )
            if key is None:
                return None
            cached = _read_cached_event(cache_path, key, frame)
            if cached is None:
                return None
            events.append(cached)
        return events

    if provider == "openai_vision":
        if not fallback_base_url or not fallback_model:
            raise ValueError("openai_vision OCR requires fallback_base_url and fallback_model")
        fallback = OpenAICompatibleVisionOCRProvider(
            base_url=fallback_base_url,
            model=fallback_model,
            api_key=fallback_api_key,
            timeout=int(request_timeout_seconds),
            max_image_long_side=max_image_long_side,
        )
        endpoint_family = f"{fallback.base_url}:{fallback.model}"
        cached_events = read_all_cached("openai_vision", endpoint_family)
        if cached_events is not None:
            if progress_callback:
                for event in cached_events:
                    progress_callback(event)
            return cached_events
        if not fallback.probe():
            events = _unavailable_events(
                frames,
                f"OpenAI-compatible OCR endpoint is not reachable: {fallback_base_url}",
                "openai_vision",
            )
            if progress_callback:
                for event in events:
                    progress_callback(event)
            return events
        events = [
            cached_or_analyze(
                frame,
                "openai_vision",
                endpoint_family,
                fallback.analyze_frame,
            )
            for frame in frames
        ]
        if progress_callback:
            for event in events:
                progress_callback(event)
        return events

    dots_endpoints = _resolve_dots_endpoints(base_url, base_urls)
    dots_endpoint_family = ",".join(dots_endpoints) if dots_endpoints else base_url
    provider_name = requested_provider if requested_provider in {"unlimited_ocr", "dots_ocr"} else "dots_mocr_vllm"
    cached_events = read_all_cached(provider_name, dots_endpoint_family)
    if cached_events is not None:
        if progress_callback:
            for event in cached_events:
                progress_callback(event)
        return cached_events

    dots_providers = _probe_dots_providers(
        endpoints=dots_endpoints,
        model=model,
        prompt_mode=prompt_mode,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=max_tokens,
        max_image_long_side=max_image_long_side,
        probe_timeout_seconds=probe_timeout_seconds,
        warmup_timeout_seconds=warmup_timeout_seconds,
        warmup_retry_interval_seconds=warmup_retry_interval_seconds,
        provider_name=provider_name,
        image_mode=image_mode,
    )
    if not dots_providers:
        error = f"DotsMOCR vLLM endpoint was not ready after {warmup_timeout_seconds}s"
        if provider == "auto" and fallback_base_url and fallback_model:
            logger.warning("%s Falling back to OpenAI-compatible vision OCR.", error)
            fallback = OpenAICompatibleVisionOCRProvider(
                base_url=fallback_base_url,
                model=fallback_model,
                api_key=fallback_api_key,
                timeout=int(request_timeout_seconds),
                max_image_long_side=max_image_long_side,
            )
            if fallback.probe():
                events = [
                    cached_or_analyze(
                        frame,
                        "openai_vision",
                        f"{fallback.base_url}:{fallback.model}",
                        fallback.analyze_frame,
                    )
                    for frame in frames
                ]
                for event in events:
                    if event.error:
                        event.error = f"DotsMOCR unavailable first: {error}; fallback error: {event.error}"
                    if progress_callback:
                        progress_callback(event)
                return events
        events = _unavailable_events(frames, error, provider_name)
        if progress_callback:
            for event in events:
                progress_callback(event)
        return events

    cached_by_number: Dict[int, OCREvent] = {}
    pending_frames: List[Frame] = []
    if cache_mode == "on":
        for frame in frames:
            key = _ocr_cache_key(
                frame,
                provider_name,
                model,
                prompt_mode,
                dots_endpoint_family,
                max_tokens,
                max_image_long_side,
                image_mode,
            )
            cached = _read_cached_event(cache_path, key, frame) if key else None
            if cached:
                cached_by_number[frame.number] = cached
            else:
                pending_frames.append(frame)
    else:
        pending_frames = list(frames)

    results: Dict[int, OCREvent] = dict(cached_by_number)
    concurrency = _resolve_ocr_concurrency(ocr_concurrency)
    max_workers = max(1, len(dots_providers) * concurrency)

    def analyze_with_endpoint_retry(index: int, frame: Frame) -> OCREvent:
        attempts = len(dots_providers) if retry_endpoints else 1
        errors = []
        for offset in range(attempts):
            dots = dots_providers[(index + offset) % len(dots_providers)]
            event = dots.analyze_frame(frame)
            if event.status == "ok":
                if offset:
                    logger.info(
                        "OCR frame %s succeeded via retry endpoint %s after %s failed attempt(s)",
                        frame.number,
                        event.provider,
                        offset,
                    )
                return event
            errors.append(f"{event.provider}: {event.error or event.status}")
            if offset + 1 < attempts:
                logger.warning(
                    "OCR frame %s failed on %s; retrying next endpoint",
                    frame.number,
                    event.provider,
                )
        last = event
        last.error = "; ".join(errors)
        return last

    def analyze_pending(index_frame: tuple[int, Frame]) -> OCREvent:
        index, frame = index_frame
        return cached_or_analyze(
            frame,
            provider_name,
            dots_endpoint_family,
            lambda item: analyze_with_endpoint_retry(index, item),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(analyze_pending, item)
            for item in enumerate(pending_frames)
        ]
        for future in as_completed(futures):
            event = future.result()
            results[event.frame_number] = event
            if progress_callback:
                progress_callback(event)

    return [results[frame.number] for frame in frames]
