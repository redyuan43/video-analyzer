from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

FAILURE_FILE_ENV = "VIDEO_ANALYZER_FAILURE_FILE"
TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def http_failure_kind(status_code: int | None, provider_code: str | None = None) -> str:
    normalized_code = str(provider_code or "").strip().lower()
    if status_code == 402 or "balance" in normalized_code or "credit" in normalized_code:
        return "permanent_billing"
    if status_code in {401, 403}:
        return "permanent_auth"
    if status_code in TRANSIENT_HTTP_STATUSES:
        return "transient_http"
    if status_code is not None and 400 <= status_code < 500:
        return "permanent_request"
    if status_code is not None and status_code >= 500:
        return "transient_http"
    return "unknown"


def failure_is_retryable(kind: str) -> bool:
    return kind in {"transient_http", "transient_network", "transient_resource", "interrupted"}


def write_failure_envelope(payload: dict[str, Any]) -> None:
    target_value = os.environ.get(FAILURE_FILE_ENV)
    if not target_value:
        return
    target = Path(target_value).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, target)


def read_failure_envelope(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
