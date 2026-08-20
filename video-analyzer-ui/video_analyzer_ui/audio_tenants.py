from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
from pathlib import Path
from typing import Any


TENANT_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def normalize_tenant_id(value: Any) -> str:
    tenant_id = str(value or "").strip().lower()
    if not TENANT_ID_PATTERN.fullmatch(tenant_id):
        raise ValueError("audio tenant id is invalid")
    return tenant_id


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AudioTenantRegistry:
    """Resolve pipeline credentials without storing raw tenant tokens."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source_key: tuple[str, int, int, str] | None = None
        self._digests: dict[str, str] = {}

    def resolve(self, supplied_token: str) -> str | None:
        digests = self._load()
        if not digests:
            # Preserve the current development behavior when auth is unconfigured.
            return "nx1"
        supplied_digest = token_digest(str(supplied_token or "").strip())
        for tenant_id, expected_digest in digests.items():
            if hmac.compare_digest(supplied_digest, expected_digest):
                return tenant_id
        return None

    def _load(self) -> dict[str, str]:
        path_value = os.environ.get("VIDEO_ANALYZER_AUDIO_TENANTS_FILE", "").strip()
        legacy_token = os.environ.get("VIDEO_ANALYZER_AUDIO_PIPELINE_TOKEN", "").strip()
        path = Path(path_value).expanduser() if path_value else None
        try:
            stat = path.stat() if path else None
            source_key = (
                str(path or ""),
                int(stat.st_mtime_ns) if stat else 0,
                int(stat.st_size) if stat else 0,
                token_digest(legacy_token) if legacy_token else "",
            )
        except OSError:
            source_key = (str(path or ""), -1, -1, token_digest(legacy_token) if legacy_token else "")
        with self._lock:
            if source_key == self._source_key:
                return dict(self._digests)
            digests: dict[str, str] = {}
            if path:
                payload = json.loads(path.read_text(encoding="utf-8"))
                tenants = payload.get("tenants") if isinstance(payload, dict) else None
                if not isinstance(tenants, dict):
                    raise ValueError("audio tenant registry must contain a tenants object")
                for raw_tenant_id, raw_config in tenants.items():
                    tenant_id = normalize_tenant_id(raw_tenant_id)
                    config = raw_config if isinstance(raw_config, dict) else {}
                    digest = str(config.get("token_sha256") or "").strip().lower()
                    if not re.fullmatch(r"[a-f0-9]{64}", digest):
                        raise ValueError(
                            f"audio tenant {tenant_id} must define token_sha256"
                        )
                    digests[tenant_id] = digest
            if legacy_token:
                digests.setdefault("nx1", token_digest(legacy_token))
            self._source_key = source_key
            self._digests = digests
            return dict(digests)
