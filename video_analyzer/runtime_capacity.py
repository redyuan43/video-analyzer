"""Resolve request concurrency from the worker topology exposed by local proxies."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit, urlunsplit

import requests

logger = logging.getLogger(__name__)


def health_url_for_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, "/api/health", "", ""))


def endpoint_worker_capacity(endpoint: str, default: int = 1) -> int:
    if not endpoint:
        return max(1, default)
    session = requests.Session()
    host = urlsplit(endpoint).hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        session.trust_env = False
    try:
        response = session.get(health_url_for_endpoint(endpoint), timeout=3)
        response.raise_for_status()
        payload = response.json()
        workers = payload.get("workers") if isinstance(payload, dict) else None
        if isinstance(workers, list) and workers:
            return len(workers)
        worker_count = payload.get("worker_count") if isinstance(payload, dict) else None
        if isinstance(worker_count, int) and worker_count > 0:
            return worker_count
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Could not resolve worker capacity from %s: %s", endpoint, exc)
    finally:
        session.close()
    return max(1, default)


def resolve_endpoint_concurrency(value: int | str, endpoints: list[str]) -> int:
    if str(value).strip().lower() != "auto":
        return max(1, int(value))
    capacities = [endpoint_worker_capacity(endpoint) for endpoint in endpoints if endpoint]
    return max(1, sum(capacities) if capacities else 1)
