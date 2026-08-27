"""Translate immutable Nano Audio workflow snapshots into AI runtime profiles."""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import re
import socket
from typing import Any
from urllib.parse import urlparse

from .model_settings import build_settings_document, expand_runtime_profile

SCHEMA_VERSION = 1
WORKFLOW_IDS = {"audio_short_v1", "audio_long_v1"}
ROUTE_POLICIES = {"auto", "local_only", "cloud_only", "background_local"}
NODE_KINDS = {
    "asr": "asr",
    "diarization": "diarization",
    "selector": "selector",
    "summary": "text",
    "tts": "tts",
}
KIND_PROTOCOLS = {
    "asr": {
        "vibevoice_http",
        "qwen3_asr_http",
        "generic_http",
        "firered_asr2_http",
        "firered_3dspeaker_http",
        "openai_audio",
        "tencent_hy_asr_ws",
        "faster_whisper",
        "none",
    },
    "diarization": {
        "asr_embedded",
        "three_d_speaker",
        "three_d_speaker_http",
        "pyannote_community",
        "wespeaker_diarization",
        "none",
    },
    "selector": {"openai_compatible", "ollama_chat", "inherit_text", "none"},
    "text": {"openai_compatible", "ollama_chat", "none"},
    "tts": {"openai_speech", "xiaomi_mimo_tts", "none"},
}
MODEL_ALIASES = {
    "nano-firered-asr2": "asr-firered2-local",
    "ai-vibevoice-asr": "asr-vibevoice-local",
    "ai-tencent-hy-asr": "asr-tencent-hy3-preview-cloud",
    "ai-3dspeaker": "diarization-3dspeaker-local",
    "ai-asr-embedded-speaker": "diarization-asr-embedded",
    "ai-local-selector": "selector-inherit-text",
    "ai-local-text": "text-local-bonsai-27b-6gpu",
    "ai-cloud-text": "text-deepseek-v4-flash",
    "nano-cloud-text": "text-deepseek-v4-flash",
    "ai-indextts": "tts-indextts25-ray-p40",
    "ai-xiaomi-mimo-tts": "tts-xiaomi-mimo-v25-cloud",
}
SECRET_FIELDS = {
    "api_key",
    "apikey",
    "token",
    "access_token",
    "password",
    "secret",
    "secret_key",
}
PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_SNAPSHOT_BYTES = 256 * 1024


class AudioWorkflowSnapshotError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_secrets(value: Any, path: str = "workflow_snapshot") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in SECRET_FIELDS:
                raise AudioWorkflowSnapshotError(f"{path}.{key} is not allowed")
            _reject_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")


def _validate_private_endpoint(value: str, path: str) -> None:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.hostname:
        raise AudioWorkflowSnapshotError(f"{path} is not a supported URL")
    if parsed.username or parsed.password:
        raise AudioWorkflowSnapshotError(f"{path} must not contain credentials")
    host = parsed.hostname.strip().lower().rstrip(".")
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    host,
                    parsed.port,
                    type=socket.SOCK_STREAM,
                )
            }
        except OSError as exc:
            raise AudioWorkflowSnapshotError(
                f"{path} host cannot be resolved"
            ) from exc
    if not addresses:
        raise AudioWorkflowSnapshotError(f"{path} has no usable address")
    tailscale = ipaddress.ip_network("100.64.0.0/10")
    for address in addresses:
        if (
            address.is_link_local
            or address.is_multicast
            or address.is_unspecified
        ):
            raise AudioWorkflowSnapshotError(f"{path} address is not allowed")
        if address.is_loopback or address.is_private:
            continue
        if isinstance(address, ipaddress.IPv4Address) and address in tailscale:
            continue
        raise AudioWorkflowSnapshotError(
            f"{path} must resolve to loopback, LAN, or Tailscale"
        )


def parse_audio_workflow_snapshot(value: Any) -> dict[str, Any] | None:
    if value in (None, "", {}):
        return None
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            raise AudioWorkflowSnapshotError("workflow_snapshot is too large")
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AudioWorkflowSnapshotError(
                "workflow_snapshot must be valid JSON"
            ) from exc
    if not isinstance(value, dict):
        raise AudioWorkflowSnapshotError("workflow_snapshot must be an object")
    _reject_secrets(value)
    snapshot = copy.deepcopy(value)
    if int(snapshot.get("schema_version") or 0) != SCHEMA_VERSION:
        raise AudioWorkflowSnapshotError("unsupported workflow_snapshot schema")
    if str(snapshot.get("workflow_id") or "") not in WORKFLOW_IDS:
        raise AudioWorkflowSnapshotError("unsupported audio workflow")
    profile_id = str(snapshot.get("profile_id") or "")
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise AudioWorkflowSnapshotError("workflow_snapshot profile_id is invalid")
    if int(snapshot.get("profile_revision") or 0) <= 0:
        raise AudioWorkflowSnapshotError(
            "workflow_snapshot profile_revision is invalid"
        )
    supplied_fingerprint = str(snapshot.pop("fingerprint", "") or "")
    actual_fingerprint = hashlib.sha256(_canonical(snapshot)).hexdigest()
    if supplied_fingerprint != actual_fingerprint:
        raise AudioWorkflowSnapshotError("workflow_snapshot fingerprint mismatch")
    snapshot["fingerprint"] = supplied_fingerprint
    nodes = snapshot.get("node_configs")
    if not isinstance(nodes, dict):
        raise AudioWorkflowSnapshotError(
            "workflow_snapshot node_configs must be an object"
        )
    unknown_nodes = set(nodes) - set(NODE_KINDS)
    if unknown_nodes:
        raise AudioWorkflowSnapshotError(
            f"workflow_snapshot contains unknown nodes: {sorted(unknown_nodes)}"
        )
    for node_id, expected_kind in NODE_KINDS.items():
        node = nodes.get(node_id)
        if node is None:
            continue
        if not isinstance(node, dict):
            raise AudioWorkflowSnapshotError(f"{node_id} node must be an object")
        policy = str(node.get("route_policy") or "")
        if policy not in ROUTE_POLICIES:
            raise AudioWorkflowSnapshotError(
                f"{node_id} route policy is invalid"
            )
        for route in ("local_model", "cloud_model"):
            model = node.get(route)
            if model is None:
                continue
            if not isinstance(model, dict):
                raise AudioWorkflowSnapshotError(
                    f"{node_id}.{route} must be an object"
                )
            if str(model.get("kind") or "") != expected_kind:
                raise AudioWorkflowSnapshotError(
                    f"{node_id}.{route} model kind is invalid"
                )
            protocol = str(model.get("protocol") or "")
            if protocol not in KIND_PROTOCOLS[expected_kind]:
                raise AudioWorkflowSnapshotError(
                    f"{node_id}.{route} protocol is invalid"
                )
            endpoints = model.get("endpoints") or []
            if not isinstance(endpoints, list) or len(endpoints) > 8:
                raise AudioWorkflowSnapshotError(
                    f"{node_id}.{route} endpoints are invalid"
                )
            if any(not isinstance(endpoint, str) for endpoint in endpoints):
                raise AudioWorkflowSnapshotError(
                    f"{node_id}.{route} endpoints are invalid"
                )
            for index, endpoint in enumerate(endpoints):
                _validate_private_endpoint(
                    endpoint,
                    f"{node_id}.{route}.endpoints[{index}]",
                )
    return snapshot


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _openai_endpoint(model: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(model)
    if str(normalized.get("protocol") or "") != "ollama_chat":
        return normalized
    normalized["protocol"] = "openai_compatible"
    endpoints = []
    for endpoint in normalized.get("endpoints") or []:
        endpoint = str(endpoint).rstrip("/")
        if endpoint.endswith("/api/chat"):
            endpoint = endpoint[: -len("/api/chat")] + "/v1"
        endpoints.append(endpoint)
    normalized["endpoints"] = endpoints
    return normalized


def _capability_for(
    model: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    model_id = str(model.get("id") or "")
    alias = MODEL_ALIASES.get(model_id)
    if alias and isinstance(catalog.get(alias), dict):
        return copy.deepcopy(catalog[alias])
    protocol = str(model.get("protocol") or "")
    model_name = str(model.get("model") or "")
    candidates = [
        item
        for item in catalog.values()
        if isinstance(item, dict)
        and item.get("kind") == model.get("kind")
        and item.get("protocol") == protocol
        and (not model_name or item.get("model") == model_name)
    ]
    candidates.sort(
        key=lambda item: (
            not bool(item.get("endpoints")),
            not bool(item.get("api_key_env")),
        )
    )
    return copy.deepcopy(candidates[0]) if candidates else {}


def _runtime_model(
    model: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = _openai_endpoint(model)
    capability = _capability_for(normalized, catalog)
    runtime = _deep_merge(capability, normalized)
    runtime["name"] = str(
        runtime.get("name") or runtime.get("label") or runtime.get("id") or "Audio model"
    )
    if not runtime.get("endpoints"):
        runtime["endpoints"] = copy.deepcopy(capability.get("endpoints") or [])
    if not runtime.get("api_key_env"):
        runtime["api_key_env"] = capability.get("api_key_env")
    runtime["options"] = _deep_merge(
        dict(capability.get("options") or {}),
        dict(normalized.get("options") or {}),
    )
    return runtime


def resolve_audio_workflow_profile(
    config: dict[str, Any],
    base_profile_name: str,
    snapshot: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    profiles = config.get("runtime_profiles") or {}
    base_profile = profiles.get(base_profile_name)
    if not isinstance(base_profile, dict):
        raise AudioWorkflowSnapshotError(
            f"AI capability profile is unavailable: {base_profile_name}"
        )
    working_config = copy.deepcopy(config)
    working_profile = copy.deepcopy(base_profile)
    catalog = build_settings_document(working_config)["models"]
    model_catalog = working_config.setdefault("model_catalog", {})
    nodes = snapshot["node_configs"]
    selected_ids: dict[str, str] = {}
    runtime_model_ids: list[str] = []

    def install(node_id: str, route: str) -> str:
        model = (nodes.get(node_id) or {}).get(route)
        if not isinstance(model, dict):
            return ""
        runtime_id = f"audio-snapshot-{node_id}-{route.removesuffix('_model')}"
        model_catalog[runtime_id] = _runtime_model(model, catalog)
        runtime_model_ids.append(runtime_id)
        selected_ids[f"{node_id}.{route}"] = str(model.get("id") or runtime_id)
        return runtime_id

    def primary_and_fallback(node_id: str) -> tuple[str, str]:
        node = nodes.get(node_id) or {}
        if not node.get("enabled"):
            return "", ""
        policy = str(node.get("route_policy") or "")
        local_id = install(node_id, "local_model")
        cloud_id = install(node_id, "cloud_model")
        if policy == "cloud_only":
            return cloud_id, ""
        if policy == "auto":
            return local_id, cloud_id
        if policy == "background_local" and node_id == "tts":
            return local_id, cloud_id
        return local_id, ""

    asr_primary, asr_fallback = primary_and_fallback("asr")
    diarization_primary, diarization_fallback = primary_and_fallback(
        "diarization"
    )
    selector_primary, selector_fallback = primary_and_fallback("selector")
    text_primary, text_fallback = primary_and_fallback("summary")
    tts_primary, tts_fallback = primary_and_fallback("tts")

    working_profile.update(
        {
            "workflow_id": "audio_nx1",
            "asr_model_id": asr_primary or "asr-disabled",
            "diarization_model_id": (
                diarization_primary or "diarization-disabled"
            ),
            "asr_fallback_model_id": asr_fallback or "asr-disabled",
            "diarization_fallback_model_id": (
                diarization_fallback or "diarization-disabled"
            ),
            "template_selector_model_id": (
                selector_primary or "selector-disabled"
            ),
            "text_model_id": text_primary or "text-disabled",
            "text_fallback_model_id": text_fallback or "text-disabled",
            "tts_model_id": tts_primary or "tts-disabled",
        }
    )
    resolved = expand_runtime_profile(working_config, working_profile)
    if selector_fallback:
        selector_resource = model_catalog[selector_fallback]
        selector_endpoints = list(selector_resource.get("endpoints") or [])
        resolved.update(
            {
                "template_selector_fallback_enabled": True,
                "template_selector_fallback_model_id": selector_fallback,
                "template_selector_fallback_base_url": (
                    selector_endpoints[0] if selector_endpoints else ""
                ),
                "template_selector_fallback_model": str(
                    selector_resource.get("model") or ""
                ),
                "template_selector_fallback_api_key_env": str(
                    selector_resource.get("api_key_env") or ""
                ),
                "template_selector_fallback_options": copy.deepcopy(
                    selector_resource.get("options") or {}
                ),
            }
        )
    else:
        resolved["template_selector_fallback_enabled"] = False
        resolved["template_selector_fallback_model_id"] = ""
    if tts_fallback:
        tts_resource = model_catalog[tts_fallback]
        tts_endpoints = list(tts_resource.get("endpoints") or [])
        tts_options = copy.deepcopy(tts_resource.get("options") or {})
        resolved.update(
            {
                "tts_fallback_enabled": True,
                "tts_fallback_model_id": tts_fallback,
                "tts_fallback_provider": str(tts_resource.get("protocol") or ""),
                "tts_fallback_base_url": (
                    tts_endpoints[0] if tts_endpoints else ""
                ),
                "tts_fallback_model": str(tts_resource.get("model") or ""),
                "tts_fallback_api_key_env": str(
                    tts_resource.get("api_key_env") or ""
                ),
                "tts_fallback_voice": str(tts_options.get("voice") or "冰糖"),
                "tts_fallback_speed": float(tts_options.get("speed") or 0.9),
                "tts_fallback_timeout_seconds": int(
                    tts_options.get("timeout_seconds") or 180
                ),
                "tts_fallback_extra_params": copy.deepcopy(
                    tts_options.get("extra_params") or {}
                ),
                "tts_fallback_style_prompt": str(
                    tts_options.get("style_prompt") or ""
                ),
            }
        )
    else:
        resolved["tts_fallback_enabled"] = False
        resolved["tts_fallback_model_id"] = ""
    metadata = {
        "source": "nano_workflow_snapshot",
        "workflow_id": snapshot["workflow_id"],
        "profile_id": snapshot["profile_id"],
        "profile_revision": snapshot["profile_revision"],
        "fingerprint": snapshot["fingerprint"],
        "route_policies": {
            node_id: str((nodes.get(node_id) or {}).get("route_policy") or "")
            for node_id in NODE_KINDS
        },
        "selected_model_ids": selected_ids,
    }
    runtime_models = {
        model_id: copy.deepcopy(model_catalog[model_id])
        for model_id in runtime_model_ids
    }
    return resolved, metadata, runtime_models
