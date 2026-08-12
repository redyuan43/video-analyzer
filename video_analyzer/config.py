import argparse
from pathlib import Path
import json
import os
import copy
import re
from typing import Any
import logging
from importlib import resources
import ipaddress
from urllib.parse import urlparse

from .model_settings import apply_disabled_runtime_profiles, expand_runtime_profile

logger = logging.getLogger(__name__)

DEFAULT_DEEPSEEK_ENV = Path("~/.config/video-analyzer/deepseek.env").expanduser()
ENDPOINT_REF_RE = re.compile(r"\{([A-Za-z0-9_.-]+)\}")


class Config:
    def __init__(self, config_dir: str = "config"):
        if config_dir == "config":
            config_dir = os.environ.get("VIDEO_ANALYZER_CONFIG_DIR", config_dir)
        # Handle user-provided config directory
        self.config_dir = Path(config_dir)
        self.user_config = self.config_dir / "config.json"
        
        # First try to find default_config.json in the user-provided directory
        self.default_config = self.config_dir / "default_config.json"
        
        # If not found, fallback to package's default config
        if not self.default_config.exists():
            try:
                default_config_path = resources.files('video_analyzer').joinpath('config', 'default_config.json')
                self.default_config = Path(default_config_path)
                logger.debug(f"Using packaged default config from {self.default_config}")
            except Exception as e:
                logger.error(f"Error finding default config: {e}")
                raise
            
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file with cascade:
        1. Load default config (default_config.json)
        2. Merge user config (config.json) as an override when present
        """
        try:
            self.loaded_user_config = self.user_config.exists()
            logger.debug(f"Loading default config from {self.default_config}")
            with open(self.default_config) as f:
                default_config = json.load(f)
            self.user_config_data = {}
            if self.loaded_user_config:
                logger.debug(f"Loading user config from {self.user_config}")
                with open(self.user_config) as f:
                    self.user_config_data = json.load(f)
                self.config = deep_merge(default_config, self.user_config_data)
            else:
                logger.debug("No user config found, using default config")
                self.config = default_config
            self.config = resolve_endpoint_config(apply_disabled_runtime_profiles(self.config))
                    
            # Ensure prompts is a list
            if not isinstance(self.config.get("prompts", []), list):
                logger.warning("Prompts in config is not a list, setting to empty list")
                self.config["prompts"] = []
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)

    def get_runtime_profile(self, profile_name: str | None = None) -> dict[str, Any]:
        """Return a named runtime profile for local endpoint/model defaults."""
        return get_runtime_profile(self.config, profile_name)

    def update_from_args(self, args: argparse.Namespace):
        """Update configuration with command line arguments."""
        for key, value in vars(args).items():
            if value is not None:  # Only update if argument was provided
                if key == "client":
                    self.config["clients"]["default"] = value
                elif key == "output":
                    self.config["output_dir"] = value
                elif key == "ollama_url":
                    self.config["clients"]["ollama"]["url"] = value
                elif key == "api_key":
                    self.config["clients"]["openai_api"]["api_key"] = value
                    # If key is provided but no client specified, use OpenAI API
                    if not args.client:
                        self.config["clients"]["default"] = "openai_api"
                elif key == "api_url":
                    self.config["clients"]["openai_api"]["api_url"] = value
                elif key == "llm_base_url":
                    self.config.setdefault("operation_manual", {})["llm_base_url"] = value
                    self.config["clients"]["openai_api"]["api_url"] = value
                    self.config["clients"]["openai_api"]["api_key"] = self.config["clients"]["openai_api"].get("api_key") or "0"
                    if not args.client:
                        self.config["clients"]["default"] = "openai_api"
                elif key == "vision_base_url":
                    self.config.setdefault("operation_manual", {})["vision_base_url"] = value
                    self.config["clients"]["openai_api"]["api_url"] = value
                    self.config["clients"]["openai_api"]["api_key"] = self.config["clients"]["openai_api"].get("api_key") or "0"
                    if not args.client:
                        self.config["clients"]["default"] = "openai_api"
                elif key == "text_base_url":
                    self.config.setdefault("operation_manual", {})["text_base_url"] = value
                elif key == "vision_model":
                    self.config.setdefault("operation_manual", {})["vision_model"] = value
                    self.config["clients"]["openai_api"]["model"] = value
                    self.config.setdefault("ocr", {})["fallback_model"] = value
                elif key == "text_model":
                    self.config.setdefault("operation_manual", {})["text_model"] = value
                elif key == "ocr_provider":
                    self.config.setdefault("ocr", {})["provider"] = value
                elif key == "ocr_base_url":
                    endpoints = normalize_string_list(value)
                    if endpoints:
                        self.config.setdefault("ocr", {})["base_url"] = endpoints[0]
                        self.config.setdefault("ocr", {})["base_urls"] = endpoints
                elif key == "ocr_concurrency":
                    self.config.setdefault("ocr", {})["concurrency"] = value
                elif key == "ocr_cache":
                    self.config.setdefault("ocr", {})["cache"] = value
                elif key == "ocr_cache_dir":
                    self.config.setdefault("ocr", {})["cache_dir"] = value
                elif key == "ocr_timeout_seconds":
                    self.config.setdefault("ocr", {})["timeout_seconds"] = value
                elif key == "ocr_prompt_mode":
                    self.config.setdefault("ocr", {})["prompt_mode"] = value
                elif key == "ocr_max_tokens":
                    self.config.setdefault("ocr", {})["max_tokens"] = value
                elif key == "ocr_max_image_long_side":
                    self.config.setdefault("ocr", {})["max_image_long_side"] = value
                elif key == "ocr_image_mode":
                    self.config.setdefault("ocr", {})["image_mode"] = value
                elif key == "ocr_retry_endpoints":
                    self.config.setdefault("ocr", {})["retry_endpoints"] = value
                elif key == "asr_provider":
                    self.config.setdefault("asr", {})["provider"] = value
                elif key == "asr_strategy":
                    self.config.setdefault("asr", {})["strategy"] = value
                elif key == "vibevoice_url":
                    self.config.setdefault("asr", {}).setdefault("vibevoice", {})["deep_remote_urls"] = value
                elif key == "remote_asr_url":
                    self.config.setdefault("asr", {}).setdefault("vibevoice", {})["remote_urls"] = value
                elif key == "model":
                    client = self.config["clients"]["default"]
                    self.config["clients"][client]["model"] = value
                elif key == "prompt":
                    self.config["prompt"] = value
                #overide audio config
                elif key == "whisper_model":
                    self.config["audio"]["whisper_model"] = value  # default is 'medium'
                elif key == "language":
                    if value is not None:
                        self.config["audio"]["language"] = value
                elif key == "device":
                    self.config["audio"]["device"] = value
                elif key == "temperature":
                    self.config["clients"]["temperature"] = value
                elif key not in ["start_stage", "max_frames"]:  # Ignore these as they're command-line only
                    self.config[key] = value

        if self.config.get("task") == "operation_manual" and not args.client:
            manual_config = self.config.setdefault("operation_manual", {})
            profile = self.get_runtime_profile(getattr(args, "profile", None))
            user_manual_config = (getattr(self, "user_config_data", {}).get("operation_manual") or {})
            user_ocr_config = (getattr(self, "user_config_data", {}).get("ocr") or {})

            ocr_config = self.config.setdefault("ocr", {})
            for profile_key, runtime_key, cli_key in (
                ("ocr_provider", "provider", "ocr_provider"),
                ("ocr_base_url", "base_url", "ocr_base_url"),
                ("ocr_base_urls", "base_urls", "ocr_base_url"),
                ("ocr_concurrency", "concurrency", "ocr_concurrency"),
                ("ocr_cache", "cache", "ocr_cache"),
                ("ocr_timeout_seconds", "timeout_seconds", "ocr_timeout_seconds"),
                ("ocr_max_tokens", "max_tokens", "ocr_max_tokens"),
                ("ocr_max_image_long_side", "max_image_long_side", "ocr_max_image_long_side"),
                ("ocr_image_mode", "image_mode", "ocr_image_mode"),
            ):
                if getattr(args, cli_key, None) is not None:
                    continue
                if runtime_key in user_ocr_config:
                    continue
                if profile.get(profile_key) is not None:
                    ocr_config[runtime_key] = copy.deepcopy(profile[profile_key])

            def profile_value(manual_key: str, profile_key: str, cli_key: str | None = None, fallback: Any = None) -> Any:
                if cli_key and getattr(args, cli_key, None) is not None:
                    return manual_config.get(manual_key)
                if manual_key in user_manual_config:
                    return manual_config.get(manual_key)
                return profile.get(profile_key, manual_config.get(manual_key, fallback))

            default_llm_base_url = (self.config.get("endpoints") or {}).get("services", {}).get(
                "amd_fast_base_url"
            )
            llm_base_url = profile_value(
                "llm_base_url",
                "llm_base_url",
                "llm_base_url",
                default_llm_base_url,
            )
            vision_base_url = profile_value(
                "vision_base_url",
                "vision_base_url",
                "vision_base_url",
                llm_base_url,
            )
            text_base_url = profile_value(
                "text_base_url",
                "text_base_url",
                "text_base_url",
                llm_base_url,
            )
            vision_model = profile_value(
                "vision_model",
                "vision_model",
                "vision_model",
                "hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive",
            )
            text_model = profile_value("text_model", "text_model", "text_model", vision_model)
            text_temperature = profile_value("text_temperature", "text_temperature")
            text_timeout_seconds = profile_value("text_timeout_seconds", "text_timeout_seconds")
            manual_config["llm_base_url"] = llm_base_url
            manual_config["vision_base_url"] = vision_base_url
            manual_config["text_base_url"] = text_base_url
            manual_config["vision_model"] = vision_model
            manual_config["text_model"] = text_model
            if text_temperature is not None:
                manual_config["text_temperature"] = text_temperature
            if text_timeout_seconds is not None:
                manual_config["text_timeout_seconds"] = int(text_timeout_seconds)
            if profile.get("text_api_key_env") and _is_deepseek_api(text_base_url):
                manual_config["text_api_key_env"] = profile["text_api_key_env"]
            elif (
                not _is_deepseek_api(text_base_url)
                and not (getattr(self, "user_config_data", {}).get("operation_manual") or {}).get("text_api_key_env")
            ):
                manual_config.pop("text_api_key_env", None)
            for extra_key in ("deepseek_thinking", "reasoning_effort"):
                if profile.get(extra_key):
                    manual_config[extra_key] = profile[extra_key]
            if "extra_body" in profile:
                manual_config["extra_body"] = copy.deepcopy(profile["extra_body"])
            for fallback_key in (
                "text_fallback_enabled",
                "text_fallback_base_url",
                "text_fallback_model",
                "text_fallback_api_key_env",
                "text_fallback_text_temperature",
                "text_fallback_text_timeout_seconds",
                "text_fallback_deepseek_thinking",
                "text_fallback_reasoning_effort",
            ):
                if fallback_key in profile:
                    manual_config[fallback_key] = copy.deepcopy(
                        profile[fallback_key]
                    )
            if profile.get("api_key_env"):
                self.config["clients"]["openai_api"]["api_key_env"] = profile["api_key_env"]
            self.config["clients"]["default"] = "openai_api"
            self.config["clients"]["openai_api"]["api_url"] = vision_base_url
            self.config["clients"]["openai_api"]["api_key"] = self.config["clients"]["openai_api"].get("api_key") or "0"
            self.config["clients"]["openai_api"]["model"] = vision_model
            user_config_asr_provider = "provider" in (getattr(self, "user_config_data", {}).get("asr") or {})
            if not getattr(args, "asr_provider", None) and not user_config_asr_provider:
                self.config.setdefault("asr", {})["provider"] = profile.get("asr_provider", "auto")
            services = (self.config.get("endpoints") or {}).get("services") or {}
            vibevoice = self.config.setdefault("asr", {}).setdefault("vibevoice", {})
            profile_vibevoice_urls = normalize_string_list(profile.get("vibevoice_urls") or profile.get("vibevoice_url"))
            if profile_vibevoice_urls and not getattr(args, "vibevoice_url", None):
                vibevoice["deep_remote_urls"] = profile_vibevoice_urls
            for key in (
                "worker_count",
                "use_native_chunking",
                "single_pass_max_duration_sec",
                "chunk_duration_sec",
                "chunk_overlap_sec",
                "chunk_parallel_workers",
            ):
                if key in profile:
                    vibevoice[key] = copy.deepcopy(profile[key])
            profile_remote_urls = normalize_string_list(profile.get("remote_asr_urls") or profile.get("remote_asr_url"))
            if profile_remote_urls and not getattr(args, "remote_asr_url", None):
                vibevoice["remote_urls"] = profile_remote_urls
            if profile.get("firered_3dspeaker_url"):
                vibevoice["firered_3dspeaker_url"] = profile["firered_3dspeaker_url"]
            if profile.get("firered_asr2_url"):
                vibevoice["firered_asr2_url"] = profile["firered_asr2_url"]
                vibevoice["firered_asr2_options"] = profile.get("firered_asr2_options") or {}
            if profile.get("qwen3_asr_url"):
                vibevoice["qwen3_asr_url"] = profile["qwen3_asr_url"]
                vibevoice["qwen3_asr_model"] = profile.get("qwen3_asr_model")
                vibevoice["qwen3_asr_options"] = profile.get("qwen3_asr_options") or {}
            if profile.get("openai_audio_url"):
                vibevoice["openai_audio_url"] = profile["openai_audio_url"]
                vibevoice["openai_audio_model"] = profile.get("openai_audio_model")
                vibevoice["asr_api_key_env"] = profile.get("asr_api_key_env")
            if services.get("capswriter_url"):
                vibevoice.setdefault("capswriter_url", services["capswriter_url"])
            if isinstance(profile.get("speaker_diarization"), dict):
                self.config["speaker_diarization"] = deep_merge(
                    self.config.get("speaker_diarization") or {},
                    profile["speaker_diarization"],
                )
            if profile.get("ocr_engine"):
                self.config.setdefault("ocr", {})["engine"] = profile["ocr_engine"]
            if profile.get("ocr_worker_count"):
                self.config.setdefault("ocr", {})["worker_count"] = profile["ocr_worker_count"]
            if isinstance(profile.get("vision_runtime"), dict):
                manual_config["vision_runtime"] = dict(profile["vision_runtime"])

    def save_user_config(self):
        """Save current configuration to user config file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.user_config, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.debug(f"Saved user config to {self.user_config}")
        except Exception as e:
            logger.error(f"Error saving user config: {e}")
            raise


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge dictionaries while replacing non-dict values."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value)
        return merged
    return override


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list):
        raw_values = value
    else:
        raw_values = [str(value)]

    values: list[str] = []
    for item in raw_values:
        for part in str(item).split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
    return values


def resolve_endpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve endpoint placeholders in configuration sections that contain URLs."""
    resolved = copy.deepcopy(config)
    registry = build_endpoint_registry(resolved)
    for key in (
        "clients",
        "operation_manual",
        "runtime_profiles",
        "model_catalog",
        "ocr",
        "asr",
        "study_cards",
    ):
        if key in resolved:
            resolved[key] = resolve_endpoint_refs(resolved[key], registry)
    if "endpoints" in resolved:
        resolved["endpoints"] = resolve_endpoint_refs(resolved["endpoints"], registry)
    return resolved


def build_endpoint_registry(config: dict[str, Any]) -> dict[str, Any]:
    endpoints = config.get("endpoints") or {}
    hosts = endpoints.get("hosts") or {}
    services = endpoints.get("services") or {}
    registry: dict[str, Any] = {}
    for name, value in hosts.items():
        registry[name] = value
        registry[f"hosts.{name}"] = value
    for name, value in services.items():
        resolved_value = resolve_endpoint_refs(value, registry)
        registry[name] = resolved_value
        registry[f"services.{name}"] = resolved_value
    return registry


def resolve_endpoint_refs(value: Any, registry: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: resolve_endpoint_refs(item, registry) for key, item in value.items()}
    if isinstance(value, list):
        resolved: list[Any] = []
        for item in value:
            item_value = resolve_endpoint_refs(item, registry)
            if isinstance(item_value, list):
                resolved.extend(item_value)
            else:
                resolved.append(item_value)
        return resolved
    if not isinstance(value, str):
        return value
    match = ENDPOINT_REF_RE.fullmatch(value)
    if match:
        key = match.group(1)
        if key not in registry:
            raise ValueError(f"Unknown endpoint placeholder {{{key}}}")
        return copy.deepcopy(registry[key])

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in registry:
            raise ValueError(f"Unknown endpoint placeholder {{{key}}}")
        replacement = registry[key]
        if isinstance(replacement, (dict, list)):
            raise ValueError(f"Endpoint placeholder {{{key}}} cannot be embedded in a string")
        return str(replacement)

    return ENDPOINT_REF_RE.sub(replace, value)


def get_runtime_profile(config: dict[str, Any], profile_name: str | None = None) -> dict[str, Any]:
    profiles = config.get("runtime_profiles") or {}
    name = profile_name or config.get("active_runtime_profile") or "spark"
    profile = profiles.get(name)
    if profile is None:
        available = ", ".join(sorted(profiles)) or "(none)"
        raise ValueError(f"Unknown runtime profile '{name}'. Available profiles: {available}")
    return expand_runtime_profile(config, dict(profile))


def build_openai_extra_body(settings: dict[str, Any], api_url: str | None = None, prefix: str = "") -> dict[str, Any]:
    """Build provider-specific OpenAI-compatible request body extensions."""
    extra_body = dict(settings.get(f"{prefix}extra_body") or {})
    thinking = settings.get(f"{prefix}deepseek_thinking")
    reasoning_effort = settings.get(f"{prefix}reasoning_effort")
    if _is_deepseek_api(api_url) and thinking:
        extra_body["thinking"] = {"type": str(thinking)}
        if reasoning_effort:
            extra_body["reasoning_effort"] = str(reasoning_effort)
    elif reasoning_effort:
        extra_body["reasoning_effort"] = str(reasoning_effort)
    return extra_body


def resolve_temperature(settings: dict[str, Any], fallback: float = 0.2, key: str = "text_temperature") -> float:
    value = settings.get(key)
    if value is None:
        return fallback
    return float(value)


def _is_deepseek_api(api_url: str | None) -> bool:
    parsed = urlparse(api_url or "")
    host = parsed.hostname or ""
    return host == "api.deepseek.com" or host.endswith(".api.deepseek.com")


def load_default_deepseek_env() -> bool:
    """Load the standard DeepSeek key env file without overriding the shell."""
    env_path = Path(os.environ.get("VIDEO_ANALYZER_DEEPSEEK_ENV", DEFAULT_DEEPSEEK_ENV)).expanduser()
    if not env_path.exists():
        return False
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key.split(None, 1)[1].strip()
            if not key or key in os.environ:
                continue
            value = value.strip().strip("'\"")
            os.environ[key] = value
        return True
    except OSError as exc:
        logger.warning("Could not load DeepSeek env file %s: %s", env_path, exc)
        return False


def resolve_api_key(
    api_key: str | None = None,
    api_key_env: str | None = None,
    api_url: str | None = None,
) -> str:
    """Resolve an API key from explicit config, an env var, or local placeholders."""
    env_name = (api_key_env or "").strip()
    parsed = urlparse(api_url or "")
    host = parsed.hostname or ""
    if not env_name and host.endswith("deepseek.com"):
        env_name = "DEEPSEEK_API_KEY"
    if env_name:
        if env_name == "DEEPSEEK_API_KEY" and not os.environ.get(env_name):
            load_default_deepseek_env()
        value = os.environ.get(env_name)
        if value:
            return value
        raise ValueError(f"API key environment variable {env_name} is required")

    key = (api_key or "").strip()
    if key.startswith("${") and key.endswith("}"):
        value = os.environ.get(key[2:-1])
        if value:
            return value
        raise ValueError(f"API key environment variable {key[2:-1]} is required")
    if key.startswith("$") and len(key) > 1:
        value = os.environ.get(key[1:])
        if value:
            return value
        raise ValueError(f"API key environment variable {key[1:]} is required")
    if key:
        return key
    if _allows_placeholder_api_key(api_url):
        return "0"
    raise ValueError("API key is required when using OpenAI API client")


def _allows_placeholder_api_key(api_url: str | None) -> bool:
    parsed = urlparse(api_url or "")
    host = parsed.hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_private or address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return host.endswith(".local") or host.endswith(".lan") or host.endswith(".taild500c8.ts.net")


def get_client(config: Config) -> dict:
    """Get the appropriate client configuration based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    
    if client_type == "ollama":
        return {"url": client_config.get("url", "http://localhost:11434")}
    elif client_type == "openai_api":
        api_url = client_config.get("api_url")
        if not api_url:
            raise ValueError("API URL is required when using OpenAI API client")
        api_key = resolve_api_key(
            client_config.get("api_key"),
            client_config.get("api_key_env"),
            api_url,
        )
        return {
            "api_key": api_key,
            "api_url": api_url,
            "timeout_seconds": int(client_config.get("timeout_seconds", 600)),
        }
    else:
        raise ValueError(f"Unknown client type: {client_type}")

def get_model(config: Config) -> str:
    """Get the appropriate model based on client type and configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    return client_config.get("model", "llama3.2-vision")
