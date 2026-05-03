import argparse
from pathlib import Path
import json
from typing import Any
import logging
from importlib import resources

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_dir: str = "config"):
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
            if self.loaded_user_config:
                logger.debug(f"Loading user config from {self.user_config}")
                with open(self.user_config) as f:
                    user_config = json.load(f)
                self.config = deep_merge(default_config, user_config)
            else:
                logger.debug("No user config found, using default config")
                self.config = default_config
                    
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
                elif key == "vision_model":
                    self.config.setdefault("operation_manual", {})["vision_model"] = value
                    self.config["clients"]["openai_api"]["model"] = value
                    self.config.setdefault("ocr", {})["fallback_model"] = value
                elif key == "text_model":
                    self.config.setdefault("operation_manual", {})["text_model"] = value
                elif key == "ocr_provider":
                    self.config.setdefault("ocr", {})["provider"] = value
                elif key == "ocr_base_url":
                    self.config.setdefault("ocr", {})["base_url"] = value
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
            llm_base_url = manual_config.get("llm_base_url") or profile.get("llm_base_url", "http://127.0.0.1:1234/v1")
            vision_model = manual_config.get("vision_model") or profile.get("vision_model", "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive@?")
            self.config["clients"]["default"] = "openai_api"
            self.config["clients"]["openai_api"]["api_url"] = llm_base_url
            self.config["clients"]["openai_api"]["api_key"] = self.config["clients"]["openai_api"].get("api_key") or "0"
            self.config["clients"]["openai_api"]["model"] = vision_model
            user_config_asr_provider = self.loaded_user_config and "provider" in self.config.get("asr", {})
            if not getattr(args, "asr_provider", None) and not user_config_asr_provider:
                self.config.setdefault("asr", {})["provider"] = "auto"

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


def get_runtime_profile(config: dict[str, Any], profile_name: str | None = None) -> dict[str, Any]:
    profiles = config.get("runtime_profiles") or {}
    name = profile_name or config.get("active_runtime_profile") or "local_lan"
    profile = profiles.get(name)
    if profile is None:
        available = ", ".join(sorted(profiles)) or "(none)"
        raise ValueError(f"Unknown runtime profile '{name}'. Available profiles: {available}")
    return dict(profile)

def get_client(config: Config) -> dict:
    """Get the appropriate client configuration based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    
    if client_type == "ollama":
        return {"url": client_config.get("url", "http://localhost:11434")}
    elif client_type == "openai_api":
        api_key = client_config.get("api_key")
        api_url = client_config.get("api_url")
        if not api_key and api_url and "127.0.0.1" in api_url:
            api_key = "0"
        if not api_key:
            raise ValueError("API key is required when using OpenAI API client")
        if not api_url:
            raise ValueError("API URL is required when using OpenAI API client")
        return {
            "api_key": api_key,
            "api_url": api_url
        }
    else:
        raise ValueError(f"Unknown client type: {client_type}")

def get_model(config: Config) -> str:
    """Get the appropriate model based on client type and configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    return client_config.get("model", "llama3.2-vision")
