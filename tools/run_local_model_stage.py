#!/usr/bin/env python3
"""Run a command while a configured local model stage owns the GPU runtime."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analyzer.config import Config
from video_analyzer.local_model_runtime import local_model_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command under the local model stage lock")
    parser.add_argument("--stage", choices=("text", "tts"), required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Switch the local model stage under the shared lock, then release it without running a command",
    )
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command and not args.prepare_only:
        parser.error("command is required after --")
    if args.command and args.prepare_only:
        parser.error("--prepare-only cannot be combined with a command")
    return args


def prepare_runtime_config(config: Config, profile_name: str | None) -> dict:
    profile = config.get_runtime_profile(profile_name)
    manual = config.config.setdefault("operation_manual", {})
    for key in ("text_base_url", "llm_base_url", "text_worker_count", "text_gpu_ids", "text_port"):
        value = profile.get(key)
        if value is not None:
            manual[key] = value
    text_base_url = profile.get("text_base_url") or profile.get("llm_base_url")
    if text_base_url:
        manual["text_base_url"] = text_base_url
    config.config["tts"] = {
        "enabled": bool(profile.get("tts_enabled")),
        "base_url": profile.get("tts_base_url"),
    }
    return profile


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    config = Config(args.config)
    profile = prepare_runtime_config(config, args.profile)
    env = os.environ.copy()
    if profile.get("text_timeout_seconds") is not None:
        env["VIDEO_ANALYZER_TEXT_TIMEOUT_SECONDS"] = str(profile["text_timeout_seconds"])

    if args.prepare_only:
        logger.info("Preparing local model stage=%s", args.stage)
        with local_model_stage(args.stage, config.config, logger, "prepare-only"):
            pass
        return 0

    logger.info("Running command under local model stage=%s: %s", args.stage, " ".join(args.command))
    with local_model_stage(args.stage, config.config, logger, " ".join(args.command)):
        return subprocess.run(args.command, cwd=ROOT_DIR, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
