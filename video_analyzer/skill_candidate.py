"""Backward-compatible entry points for skill distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .skill_distillation import (
    DEFAULT_DISTILLATION_PROFILE,
    SkillDistillationPipeline,
    distillation_summary,
    enable_distilled_skills,
    initialize_distillation,
)


def build_tool_skill_candidate(
    run_dir: Path,
    *,
    force: bool = True,
    profile_name: str = DEFAULT_DISTILLATION_PROFILE,
) -> dict[str, Any]:
    """Start the replacement distillation pipeline and run to its first review gate."""
    initialize_distillation(run_dir, profile_name=profile_name, force=force)
    SkillDistillationPipeline(run_dir).run_until_pause()
    return distillation_summary(run_dir)


def candidate_summary(run_dir: Path) -> dict[str, Any]:
    return distillation_summary(run_dir)


def enable_tool_skill_candidate(
    run_dir: Path,
    repo_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    return enable_distilled_skills(run_dir, repo_root, overwrite=overwrite)
