"""Evidence-grounded RIA-TV++ compatible skill distillation."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .clients.generic_openai_api import GenericOpenAIAPIClient
from .config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature
from .local_model_runtime import (
    DEFAULT_LOCK_PATH,
    is_loopback_endpoint,
    local_model_stage_needed,
    prepare_local_model_stage,
    unload_local_model_stage,
)

DEFAULT_DISTILLATION_PROFILE = "deepseek_v4_pro"
PACK_DIR = Path("skills") / "cangjie_pack"
STATE_NAME = "PIPELINE_STATE.json"
STATE_MD_NAME = "PIPELINE_STATE.md"
SOURCE_MANIFEST_NAME = "source_manifest.json"
SOURCE_RECORDS_NAME = "evidence_records.jsonl"
PROJECT_SOURCE_RECORDS_NAME = "skill_project_evidence.jsonl"
EVIDENCE_EVENTS_NAME = "evidence_events.json"
OVERVIEW_JSON_NAME = "BOOK_OVERVIEW.json"
OVERVIEW_MD_NAME = "BOOK_OVERVIEW.md"
VERIFIED_JSON_NAME = "verified.json"
VERIFIED_MD_NAME = "verified.md"
DISTILLATION_MANIFEST_NAME = "distillation_manifest.json"
METHOD_NAME = "RIA-TV++ compatible"
METHOD_SOURCE = "kangarooking/cangjie-skill"
PIPELINE_VERSION = 1
MAX_REPAIR_ROUNDS = 2
DEFAULT_CHUNK_CHARS = 30000
DEFAULT_EVENT_WINDOW_SECONDS = 30
DEFAULT_MULTIMODAL_IMAGES = 3
DEFAULT_VERIFICATION_BATCH_SIZE = 6
DEFAULT_VERIFICATION_BATCH_MAX_TOKENS = 6000
logger = logging.getLogger(__name__)

PIPELINE_STAGES = (
    "source",
    "overview",
    "extract",
    "verify",
    "build",
    "link",
    "test",
    "deliver",
)

EXTRACTOR_SPECS = {
    "frameworks": (
        "framework",
        "识别可重复使用的推理框架、决策结构和问题求解模型。不要提取普通事实或泛泛建议。",
    ),
    "principles": (
        "principle",
        "识别具有明确条件、约束或检查标准的原则、规则和清单。不要把口号当作原则。",
    ),
    "cases": (
        "case",
        "识别作者或演示者真实执行方法、遭遇问题、采取动作并得到结果的案例。",
    ),
    "counter_examples": (
        "counter-example",
        "识别失败尝试、风险、陷阱、返工原因以及明确不应采用的做法。",
    ),
    "glossary": (
        "term",
        "识别理解和执行该领域流程所必需的专有概念、部件、参数和术语。",
    ),
}


class DistillationError(RuntimeError):
    """Raised when a distillation stage cannot continue."""


class DistillationCancelled(DistillationError):
    """Raised when the caller cancels a running distillation."""


class DistillationResourceConflict(DistillationError):
    """Raised when a separate Skill worker would contend with a local model job."""


@dataclass(frozen=True)
class ModelRuntime:
    profile_name: str
    base_url: str
    generation_model: str
    review_model: str
    generation_temperature: float
    review_temperature: float
    generation_client: Any
    review_client: Any
    concurrency: int
    vision_base_url: str = ""
    vision_model: str = ""
    vision_temperature: float = 0.1
    vision_client: Any = None
    vision_concurrency: int = 1
    multimodal_enabled: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pack_dir(run_dir: Path) -> Path:
    return run_dir.expanduser().resolve() / PACK_DIR


def state_path(run_dir: Path) -> Path:
    return pack_dir(run_dir) / STATE_NAME


def load_state(run_dir: Path) -> dict[str, Any] | None:
    path = state_path(run_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistillationError(f"Invalid skill distillation state: {path}") from exc


def distillation_summary(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    state = load_state(run_dir)
    if not state:
        legacy_dir = run_dir / "skills" / "tool_skill_candidate"
        return {
            "available": False,
            "status": "not_started",
            "profile": DEFAULT_DISTILLATION_PROFILE,
            "generation_model": None,
            "review_model": None,
            "legacy_candidate_available": (legacy_dir / "SKILL.md").is_file(),
            "warnings": [],
        }
    result = {
        "available": state.get("status") in {"succeeded", "completed_no_skills"},
        "status": state.get("status"),
        "current_stage": state.get("current_stage"),
        "profile": state.get("profile") or DEFAULT_DISTILLATION_PROFILE,
        "generation_model": state.get("generation_model"),
        "review_model": state.get("review_model"),
        "vision_model": state.get("vision_model"),
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "error": state.get("error"),
        "retryable": bool(state.get("retryable")),
        "warnings": state.get("warnings") or [],
        "progress": stage_progress(state),
        "pack_dir": str(PACK_DIR),
        "overview": state.get("overview") or {},
        "candidates": state.get("candidates") or {},
        "skills": state.get("skills") or {},
        "installed": state.get("installed") or {},
        "artifacts": state.get("artifacts") or {},
    }
    result["enabled"] = bool((result["installed"] or {}).get("paths"))
    return result


def stage_progress(state: dict[str, Any]) -> dict[str, Any]:
    stages = state.get("stages") or {}
    completed = sum(
        1 for name in PIPELINE_STAGES if (stages.get(name) or {}).get("status") == "succeeded"
    )
    active_fraction = 0.0
    current_stage = str(state.get("current_stage") or "")
    current_info = stages.get(current_stage) or {}
    if current_info.get("status") == "running":
        active_fraction = max(
            0.0,
            min(float(current_info.get("progress_percent") or 0) / 100, 0.99),
        )
    waiting = state.get("status") in {"waiting_overview_review", "waiting_candidate_review"}
    percent = round(((completed + active_fraction) / len(PIPELINE_STAGES)) * 100)
    if waiting:
        percent = max(percent, 12 if state.get("status") == "waiting_overview_review" else 50)
    return {
        "completed": completed,
        "total": len(PIPELINE_STAGES),
        "percent": min(percent, 100),
        "stages": stages,
    }


def initialize_distillation(
    run_dir: Path,
    *,
    profile_name: str = DEFAULT_DISTILLATION_PROFILE,
    config_dir: str = "config",
    force: bool = False,
    target_brief: dict[str, Any] | None = None,
    assessment: dict[str, Any] | None = None,
    source_records: list[dict[str, Any]] | None = None,
    reference_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    root = pack_dir(run_dir)
    existing = load_state(run_dir)
    if existing and not force:
        raise FileExistsError("Skill distillation already exists; resume it or restart explicitly")
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    project_records_path = None
    if source_records is not None:
        project_records_path = PROJECT_SOURCE_RECORDS_NAME
        write_jsonl(run_dir / project_records_path, source_records)
    runtime = resolve_model_runtime(profile_name, config_dir=config_dir)
    now = utc_now()
    state = {
        "version": PIPELINE_VERSION,
        "method": METHOD_NAME,
        "method_source": METHOD_SOURCE,
        "run_dir": str(run_dir),
        "status": "ready",
        "current_stage": "source",
        "profile": runtime.profile_name,
        "config_dir": config_dir,
        "generation_model": runtime.generation_model,
        "review_model": runtime.review_model,
        "vision_model": runtime.vision_model,
        "created_at": now,
        "updated_at": now,
        "stages": {name: {"status": "pending"} for name in PIPELINE_STAGES},
        "warnings": [],
        "overview": {"reviewed": False},
        "candidates": {"reviewed": False, "selected_ids": []},
        "skills": {"count": 0, "passed": 0, "failed": 0, "items": []},
        "installed": {"paths": []},
        "artifacts": {},
        "target_brief": dict(target_brief or {}),
        "assessment": dict(assessment or {}),
        "project_source_records_path": project_records_path,
        "project_reference_context": list(reference_context or []),
    }
    save_state(run_dir, state)
    return state


def resolve_model_runtime(
    profile_name: str,
    *,
    config_dir: str = "config",
    client_factory: Callable[..., Any] = GenericOpenAIAPIClient,
) -> ModelRuntime:
    config = Config(config_dir)
    profile = config.get_runtime_profile(profile_name or DEFAULT_DISTILLATION_PROFILE)
    base_url = str(profile.get("text_base_url") or profile.get("llm_base_url") or "").rstrip("/")
    review_base_url = str(profile.get("review_base_url") or base_url).rstrip("/")
    generation_model = str(profile.get("text_model") or "").strip()
    review_model = str(profile.get("review_model") or generation_model).strip()
    if not base_url or not generation_model:
        raise DistillationError(
            f"Profile {profile_name} must provide text_base_url/llm_base_url and text_model"
        )
    api_key = resolve_api_key(
        profile.get("api_key"),
        profile.get("text_api_key_env") or profile.get("api_key_env"),
        base_url,
    )
    timeout = int(profile.get("text_timeout_seconds") or 600)
    generation_client = client_factory(
        api_key,
        base_url,
        timeout_seconds=timeout,
        extra_body=build_openai_extra_body(profile, base_url),
    )
    review_api_key = resolve_api_key(
        profile.get("review_api_key"),
        profile.get("review_api_key_env")
        or profile.get("text_api_key_env")
        or profile.get("api_key_env"),
        review_base_url,
    )
    review_client = client_factory(
        review_api_key,
        review_base_url,
        timeout_seconds=int(profile.get("review_timeout_seconds") or timeout),
        extra_body=build_openai_extra_body(profile, review_base_url, prefix="review_"),
    )
    configured_concurrency = int(
        (config.get("skill_distillation") or {}).get("concurrency") or 0
    )
    distillation_config = config.get("skill_distillation") or {}
    concurrency = configured_concurrency or (1 if is_local_endpoint(base_url) else 5)
    vision_base_url = str(profile.get("vision_base_url") or "").rstrip("/")
    vision_model = str(profile.get("vision_model") or "").strip()
    multimodal_enabled = bool(
        distillation_config.get("multimodal_verification", True)
        and vision_base_url
        and vision_model
    )
    vision_client = None
    if multimodal_enabled:
        vision_api_key = resolve_api_key(
            profile.get("vision_api_key"),
            profile.get("vision_api_key_env") or profile.get("api_key_env"),
            vision_base_url,
        )
        vision_client = client_factory(
            vision_api_key,
            vision_base_url,
            timeout_seconds=int(profile.get("vision_timeout_seconds") or 600),
            extra_body=build_openai_extra_body(profile, vision_base_url, prefix="vision_"),
        )
    vision_concurrency = int(
        distillation_config.get("vision_concurrency")
        or profile.get("vl_concurrency")
        or 1
    )
    return ModelRuntime(
        profile_name=profile_name,
        base_url=base_url,
        generation_model=generation_model,
        review_model=review_model,
        generation_temperature=resolve_temperature(profile, 0.2),
        review_temperature=resolve_temperature(
            profile,
            resolve_temperature(profile, 0.2),
            key="review_temperature",
        ),
        generation_client=generation_client,
        review_client=review_client,
        concurrency=max(1, min(concurrency, 5)),
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        vision_temperature=float(profile.get("vision_temperature") or 0.1),
        vision_client=vision_client,
        vision_concurrency=max(1, min(vision_concurrency, 5)),
        multimodal_enabled=multimodal_enabled,
    )


def is_local_endpoint(base_url: str) -> bool:
    return is_loopback_endpoint(base_url)


@contextlib.contextmanager
def skill_local_model_stage(stage: str, config: dict, owner: str) -> Iterable[None]:
    """Use a local model only when its shared runtime is currently idle.

    Skill workers must not queue behind the primary Video Analyzer pipeline:
    they persist a resource-conflict checkpoint and wait for an explicit user
    decision instead.
    """
    if not local_model_stage_needed(stage, config):
        yield
        return
    runtime = config.get("local_model_runtime") or {}
    lock_path = Path(
        os.environ.get("VIDEO_ANALYZER_LOCAL_MODEL_LOCK")
        or runtime.get("lock_path")
        or DEFAULT_LOCK_PATH
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            try:
                holder = json.loads(lock_path.read_text(encoding="utf-8") or "{}")
            except (OSError, ValueError, json.JSONDecodeError):
                holder = {}
            details = "本地模型正在被主视频流程使用"
            if holder.get("stage") or holder.get("owner"):
                details += (
                    f"（stage={holder.get('stage') or 'unknown'}"
                    f"，owner={holder.get('owner') or 'unknown'}）"
                )
            raise DistillationResourceConflict(
                f"{details}；Skill 已暂停，等待你的资源裁定。"
            ) from exc
        prepare_local_model_stage(stage, config, logger)
        try:
            yield
        finally:
            unload_local_model_stage(config, logger)
    finally:
        try:
            if acquired:
                os.ftruncate(fd, 0)
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class SkillDistillationPipeline:
    def __init__(
        self,
        run_dir: Path,
        *,
        config_dir: str = "config",
        client_factory: Callable[..., Any] = GenericOpenAIAPIClient,
    ):
        self.run_dir = run_dir.expanduser().resolve()
        self.root = pack_dir(self.run_dir)
        state = load_state(self.run_dir)
        if not state:
            raise DistillationError("Skill distillation is not initialized")
        if config_dir == "config" and state.get("config_dir"):
            config_dir = str(state["config_dir"])
        self.state = state
        self.config = Config(config_dir).config
        self.runtime = resolve_model_runtime(
            str(state.get("profile") or DEFAULT_DISTILLATION_PROFILE),
            config_dir=config_dir,
            client_factory=client_factory,
        )
        self._cancel_event: threading.Event | None = None

    def run_until_pause(
        self,
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        self._cancel_event = cancel_event
        try:
            if is_loopback_endpoint(self.runtime.base_url):
                raise DistillationResourceConflict(
                    "当前 Skill 文本模型使用本地共享运行时；Skill 已暂停，等待你的资源裁定。"
                )
            return self._run_until_pause_locked(cancel_event)
        except DistillationResourceConflict as exc:
            self._set_state(
                status="waiting_resource_decision",
                retryable=True,
                error=str(exc),
            )
            self._append_log(
                str(self.state.get("current_stage") or "pipeline"),
                "resource_conflict",
                str(exc),
            )
            return self.state
        finally:
            self._cancel_event = None

    def _run_until_pause_locked(
        self,
        cancel_event: threading.Event | None,
    ) -> dict[str, Any]:
        try:
            self._set_state(status="running", error=None, retryable=False)
            while True:
                self._check_cancel(cancel_event)
                stage = str(self.state.get("current_stage") or "")
                if stage == "source":
                    self._run_stage("source", self._build_source_bundle)
                    self._advance("overview")
                    continue
                if stage == "overview":
                    self._run_stage("overview", self._build_overview)
                    self._set_state(status="waiting_overview_review")
                    return self.state
                if stage == "extract":
                    self._run_stage("extract", self._extract_candidates)
                    self._advance("verify")
                    continue
                if stage == "verify":
                    self._run_stage("verify", self._verify_candidates)
                    self._set_state(status="waiting_candidate_review")
                    return self.state
                if stage == "build":
                    self._run_stage("build", self._build_skills)
                    self._advance("link")
                    continue
                if stage == "link":
                    self._run_stage("link", self._link_skills)
                    self._advance("test")
                    continue
                if stage == "test":
                    self._run_stage("test", self._test_skills)
                    self._advance("deliver")
                    continue
                if stage == "deliver":
                    self._run_stage("deliver", self._deliver)
                    final_status = (
                        "succeeded"
                        if int((self.state.get("skills") or {}).get("passed") or 0) > 0
                        else "completed_no_skills"
                    )
                    self._set_state(status=final_status, current_stage=None)
                    return self.state
                if not stage:
                    return self.state
                raise DistillationError(f"Unknown distillation stage: {stage}")
        except DistillationResourceConflict as exc:
            self._set_state(
                status="waiting_resource_decision",
                retryable=True,
                error=str(exc),
            )
            self._append_log(str(self.state.get("current_stage") or "pipeline"), "resource_conflict", str(exc))
            return self.state
        except DistillationCancelled:
            self._set_state(status="cancelled", retryable=True, error="distillation cancelled")
            return self.state
        except Exception as exc:
            self._set_state(status="failed", retryable=True, error=str(exc))
            self._append_log(str(self.state.get("current_stage") or "pipeline"), "error", str(exc))
            raise

    def review_overview(self, action: str, feedback: str = "") -> dict[str, Any]:
        if self.state.get("status") != "waiting_overview_review":
            raise DistillationError("Overview is not waiting for review")
        action = str(action or "confirm").strip().lower()
        if action == "regenerate":
            overview = dict(self.state.get("overview") or {})
            overview["feedback"] = feedback.strip()
            overview["reviewed"] = False
            self.state["overview"] = overview
            self._invalidate_from("overview")
            self._set_state(status="ready", current_stage="overview")
            return self.state
        if action != "confirm":
            raise ValueError("overview action must be confirm or regenerate")
        overview = dict(self.state.get("overview") or {})
        overview.update({"reviewed": True, "feedback": feedback.strip(), "reviewed_at": utc_now()})
        self.state["overview"] = overview
        self._set_state(status="ready", current_stage="extract")
        return self.state

    def review_candidates(self, selected_ids: Iterable[str]) -> dict[str, Any]:
        if self.state.get("status") != "waiting_candidate_review":
            raise DistillationError("Candidates are not waiting for review")
        verified = read_json(self.root / VERIFIED_JSON_NAME) or {}
        all_candidates = (
            list(verified.get("accepted") or [])
            + list(verified.get("single_case") or [])
            + list(verified.get("rejected") or [])
        )
        allowed = {str(item.get("id")) for item in all_candidates if item.get("id")}
        selected = list(dict.fromkeys(str(value).strip() for value in selected_ids if str(value).strip()))
        unknown = [value for value in selected if value not in allowed]
        if unknown:
            raise ValueError(f"Unknown candidate ids: {unknown}")
        candidates = dict(self.state.get("candidates") or {})
        candidates.update(
            {
                "reviewed": True,
                "reviewed_at": utc_now(),
                "selected_ids": selected,
                "selected_count": len(selected),
            }
        )
        self.state["candidates"] = candidates
        self._set_state(status="ready", current_stage="build")
        return self.state

    def mark_interrupted(self) -> dict[str, Any]:
        if self.state.get("status") == "running":
            self._set_state(
                status="interrupted",
                retryable=True,
                error="server restarted while skill distillation was running",
            )
        return self.state

    def _run_stage(self, stage: str, func: Callable[[], None]) -> None:
        info = dict((self.state.get("stages") or {}).get(stage) or {})
        if info.get("status") == "succeeded":
            return
        started = utc_now()
        self._update_stage(stage, status="running", started_at=started, error=None)
        self._append_log(stage, "start", f"stage={stage}")
        try:
            func()
        except Exception as exc:
            self._update_stage(stage, status="failed", finished_at=utc_now(), error=str(exc))
            raise
        self._update_stage(stage, status="succeeded", finished_at=utc_now(), error=None)
        self._append_log(stage, "done", f"stage={stage}")

    def _build_source_bundle(self) -> None:
        project_source_path = str(self.state.get("project_source_records_path") or "")
        if project_source_path:
            source_path = (self.run_dir / project_source_path).resolve()
            try:
                source_path.relative_to(self.run_dir)
            except ValueError as exc:
                raise DistillationError("Project source records path escapes run directory") from exc
            records = read_jsonl(source_path)
            sources = [{"type": "skill_project", "paths": [project_source_path], "records": len(records)}]
        else:
            records, sources = load_evidence_records(self.run_dir)
        if not records:
            raise FileNotFoundError(
                "No substantive transcript, subtitle, OCR, VL, or page evidence is available"
            )
        high_confidence = [
            record
            for record in records
            if record.get("confidence") == "high"
            and record.get("source_type") in {"transcript", "subtitle", "ocr", "visual"}
        ]
        if not high_confidence:
            raise FileNotFoundError(
                "Skill distillation requires transcript/subtitle/OCR/VL evidence; index and derived documents are insufficient"
            )
        event_window_seconds = int(
            (self.config.get("skill_distillation") or {}).get(
                "evidence_event_window_seconds",
                DEFAULT_EVENT_WINDOW_SECONDS,
            )
        )
        records, events = assign_evidence_events(
            records,
            window_seconds=max(1, event_window_seconds),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.root / SOURCE_RECORDS_NAME, records)
        write_json(self.root / EVIDENCE_EVENTS_NAME, {"version": 1, "events": events})
        fingerprint = evidence_fingerprint(records)
        manifest = {
            "version": 1,
            "generated_at": utc_now(),
            "fingerprint": fingerprint,
            "record_count": len(records),
            "high_confidence_count": len(high_confidence),
            "sources": sources,
            "records_file": SOURCE_RECORDS_NAME,
            "events_file": EVIDENCE_EVENTS_NAME,
            "event_count": len(events),
        }
        write_json(self.root / SOURCE_MANIFEST_NAME, manifest)
        self.state["source_fingerprint"] = fingerprint
        self.state.setdefault("artifacts", {})["source_manifest"] = str(
            PACK_DIR / SOURCE_MANIFEST_NAME
        )
        self.state["artifacts"]["source_records"] = str(PACK_DIR / SOURCE_RECORDS_NAME)
        self.state["artifacts"]["evidence_events"] = str(PACK_DIR / EVIDENCE_EVENTS_NAME)
        save_state(self.run_dir, self.state)

    def _build_overview(self) -> None:
        records = read_jsonl(self.root / SOURCE_RECORDS_NAME)
        chunks = build_evidence_chunks(records)
        digest_dir = self.root / "overview_chunks"
        digest_dir.mkdir(parents=True, exist_ok=True)
        digests = []
        for index, chunk in enumerate(chunks, start=1):
            checkpoint = digest_dir / f"chunk_{index:03d}.json"
            cached = read_json(checkpoint)
            if cached:
                digests.append(cached)
                continue
            prompt = overview_chunk_prompt(index, len(chunks), chunk)
            digest = call_json(
                self.runtime.generation_client,
                self.runtime.generation_model,
                prompt,
                self.runtime.generation_temperature,
                5000,
            )
            digest["chunk_index"] = index
            write_json(checkpoint, digest)
            digests.append(digest)
            self._append_log("overview", "chunk", f"{index}/{len(chunks)}")
        feedback = str((self.state.get("overview") or {}).get("feedback") or "")
        generated_overview = call_json(
            self.runtime.generation_client,
            self.runtime.generation_model,
            overview_synthesis_prompt(
                digests,
                feedback,
                self._target_brief(),
                self._reference_context(),
            ),
            self.runtime.generation_temperature,
            7000,
        )
        overview = normalize_overview(
            generated_overview,
            {record["id"]: record for record in records},
        )
        declared_fields = ("structure", "methods", "concepts", "cases", "failures")
        if (
            all(field in generated_overview for field in declared_fields)
            and not overview_has_substantive_content(overview)
        ):
            raise DistillationError(
                "Overview synthesis did not produce evidence-grounded structure, methods, concepts, cases, or risks"
            )
        overview["method"] = METHOD_NAME
        overview["source_fingerprint"] = self.state.get("source_fingerprint")
        write_json(self.root / OVERVIEW_JSON_NAME, overview)
        (self.root / OVERVIEW_MD_NAME).write_text(render_overview(overview), encoding="utf-8")
        self.state["overview"] = {
            **dict(self.state.get("overview") or {}),
            "reviewed": False,
            "title": overview.get("title"),
            "summary": overview.get("summary"),
            "artifact": str(PACK_DIR / OVERVIEW_MD_NAME),
            "chunk_count": len(chunks),
        }
        self.state.setdefault("artifacts", {})["overview"] = str(PACK_DIR / OVERVIEW_MD_NAME)
        save_state(self.run_dir, self.state)

    def _extract_candidates(self) -> None:
        overview = read_json(self.root / OVERVIEW_JSON_NAME) or {}
        digest_paths = sorted((self.root / "overview_chunks").glob("chunk_*.json"))
        digests = [read_json(path) for path in digest_paths]
        digests = [item for item in digests if item]
        candidates_dir = self.root / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        def run_extractor(name: str, spec: tuple[str, str]) -> tuple[str, dict[str, Any]]:
            candidate_type, instruction = spec
            result = call_json(
                self.runtime.generation_client,
                self.runtime.generation_model,
                extractor_prompt(
                    name,
                    candidate_type,
                    instruction,
                    overview,
                    digests,
                    self._target_brief(),
                ),
                self.runtime.generation_temperature,
                7000,
            )
            normalized = normalize_extractor_result(name, candidate_type, result)
            write_json(candidates_dir / f"{name}.json", normalized)
            (candidates_dir / f"{name}.md").write_text(
                render_candidates(name, normalized.get("candidates") or []),
                encoding="utf-8",
            )
            return name, normalized

        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.runtime.concurrency) as executor:
            futures = {
                executor.submit(run_extractor, name, spec): name
                for name, spec in EXTRACTOR_SPECS.items()
            }
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result
                self._append_log("extract", "extractor", name)
        count = sum(len((result or {}).get("candidates") or []) for result in results.values())
        self.state["candidates"] = {
            **dict(self.state.get("candidates") or {}),
            "reviewed": False,
            "extracted_count": count,
        }
        save_state(self.run_dir, self.state)

    def _verify_candidates(self) -> None:
        record_list = read_jsonl(self.root / SOURCE_RECORDS_NAME)
        if any(not item.get("event_id") for item in record_list):
            event_window_seconds = int(
                (self.config.get("skill_distillation") or {}).get(
                    "evidence_event_window_seconds",
                    DEFAULT_EVENT_WINDOW_SECONDS,
                )
            )
            record_list, events = assign_evidence_events(
                record_list,
                window_seconds=max(1, event_window_seconds),
            )
            write_jsonl(self.root / SOURCE_RECORDS_NAME, record_list)
            write_json(self.root / EVIDENCE_EVENTS_NAME, {"version": 1, "events": events})
        records = {item["id"]: item for item in record_list}
        candidate_payloads = []
        for path in sorted((self.root / "candidates").glob("*.json")):
            payload = read_json(path) or {}
            candidate_payloads.extend(payload.get("candidates") or [])
        glossary = [
            item for item in candidate_payloads if str(item.get("type") or "") == "term"
        ]
        skill_candidates = [
            item for item in candidate_payloads if str(item.get("type") or "") != "term"
        ]
        result = self._review_candidate_batches(skill_candidates)
        audits = self._run_multimodal_candidate_audits(skill_candidates, records)
        verified = normalize_verification(
            result,
            records,
            skill_candidates,
            multimodal_audits=audits,
            glossary=glossary,
        )
        write_json(self.root / VERIFIED_JSON_NAME, verified)
        (self.root / VERIFIED_MD_NAME).write_text(render_verified(verified), encoding="utf-8")
        write_json(
            self.root / "multimodal_audits.json",
            {"version": 1, "audits": audits},
        )
        rejected_dir = self.root / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for item in verified["rejected"]:
            (rejected_dir / f"{safe_slug(item.get('id') or 'candidate')}.md").write_text(
                render_rejected(item),
                encoding="utf-8",
            )
        self.state["candidates"] = {
            **dict(self.state.get("candidates") or {}),
            "reviewed": False,
            "accepted": verified["accepted"],
            "single_case": verified["single_case"],
            "rejected": verified["rejected"],
            "glossary": verified["glossary"],
            "accepted_count": len(verified["accepted"]),
            "single_case_count": len(verified["single_case"]),
            "rejected_count": len(verified["rejected"]),
            "glossary_count": len(verified["glossary"]),
            "selected_ids": [item["id"] for item in verified["accepted"]],
        }
        self.state.setdefault("artifacts", {})["verified"] = str(PACK_DIR / VERIFIED_MD_NAME)
        self.state["artifacts"]["multimodal_audits"] = str(
            PACK_DIR / "multimodal_audits.json"
        )
        save_state(self.run_dir, self.state)

    def _review_candidate_batches(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        settings = self.config.get("skill_distillation") or {}
        batch_size = max(
            1,
            int(settings.get("verification_batch_size") or DEFAULT_VERIFICATION_BATCH_SIZE),
        )
        max_tokens = max(
            1000,
            int(
                settings.get("verification_batch_max_tokens")
                or DEFAULT_VERIFICATION_BATCH_MAX_TOKENS
            ),
        )
        batches = [
            candidates[index : index + batch_size]
            for index in range(0, len(candidates), batch_size)
        ]
        if not batches:
            self._append_log("verify", "review_skipped", "没有可评审候选")
            return {"evaluations": []}
        checkpoint_dir = self.root / "verification_batches"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        total = len(batches)
        for index, batch in enumerate(batches, start=1):
            self._check_cancel(self._cancel_event)
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "candidates": batch,
                        "target_brief": self._target_brief(),
                        "review_model": self.runtime.review_model,
                        "schema": 1,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            checkpoint = checkpoint_dir / f"batch_{index:03d}.json"
            cached = read_json(checkpoint) or {}
            if cached.get("signature") == signature and cached.get("status") == "succeeded":
                result = dict(cached.get("result") or {})
                self._append_log(
                    "verify",
                    "review_batch_cached",
                    f"文本评审批次 {index}/{total}",
                )
            else:
                self._append_log(
                    "verify",
                    "review_batch_start",
                    f"文本评审批次 {index}/{total} · {len(batch)} 个候选",
                )
                self._update_stage(
                    "verify",
                    progress_percent=round(((index - 1) / total) * 55),
                    message=f"正在文本评审第 {index}/{total} 批",
                )
                result = call_json(
                    self.runtime.review_client,
                    self.runtime.review_model,
                    verification_prompt(batch, self._target_brief()),
                    self.runtime.review_temperature,
                    max_tokens,
                )
                write_json(
                    checkpoint,
                    {
                        "version": 1,
                        "status": "succeeded",
                        "signature": signature,
                        "completed_at": utc_now(),
                        "result": result,
                    },
                )
                self._append_log(
                    "verify",
                    "review_batch_done",
                    f"文本评审批次 {index}/{total}",
                )
            results.append(result)
            self._update_stage(
                "verify",
                progress_percent=round((index / total) * 55),
                message=f"文本评审已完成 {index}/{total} 批",
            )
        merged = merge_verification_batch_results(results)
        write_json(self.root / "verification_review.json", merged)
        return merged

    def _run_multimodal_candidate_audits(
        self,
        candidates: list[dict[str, Any]],
        records: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not self.runtime.multimodal_enabled or not self.runtime.vision_client:
            return {}
        max_images = int(
            (self.config.get("skill_distillation") or {}).get(
                "max_candidate_images",
                DEFAULT_MULTIMODAL_IMAGES,
            )
        )

        def plan(candidate: dict[str, Any]) -> dict[str, Any]:
            candidate_id = str(candidate.get("id") or "")
            evidence = [
                records[evidence_id]
                for evidence_id in candidate.get("source_ids") or []
                if evidence_id in records
            ]
            event_ids = unique_strings(
                item.get("event_id") for item in evidence if item.get("event_id")
            )
            image_paths = candidate_frame_paths(
                self.run_dir,
                evidence,
                records,
                max_images=max(1, max_images),
            )
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "candidate": candidate,
                        "event_ids": event_ids,
                        "image_paths": [str(path) for path in image_paths],
                        "vision_model": self.runtime.vision_model,
                        "schema": 2,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            checkpoint = self.root / "multimodal_audits" / f"{safe_slug(candidate_id)}.json"
            return {
                "candidate": candidate,
                "candidate_id": candidate_id,
                "evidence": evidence,
                "event_ids": event_ids,
                "image_paths": image_paths,
                "signature": signature,
                "checkpoint": checkpoint,
            }

        def audit(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            candidate = item["candidate"]
            candidate_id = item["candidate_id"]
            evidence = item["evidence"]
            event_ids = item["event_ids"]
            image_paths = item["image_paths"]
            signature = item["signature"]
            checkpoint = item["checkpoint"]
            try:
                result = call_json(
                    self.runtime.vision_client,
                    self.runtime.vision_model,
                    multimodal_verification_prompt(candidate, evidence, event_ids),
                    self.runtime.vision_temperature,
                    2500,
                    image_paths=[str(path) for path in image_paths],
                )
                normalized = normalize_multimodal_audit(result, event_ids)
                normalized["status"] = "succeeded"
            except Exception as exc:
                normalized = {
                    "status": "failed",
                    "error": str(exc),
                    "event_ids": event_ids,
                }
            normalized["signature"] = signature
            normalized["image_paths"] = [
                str(path.relative_to(self.run_dir)) for path in image_paths
            ]
            write_json(checkpoint, normalized)
            return candidate_id, normalized

        vision_config = {
            **self.config,
            "operation_manual": {
                **dict(self.config.get("operation_manual") or {}),
                "vision_base_url": self.runtime.vision_base_url,
                "vision_model": self.runtime.vision_model,
            },
        }
        audits: dict[str, dict[str, Any]] = {}
        pending: list[dict[str, Any]] = []
        for candidate in candidates:
            item = plan(candidate)
            candidate_id = item["candidate_id"]
            cached = read_json(item["checkpoint"]) or {}
            if cached.get("signature") == item["signature"] and cached.get("status") == "succeeded":
                audits[candidate_id] = cached
                self._append_log("verify", "multimodal_cached", candidate_id)
                continue
            if not item["image_paths"]:
                result = {
                    "status": "no_frames",
                    "event_ids": item["event_ids"],
                    "image_paths": [],
                    "signature": item["signature"],
                }
                write_json(item["checkpoint"], result)
                audits[candidate_id] = result
                self._append_log("verify", "multimodal_skipped", f"{candidate_id} · 无可复核帧")
                continue
            pending.append(item)
        if not pending:
            self._update_stage(
                "verify",
                progress_percent=100,
                message="所有候选均无可复核帧，已跳过本地视觉模型",
            )
            self._append_log("verify", "multimodal_skipped", "所有候选均无可复核帧，不占用本地视觉模型")
            return audits

        self._append_log(
            "verify",
            "local_vision_check",
            f"{len(pending)} 个候选需要视觉复核，正在检查本地视觉模型是否空闲",
        )
        with skill_local_model_stage(
            "vl",
            vision_config,
            f"skill-distillation-vision:{self.run_dir}",
        ):
            self._append_log("verify", "local_vision_acquired", f"开始视觉复核 {len(pending)} 个候选")
            with ThreadPoolExecutor(max_workers=self.runtime.vision_concurrency) as executor:
                futures = [executor.submit(audit, item) for item in pending]
                for future in as_completed(futures):
                    self._check_cancel(self._cancel_event)
                    candidate_id, result = future.result()
                    audits[candidate_id] = result
                    self._append_log("verify", "multimodal", candidate_id)
                    self._update_stage(
                        "verify",
                        progress_percent=55 + round((len(audits) / len(candidates)) * 45),
                        message=f"视觉复核 {len(audits)}/{len(candidates)}",
                    )
        return audits

    def _build_skills(self) -> None:
        verified = read_json(self.root / VERIFIED_JSON_NAME) or {}
        all_candidates = {
            str(item.get("id")): item
            for item in (
                list(verified.get("accepted") or [])
                + list(verified.get("single_case") or [])
                + list(verified.get("rejected") or [])
            )
            if item.get("id")
        }
        selected_ids = list((self.state.get("candidates") or {}).get("selected_ids") or [])
        records = {item["id"]: item for item in read_jsonl(self.root / SOURCE_RECORDS_NAME)}
        overview = read_json(self.root / OVERVIEW_JSON_NAME) or {}
        selected = [all_candidates[item_id] for item_id in selected_ids if item_id in all_candidates]
        skills_dir = self.root / "distilled_skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for candidate in selected:
            candidate_id = str(candidate["id"])
            existing_slug = existing_skill_slug_for_candidate(skills_dir, candidate_id)
            if existing_slug:
                existing_dir = skills_dir / existing_slug
                existing_skill = read_json(existing_dir / "skill.json") or {}
                if (
                    str(existing_skill.get("candidate_id") or "") == candidate_id
                    and (existing_dir / "SKILL.md").is_file()
                ):
                    items.append(
                        {
                            "candidate_id": candidate_id,
                            "name": existing_slug,
                            "title": existing_skill.get("title") or candidate.get("title"),
                            "status": "built",
                            "path": str(
                                PACK_DIR / "distilled_skills" / existing_slug / "SKILL.md"
                            ),
                        }
                    )
                    self._append_log("build", "reuse", existing_slug)
                    continue
            evidence = [
                records[evidence_id]
                for evidence_id in candidate.get("source_ids") or []
                if evidence_id in records
            ][:12]
            result = call_json(
                self.runtime.generation_client,
                self.runtime.generation_model,
                skill_build_prompt(
                    candidate,
                    evidence,
                    selected,
                    overview,
                    self._target_brief(),
                ),
                self.runtime.generation_temperature,
                9000,
            )
            skill = normalize_skill(result, candidate)
            slug = existing_slug or unique_skill_slug(
                skills_dir,
                safe_slug(
                    skill.get("name")
                    or skill.get("title")
                    or candidate.get("title")
                    or candidate["id"]
                ),
            )
            skill["name"] = slug
            skill_dir = skills_dir / slug
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(render_skill(skill), encoding="utf-8")
            write_json(skill_dir / "skill.json", skill)
            items.append(
                {
                    "candidate_id": candidate["id"],
                    "name": slug,
                    "title": skill.get("title"),
                    "status": "built",
                    "path": str((PACK_DIR / "distilled_skills" / slug / "SKILL.md")),
                }
            )
            self._append_log("build", "skill", slug)
        self.state["skills"] = {
            "count": len(items),
            "passed": 0,
            "failed": 0,
            "items": items,
        }
        save_state(self.run_dir, self.state)

    def _link_skills(self) -> None:
        skill_paths = active_skill_paths(self.root, self.state)
        skills = [read_json(path) for path in skill_paths]
        skills = [item for item in skills if item]
        if skills:
            result = call_json(
                self.runtime.generation_client,
                self.runtime.generation_model,
                linking_prompt(skills),
                self.runtime.generation_temperature,
                5000,
            )
        else:
            result = {"links": []}
        links_by_name: dict[str, list[dict[str, Any]]] = {}
        for link in result.get("links") or []:
            source = str(link.get("source") or "")
            target = str(link.get("target") or "")
            if source and target and source != target:
                links_by_name.setdefault(source, []).append(
                    {"target": target, "relation": str(link.get("relation") or "related")}
                )
        for path in skill_paths:
            skill = read_json(path) or {}
            skill["related_skills"] = links_by_name.get(skill.get("name"), [])
            write_json(path, skill)
            (path.parent / "SKILL.md").write_text(render_skill(skill), encoding="utf-8")
        (self.root / "INDEX.md").write_text(render_index(skills, links_by_name), encoding="utf-8")
        (self.root / "GLOSSARY.md").write_text(render_glossary(self.root), encoding="utf-8")
        self.state.setdefault("artifacts", {})["index"] = str(PACK_DIR / "INDEX.md")
        self.state["artifacts"]["glossary"] = str(PACK_DIR / "GLOSSARY.md")
        save_state(self.run_dir, self.state)

    def _test_skills(self) -> None:
        skill_paths = active_skill_paths(self.root, self.state)
        names_and_descriptions = [
            {
                "name": (read_json(path) or {}).get("name"),
                "description": (read_json(path) or {}).get("description"),
            }
            for path in skill_paths
        ]
        items = []
        passed_count = 0
        total_skills = len(skill_paths)
        self._set_test_progress(
            total=total_skills,
            completed=0,
            current_skill=None,
            phase="preparing",
            phase_fraction=0.0,
        )
        for skill_index, skill_path in enumerate(skill_paths):
            skill = read_json(skill_path) or {}
            skill_dir = skill_path.parent
            cached_result = read_json(skill_dir / "test-results.json") or {}
            if (
                cached_result.get("passed")
                and (skill_dir / "test-results.json").stat().st_mtime
                >= skill_path.stat().st_mtime
            ):
                passed_count += 1
                items.append(
                    {
                        "candidate_id": skill.get("candidate_id"),
                        "name": skill["name"],
                        "title": skill.get("title"),
                        "status": "passed",
                        "pass_rate": cached_result.get("pass_rate"),
                        "path": str(
                            PACK_DIR
                            / "distilled_skills"
                            / skill["name"]
                            / "SKILL.md"
                        ),
                    }
                )
                self._append_log("test", "checkpoint", f"{skill['name']} passed=True")
                self._set_test_progress(
                    total=total_skills,
                    completed=skill_index + 1,
                    current_skill=skill.get("title") or skill.get("name"),
                    phase="skill_completed",
                    phase_fraction=0.0,
                    repair_round=cached_result.get("repair_round"),
                )
                continue
            self._set_test_progress(
                total=total_skills,
                completed=skill_index,
                current_skill=skill.get("title") or skill.get("name"),
                phase="generating_tests",
                phase_fraction=0.1,
            )
            test_spec = call_json(
                self.runtime.generation_client,
                self.runtime.generation_model,
                test_generation_prompt(
                    skill,
                    names_and_descriptions,
                    self._target_brief(),
                ),
                self.runtime.generation_temperature,
                5000,
            )
            tests = normalize_tests(skill["name"], test_spec, names_and_descriptions)
            write_json(skill_dir / "test-prompts.json", tests)
            final_result = None
            for repair_round in range(MAX_REPAIR_ROUNDS + 1):
                self._set_test_progress(
                    total=total_skills,
                    completed=skill_index,
                    current_skill=skill.get("title") or skill.get("name"),
                    phase="blind_judging",
                    phase_fraction=min(0.85, 0.35 + (repair_round * 0.25)),
                    repair_round=repair_round,
                )
                judged = call_json(
                    self.runtime.review_client,
                    self.runtime.review_model,
                    blind_test_prompt(
                        skill,
                        tests,
                        names_and_descriptions,
                        self._target_brief(),
                    ),
                    self.runtime.review_temperature,
                    7000,
                )
                final_result = score_tests(tests, judged)
                final_result["repair_round"] = repair_round
                if final_result["passed"]:
                    break
                if repair_round >= MAX_REPAIR_ROUNDS:
                    break
                self._set_test_progress(
                    total=total_skills,
                    completed=skill_index,
                    current_skill=skill.get("title") or skill.get("name"),
                    phase="repairing",
                    phase_fraction=min(0.75, 0.25 + ((repair_round + 1) * 0.25)),
                    repair_round=repair_round + 1,
                )
                repaired = call_json(
                    self.runtime.generation_client,
                    self.runtime.generation_model,
                    repair_skill_prompt(skill, tests, final_result),
                    self.runtime.generation_temperature,
                    7000,
                )
                skill = normalize_skill(repaired, skill)
                skill["name"] = skill_path.parent.name
                write_json(skill_path, skill)
                (skill_dir / "SKILL.md").write_text(render_skill(skill), encoding="utf-8")
            write_json(skill_dir / "test-results.json", final_result or {})
            (skill_dir / "test-results.md").write_text(
                render_test_results(final_result or {}),
                encoding="utf-8",
            )
            passed = bool((final_result or {}).get("passed"))
            passed_count += int(passed)
            items.append(
                {
                    "candidate_id": skill.get("candidate_id"),
                    "name": skill["name"],
                    "title": skill.get("title"),
                    "status": "passed" if passed else "failed",
                    "pass_rate": (final_result or {}).get("pass_rate"),
                    "path": str(PACK_DIR / "distilled_skills" / skill["name"] / "SKILL.md"),
                }
            )
            self._append_log("test", "skill", f"{skill['name']} passed={passed}")
            self._set_test_progress(
                total=total_skills,
                completed=skill_index + 1,
                current_skill=skill.get("title") or skill.get("name"),
                phase="skill_completed",
                phase_fraction=0.0,
                repair_round=(final_result or {}).get("repair_round"),
            )
        test_progress = {
            **dict((self.state.get("skills") or {}).get("test_progress") or {}),
            "total": total_skills,
            "completed": total_skills,
            "current_skill": None,
            "phase": "completed",
            "stage_percent": 100,
            "updated_at": utc_now(),
        }
        self.state["skills"] = {
            "count": len(items),
            "passed": passed_count,
            "failed": len(items) - passed_count,
            "items": items,
            "test_progress": test_progress,
        }
        save_state(self.run_dir, self.state)

    def _set_test_progress(
        self,
        *,
        total: int,
        completed: int,
        current_skill: str | None,
        phase: str,
        phase_fraction: float,
        repair_round: int | None = None,
    ) -> None:
        denominator = max(total, 1)
        stage_percent = min(
            100,
            round(((completed + max(0.0, min(phase_fraction, 0.99))) / denominator) * 100),
        )
        skills = dict(self.state.get("skills") or {})
        skills["test_progress"] = {
            "total": total,
            "completed": completed,
            "current_skill": current_skill,
            "phase": phase,
            "repair_round": repair_round,
            "stage_percent": stage_percent,
            "updated_at": utc_now(),
        }
        self.state["skills"] = skills
        self._update_stage(
            "test",
            progress_percent=stage_percent,
            message=phase,
        )

    def _deliver(self) -> None:
        overview = read_json(self.root / OVERVIEW_JSON_NAME) or {}
        skills = self.state.get("skills") or {}
        (self.root / "DIGEST.md").write_text(render_digest(overview, skills), encoding="utf-8")
        manifest = {
            "version": 1,
            "method": METHOD_NAME,
            "method_source": METHOD_SOURCE,
            "generated_at": utc_now(),
            "profile": self.state.get("profile"),
            "generation_model": self.state.get("generation_model"),
            "review_model": self.state.get("review_model"),
            "source_fingerprint": self.state.get("source_fingerprint"),
            "skills": skills,
            "artifacts": {
                **dict(self.state.get("artifacts") or {}),
                "digest": str(PACK_DIR / "DIGEST.md"),
            },
        }
        write_json(self.root / DISTILLATION_MANIFEST_NAME, manifest)
        self.state.setdefault("artifacts", {})["digest"] = str(PACK_DIR / "DIGEST.md")
        self.state["artifacts"]["manifest"] = str(PACK_DIR / DISTILLATION_MANIFEST_NAME)
        save_state(self.run_dir, self.state)

    def _invalidate_from(self, stage: str) -> None:
        start = PIPELINE_STAGES.index(stage)
        for name in PIPELINE_STAGES[start:]:
            self.state.setdefault("stages", {})[name] = {"status": "pending"}
        if stage == "overview":
            for path in (self.root / OVERVIEW_JSON_NAME, self.root / OVERVIEW_MD_NAME):
                path.unlink(missing_ok=True)
        save_state(self.run_dir, self.state)

    def _advance(self, stage: str | None) -> None:
        self._set_state(current_stage=stage)

    def _update_stage(self, stage: str, **updates: Any) -> None:
        info = dict((self.state.get("stages") or {}).get(stage) or {})
        info.update(updates)
        self.state.setdefault("stages", {})[stage] = info
        self.state["updated_at"] = utc_now()
        save_state(self.run_dir, self.state)

    def _set_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_at"] = utc_now()
        save_state(self.run_dir, self.state)

    def _append_log(self, stage: str, event: str, message: str) -> None:
        log_dir = self.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {"time": utc_now(), "stage": stage, "event": event, "message": message}
        with (log_dir / f"{safe_slug(stage)}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event and cancel_event.is_set():
            raise DistillationCancelled("distillation cancelled")

    def _target_brief(self) -> dict[str, Any]:
        return dict(self.state.get("target_brief") or {})

    def _reference_context(self) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in (self.state.get("project_reference_context") or [])
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]


def enable_distilled_skills(
    run_dir: Path,
    repo_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    state = load_state(run_dir)
    if not state or state.get("status") != "succeeded":
        raise FileNotFoundError("A completed skill distillation pack is not available")
    passed = [
        item for item in (state.get("skills") or {}).get("items") or [] if item.get("status") == "passed"
    ]
    if not passed:
        raise FileNotFoundError("No pressure-tested skills are available to enable")
    source_root = pack_dir(run_dir) / "distilled_skills"
    target_root = (repo_root / ".codex" / "skills").resolve()
    targets = [(source_root / item["name"], (target_root / item["name"]).resolve()) for item in passed]
    for source, target in targets:
        if not source.is_dir():
            raise FileNotFoundError(f"Skill source is missing: {source}")
        try:
            target.relative_to(target_root)
        except ValueError as exc:
            raise ValueError("Skill destination escapes .codex/skills") from exc
    conflicts = [str(target) for _, target in targets if target.exists()]
    if conflicts and not overwrite:
        error = FileExistsError("Skill targets already exist")
        error.conflicts = conflicts  # type: ignore[attr-defined]
        raise error
    target_root.mkdir(parents=True, exist_ok=True)
    installed = []
    with tempfile.TemporaryDirectory(prefix="skill-install-", dir=target_root) as tmp:
        staging_root = Path(tmp)
        for source, target in targets:
            staged = staging_root / target.name
            shutil.copytree(source, staged)
        for _, target in targets:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(staging_root / target.name, target)
            installed.append(str(target))
    state["installed"] = {"paths": installed, "installed_at": utc_now()}
    save_state(run_dir, state)
    return distillation_summary(run_dir)


def load_evidence_records(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        source_type: str,
        path: Path,
        text: str,
        *,
        index: int,
        confidence: str,
        timestamp: float | None = None,
        end_timestamp: float | None = None,
        frame_number: int | None = None,
        speaker: str | None = None,
    ) -> None:
        cleaned = clean_evidence_text(text)
        if not cleaned:
            return
        stable = f"{source_type}:{index:04d}"
        if stable in seen:
            return
        seen.add(stable)
        records.append(
            {
                "id": stable,
                "source_type": source_type,
                "path": str(path.relative_to(run_dir)),
                "confidence": confidence,
                "timestamp": timestamp,
                "end_timestamp": end_timestamp,
                "frame_number": frame_number,
                "speaker": speaker,
                "text": cleaned,
            }
        )

    transcript_path = run_dir / "orin" / "transcript.json"
    transcript = read_json(transcript_path) or {}
    segments = transcript.get("segments") if isinstance(transcript, dict) else []
    for index, segment in enumerate(segments or []):
        add(
            "transcript",
            transcript_path,
            first_value(segment, "text", "Text", "content", "Content", "transcript", "Transcript"),
            index=index,
            confidence="high",
            timestamp=as_float(first_value(segment, "start_time", "start", "Start")),
            end_timestamp=as_float(first_value(segment, "end_time", "end", "End")),
            speaker=str(first_value(segment, "speaker", "Speaker") or "") or None,
        )
    if not segments and isinstance(transcript, dict) and transcript.get("text"):
        for index, text in enumerate(split_paragraphs(str(transcript["text"]))):
            add("transcript", transcript_path, text, index=index, confidence="high")

    ocr_path = run_dir / "orin" / "ocr_events.json"
    for index, event in enumerate(read_json(ocr_path) or []):
        add(
            "ocr",
            ocr_path,
            str(event.get("text") or ""),
            index=index,
            confidence="high",
            timestamp=as_float(event.get("timestamp")),
            frame_number=as_int(event.get("frame_number")),
        )

    visual_path = run_dir / "orin" / "frame_analyses.json"
    visual_events = read_json(visual_path) or []
    if not visual_events:
        visual_path = run_dir / "orin" / "visual_events.json"
        visual_events = read_json(visual_path) or []
    for index, event in enumerate(visual_events):
        if str(event.get("status") or "").lower() not in {"", "ok", "succeeded"}:
            continue
        add(
            "visual",
            visual_path,
            str(event.get("response") or ""),
            index=index,
            confidence="high",
            timestamp=as_float(event.get("timestamp")),
            frame_number=as_int(event.get("frame_number")),
        )

    context_path = run_dir / "orin" / "page_context.md"
    context_text = read_text(context_path)
    if context_text:
        for index, text in enumerate(split_page_context(context_text)):
            add("page", context_path, text, index=index, confidence="medium")

    comments_path = run_dir / "orin" / "comments.md"
    comments_text = read_text(comments_path)
    if comments_text:
        for index, text in enumerate(split_paragraphs(comments_text)):
            add("comments", comments_path, text, index=index, confidence="low")

    for source_type in ("transcript", "ocr", "visual", "page", "comments"):
        selected = [record for record in records if record["source_type"] == source_type]
        if selected:
            paths = sorted({record["path"] for record in selected})
            sources.append(
                {
                    "type": source_type,
                    "paths": paths,
                    "records": len(selected),
                    "chars": sum(len(record["text"]) for record in selected),
                    "confidence": selected[0]["confidence"],
                }
            )
    return records, sources


def assign_evidence_events(
    records: list[dict[str, Any]],
    *,
    window_seconds: int = DEFAULT_EVENT_WINDOW_SECONDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in records:
        record = dict(raw)
        event_id = evidence_event_id(
            record,
            str(record.get("id") or ""),
            window_seconds=max(1, window_seconds),
        )
        record["event_id"] = event_id
        record["case_id"] = evidence_case_id(record, str(record.get("id") or ""))
        normalized.append(record)
        grouped.setdefault(event_id, []).append(record)
    events = []
    for event_id, items in sorted(grouped.items()):
        timestamps = [
            float(item["timestamp"])
            for item in items
            if item.get("timestamp") is not None
        ]
        frame_numbers = sorted(
            {
                int(item["frame_number"])
                for item in items
                if item.get("frame_number") is not None
            }
        )
        events.append(
            {
                "id": event_id,
                "source_types": sorted({str(item.get("source_type") or "") for item in items}),
                "case_ids": sorted({str(item.get("case_id") or "") for item in items}),
                "record_ids": [str(item.get("id")) for item in items],
                "start_timestamp": min(timestamps) if timestamps else None,
                "end_timestamp": max(timestamps) if timestamps else None,
                "frame_numbers": frame_numbers,
            }
        )
    return normalized, events


def evidence_event_id(
    record: dict[str, Any],
    evidence_id: str,
    *,
    window_seconds: int = DEFAULT_EVENT_WINDOW_SECONDS,
) -> str:
    timestamp = record.get("timestamp")
    source_type = str(record.get("source_type") or "unknown")
    if timestamp is not None and source_type in {"transcript", "subtitle", "ocr", "visual"}:
        window = max(1, window_seconds)
        bucket = int((float(timestamp) + (window / 2)) // window)
        return f"video-event:{bucket:04d}"
    path = safe_slug(record.get("path") or source_type)
    if source_type in {"page", "transcript", "subtitle"}:
        return f"document-event:{path}"
    if source_type == "comments":
        return f"comment-event:{safe_slug(evidence_id)}"
    return f"record-event:{safe_slug(evidence_id)}"


def evidence_case_id(record: dict[str, Any], evidence_id: str) -> str:
    source_type = str(record.get("source_type") or "unknown")
    path = safe_slug(record.get("path") or source_type)
    if source_type in {"transcript", "subtitle", "ocr", "visual"}:
        return "video-case:primary"
    if source_type == "page":
        return f"document-case:{path}"
    if source_type == "comments":
        return f"comment-case:{safe_slug(evidence_id)}"
    return f"record-case:{safe_slug(evidence_id)}"


def candidate_frame_paths(
    run_dir: Path,
    evidence: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    *,
    max_images: int = DEFAULT_MULTIMODAL_IMAGES,
) -> list[Path]:
    event_ids = {str(item.get("event_id") or "") for item in evidence if item.get("event_id")}
    related = list(evidence)
    related.extend(
        item
        for item in records.values()
        if item.get("event_id") in event_ids and item not in related
    )
    frame_numbers = []
    for item in related:
        frame_number = item.get("frame_number")
        if frame_number is not None:
            frame_numbers.append(int(frame_number))
    paths = []
    for frame_number in dict.fromkeys(frame_numbers):
        candidates = [
            run_dir / "manual_assets" / f"frame_{frame_number:03d}.jpg",
            run_dir / "manual_assets" / f"frame_{frame_number:03d}.png",
            run_dir / "frames" / f"frame_{frame_number}.jpg",
            run_dir / "frames" / f"frame_{frame_number:03d}.jpg",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path:
            paths.append(path)
        if len(paths) >= max_images:
            break
    return paths


def normalize_multimodal_audit(
    result: dict[str, Any],
    event_ids: list[str],
) -> dict[str, Any]:
    confidence = result.get("confidence")
    try:
        confidence = max(0.0, min(float(confidence), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    instructional_value = str(result.get("instructional_value") or "low").strip().lower()
    if instructional_value not in {"high", "medium", "low"}:
        instructional_value = "low"
    return {
        "claim_supported": bool(result.get("claim_supported")),
        "execution_supported": bool(result.get("execution_supported")),
        "visual_support": str(result.get("visual_support") or "").strip(),
        "transcript_support": str(result.get("transcript_support") or "").strip(),
        "unsupported_details": unique_strings(result.get("unsupported_details") or []),
        "contradiction": bool(result.get("contradiction")),
        "contradiction_reason": str(result.get("contradiction_reason") or "").strip(),
        "instructional_value": instructional_value,
        "confidence": confidence,
        "event_ids": unique_strings(result.get("event_ids") or event_ids),
    }


def clean_evidence_text(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text.lower() in {"none", "null", "no ocr text"}:
        return ""
    if text.startswith("VL analysis skipped") and len(text) < 100:
        return ""
    return text[:6000]


def split_page_context(text: str) -> list[str]:
    allowed_headings = {
        "metadata summary",
        "original description",
        "page description",
        "chapters",
        "author subtitles",
        "subtitles",
    }
    sections = re.split(r"(?m)^##\s+", text)
    selected = []
    for section in sections:
        heading, _, body = section.partition("\n")
        normalized = heading.strip().lower()
        if not selected and text.startswith("#"):
            selected.append(section[:1200])
        if normalized in allowed_headings or normalized.startswith("original description"):
            selected.extend(split_paragraphs(f"## {heading}\n{body}"))
    if len(selected) <= 1:
        selected.extend(split_paragraphs(text[:30000]))
    return selected


def split_paragraphs(text: str, max_chars: int = 4000) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        if paragraph:
            chunks.append(paragraph)
    return chunks


def build_evidence_chunks(
    records: list[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    used = 0
    for record in records:
        rendered = render_evidence_record(record)
        if current and used + len(rendered) > max_chars:
            chunks.append("\n".join(current))
            current = []
            used = 0
        current.append(rendered)
        used += len(rendered)
    if current:
        chunks.append("\n".join(current))
    return chunks


def render_evidence_record(record: dict[str, Any]) -> str:
    location = []
    if record.get("timestamp") is not None:
        location.append(format_seconds(float(record["timestamp"])))
    if record.get("frame_number") is not None:
        location.append(f"frame {record['frame_number']}")
    where = ", ".join(location) or record.get("path") or ""
    return (
        f"[{record['id']}] type={record['source_type']} confidence={record['confidence']} "
        f"location={where}\n{record['text']}\n"
    )


def overview_chunk_prompt(index: int, total: int, chunk: str) -> str:
    return f"""你是证据分析器。以下是视频原始证据的第 {index}/{total} 块。
只根据给出的证据工作，保留可追溯 evidence id，不要使用常识补全缺失步骤。

返回严格 JSON 对象：
{{
  "summary": "本块核心内容",
  "structure": [{{"title":"主题","summary":"说明","source_ids":["..."]}}],
  "methods": [{{"title":"可执行方法或流程","summary":"说明","source_ids":["..."]}}],
  "concepts": [{{"term":"术语","meaning":"含义","source_ids":["..."]}}],
  "cases": [{{"title":"案例","problem":"问题","action":"动作","result":"结果","source_ids":["..."]}}],
  "failures": [{{"title":"失败或风险","reason":"原因","source_ids":["..."]}}],
  "limitations": [{{"issue":"证据限制","source_ids":["..."]}}]
}}

证据：
{chunk}
"""


def overview_synthesis_prompt(
    digests: list[dict[str, Any]],
    feedback: str,
    target_brief: dict[str, Any] | None = None,
    reference_context: list[dict[str, Any]] | None = None,
) -> str:
    feedback_text = feedback or "无额外修订意见"
    return f"""你负责把一组视频证据摘要合成为面向 skill 蒸馏的全局理解。
不得创造摘要中不存在的事实；所有重要判断保留 source_ids。
排除赞助广告、偶然出现的网页/频道、识别噪声以及与主要项目流程无关的素材。
source_ids 只能使用 transcript:/ocr:/visual:/page:/comments: 开头的真实证据 id，
不得输出 chunk_index、块号或自造引用。

返回严格 JSON：
{{
  "title":"内容标题",
  "summary":"整体主旨",
  "source_kind":"video",
  "structure":[{{"title":"部分","summary":"说明","source_ids":["..."]}}],
  "methods":[{{"title":"方法","summary":"说明","source_ids":["..."]}}],
  "concepts":[{{"term":"术语","meaning":"含义","source_ids":["..."]}}],
  "cases":[{{"title":"案例","summary":"问题-动作-结果","source_ids":["..."]}}],
  "failures":[{{"title":"失败或风险","reason":"原因","source_ids":["..."]}}],
  "critique":[{{"issue":"盲点、适用限制或证据不足","source_ids":["..."]}}],
  "coverage":{{"included_source_types":[],"known_gaps":[]}}
}}

用户修订意见：{feedback_text}

目标约束（为空时保持通用蒸馏）：
{json.dumps(target_brief or {}, ensure_ascii=False)}

辅助资料（仅用于理解主题、学习结构和术语；不得作为事实依据或 source_ids）：
{json.dumps(reference_context or [], ensure_ascii=False)}

分块摘要：
{json.dumps(digests, ensure_ascii=False)}
"""


def extractor_prompt(
    name: str,
    candidate_type: str,
    instruction: str,
    overview: dict[str, Any],
    digests: list[dict[str, Any]],
    target_brief: dict[str, Any] | None = None,
) -> str:
    return f"""你是独立的 {name} 提取器。你的唯一职责：{instruction}
候选必须来自输入中的 source_ids。宁可少，不要把常识、广告或无执行意义的事实做成 skill。

返回严格 JSON：
{{
  "candidates":[
    {{
      "title":"候选标题",
      "type":"{candidate_type}",
      "summary":"自己的话解释",
      "source_ids":["至少一个证据 id"],
      "source_quote":"不超过 150 个中文字符或 100 个英文词的短证据摘录",
      "tags":["tag"],
      "execution_hint":"可执行动作",
      "boundaries":["不适用情况"]
    }}
  ]
}}

全局理解：
{json.dumps(overview, ensure_ascii=False)}

目标约束：
{json.dumps(target_brief or {}, ensure_ascii=False)}

证据摘要：
{json.dumps(digests, ensure_ascii=False)}
"""


def verification_prompt(
    candidates: list[dict[str, Any]],
    target_brief: dict[str, Any] | None = None,
) -> str:
    return f"""你是严格的 skill 候选评审器。逐条执行三项验证：
- V1：至少两个独立语境支持，不得把同一案例换种说法重复计数。
- V2：方法可以对输入未直接讨论的新场景产生非平庸结论。
- V3：不是任何通用模型都能给出的常识，必须有明确差异化结构。

必须覆盖输入中的每一个候选。即使 V1 失败，也必须继续给出 V2 和 V3 判断。
返回严格 JSON：
{{
  "evaluations":[{{
    "id":"候选 id",
    "title":"标题",
    "type":"类型",
    "source_ids":["证据 id"],
    "source_quote":"短引用",
    "summary":"说明",
    "tags":[],
    "v1":{{"passed":false,"reason":"理由","evidence_ids":["证据 id"]}},
    "v2":{{"passed":true,"novel_question":"新问题","derived_answer":"推导答案"}},
    "v3":{{"passed":true,"reason":"为什么不是常识"}}
  }}]
}}

候选：
{json.dumps(candidates, ensure_ascii=False)}

目标约束：
{json.dumps(target_brief or {}, ensure_ascii=False)}
"""


def multimodal_verification_prompt(
    candidate: dict[str, Any],
    evidence: list[dict[str, Any]],
    event_ids: list[str],
) -> str:
    compact_evidence = [
        {
            "id": item.get("id"),
            "event_id": item.get("event_id"),
            "source_type": item.get("source_type"),
            "timestamp": item.get("timestamp"),
            "text": str(item.get("text") or "")[:1200],
        }
        for item in evidence
    ]
    return f"""你是候选 Skill 的多模态证据复核器。图片按证据时间顺序给出。
只判断图片、口播和 OCR 是否支持候选，不要补写图片中看不到的工程细节。

返回严格 JSON：
{{
  "claim_supported": true,
  "execution_supported": false,
  "visual_support": "画面中可直接观察到的支持",
  "transcript_support": "口播或 OCR 的支持",
  "unsupported_details": ["候选中没有证据支撑的步骤、参数或工具细节"],
  "contradiction": false,
  "contradiction_reason": "",
  "instructional_value": "high|medium|low",
  "confidence": 0.0,
  "event_ids": ["证据事件 id"]
}}

候选：
{json.dumps(candidate, ensure_ascii=False)}

事件：
{json.dumps(event_ids, ensure_ascii=False)}

证据：
{json.dumps(compact_evidence, ensure_ascii=False)}
"""


def skill_build_prompt(
    candidate: dict[str, Any],
    evidence: list[dict[str, Any]],
    siblings: list[dict[str, Any]],
    overview: dict[str, Any],
    target_brief: dict[str, Any] | None = None,
) -> str:
    sibling_titles = [item.get("title") for item in siblings if item.get("id") != candidate.get("id")]
    return f"""把一个已审核的方法单元构造成简洁、可触发、可执行的 Codex skill。
SKILL.md 的 frontmatter 只依赖 name 和 description；正文应短而具体，详细证据放入审计 JSON。
description 必须同时说明何时使用、何时不使用以及用户语言信号。
如果候选包含 multimodal_audit.unsupported_details，必须删除或降级这些无证据细节，
不得把它们写入执行步骤、完成标准、边界或限制。single_case 只能表述为来源案例中的经验，
不得伪装成已经由多个独立案例验证的通用定律。

返回严格 JSON：
{{
  "name":"lowercase-kebab-case，不超过64字符",
  "title":"显示标题",
  "description":"触发描述，不超过300字",
  "candidate_id":"{candidate.get('id')}",
  "reading":{{"quote":"短原始证据","source_ids":["..."],"source_note":"时间戳或帧"}},
  "interpretation":"自己的话解释方法骨架",
  "applications":[{{"problem":"问题","action":"如何使用","conclusion":"结论","result":"结果","source_ids":["..."]}}],
  "triggers":{{"scenarios":[],"language_signals":[],"distinctions":[]}},
  "execution":[{{"title":"动作","instruction":"具体执行","done_when":"完成标准","stop_condition":""}}],
  "boundaries":{{"do_not_use":[],"failure_modes":[],"limitations":[]}},
  "tags":[],
  "related_skills":[]
}}

候选：
{json.dumps(candidate, ensure_ascii=False)}

对应原始证据：
{json.dumps(evidence, ensure_ascii=False)}

相邻候选标题：
{json.dumps(sibling_titles, ensure_ascii=False)}

全局批判与限制：
{json.dumps(overview.get('critique') or [], ensure_ascii=False)}

目标约束：
{json.dumps(target_brief or {}, ensure_ascii=False)}
"""


def linking_prompt(skills: list[dict[str, Any]]) -> str:
    compact = [
        {
            "name": item.get("name"),
            "title": item.get("title"),
            "description": item.get("description"),
            "tags": item.get("tags") or [],
        }
        for item in skills
    ]
    return f"""分析同一来源的 skills 之间的关系。只返回真实有用的关系，避免把所有项互相连接。
返回严格 JSON：
{{"links":[{{"source":"skill-name","target":"skill-name","relation":"depends-on|contrasts-with|composes-with"}}]}}

Skills：
{json.dumps(compact, ensure_ascii=False)}
"""


def test_generation_prompt(
    skill: dict[str, Any],
    siblings: list[dict[str, Any]],
    target_brief: dict[str, Any] | None = None,
) -> str:
    return f"""为以下 skill 生成触发压力测试。至少 3 条 should_trigger、2 条 should_not_trigger、
1 条 edge_case；其中至少一条 should_not_trigger 应当触发兄弟 skill，测试相互抢调用。

返回严格 JSON：
{{"test_cases":[{{
  "id":"should-trigger-01",
  "type":"should_trigger|should_not_trigger|edge_case",
  "prompt":"真实用户表达",
  "expected_behavior":"预期是否触发以及动作",
  "expected_skill":"skill-name 或 null",
  "notes":"边界说明"
}}]}}

当前 skill：
{json.dumps(skill, ensure_ascii=False)}

兄弟 skills：
{json.dumps(siblings, ensure_ascii=False)}

目标约束：
{json.dumps(target_brief or {}, ensure_ascii=False)}
"""


def blind_test_prompt(
    skill: dict[str, Any],
    tests: dict[str, Any],
    siblings: list[dict[str, Any]],
    target_brief: dict[str, Any] | None = None,
) -> str:
    blind_cases = [{"id": item["id"], "prompt": item["prompt"]} for item in tests["test_cases"]]
    catalog = [
        {
            "name": item.get("name"),
            "description": item.get("description"),
        }
        for item in siblings
    ]
    return f"""你是独立触发评测器。你只能看到 skill 目录和用户 prompt，看不到测试类型和预期答案。
对每条 prompt 判断应激活哪个 skill；若都不应激活，selected_skill 返回 null。

返回严格 JSON：
{{"results":[{{"id":"case id","selected_skill":"name 或 null","would_trigger":true,"reason":"理由","action":"触发后动作"}}]}}

候选目录：
{json.dumps(catalog, ensure_ascii=False)}

被测 skill 正文：
{json.dumps({"name": skill.get("name"), "description": skill.get("description"), "execution": skill.get("execution"), "boundaries": skill.get("boundaries")}, ensure_ascii=False)}

测试 prompts：
{json.dumps(blind_cases, ensure_ascii=False)}

目标约束：
{json.dumps(target_brief or {}, ensure_ascii=False)}
"""


def repair_skill_prompt(
    skill: dict[str, Any],
    tests: dict[str, Any],
    result: dict[str, Any],
) -> str:
    return f"""根据盲测失败修订 skill 的触发描述、执行步骤或边界。不要改变方法论本身，不要删除合理的诱饵测试。
返回与输入 skill 相同字段的严格 JSON。

Skill：
{json.dumps(skill, ensure_ascii=False)}

测试：
{json.dumps(tests, ensure_ascii=False)}

失败结果：
{json.dumps(result, ensure_ascii=False)}
"""


def call_json(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    *,
    image_paths: list[str] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    current_prompt = prompt
    for attempt in range(2):
        response = client.generate(
            prompt=current_prompt,
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            image_paths=image_paths,
        )
        text = str((response or {}).get("response") or "")
        try:
            payload = parse_json_response(text)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            return payload
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            current_prompt = (
                "把下面响应修复为严格 JSON 对象。不要解释，不要使用 Markdown 代码围栏，不要丢失信息。\n\n"
                + text
            )
    raise DistillationError(f"Model returned invalid JSON: {last_error}")


def parse_json_response(text: str) -> Any:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [index for index in (text.find("{"), text.find("[")) if index >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        closing = "}" if text[start] == "{" else "]"
        end = text.rfind(closing)
        if end <= start:
            raise
        return json.loads(text[start : end + 1])


def normalize_extractor_result(
    name: str,
    candidate_type: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    prefix = {
        "frameworks": "f",
        "principles": "p",
        "cases": "c",
        "counter_examples": "x",
        "glossary": "g",
    }[name]
    for index, item in enumerate(result.get("candidates") or [], start=1):
        if not isinstance(item, dict) or not str(item.get("title") or "").strip():
            continue
        candidates.append(
            {
                **item,
                "id": f"{prefix}{index:03d}",
                "type": str(item.get("type") or candidate_type),
                "source_ids": unique_strings(item.get("source_ids") or []),
                "tags": unique_strings(item.get("tags") or []),
            }
        )
    return {"extractor": name, "type": candidate_type, "candidates": candidates}


def normalize_overview(
    overview: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = dict(overview or {})
    valid_source_ids = set(records)
    narrative_types = {"transcript", "subtitle", "page"}
    has_narrative_evidence = any(
        record.get("source_type") in narrative_types for record in records.values()
    )
    for field in ("structure", "methods", "concepts", "cases", "failures", "critique"):
        items = []
        for raw in normalized.get(field) or []:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item["source_ids"] = [
                value
                for value in unique_strings(item.get("source_ids") or [])
                if value in valid_source_ids
            ]
            has_narrative_anchor = any(
                (records.get(value) or {}).get("source_type") in narrative_types
                for value in item["source_ids"]
            )
            if (
                has_narrative_evidence
                and field != "critique"
                and not has_narrative_anchor
            ):
                continue
            if item["source_ids"]:
                items.append(item)
        normalized[field] = items
    coverage = dict(normalized.get("coverage") or {})
    coverage["included_source_types"] = [
        value
        for value in unique_strings(coverage.get("included_source_types") or [])
        if value in {"transcript", "subtitle", "ocr", "visual", "page", "comments"}
    ]
    coverage["known_gaps"] = unique_strings(coverage.get("known_gaps") or [])
    normalized["coverage"] = coverage
    return normalized


def overview_has_substantive_content(overview: dict[str, Any]) -> bool:
    return any(
        bool(overview.get(field))
        for field in ("structure", "methods", "concepts", "cases", "failures")
    )


def merge_verification_batch_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    legacy_rejected_ids: list[str] = []
    for result in results:
        payload = dict(result or {})
        batch_evaluations = [
            item for item in payload.get("evaluations") or [] if isinstance(item, dict)
        ]
        if batch_evaluations:
            evaluations.extend(batch_evaluations)
            continue
        evaluations.extend(
            item for item in payload.get("accepted") or [] if isinstance(item, dict)
        )
        for item in payload.get("rejected") or []:
            if not isinstance(item, dict):
                continue
            evaluations.append(item)
            candidate_id = str(item.get("id") or "").strip()
            if candidate_id:
                legacy_rejected_ids.append(candidate_id)
    return {
        "evaluations": evaluations,
        "legacy_rejected_ids": unique_strings(legacy_rejected_ids),
    }


def normalize_verification(
    result: dict[str, Any],
    records: dict[str, dict[str, Any]],
    candidates: Iterable[dict[str, Any]] = (),
    *,
    multimodal_audits: dict[str, dict[str, Any]] | None = None,
    glossary: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    accepted = []
    single_case = []
    rejected = []
    candidate_items = [dict(item) for item in candidates if isinstance(item, dict)]
    candidate_by_id = {
        str(item.get("id")): item for item in candidate_items if item.get("id")
    }
    audits = multimodal_audits or {}
    used_ids: set[str] = set()
    evaluations = list(result.get("evaluations") or [])
    legacy_rejected_ids: set[str] = set(
        unique_strings(result.get("legacy_rejected_ids") or [])
    )
    if not evaluations:
        evaluations.extend(result.get("accepted") or [])
        for raw in result.get("rejected") or []:
            if isinstance(raw, dict):
                legacy_rejected_ids.add(str(raw.get("id") or ""))
                evaluations.append(raw)
    for index, raw in enumerate(evaluations, start=1):
        if not isinstance(raw, dict):
            continue
        original_id = str(raw.get("id") or "")
        candidate = candidate_by_id.get(original_id) or {}
        item = {**candidate, **raw}
        item_id = safe_candidate_id(original_id or f"evaluation-{index}", used_ids)
        item["id"] = item_id
        used_ids.add(item_id)
        source_ids = [
            value
            for value in unique_strings(item.get("source_ids") or candidate.get("source_ids") or [])
            if value in records
        ]
        item["source_ids"] = source_ids
        v1 = dict(item.get("v1") or {})
        v2 = dict(item.get("v2") or {})
        v3 = dict(item.get("v3") or {})
        v1_ids = [
            value
            for value in unique_strings(v1.get("evidence_ids") or source_ids)
            if value in records
        ]
        independent = independent_evidence_count(v1_ids, records)
        event_ids = unique_strings(
            records[value].get("event_id")
            for value in v1_ids
            if records[value].get("event_id")
        )
        case_ids = unique_strings(
            records[value].get("case_id")
            or evidence_case_id(records[value], value)
            for value in v1_ids
        )
        model_v1_passed = bool(v1.get("passed"))
        v1.update(
            {
                "evidence_ids": v1_ids,
                "event_ids": event_ids,
                "event_count": len(event_ids),
                "case_ids": case_ids,
                "independent_context_count": independent,
                "passed": model_v1_passed and independent >= 2,
            }
        )
        v2["passed"] = bool(v2.get("passed"))
        v3["passed"] = bool(v3.get("passed"))
        audit = dict(audits.get(original_id) or {})
        audit_has_frames = bool(audit.get("image_paths"))
        multimodal_claim_blocked = bool(
            (audit_has_frames and audit.get("status") == "failed")
            or (
                audit.get("status") == "succeeded"
                and (
                    audit.get("contradiction")
                    or (audit_has_frames and not audit.get("claim_supported"))
                )
            )
        )
        execution_unverified = bool(
            audit_has_frames
            and audit.get("status") == "succeeded"
            and not audit.get("execution_supported")
        )
        item.update(
            {
                "v1": v1,
                "v2": v2,
                "v3": v3,
                "multimodal_audit": audit,
                "grounding_required": execution_unverified,
            }
        )
        failed_checks = [
            name for name in ("v1", "v2", "v3") if not item[name].get("passed")
        ]
        if multimodal_claim_blocked:
            failed_checks.append("multimodal")
        elif execution_unverified:
            failed_checks.append("multimodal_execution")
        item["failed_checks"] = failed_checks
        if (
            v1["passed"]
            and v2["passed"]
            and v3["passed"]
            and not multimodal_claim_blocked
            and not execution_unverified
        ):
            item["evidence_level"] = "verified"
            item.pop("reason", None)
            accepted.append(item)
            continue
        if (
            independent >= 1
            and v2["passed"]
            and v3["passed"]
            and not multimodal_claim_blocked
            and original_id not in legacy_rejected_ids
        ):
            item["evidence_level"] = "single_case"
            item["reason"] = (
                "方法具备迁移价值，但仅有一个独立案例，需人工确认。"
                + (
                    " 构建时必须删除多模态复核指出的无证据执行细节。"
                    if execution_unverified
                    else ""
                )
            )
            single_case.append(item)
            continue
        item["evidence_level"] = "rejected"
        item["reason"] = str(item.get("reason") or "三重验证或多模态复核未通过")
        if not item["failed_checks"]:
            item["failed_checks"] = ["unspecified"]
        rejected.append(item)
    covered_ids = {
        value
        for item in accepted + single_case + rejected
        for value in [item.get("id"), *(item.get("merged_from") or [])]
        if value
    }
    for raw in candidate_items:
        if not isinstance(raw, dict) or raw.get("id") in covered_ids:
            continue
        source_ids = [
            value
            for value in unique_strings(raw.get("source_ids") or [])
            if value in records
        ]
        rejected.append(
            {
                **raw,
                "source_ids": source_ids,
                "evidence_level": "rejected",
                "failed_checks": ["verification_omitted"],
                "reason": "评审模型未返回该候选，按未通过处理以保证验证结果完整。",
            }
        )
    glossary_items = [
        {
            **item,
            "evidence_level": "glossary",
        }
        for item in glossary
        if isinstance(item, dict)
    ]
    return {
        "version": 1,
        "generated_at": utc_now(),
        "accepted": accepted,
        "single_case": single_case,
        "rejected": rejected,
        "glossary": glossary_items,
    }


def independent_evidence_count(
    evidence_ids: list[str],
    records: dict[str, dict[str, Any]],
) -> int:
    cases = set()
    for evidence_id in evidence_ids:
        record = records.get(evidence_id) or {}
        cases.add(
            record.get("case_id")
            or evidence_case_id(record, evidence_id)
        )
    return len(cases)


def normalize_skill(result: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    skill = dict(result or {})
    skill["candidate_id"] = skill.get("candidate_id") or candidate.get("candidate_id") or candidate.get("id")
    skill["title"] = str(skill.get("title") or candidate.get("title") or "Distilled Skill").strip()
    skill["name"] = safe_slug(skill.get("name") or skill["title"])
    skill["description"] = str(skill.get("description") or "").strip()[:600]
    if not skill["description"]:
        skill["description"] = (
            f"Use when the user needs the {skill['title']} workflow. "
            "Do not use for unrelated factual lookup."
        )
    skill["interpretation"] = str(skill.get("interpretation") or candidate.get("summary") or "").strip()
    skill["applications"] = list(skill.get("applications") or [])
    skill["triggers"] = dict(skill.get("triggers") or {})
    skill["execution"] = list(skill.get("execution") or [])
    skill["boundaries"] = dict(skill.get("boundaries") or {})
    skill["tags"] = unique_strings(skill.get("tags") or candidate.get("tags") or [])
    skill["related_skills"] = list(skill.get("related_skills") or [])
    return skill


def normalize_tests(
    skill_name: str,
    result: dict[str, Any],
    siblings: list[dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    counters = {"should_trigger": 0, "should_not_trigger": 0, "edge_case": 0}
    sibling_names = {item.get("name") for item in siblings if item.get("name")}
    for index, item in enumerate(result.get("test_cases") or [], start=1):
        if not isinstance(item, dict):
            continue
        case_type = str(item.get("type") or "")
        if case_type not in counters or not str(item.get("prompt") or "").strip():
            continue
        counters[case_type] += 1
        expected_skill = item.get("expected_skill")
        if case_type == "should_trigger":
            expected_skill = skill_name
        elif expected_skill not in sibling_names:
            expected_skill = None
        cases.append(
            {
                **item,
                "id": str(item.get("id") or f"{case_type}-{index:02d}"),
                "type": case_type,
                "expected_skill": expected_skill,
            }
        )
    if counters["should_trigger"] < 3 or counters["should_not_trigger"] < 2 or counters["edge_case"] < 1:
        raise DistillationError(
            f"Test generation for {skill_name} did not meet minimum case counts: {counters}"
        )
    other_skill_names = sibling_names - {skill_name}
    if other_skill_names and not any(
        item["type"] == "should_not_trigger" and item.get("expected_skill") in other_skill_names
        for item in cases
    ):
        sibling = next(
            (
                item
                for item in sorted(siblings, key=lambda value: str(value.get("name") or ""))
                if item.get("name") in other_skill_names
            ),
            None,
        )
        if sibling:
            scenarios = list((sibling.get("triggers") or {}).get("scenarios") or [])
            prompt = str(
                scenarios[0]
                if scenarios
                else sibling.get("description") or sibling.get("title") or sibling["name"]
            ).strip()
            cases.append(
                {
                    "id": "should-not-trigger-sibling-confusion",
                    "type": "should_not_trigger",
                    "prompt": f"我需要处理这样一个需求：{prompt}",
                    "expected_behavior": (
                        f"应触发 {sibling['name']}，而不是 {skill_name}，"
                        "用于验证相邻 Skill 的边界。"
                    ),
                    "expected_skill": sibling["name"],
                    "notes": "由测试规范补全的兄弟 Skill 混淆用例。",
                }
            )
    return {
        "skill": skill_name,
        "version": "0.1.0",
        "test_cases": cases,
        "minimum_pass_rate": 0.8,
        "negative_cases_must_all_pass": True,
    }


def score_tests(spec: dict[str, Any], judged: dict[str, Any]) -> dict[str, Any]:
    results_by_id = {
        str(item.get("id")): item for item in judged.get("results") or [] if isinstance(item, dict)
    }
    results = []
    passed_count = 0
    negative_failed = False
    for case in spec.get("test_cases") or []:
        judged_case = results_by_id.get(case["id"]) or {}
        selected = judged_case.get("selected_skill")
        passed = selected == case.get("expected_skill")
        if case["type"] == "edge_case" and case.get("expected_skill") is None:
            passed = selected in {None, "", "null"}
        if case["type"] == "should_not_trigger" and not passed:
            negative_failed = True
        passed_count += int(passed)
        results.append(
            {
                "id": case["id"],
                "type": case["type"],
                "prompt": case["prompt"],
                "expected_skill": case.get("expected_skill"),
                "selected_skill": selected,
                "passed": passed,
                "reason": judged_case.get("reason"),
                "action": judged_case.get("action"),
            }
        )
    total = len(results)
    pass_rate = passed_count / total if total else 0.0
    passed = total > 0 and pass_rate >= float(spec.get("minimum_pass_rate") or 0.8)
    if spec.get("negative_cases_must_all_pass") and negative_failed:
        passed = False
    return {
        "passed": passed,
        "pass_rate": round(pass_rate, 4),
        "passed_count": passed_count,
        "total": total,
        "negative_failed": negative_failed,
        "results": results,
    }


def render_overview(overview: dict[str, Any]) -> str:
    lines = [
        f"# {overview.get('title') or '内容整体理解'}",
        "",
        overview.get("summary") or "",
        "",
        "## 结构",
        "",
    ]
    for item in overview.get("structure") or []:
        lines.append(f"- **{item.get('title') or '未命名'}**：{item.get('summary') or ''} {source_suffix(item)}")
    lines.extend(["", "## 方法与流程", ""])
    for item in overview.get("methods") or []:
        lines.append(f"- **{item.get('title') or '未命名'}**：{item.get('summary') or ''} {source_suffix(item)}")
    lines.extend(["", "## 概念", ""])
    for item in overview.get("concepts") or []:
        lines.append(f"- **{item.get('term') or '未命名'}**：{item.get('meaning') or ''} {source_suffix(item)}")
    lines.extend(["", "## 风险与局限", ""])
    for item in list(overview.get("failures") or []) + list(overview.get("critique") or []):
        lines.append(
            f"- **{item.get('title') or item.get('issue') or '需复核'}**："
            f"{item.get('reason') or ''} {source_suffix(item)}"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_candidates(name: str, candidates: list[dict[str, Any]]) -> str:
    lines = [f"# {name}", ""]
    for item in candidates:
        lines.extend(
            [
                f"## {item.get('id')} {item.get('title')}",
                "",
                item.get("summary") or "",
                "",
                f"- 类型：`{item.get('type')}`",
                f"- 证据：{', '.join(item.get('source_ids') or []) or '无'}",
                f"- 执行线索：{item.get('execution_hint') or '无'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_verified(verified: dict[str, Any]) -> str:
    lines = ["# 三重验证与多模态复核结果", "", "## 已验证 Skills", ""]
    for item in verified.get("accepted") or []:
        lines.append(
            f"- `{item.get('id')}` **{item.get('title')}**："
            f"V1/V2/V3 通过；独立案例 "
            f"{(item.get('v1') or {}).get('independent_context_count', 0)}；"
            f"证据 {', '.join(item.get('source_ids') or [])}"
        )
    lines.extend(["", "## 单案例候选", ""])
    for item in verified.get("single_case") or []:
        audit = item.get("multimodal_audit") or {}
        lines.append(
            f"- `{item.get('id')}` **{item.get('title')}**："
            f"{item.get('reason')} 多模态复核 `{audit.get('status') or 'not_run'}`"
        )
    lines.extend(["", "## 淘汰", ""])
    for item in verified.get("rejected") or []:
        lines.append(
            f"- `{item.get('id')}` **{item.get('title')}**："
            f"{item.get('reason') or '未通过'}（{', '.join(item.get('failed_checks') or [])}）"
        )
    lines.extend(["", "## Glossary", ""])
    for item in verified.get("glossary") or []:
        lines.append(f"- `{item.get('id')}` **{item.get('title')}**")
    return "\n".join(lines).rstrip() + "\n"


def render_rejected(item: dict[str, Any]) -> str:
    return (
        f"# {item.get('title') or item.get('id')}\n\n"
        f"- ID: `{item.get('id')}`\n"
        f"- 未通过：{', '.join(item.get('failed_checks') or [])}\n"
        f"- 原因：{item.get('reason') or '未说明'}\n"
        f"- 证据：{', '.join(item.get('source_ids') or []) or '无'}\n"
    )


def render_skill(skill: dict[str, Any]) -> str:
    description = json.dumps(str(skill.get("description") or ""), ensure_ascii=False)
    lines = [
        "---",
        f"name: {skill['name']}",
        f"description: {description}",
        "---",
        "",
        f"# {skill.get('title') or skill['name']}",
        "",
        "## 方法骨架",
        "",
        skill.get("interpretation") or "",
        "",
        "## 触发场景",
        "",
    ]
    triggers = skill.get("triggers") or {}
    for scenario in triggers.get("scenarios") or []:
        lines.append(f"- {scenario}")
    signals = triggers.get("language_signals") or []
    if signals:
        lines.extend(["", "语言信号：" + "；".join(str(item) for item in signals)])
    lines.extend(["", "## 执行步骤", ""])
    for index, step in enumerate(skill.get("execution") or [], start=1):
        lines.append(f"{index}. **{step.get('title') or f'步骤 {index}'}**：{step.get('instruction') or ''}")
        if step.get("done_when"):
            lines.append(f"   - 完成标准：{step['done_when']}")
        if step.get("stop_condition"):
            lines.append(f"   - 判停条件：{step['stop_condition']}")
    lines.extend(["", "## 边界", ""])
    boundaries = skill.get("boundaries") or {}
    for value in boundaries.get("do_not_use") or []:
        lines.append(f"- 不适用：{value}")
    for value in boundaries.get("failure_modes") or []:
        lines.append(f"- 失败模式：{value}")
    for value in boundaries.get("limitations") or []:
        lines.append(f"- 局限：{value}")
    applications = skill.get("applications") or []
    if applications:
        lines.extend(["", "## 来源中的应用", ""])
        for item in applications[:3]:
            lines.append(
                f"- {item.get('problem') or '问题'} → {item.get('action') or '动作'} → "
                f"{item.get('result') or item.get('conclusion') or '结果'}"
            )
    reading = skill.get("reading") or {}
    if reading.get("quote"):
        lines.extend(
            [
                "",
                "## 证据摘录",
                "",
                f"> {reading.get('quote')}",
                "",
                f"来源：{reading.get('source_note') or ', '.join(reading.get('source_ids') or [])}",
            ]
        )
    related = skill.get("related_skills") or []
    if related:
        lines.extend(["", "## 相关 Skills", ""])
        for item in related:
            if isinstance(item, dict):
                lines.append(f"- `{item.get('relation') or 'related'}` → `{item.get('target')}`")
            else:
                lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def render_index(
    skills: list[dict[str, Any]],
    links_by_name: dict[str, list[dict[str, Any]]],
) -> str:
    lines = ["# Skills Index", ""]
    for skill in skills:
        name = skill.get("name")
        lines.append(f"- [`{name}`](distilled_skills/{name}/SKILL.md)：{skill.get('description') or ''}")
        for link in links_by_name.get(name, []):
            lines.append(f"  - {link['relation']} → `{link['target']}`")
    return "\n".join(lines).rstrip() + "\n"


def render_glossary(root: Path) -> str:
    payload = read_json(root / "candidates" / "glossary.json") or {}
    lines = ["# Glossary", ""]
    for item in payload.get("candidates") or []:
        lines.append(f"- **{item.get('title')}**：{item.get('summary') or ''}")
    return "\n".join(lines).rstrip() + "\n"


def render_test_results(result: dict[str, Any]) -> str:
    lines = [
        "# Test Results",
        "",
        f"- Passed: `{bool(result.get('passed'))}`",
        f"- Pass rate: `{float(result.get('pass_rate') or 0):.0%}`",
        f"- Repair round: `{result.get('repair_round', 0)}`",
        "",
    ]
    for item in result.get("results") or []:
        mark = "PASS" if item.get("passed") else "FAIL"
        lines.append(
            f"- `{mark}` `{item.get('id')}` expected `{item.get('expected_skill')}` "
            f"selected `{item.get('selected_skill')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_digest(overview: dict[str, Any], skills: dict[str, Any]) -> str:
    lines = [
        f"# {overview.get('title') or 'Skills 蒸馏交付'}",
        "",
        overview.get("summary") or "",
        "",
        f"- Skills：{skills.get('count') or 0}",
        f"- 压力测试通过：{skills.get('passed') or 0}",
        f"- 未通过：{skills.get('failed') or 0}",
        "",
        "## 可用 Skills",
        "",
    ]
    for item in skills.get("items") or []:
        lines.append(
            f"- `{item.get('name')}`：{item.get('title') or ''} "
            f"（{item.get('status')}，pass rate {item.get('pass_rate')}）"
        )
    return "\n".join(lines).rstrip() + "\n"


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    root = pack_dir(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = state.get("updated_at") or utc_now()
    write_json(root / STATE_NAME, state, atomic=True)
    (root / STATE_MD_NAME).write_text(render_state_markdown(state), encoding="utf-8")


def render_state_markdown(state: dict[str, Any]) -> str:
    lines = [
        "# Skill Distillation State",
        "",
        f"- Status: `{state.get('status')}`",
        f"- Stage: `{state.get('current_stage') or '-'}`",
        f"- Profile: `{state.get('profile')}`",
        f"- Generation model: `{state.get('generation_model')}`",
        f"- Review model: `{state.get('review_model')}`",
        f"- Updated: `{state.get('updated_at')}`",
        "",
        "## Stages",
        "",
    ]
    for name in PIPELINE_STAGES:
        info = (state.get("stages") or {}).get(name) or {}
        lines.append(f"- `{name}`：`{info.get('status') or 'pending'}`")
    if state.get("error"):
        lines.extend(["", "## Error", "", str(state["error"])])
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, payload: Any, *, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if not atomic:
        path.write_text(text, encoding="utf-8")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip() if path.is_file() else ""


def evidence_fingerprint(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def first_value(values: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values and values[key] not in (None, ""):
            return values[key]
    return ""


def as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def unique_strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def safe_slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return (text or "distilled-skill")[:64].rstrip("-")


def unique_skill_slug(root: Path, slug: str) -> str:
    candidate = slug
    index = 2
    while (root / candidate).exists():
        suffix = f"-{index}"
        candidate = f"{slug[:64-len(suffix)].rstrip('-')}{suffix}"
        index += 1
    return candidate


def existing_skill_slug_for_candidate(root: Path, candidate_id: str) -> str | None:
    for path in sorted(root.glob("*/skill.json")):
        skill = read_json(path) or {}
        if str(skill.get("candidate_id") or "") == candidate_id:
            return path.parent.name
    return None


def active_skill_paths(root: Path, state: dict[str, Any]) -> list[Path]:
    paths = []
    for item in (state.get("skills") or {}).get("items") or []:
        name = safe_slug(item.get("name") or "")
        path = root / "distilled_skills" / name / "skill.json"
        if path.is_file():
            paths.append(path)
    return paths


def safe_candidate_id(value: Any, used: set[str]) -> str:
    base = safe_slug(value)
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def source_suffix(item: dict[str, Any]) -> str:
    ids = unique_strings(item.get("source_ids") or [])
    return f"（{', '.join(ids)}）" if ids else ""


def format_seconds(value: float) -> str:
    seconds = max(0, int(value))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
