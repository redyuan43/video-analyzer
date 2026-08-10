"""Local, goal-oriented Skill project storage and readiness assessment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_FILE = "project.json"
SOURCES_DIR = "sources"
FROZEN_SOURCES_FILE = "source_snapshot.json"
MAX_IMPORTED_SOURCE_BYTES = 1_000_000
ALLOWED_IMPORT_SUFFIXES = {".txt", ".md", ".json", ".jsonl"}
PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
SOURCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
KNOWN_CAPABILITIES = {
    "python3": {
        "id": "python3",
        "label": "Python 3",
        "smoke_test": {"kind": "command_help", "target": "python3"},
    },
    "ffmpeg": {
        "id": "ffmpeg",
        "label": "FFmpeg",
        "smoke_test": {"kind": "command_help", "target": "ffmpeg"},
    },
    "yt-dlp": {
        "id": "yt-dlp",
        "label": "yt-dlp",
        "smoke_test": {"kind": "command_help", "target": "yt-dlp"},
    },
    "docker": {
        "id": "docker",
        "label": "Docker",
        "smoke_test": {"kind": "command_help", "target": "docker"},
    },
    "vibevoice-asr": {
        "id": "vibevoice-asr",
        "label": "VibeVoice ASR",
        "smoke_test": {
            "kind": "http_health",
            "target": "http://127.0.0.1:18012/api/health",
        },
    },
}
CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*['\"]?[a-z0-9_\-]{16,}"),
    re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),
    re.compile(r"(?i)\bghp_[a-z0-9]{30,}\b"),
)


class SkillProjectError(ValueError):
    """Raised when Skill project data is invalid or cannot continue."""


def iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_id(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not SOURCE_ID_PATTERN.fullmatch(normalized):
        raise SkillProjectError(f"{field} is invalid")
    return normalized


def _source_id() -> str:
    return f"source-{uuid.uuid4().hex[:12]}"


def _split_text(text: str, max_chars: int = 4000) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    result: list[str] = []
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            result.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        if paragraph:
            result.append(paragraph)
    return result


def _credential_warning(text: str) -> str | None:
    for pattern in CREDENTIAL_PATTERNS:
        if pattern.search(text):
            return "Imported material appears to include a credential or token"
    return None


def _goal_kind(goal: str) -> str:
    value = goal.lower()
    tool_markers = ("脚本", "命令", "命令行", "api", "接口", "服务", "docker", "cli", "工具")
    knowledge_markers = ("方法", "流程", "排错", "诊断", "分析", "操作", "工作流", "规范", "实践")
    has_tool = any(marker in value for marker in tool_markers)
    has_knowledge = any(marker in value for marker in knowledge_markers)
    if has_tool and has_knowledge:
        return "hybrid"
    if has_tool:
        return "tool_workflow"
    return "knowledge_method"


def _not_a_skill(goal: str) -> bool:
    value = goal.strip().lower()
    return bool(value) and any(
        value.startswith(prefix)
        for prefix in ("总结", "翻译", "闲聊", "聊天", "summarize", "translate", "chat")
    )


class SkillProjectStore:
    """Atomic local storage for goal-oriented skill projects."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def project_dir(self, project_id: str) -> Path:
        if not PROJECT_ID_PATTERN.fullmatch(str(project_id or "")):
            raise FileNotFoundError("Skill project is not available")
        return self.root / project_id

    def list(self) -> list[dict[str, Any]]:
        projects = []
        for path in self.root.glob(f"*/{PROJECT_FILE}"):
            try:
                projects.append(self.load(path.parent.name))
            except (FileNotFoundError, SkillProjectError):
                continue
        projects.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return projects

    def load(self, project_id: str) -> dict[str, Any]:
        path = self.project_dir(project_id) / PROJECT_FILE
        if not path.is_file():
            raise FileNotFoundError("Skill project is not available")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillProjectError("Skill project metadata is invalid") from exc
        if not isinstance(payload, dict) or payload.get("id") != project_id:
            raise SkillProjectError("Skill project metadata is invalid")
        return payload

    def save(self, project: dict[str, Any]) -> dict[str, Any]:
        project_id = str(project.get("id") or "")
        self.project_dir(project_id).mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.project_dir(project_id) / PROJECT_FILE, project)
        return project

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        goal = str(payload.get("goal") or "").strip()
        title = str(payload.get("title") or "").strip()
        origin_job_id = str(payload.get("job_id") or "").strip()
        if not goal:
            raise SkillProjectError("goal is required")
        if origin_job_id and not re.fullmatch(r"[a-f0-9]{32}", origin_job_id):
            raise SkillProjectError("job_id is invalid")
        requested_capabilities = payload.get("required_capabilities") or []
        if not isinstance(requested_capabilities, list):
            raise SkillProjectError("required_capabilities must be a list")
        requested_capabilities = [
            str(item).strip() for item in requested_capabilities if str(item).strip()
        ]
        unknown_capabilities = sorted(set(requested_capabilities) - set(KNOWN_CAPABILITIES))
        if unknown_capabilities:
            raise SkillProjectError(
                f"Unknown capabilities: {', '.join(unknown_capabilities)}"
            )
        now = iso_now()
        project_id = uuid.uuid4().hex
        project = {
            "id": project_id,
            "origin": "job" if origin_job_id else "global",
            "origin_job_id": origin_job_id or None,
            "title": title or goal[:80],
            "brief": {
                "goal": goal,
                "normalized_goal": "",
                "skill_type": _goal_kind(goal),
                "trigger_examples": [],
                "expected_output": "",
                "boundaries": [],
                "acceptance_criteria": [],
                "required_capabilities": requested_capabilities,
            },
            "sources": [],
            "assessment": {},
            "capability_checks": [],
            "distillation": {},
            "status": "draft",
            "revision": 1,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self.save(project)
            if origin_job_id:
                self.add_source(project_id, {"kind": "job", "job_id": origin_job_id})
        return self.load(project_id)

    def update(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            project = self.load(project_id)
            changed = False
            if "title" in payload:
                title = str(payload.get("title") or "").strip()
                if not title:
                    raise SkillProjectError("title cannot be empty")
                project["title"] = title[:160]
                changed = True
            if "goal" in payload:
                goal = str(payload.get("goal") or "").strip()
                if not goal:
                    raise SkillProjectError("goal cannot be empty")
                project.setdefault("brief", {})["goal"] = goal
                project["brief"]["skill_type"] = _goal_kind(goal)
                changed = True
            for key in ("expected_output", "normalized_goal"):
                if key in payload:
                    project.setdefault("brief", {})[key] = str(payload.get(key) or "").strip()
                    changed = True
            for key in ("trigger_examples", "boundaries", "acceptance_criteria", "required_capabilities"):
                if key not in payload:
                    continue
                value = payload.get(key)
                if not isinstance(value, list):
                    raise SkillProjectError(f"{key} must be a list")
                if key == "required_capabilities":
                    value = [str(item).strip() for item in value if str(item).strip()]
                    unknown = sorted(set(value) - set(KNOWN_CAPABILITIES))
                    if unknown:
                        raise SkillProjectError(f"Unknown capabilities: {', '.join(unknown)}")
                else:
                    value = [str(item).strip()[:500] for item in value if str(item).strip()]
                project.setdefault("brief", {})[key] = value
                changed = True
            if changed:
                self._invalidate_assessment(project)
                self._touch(project)
                self.save(project)
            return project

    def add_source(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("kind") or "").strip().lower()
        if kind not in {"job", "conversation"}:
            raise SkillProjectError("source kind must be job or conversation")
        with self._lock:
            project = self.load(project_id)
            source_id = _source_id()
            source: dict[str, Any] = {
                "id": source_id,
                "kind": kind,
                "created_at": iso_now(),
            }
            if kind == "job":
                job_id = str(payload.get("job_id") or "").strip()
                if not re.fullmatch(r"[a-f0-9]{32}", job_id):
                    raise SkillProjectError("job_id is invalid")
                if any(item.get("kind") == "job" and item.get("job_id") == job_id for item in project["sources"]):
                    raise SkillProjectError("Job is already a source for this project")
                source.update(
                    {
                        "job_id": job_id,
                        "include_qa": bool(payload.get("include_qa", True)),
                        "label": str(payload.get("label") or f"任务 {job_id[:8]}").strip()[:160],
                    }
                )
            else:
                content = str(payload.get("content") or "")
                encoded = content.encode("utf-8")
                if not content.strip():
                    raise SkillProjectError("conversation content is required")
                if len(encoded) > MAX_IMPORTED_SOURCE_BYTES:
                    raise SkillProjectError("Imported material is too large")
                warning = _credential_warning(content)
                if warning:
                    raise SkillProjectError(warning)
                filename = str(payload.get("filename") or "conversation.md").strip()
                suffix = Path(filename).suffix.lower()
                if suffix not in ALLOWED_IMPORT_SUFFIXES:
                    raise SkillProjectError("Imported material must be txt, md, json, or jsonl")
                data_path = self.project_dir(project_id) / SOURCES_DIR / f"{source_id}.json"
                _atomic_write_json(
                    data_path,
                    {"filename": Path(filename).name, "content": content, "format": suffix.lstrip(".")},
                )
                source.update(
                    {
                        "filename": Path(filename).name,
                        "data_file": str(data_path.relative_to(self.project_dir(project_id))),
                        "label": str(payload.get("label") or Path(filename).name).strip()[:160],
                        "bytes": len(encoded),
                    }
                )
            project["sources"].append(source)
            self._invalidate_assessment(project)
            self._touch(project)
            self.save(project)
            return project

    def remove_source(self, project_id: str, source_id: str) -> dict[str, Any]:
        source_id = _safe_id(source_id, field="source_id")
        with self._lock:
            project = self.load(project_id)
            source = next((item for item in project["sources"] if item.get("id") == source_id), None)
            if not source:
                raise FileNotFoundError("Skill project source is not available")
            project["sources"] = [item for item in project["sources"] if item.get("id") != source_id]
            data_file = str(source.get("data_file") or "")
            if data_file:
                path = (self.project_dir(project_id) / data_file).resolve()
                try:
                    path.relative_to(self.project_dir(project_id))
                except ValueError:
                    path = None
                if path and path.is_file():
                    path.unlink()
            self._invalidate_assessment(project)
            self._touch(project)
            self.save(project)
            return project

    def source_data(self, project_id: str, source: dict[str, Any]) -> dict[str, Any]:
        relative = str(source.get("data_file") or "")
        if not relative:
            return {}
        path = (self.project_dir(project_id) / relative).resolve()
        try:
            path.relative_to(self.project_dir(project_id))
        except ValueError as exc:
            raise SkillProjectError("Source data path escapes project") from exc
        if not path.is_file():
            raise FileNotFoundError("Imported source data is not available")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillProjectError("Imported source data is invalid") from exc
        return payload if isinstance(payload, dict) else {}

    def freeze_sources(self, project_id: str, bundle: dict[str, Any]) -> dict[str, Any]:
        project_dir = self.project_dir(project_id)
        snapshot = {
            "version": 1,
            "frozen_at": iso_now(),
            "project_revision": bundle.get("project_revision"),
            "fingerprint": bundle.get("fingerprint"),
            "sources": bundle.get("sources") or [],
            "records": bundle.get("records") or [],
        }
        _atomic_write_json(project_dir / FROZEN_SOURCES_FILE, snapshot)
        return snapshot

    @staticmethod
    def _touch(project: dict[str, Any]) -> None:
        project["revision"] = int(project.get("revision") or 0) + 1
        project["updated_at"] = iso_now()

    @staticmethod
    def _invalidate_assessment(project: dict[str, Any]) -> None:
        project["assessment"] = {}
        project["capability_checks"] = []
        if project.get("status") not in {"distilling", "completed"}:
            project["status"] = "draft"


def build_source_bundle(
    project: dict[str, Any],
    project_dir: Path,
    *,
    job_records: Callable[[str], tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    qa_history: Callable[[str], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Materialize project references as immutable, namespaced pipeline records."""

    records: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source in project.get("sources") or []:
        source_id = str(source.get("id") or "")
        kind = str(source.get("kind") or "")
        if kind == "job":
            job_id = str(source.get("job_id") or "")
            raw_records, raw_sources = job_records(job_id)
            for index, raw in enumerate(raw_records):
                item = _json_copy(raw)
                original_id = str(item.get("id") or f"record-{index:04d}")
                item["id"] = f"job:{job_id}:{original_id}"
                item["path"] = f"jobs/{job_id}/{item.get('path') or original_id}"
                item["case_id"] = f"job:{job_id}"
                item["event_id"] = f"job:{job_id}:event:{index:05d}"
                item["project_source_id"] = source_id
                records.append(item)
            qa_count = 0
            if source.get("include_qa"):
                for index, entry in enumerate(qa_history(job_id)):
                    question = str(entry.get("question") or "").strip()
                    answer = str(entry.get("answer") or "").strip()
                    text = "\n".join(value for value in (question, answer) if value)
                    if not text:
                        continue
                    records.append(
                        {
                            "id": f"job:{job_id}:qa:{index:04d}",
                            "source_type": "qa",
                            "path": f"jobs/{job_id}/qa/chat_history",
                            "confidence": "low",
                            "text": text[:6000],
                            "case_id": f"job:{job_id}",
                            "event_id": f"job:{job_id}:qa:{index:04d}",
                            "project_source_id": source_id,
                        }
                    )
                    qa_count += 1
            source_summaries.append(
                {
                    "id": source_id,
                    "kind": "job",
                    "job_id": job_id,
                    "records": len(raw_records),
                    "qa_records": qa_count,
                    "raw_sources": raw_sources,
                }
            )
            continue
        if kind == "conversation":
            relative = str(source.get("data_file") or "")
            path = (project_dir / relative).resolve()
            try:
                path.relative_to(project_dir.resolve())
            except ValueError as exc:
                raise SkillProjectError("Imported source data path escapes project") from exc
            if not path.is_file():
                raise FileNotFoundError("Imported source data is not available")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SkillProjectError("Imported source data is invalid") from exc
            content = str((data or {}).get("content") or "")
            chunks = _split_text(content)
            for index, chunk in enumerate(chunks):
                records.append(
                    {
                        "id": f"conversation:{source_id}:{index:04d}",
                        "source_type": "conversation",
                        "path": relative,
                        "confidence": "medium",
                        "text": chunk,
                        "case_id": f"conversation:{source_id}",
                        "event_id": f"conversation:{source_id}:{index:04d}",
                        "project_source_id": source_id,
                    }
                )
            source_summaries.append(
                {
                    "id": source_id,
                    "kind": "conversation",
                    "filename": source.get("filename"),
                    "records": len(chunks),
                }
            )
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "project_revision": project.get("revision"),
        "records": records,
        "sources": source_summaries,
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def capability_inventory(repo_root: Path, profile_names: list[str]) -> dict[str, Any]:
    return {
        "repo_root": str(repo_root),
        "profiles": sorted(profile_names),
        "capabilities": [
            {
                "id": capability["id"],
                "label": capability["label"],
                "smoke_test": _json_copy(capability["smoke_test"]),
                "available": capability["id"] in {"vibevoice-asr"} or bool(
                    __import__("shutil").which(capability["smoke_test"]["target"])
                ),
            }
            for capability in KNOWN_CAPABILITIES.values()
        ],
    }


def assess_project(
    project: dict[str, Any],
    bundle: dict[str, Any],
    inventory: dict[str, Any],
    *,
    model_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a conservative readiness verdict without fabricating certainty."""

    brief = dict(project.get("brief") or {})
    goal = str(brief.get("goal") or "").strip()
    records = list(bundle.get("records") or [])
    high_records = [
        item
        for item in records
        if item.get("confidence") == "high"
        and item.get("source_type") in {"transcript", "subtitle", "ocr", "visual"}
    ]
    independent_cases = sorted({str(item.get("case_id") or "") for item in high_records if item.get("case_id")})
    model_assessment = model_assessment if isinstance(model_assessment, dict) else {}
    skill_type = str(model_assessment.get("skill_type") or brief.get("skill_type") or _goal_kind(goal))
    if skill_type not in {"knowledge_method", "tool_workflow", "hybrid"}:
        skill_type = _goal_kind(goal)
    requested = list(brief.get("required_capabilities") or [])
    requested.extend(model_assessment.get("required_capabilities") or [])
    requested = list(dict.fromkeys(str(item).strip() for item in requested if str(item).strip()))
    checks = []
    inventory_map = {item["id"]: item for item in inventory.get("capabilities") or []}
    for capability_id in requested:
        known = KNOWN_CAPABILITIES.get(capability_id)
        if not known:
            checks.append(
                {
                    "id": capability_id,
                    "label": capability_id,
                    "status": "unknown",
                    "verification": "unverified",
                    "smoke_test": None,
                }
            )
            continue
        available = bool((inventory_map.get(capability_id) or {}).get("available"))
        checks.append(
            {
                "id": capability_id,
                "label": known["label"],
                "status": "unverified" if available else "missing",
                "verification": "inventory",
                "smoke_test": _json_copy(known["smoke_test"]),
            }
        )
    questions = [
        str(item).strip()
        for item in (model_assessment.get("questions") or [])
        if str(item).strip()
    ][:3]
    material_requests = []
    if not goal:
        material_requests.append(
            {
                "id": "clarify-goal",
                "reason": "缺少要沉淀的目标",
                "minimum_cases": 0,
                "acceptance": "给出一个可触发、可验收的自然语言目标",
            }
        )
    if not high_records:
        material_requests.append(
            {
                "id": "raw-evidence",
                "reason": "当前没有可追溯的原始转写、OCR 或视觉证据",
                "minimum_cases": 1,
                "acceptance": "关联至少一个含原始分析证据的学习任务",
            }
        )
    elif len(independent_cases) < 2:
        material_requests.append(
            {
                "id": "independent-cases",
                "reason": "当前只有一个独立学习案例，无法完整满足 V1",
                "minimum_cases": 2,
                "acceptance": "关联另一项独立学习任务或等价原始案例",
            }
        )
    missing_capabilities = [
        item for item in checks if item.get("status") in {"unknown", "missing"}
    ]
    unverified_capabilities = [
        item for item in checks if item.get("status") == "unverified"
    ]
    if not goal:
        verdict = "blocked"
    elif _not_a_skill(goal):
        verdict = "not_a_skill"
    elif missing_capabilities:
        verdict = "needs_capability"
    elif not high_records:
        verdict = "needs_materials"
    elif len(independent_cases) < 2:
        verdict = "ready_limited"
    elif unverified_capabilities:
        verdict = "ready_limited"
    else:
        verdict = "ready"
    normalized_goal = str(model_assessment.get("normalized_goal") or goal).strip()
    return {
        "verdict": verdict,
        "summary": str(
            model_assessment.get("summary")
            or {
                "ready": "目标、证据和所需能力已满足蒸馏条件",
                "ready_limited": "可以继续，但必须明确接受资料或能力验证限制",
                "needs_materials": "需要补充原始学习证据",
                "needs_capability": "需要补齐或确认执行能力",
                "blocked": "需要先明确项目目标",
                "not_a_skill": "该目标更适合作为一次性任务，不适合沉淀为 Skill",
            }[verdict]
        ),
        "project_revision": project.get("revision"),
        "assessed_at": iso_now(),
        "normalized_goal": normalized_goal,
        "skill_type": skill_type,
        "questions": questions,
        "material_requests": material_requests,
        "evidence_gaps": [item["reason"] for item in material_requests],
        "source_coverage": {
            "records": len(records),
            "high_confidence_records": len(high_records),
            "independent_learning_cases": len(independent_cases),
            "case_ids": independent_cases,
        },
        "capabilities": checks,
        "warnings": [],
    }
