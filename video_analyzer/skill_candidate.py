"""Generate reviewable Codex skill drafts from video-analysis artifacts."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKILL_ROOT_NAME = "skills"
CANDIDATE_NAME = "tool_skill_candidate"
SKILL_MD_NAME = "SKILL.md"
REVIEW_NAME = "skill_review.json"
REFERENCE_DIR_NAME = "references"
EVIDENCE_REFERENCE_NAME = "evidence_summary.md"


def build_tool_skill_candidate(run_dir: Path, *, force: bool = True) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    artifacts = load_artifacts(run_dir)
    if not any(artifacts.get(key) for key in ("operation_manual", "study_guide", "manual_evidence", "transcript")):
        raise FileNotFoundError(f"No supported analysis artifacts found in: {run_dir}")

    candidate_dir = run_dir / SKILL_ROOT_NAME / CANDIDATE_NAME
    references_dir = candidate_dir / REFERENCE_DIR_NAME
    skill_path = candidate_dir / SKILL_MD_NAME
    review_path = candidate_dir / REVIEW_NAME
    evidence_path = references_dir / EVIDENCE_REFERENCE_NAME
    if candidate_dir.exists() and force:
        shutil.rmtree(candidate_dir)
    references_dir.mkdir(parents=True, exist_ok=True)

    title = infer_title(run_dir, artifacts)
    skill_name = safe_skill_name(title)
    if skill_name in {CANDIDATE_NAME, "tool"}:
        skill_name = safe_skill_name(f"{run_dir.parent.name}-{run_dir.name}")
    review = build_review(run_dir, artifacts, skill_name, title)
    skill_text = render_skill_md(skill_name, title, artifacts, review)
    evidence_text = render_evidence_reference(run_dir, artifacts, review)

    skill_path.write_text(skill_text, encoding="utf-8")
    evidence_path.write_text(evidence_text, encoding="utf-8")
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return candidate_summary(run_dir)


def candidate_summary(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    candidate_dir = run_dir / SKILL_ROOT_NAME / CANDIDATE_NAME
    skill_path = candidate_dir / SKILL_MD_NAME
    review_path = candidate_dir / REVIEW_NAME
    if not skill_path.is_file() or not review_path.is_file():
        return {"available": False, "candidate_dir": None, "skill_name": None, "enabled": False, "warnings": []}
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        review = {"warnings": [{"code": "invalid_review", "message": "skill_review.json is invalid"}]}
    enabled_path = review.get("enabled_path")
    return {
        "available": True,
        "candidate_dir": str(candidate_dir.relative_to(run_dir)),
        "skill_path": str(skill_path.relative_to(run_dir)),
        "review_path": str(review_path.relative_to(run_dir)),
        "skill_name": review.get("skill_name") or parse_skill_name(skill_path.read_text(encoding="utf-8", errors="replace")),
        "title": review.get("title"),
        "confidence": review.get("confidence"),
        "status": review.get("status") or "needs_review",
        "enabled": bool(enabled_path and Path(enabled_path).is_dir()),
        "enabled_path": enabled_path,
        "warnings": review.get("warnings") or [],
    }


def enable_tool_skill_candidate(run_dir: Path, repo_root: Path, *, overwrite: bool = True) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    summary = candidate_summary(run_dir)
    if not summary.get("available"):
        raise FileNotFoundError("Skill candidate is not available")
    skill_name = safe_skill_name(summary.get("skill_name") or summary.get("title") or CANDIDATE_NAME)
    source_dir = run_dir / summary["candidate_dir"]
    target_dir = repo_root / ".codex" / "skills" / skill_name
    target_root = (repo_root / ".codex" / "skills").resolve()
    target_dir = target_dir.resolve()
    try:
        target_dir.relative_to(target_root)
    except ValueError as exc:
        raise ValueError("Skill destination escapes .codex/skills") from exc
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Skill already exists: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir)

    review_path = source_dir / REVIEW_NAME
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.update(
        {
            "status": "enabled",
            "enabled_at": datetime.now(timezone.utc).isoformat(),
            "enabled_path": str(target_dir),
        }
    )
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (target_dir / REVIEW_NAME).write_text(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return candidate_summary(run_dir)


def load_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "operation_manual": read_text(run_dir / "operation_manual.md")
        or read_text(run_dir / "operation_manual.quality_failed.md"),
        "manual_evidence": read_text(run_dir / "manual_evidence.md"),
        "transcript": read_text(run_dir / "transcript.md"),
        "page_context": read_text(run_dir / "orin" / "page_context.md") or read_text(run_dir.parent / "page_context.md"),
        "comments": read_text(run_dir / "orin" / "comments.md"),
        "study_guide": read_json(run_dir / "study_guide.json"),
        "qa_index": read_json(run_dir / "qa" / "answer_index.json"),
        "evidence_gaps": read_json(run_dir / "evidence_gaps.json"),
        "web_evidence": read_json(run_dir / "web_evidence.json"),
        "publish_decision": read_json(run_dir / "publish_decision.json"),
    }


def build_review(run_dir: Path, artifacts: dict[str, Any], skill_name: str, title: str) -> dict[str, Any]:
    warnings: list[dict[str, str]] = []
    if not artifacts.get("manual_evidence"):
        warnings.append({"code": "missing_manual_evidence", "message": "缺少 manual_evidence.md，操作步骤证据需要人工复核。"})
    if not artifacts.get("transcript"):
        warnings.append({"code": "missing_transcript", "message": "缺少 transcript.md，口播解释无法交叉验证。"})
    gaps = artifacts.get("evidence_gaps") or {}
    gap_total = ((gaps.get("summary") or {}).get("total") if isinstance(gaps, dict) else None) or 0
    if gap_total:
        warnings.append({"code": "evidence_gaps", "message": f"存在 {gap_total} 个证据缺口，启用前需人工确认。"})
    web_evidence = artifacts.get("web_evidence") or {}
    web_summary = web_evidence.get("summary") if isinstance(web_evidence, dict) else {}
    if web_summary:
        unresolved = int(web_summary.get("unresolved") or 0) + int(web_summary.get("video_only_gap") or 0)
        if unresolved:
            warnings.append({"code": "web_evidence_unresolved", "message": f"联网补证据后仍有 {unresolved} 个缺口不能视为已解决。"})
    decision = artifacts.get("publish_decision") or {}
    if isinstance(decision, dict) and decision.get("status") not in {None, "", "publishable"}:
        warnings.append({"code": "publish_decision", "message": f"发布状态为 {decision.get('status')}，不能视为强确定性流程。"})
    if artifacts.get("comments"):
        warnings.append({"code": "comments_low_confidence", "message": "评论只能作为低置信补充，不能进入确定性操作步骤。"})
    confidence = "medium" if warnings else "high"
    return {
        "version": 1,
        "status": "needs_review",
        "skill_name": skill_name,
        "title": title,
        "confidence": confidence,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "sources": available_sources(artifacts),
        "warnings": warnings,
        "review_required": True,
        "enable_target": ".codex/skills",
    }


def render_skill_md(skill_name: str, title: str, artifacts: dict[str, Any], review: dict[str, Any]) -> str:
    summary = summarize_tool(artifacts)
    workflow = extract_workflow(artifacts)
    evidence_note = "启用前已要求人工审核；评论信息只作为低置信补充。"
    description = (
        f"Use when the user asks to operate, configure, troubleshoot, or explain {title}. "
        "Follow the evidence-grounded workflow captured from video-analyzer outputs; "
        "review references/evidence_summary.md when exact steps or caveats matter."
    )
    return f"""---
name: {skill_name}
description: {description}
---

# {title}

Use this skill when the user asks for help with this tool or workflow.

## Evidence Boundary

- {evidence_note}
- Prefer verified manual/evidence/transcript content over comments.
- If a requested step is absent from the evidence, say `需复核` instead of inventing.
- Detailed source summary lives in `references/{EVIDENCE_REFERENCE_NAME}`.

## Workflow

{workflow}

## Quick Summary

{summary}
"""


def render_evidence_reference(run_dir: Path, artifacts: dict[str, Any], review: dict[str, Any]) -> str:
    sections = [
        f"# Skill Evidence Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Candidate skill: `{review['skill_name']}`",
        f"- Confidence: `{review['confidence']}`",
        "",
        "## Sources",
        "",
        *[f"- `{source}`" for source in review.get("sources") or []],
        "",
        "## Review Warnings",
        "",
        *[f"- `{item['code']}` {item['message']}" for item in review.get("warnings") or []],
        "",
        "## Extracted Workflow Text",
        "",
        extract_workflow(artifacts),
    ]
    return "\n".join(sections).rstrip() + "\n"


def infer_title(run_dir: Path, artifacts: dict[str, Any]) -> str:
    guide = artifacts.get("study_guide") or {}
    if isinstance(guide, dict) and guide.get("title"):
        return clean_title(str(guide["title"]))
    for text in (artifacts.get("page_context") or "", artifacts.get("operation_manual") or ""):
        title = first_heading(text)
        if title:
            return clean_title(title)
    return clean_title(run_dir.parent.name or "tool workflow")


def extract_workflow(artifacts: dict[str, Any]) -> str:
    guide = artifacts.get("study_guide") or {}
    chapters = guide.get("chapters") if isinstance(guide, dict) else []
    lines: list[str] = []
    for chapter in (chapters or [])[:8]:
        raw_title = chapter.get("title") or f"步骤 {chapter.get('index') or len(lines) + 1}"
        summary = sentence_trim(chapter.get("summary") or "", 180)
        title = normalize_step_title(raw_title, summary, len(lines) + 1)
        if summary:
            lines.append(f"{len(lines) + 1}. **{title}**：{summary}")
        else:
            lines.append(f"{len(lines) + 1}. **{title}**")
    if lines:
        return "\n".join(lines)
    manual = artifacts.get("operation_manual") or ""
    bullets = extract_manual_steps(manual)
    if bullets:
        return "\n".join(f"{index + 1}. {step}" for index, step in enumerate(bullets[:8]))
    return "1. 先阅读 `references/evidence_summary.md`，再按用户目标提取可复核步骤。"


def summarize_tool(artifacts: dict[str, Any]) -> str:
    guide = artifacts.get("study_guide") or {}
    overview = guide.get("overview") if isinstance(guide, dict) else {}
    if isinstance(overview, dict) and overview.get("summary"):
        summary = sentence_trim(str(overview["summary"]), 420)
        chapters = guide.get("chapters") if isinstance(guide, dict) else []
        if chapters and "自动分段" in summary:
            titles = [
                normalize_step_title(chapter.get("title") or "", chapter.get("summary") or "", index + 1)
                for index, chapter in enumerate(chapters[:6])
            ]
            return f"全片分为 {len(chapters)} 个学习章节，核心路径包括：{'、'.join(titles)}。"
        return summary
    manual = artifacts.get("operation_manual") or ""
    return sentence_trim(strip_markdown(manual), 420) or "该 skill 来自 video-analyzer 工具内容分析产物。"


def extract_manual_steps(text: str) -> list[str]:
    steps = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^(?:\d+[\.\)]|[-*])\s+", stripped):
            steps.append(sentence_trim(re.sub(r"^(?:\d+[\.\)]|[-*])\s+", "", stripped), 180))
    return [item for item in steps if item]


def normalize_step_title(title: str, summary: str, index: int) -> str:
    title = clean_title(title)
    if title and not re.match(r"^(?:自动分段|章节|步骤)\s*\d*", title):
        return title[:40]
    derived = short_title_from_text(summary)
    return derived or f"步骤 {index:02d}"


def short_title_from_text(text: str) -> str:
    text = strip_markdown(text)
    text = re.sub(r"\s+", " ", text).strip(" ，,；;。.!！?？:-")
    if not text:
        return ""
    if re.search(r"[\u4e00-\u9fff]", text):
        return re.split(r"[，,。；;：:]", text, maxsplit=1)[0].strip()[:18]
    for _ in range(4):
        updated = re.sub(
            r"^(today|now|so|and|okay|alright|right|basically|actually|you can see here|you can see|we're going to|we are going to|i'm going to)\b[\s,.:;-]*",
            "",
            text,
            flags=re.I,
        ).strip(" ,.:;-")
        if updated == text:
            break
        text = updated
    text = re.sub(r"^(be|to)\s+", "", text, flags=re.I).strip(" ,.:;-")
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)
    while words and words[0].lower() in {"the", "a", "an", "this", "that", "we", "i", "you", "it"}:
        words.pop(0)
    title_words = words[:7]
    while len(title_words) > 3 and title_words[-1].lower() in {"to", "of", "in", "on", "at", "for", "with", "here", "there"}:
        title_words.pop()
    return " ".join(word if word.isupper() else word.capitalize() for word in title_words)


def safe_skill_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value or "").strip("-").lower()
    text = re.sub(r"-+", "-", text)
    if re.search(r"[\u4e00-\u9fff]", text):
        text = "tool-" + re.sub(r"[^a-z0-9-]+", "", text.encode("ascii", "ignore").decode("ascii"))
    text = text.strip("-") or CANDIDATE_NAME
    return text[:64].strip("-") or CANDIDATE_NAME


def parse_skill_name(text: str) -> str:
    match = re.search(r"^name:\s*(.+)$", text, flags=re.M)
    return match.group(1).strip() if match else CANDIDATE_NAME


def clean_title(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip(" #\t\r\n")
    text = re.sub(r"^(?:Page Context Evidence|页面上下文证据)\s*:\s*", "", text, flags=re.I)
    return text[:80] or "Tool Workflow"


def first_heading(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return title
        if line and not line.lower().startswith(("http", "url:")):
            return line[:120]
    return ""


def sentence_trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", strip_markdown(text or "")).strip()
    if len(text) <= limit:
        return text
    boundary = max(text.rfind(mark, 0, limit) for mark in ("。", "！", "？", ";", "；", ".", "!", "?"))
    if boundary >= limit // 2:
        return text[: boundary + 1].strip()
    return text[:limit].rstrip() + "..."


def strip_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text or "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[#>*_`|]+", " ", text)
    return text


def available_sources(artifacts: dict[str, Any]) -> list[str]:
    names = []
    for key, rel in (
        ("operation_manual", "operation_manual.md"),
        ("manual_evidence", "manual_evidence.md"),
        ("transcript", "transcript.md"),
        ("study_guide", "study_guide.json"),
        ("qa_index", "qa/answer_index.json"),
        ("evidence_gaps", "evidence_gaps.json"),
        ("web_evidence", "web_evidence.json"),
        ("publish_decision", "publish_decision.json"),
        ("page_context", "orin/page_context.md"),
        ("comments", "orin/comments.md"),
    ):
        if artifacts.get(key):
            names.append(rel)
    return names


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip() if path.is_file() else ""


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
