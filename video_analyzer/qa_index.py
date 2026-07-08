"""Build and query a lightweight evidence index for video-analysis outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - sklearn is an optional runtime dependency for callers.
    TfidfVectorizer = None
    cosine_similarity = None


INDEX_VERSION = 1
QA_DIR_NAME = "qa"
ANSWER_INDEX_NAME = "answer_index.json"
CHUNKS_NAME = "source_chunks.jsonl"
DEFAULT_CHUNK_CHARS = 1800
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_TOP_K = 10
DEFAULT_CONTEXT_CHARS = 60000

DOC_SOURCES = [
    ("manual_evidence", "manual_evidence.md", "highest confidence for visible UI, OCR, VL, screenshots", "high"),
    ("transcript", "transcript.md", "timestamped ASR/subtitle transcript for spoken claims", "high"),
    ("operation_manual", "operation_manual.md", "user-facing derived manual", "medium"),
    ("review_notes", "review_notes.md", "human/model review notes and evidence boundaries", "high"),
    ("study_overview", "study_overview.md", "structured learning overview", "medium"),
    ("study_cards", "study_cards.md", "structured learning cards", "medium"),
    ("evidence_index", "evidence_index.md", "time-aligned evidence index", "high"),
    ("page_context", "orin/page_context.md", "page metadata, description, subtitle diagnostics", "medium"),
    ("comments", "orin/comments.md", "low-confidence community comments only", "low"),
    ("knowledge_notes", "docs_analysis/knowledge_notes.md", "derived knowledge notes", "medium"),
    ("deep_report", "docs_analysis/deep_report.md", "derived deep report", "medium"),
    ("manual_review", "docs_analysis/operation_manual_review.md", "derived manual review and missing-item hints", "medium"),
    ("knowledge_notes_v2", "docs_analysis_chapters/knowledge_notes_v2.md", "chapter-based knowledge notes", "medium"),
    ("deep_report_v2", "docs_analysis_chapters/deep_report_v2.md", "chapter-based deep report", "medium"),
    ("deep_report_v2_review", "docs_analysis_chapters/deep_report_v2.review.md", "chapter-based deep report review", "medium"),
    ("study_guide", "study_guide.json", "structured study guide and chapters", "medium"),
    ("evidence_gaps", "evidence_gaps.json", "structured evidence gaps", "high"),
    ("evidence_triage", "evidence_triage.json", "structured evidence triage routes and publish impact", "high"),
    ("web_evidence", "web_evidence.md", "external web evidence supplements; cannot replace video evidence", "medium"),
    ("web_evidence_json", "web_evidence.json", "structured external web evidence supplements", "medium"),
    ("publish_decision", "publish_decision.json", "publish gate and risk decision", "high"),
    ("analysis_json", "analysis.json", "canonical structured analysis fallback", "medium"),
]


@dataclass
class SourceDoc:
    name: str
    rel_path: str
    note: str
    confidence: str
    text: str


def build_qa_index(run_dir: Path, *, force: bool = True) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    qa_dir = run_dir / QA_DIR_NAME
    index_path = qa_dir / ANSWER_INDEX_NAME
    chunks_path = qa_dir / CHUNKS_NAME
    if not force and index_path.is_file() and chunks_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8"))

    docs = load_source_docs(run_dir)
    if not docs:
        raise FileNotFoundError(f"No supported video-analysis documents found in: {run_dir}")
    chunks = []
    for doc in docs:
        for index, text in enumerate(split_text(doc.text)):
            chunks.append(
                {
                    "chunk_id": f"{doc.name}:{index:04d}",
                    "source": doc.name,
                    "path": doc.rel_path,
                    "note": doc.note,
                    "confidence": doc.confidence,
                    "text": text,
                    "timestamps": extract_timestamps(text),
                    "frames": extract_frames(text),
                }
            )

    warnings = quality_warnings(run_dir)
    qa_dir.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    index = {
        "version": INDEX_VERSION,
        "run_dir": str(run_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(docs),
        "chunk_count": len(chunks),
        "sources": [
            {
                "name": doc.name,
                "path": doc.rel_path,
                "note": doc.note,
                "confidence": doc.confidence,
                "chars": len(doc.text),
            }
            for doc in docs
        ],
        "warnings": warnings,
        "files": {
            "answer_index": str(index_path.relative_to(run_dir)),
            "source_chunks": str(chunks_path.relative_to(run_dir)),
        },
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def load_source_docs(run_dir: Path) -> list[SourceDoc]:
    docs = []
    for name, rel_path, note, confidence in DOC_SOURCES:
        path = run_dir / rel_path
        if not path.is_file():
            continue
        if path.suffix == ".json":
            text = json_to_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        text = text.strip()
        if text:
            docs.append(SourceDoc(name, rel_path, note, confidence, text))
    return docs


def load_index(run_dir: Path, *, build_if_missing: bool = True) -> dict[str, Any]:
    index_path = run_dir / QA_DIR_NAME / ANSWER_INDEX_NAME
    chunks_path = run_dir / QA_DIR_NAME / CHUNKS_NAME
    if (not index_path.is_file() or not chunks_path.is_file()) and build_if_missing:
        return build_qa_index(run_dir)
    if not index_path.is_file():
        raise FileNotFoundError(f"QA index is not available: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def load_chunks(run_dir: Path) -> list[dict[str, Any]]:
    chunks_path = run_dir / QA_DIR_NAME / CHUNKS_NAME
    if not chunks_path.is_file():
        raise FileNotFoundError(f"QA chunks are not available: {chunks_path}")
    chunks = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(json.loads(line))
    return chunks


def retrieve_context(run_dir: Path, question: str, *, max_context_chars: int = DEFAULT_CONTEXT_CHARS, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    index = load_index(run_dir)
    chunks = load_chunks(run_dir)
    if not chunks:
        raise FileNotFoundError(f"QA index has no chunks: {run_dir}")

    total_chars = sum(len(chunk.get("text") or "") for chunk in chunks)
    if total_chars <= max_context_chars:
        selected = chunks
    else:
        selected = rank_chunks(question, chunks, top_k=top_k)
        selected = fit_context_budget(selected, max_context_chars)
    return {"index": index, "chunks": selected, "warnings": index.get("warnings") or []}


def rank_chunks(question: str, chunks: list[dict[str, Any]], *, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    texts = [chunk.get("text") or "" for chunk in chunks]
    if TfidfVectorizer is not None and cosine_similarity is not None:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        matrix = vectorizer.fit_transform(texts + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [dict(chunks[index], score=round(float(score), 6)) for index, score in ranked[:top_k]]

    query_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", question.lower()))
    ranked = []
    for index, text in enumerate(texts):
        terms = set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))
        score = len(query_terms & terms) / max(len(query_terms), 1)
        ranked.append((index, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [dict(chunks[index], score=round(float(score), 6)) for index, score in ranked[:top_k]]


def fit_context_budget(chunks: list[dict[str, Any]], max_context_chars: int) -> list[dict[str, Any]]:
    selected = []
    used = 0
    for chunk in chunks:
        size = len(chunk.get("text") or "")
        if selected and used + size > max_context_chars:
            break
        selected.append(chunk)
        used += size
    return selected


def split_text(text: str, *, chunk_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(paragraph) > chunk_chars:
            chunks.append(paragraph[:chunk_chars])
            paragraph = paragraph[max(0, chunk_chars - overlap) :]
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def quality_warnings(run_dir: Path) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    analysis = read_json(run_dir / "analysis.json")
    metadata = analysis.get("metadata") if isinstance(analysis, dict) else {}
    if isinstance(metadata, dict):
        frame_selection = metadata.get("frame_selection") or {}
        vl_frames = metadata.get("vl_frames_processed")
        if vl_frames == 0 or frame_selection.get("vl_frame_policy_resolved") == "none":
            warnings.append(
                {
                    "code": "vl_skipped",
                    "severity": "warning",
                    "message": "本次分析未运行 VL 视觉理解；操作细节主要依赖 ASR/OCR/截图，回答 UI 操作细节时需更谨慎。",
                }
            )
        ocr_keyframes = metadata.get("ocr_keyframes") or {}
        if int(ocr_keyframes.get("ocr_text_events_count") or 0) == 0:
            warnings.append(
                {
                    "code": "no_ocr_text_events",
                    "severity": "warning",
                    "message": "OCR 未形成有效文本事件；屏幕文字相关答案需要复核。",
                }
            )
    gaps = read_json(run_dir / "evidence_gaps.json")
    summary = gaps.get("summary") if isinstance(gaps, dict) else {}
    if summary and int(summary.get("total") or 0) > 0:
        warnings.append(
            {
                "code": "evidence_gaps",
                "severity": "info",
                "message": f"存在 {summary.get('total')} 个证据缺口；回答时应标注相关不确定性。",
            }
        )
    web_evidence = read_json(run_dir / "web_evidence.json")
    web_summary = web_evidence.get("summary") if isinstance(web_evidence, dict) else {}
    if web_summary and int(web_summary.get("processed_gaps") or 0) > 0:
        unresolved = int(web_summary.get("unresolved") or 0) + int(web_summary.get("video_only_gap") or 0)
        warnings.append(
            {
                "code": "web_evidence",
                "severity": "info",
                "message": (
                    f"联网补证据已处理 {web_summary.get('processed_gaps')} 个缺口；"
                    f"外部补强 {web_summary.get('resolved_by_external') or 0} 个，"
                    f"部分补强 {web_summary.get('partial_external_support') or 0} 个，"
                    f"仍需复核 {unresolved} 个。"
                ),
            }
        )
    decision = read_json(run_dir / "publish_decision.json")
    if isinstance(decision, dict) and decision.get("status") not in {None, "", "publishable"}:
        warnings.append(
            {
                "code": "publish_decision",
                "severity": "warning",
                "message": f"发布门禁状态为 {decision.get('status')}，风险等级 {decision.get('risk_level') or '-'}：{decision.get('reason') or '无原因'}",
            }
        )
    return warnings


def json_to_text(path: Path) -> str:
    payload = read_json(path)
    return json.dumps(payload, ensure_ascii=False, indent=2) if payload is not None else path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_timestamps(text: str) -> list[str]:
    values = re.findall(r"\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d+(?:\.\d+)?s\b", text)
    return sorted(set(values), key=values.index)[:12]


def extract_frames(text: str) -> list[str]:
    values = re.findall(r"\b[Ff]rame[_\s-]*(\d+)\b|frame_(\d+)", text)
    frames = [left or right for left, right in values if (left or right).isdigit()]
    return [f"frame_{int(value):03d}" for value in sorted(set(frames), key=lambda item: int(item))[:12]]
