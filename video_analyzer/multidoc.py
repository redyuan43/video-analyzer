#!/usr/bin/env python3
"""Run multi-round document analysis from an existing operation-manual run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature


DEFAULT_TEXT_MODEL = "redhatai_qwen3.6-35b-a3b-nvfp4"
DEFAULT_DOC_TYPES = ["knowledge_notes", "deep_report", "operation_manual_review"]
DOC_FILENAMES = {
    "knowledge_notes": "knowledge_notes.md",
    "deep_report": "deep_report.md",
    "operation_manual_review": "operation_manual_review.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-round documents from an existing video analysis run")
    parser.add_argument("run_dir", help="Existing operation-manual run directory containing analysis.json and orin/")
    parser.add_argument("--config", default="config", help="Configuration directory containing optional config.json")
    parser.add_argument("--profile", help="Runtime profile from config/default_config.json or config.json")
    parser.add_argument("--doc-types", default="all", help="all or comma-separated: knowledge_notes,deep_report,operation_manual_review")
    parser.add_argument("--language", default="zh-CN")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output", help="Output directory; default RUN_DIR/docs_analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config(args.config)
    profile = config.get_runtime_profile(args.profile)
    default_base_url = (config.get("endpoints") or {}).get("services", {}).get("amd_fast_base_url")
    llm_base_url = args.llm_base_url or profile.get("llm_base_url") or default_base_url
    run_multidoc_analysis(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output) if args.output else None,
        doc_types=parse_doc_types(args.doc_types, profile),
        language=args.language,
        llm_base_url=llm_base_url,
        text_model=args.text_model or profile.get("text_model"),
        temperature=args.temperature if args.temperature is not None else resolve_temperature(profile, 0.2),
        api_key_env=profile.get("text_api_key_env") or profile.get("api_key_env"),
        extra_body=build_openai_extra_body(profile, llm_base_url),
    )
    return 0


def run_multidoc_analysis(
    run_dir: Path,
    output_dir: Path | None = None,
    doc_types: list[str] | None = None,
    language: str = "zh-CN",
    llm_base_url: str | None = None,
    text_model: str | None = None,
    temperature: float = 0.2,
    api_key_env: str | None = None,
    extra_body: dict[str, Any] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    validate_run_dir(run_dir)
    output_dir = (output_dir or (run_dir / "docs_analysis")).expanduser().resolve()
    orin_dir = output_dir / "orin"
    output_dir.mkdir(parents=True, exist_ok=True)
    orin_dir.mkdir(parents=True, exist_ok=True)

    doc_types = doc_types or DEFAULT_DOC_TYPES
    analysis = read_json(run_dir / "analysis.json")
    metadata = analysis.get("metadata") or {}
    model = text_model or metadata.get("text_model") or DEFAULT_TEXT_MODEL
    base_url = llm_base_url or metadata.get("llm_base_url") or _default_llm_base_url()
    client = client or GenericOpenAIAPIClient(
        resolve_api_key(api_key_env=api_key_env, api_url=base_url),
        base_url,
        extra_body=extra_body,
    )

    evidence = load_evidence(run_dir, analysis)
    round1 = generate_round(
        client,
        model,
        temperature,
        build_evidence_map_prompt(evidence, language),
        orin_dir / "round_01_evidence_map.md",
    )
    write_json(orin_dir / "round_01_evidence_map.json", build_evidence_map_json(evidence))

    round2 = generate_round(
        client,
        model,
        temperature,
        build_chapter_analysis_prompt(evidence, round1, language),
        orin_dir / "round_02_chapter_analysis.md",
    )
    write_json(orin_dir / "round_02_chapter_analysis.json", {"chapters": evidence["chapters"]})

    drafts: dict[str, str] = {}
    final_docs: dict[str, str] = {}
    for doc_type in doc_types:
        draft = generate_round(
            client,
            model,
            temperature,
            build_document_prompt(doc_type, evidence, round1, round2, language),
            orin_dir / f"round_03_{doc_type}_draft.md",
        )
        drafts[doc_type] = draft

    review = generate_round(
        client,
        model,
        temperature,
        build_review_prompt(evidence, drafts, language),
        orin_dir / "round_04_review.md",
    )
    write_json(orin_dir / "round_04_review.json", {"review": review, "doc_types": doc_types})

    for doc_type, draft in drafts.items():
        final_text = render_final_document(draft, review)
        path = output_dir / DOC_FILENAMES[doc_type]
        path.write_text(final_text, encoding="utf-8")
        final_docs[doc_type] = str(path)

    summary = {
        "run_dir": str(run_dir),
        "orin_dir": str(orin_dir),
        "language": language,
        "doc_types": doc_types,
        "llm_base_url": base_url,
        "text_model": model,
        "rounds": {
            "evidence_map": str(orin_dir / "round_01_evidence_map.md"),
            "chapter_analysis": str(orin_dir / "round_02_chapter_analysis.md"),
            "drafts": {doc_type: str(orin_dir / f"round_03_{doc_type}_draft.md") for doc_type in doc_types},
            "review": str(orin_dir / "round_04_review.md"),
        },
        "outputs": final_docs,
    }
    write_json(output_dir / "analysis.json", summary)
    return summary


def _default_llm_base_url() -> str:
    return (Config("config").get("endpoints") or {}).get("services", {}).get("amd_fast_base_url")


def validate_run_dir(run_dir: Path) -> None:
    if not (run_dir / "analysis.json").exists():
        raise FileNotFoundError(f"Missing analysis.json in run directory: {run_dir}")
    if not (run_dir / "orin").is_dir():
        raise FileNotFoundError(f"Missing orin/ in run directory: {run_dir}")


def parse_doc_types(value: str, profile: dict[str, Any] | None = None) -> list[str]:
    if value.strip() == "all":
        configured = (profile or {}).get("multidoc_doc_types") or DEFAULT_DOC_TYPES
        return list(configured)
    doc_types = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in doc_types if item not in DOC_FILENAMES]
    if invalid:
        raise ValueError(f"Invalid doc type(s): {', '.join(invalid)}")
    return doc_types


def load_evidence(run_dir: Path, analysis: dict[str, Any]) -> dict[str, Any]:
    orin_dir = run_dir / "orin"
    page_context = read_text_if_exists(orin_dir / "page_context.md") or read_text_if_exists(run_dir.parent / "page_context.md")
    transcript_md = read_text_if_exists(orin_dir / "transcript.md") or read_text_if_exists(run_dir / "transcript.md")
    manual = read_text_if_exists(run_dir / "operation_manual.md") or read_text_if_exists(run_dir / "operation_manual.quality_failed.md")
    manual_evidence = read_text_if_exists(run_dir / "manual_evidence.md")
    comments = read_text_if_exists(orin_dir / "comments.md")
    ocr_events = read_json_if_exists(orin_dir / "ocr_events.json") or analysis.get("ocr_events") or []
    frame_analyses = read_json_if_exists(orin_dir / "frame_analyses.json") or analysis.get("frame_analyses") or []
    transcript = normalize_transcript(read_json_if_exists(orin_dir / "transcript.json") or analysis.get("transcript") or {})
    study_context = load_study_context(run_dir)
    chapters = parse_study_chapters(study_context) or parse_chapters(page_context, transcript)
    chapter_transcript = build_chapter_transcript_digest(chapters, transcript)
    return {
        "page_context": page_context,
        "transcript_md": transcript_md,
        "transcript": transcript,
        "manual": manual,
        "manual_evidence": manual_evidence,
        "comments": comments,
        "ocr_events": ocr_events,
        "frame_analyses": frame_analyses,
        "chapters": chapters,
        "chapter_transcript": chapter_transcript,
        "study_context": study_context,
        "study_context_text": summarize_study_context(study_context),
        "metadata": analysis.get("metadata") or {},
    }


def build_evidence_map_json(evidence: dict[str, Any]) -> dict[str, Any]:
    study_context = evidence.get("study_context") or {}
    gaps = study_context.get("evidence_gaps") or {}
    guide = study_context.get("study_guide") or {}
    return {
        "chapter_count": len(evidence["chapters"]),
        "transcript_segments": len((evidence.get("transcript") or {}).get("segments") or []),
        "ocr_event_count": len(evidence["ocr_events"]),
        "frame_analysis_count": len(evidence["frame_analyses"]),
        "has_page_context": bool(evidence["page_context"]),
        "has_manual": bool(evidence["manual"]),
        "has_study_guide": bool(guide),
        "study_chapter_count": len(guide.get("chapters") or []),
        "evidence_gap_count": (gaps.get("summary") or {}).get("total", len(gaps.get("items") or [])),
        "publish_decision": (study_context.get("publish_decision") or {}).get("status"),
    }


def build_evidence_map_prompt(evidence: dict[str, Any], language: str) -> str:
    return f"""
你是视频资料分析员。请用 {language} 输出第一轮证据索引。

任务：
- 按章节/时间线列出视频主题、关键证据、截图/OCR/转写依据。
- 标出证据强弱：OCR/VL > 作者字幕 > VibeVoice ASR > 自动字幕 > 页面简介 > 置顶/作者评论 > 普通评论。
- 评论只能作为社区补充/FAQ/风险提示，不能单独形成确定性结论。

页面上下文：
{trim(evidence['page_context'], 5000)}

带时间戳转写：
{trim(evidence['chapter_transcript'], 30000)}

OCR/视觉证据摘要：
{trim(summarize_frame_evidence(evidence), 7000)}

结构化学习证据模型：
{trim(evidence.get('study_context_text') or '', 9000)}
""".strip()


def build_chapter_analysis_prompt(evidence: dict[str, Any], round1: str, language: str) -> str:
    return f"""
你是视频内容结构分析员。请用 {language} 输出第二轮逐章分析。

要求：
- 每章说明：核心论点、展开逻辑、重要例子、可复用方法、需复核点。
- 尽量保留时间戳。
- 不要把评论里的观点当成视频事实，除非视频/转写/OCR 支持。

章节列表：
{json.dumps(evidence['chapters'], ensure_ascii=False, indent=2)}

结构化学习证据模型：
{trim(evidence.get('study_context_text') or '', 9000)}

第一轮证据索引：
{trim(round1, 9000)}

转写摘要：
{trim(evidence['chapter_transcript'], 35000)}
""".strip()


def build_document_prompt(doc_type: str, evidence: dict[str, Any], round1: str, round2: str, language: str) -> str:
    instructions = {
        "knowledge_notes": "生成知识笔记：必须覆盖全部章节；每章包含核心观点、概念解释、重要例子、可复用方法、时间戳引用。",
        "deep_report": "生成深度报告：必须覆盖全部章节，不得压缩成少数大章；包含主论点、证据链、方法评价、风险限制、适用场景、延伸问题。",
        "operation_manual_review": "生成操作手册复核稿：基于现有手册与原始证据补充遗漏、标出需复核项，不覆盖原手册。",
    }
    return f"""
你是视频文档作者。请用 {language} {instructions[doc_type]}

硬性规则：
- 保留证据来源意识和时间戳。
- 对长视频不得只给概览；需要按原始章节逐段展开，解释每段为什么重要。
- 与视频证据冲突或证据不足的内容写入“需复核”。
- 评论只进入社区补充、FAQ、风险提示。
- 不要声称看到了没有证据支持的操作、命令或结论。

第一轮证据索引：
{trim(round1, 7000)}

第二轮逐章分析：
{trim(round2, 9000)}

结构化学习证据模型与证据缺口：
{trim(evidence.get('study_context_text') or '', 9000)}

按章节转写摘录：
{trim(evidence['chapter_transcript'], 30000)}

现有手册：
{trim(evidence['manual'], 7000)}

评论补充：
{trim(evidence['comments'], 3000)}
""".strip()


def build_review_prompt(evidence: dict[str, Any], drafts: dict[str, str], language: str) -> str:
    draft_text = "\n\n".join(f"## {doc_type}\n{trim(text, 5000)}" for doc_type, text in drafts.items())
    return f"""
你是文档复核员。请用 {language} 对第三轮草稿做最终复核。

检查点：
- 是否有缺失的重要章节、时间戳、概念或步骤。
- 是否有 ASR/OCR/VL/page context 冲突未标记“需复核”。
- 是否有评论污染主结论或确定性步骤。
- 是否有应进入 FAQ/社区补充的内容。

输出一个简短复核报告，包含“通过项”“需修正项”“最终发布建议”。

草稿：
{draft_text}

原始证据摘要：
{trim(summarize_frame_evidence(evidence), 6000)}

结构化学习证据模型与发布门禁：
{trim(evidence.get('study_context_text') or '', 9000)}
""".strip()


def generate_round(client: Any, model: str, temperature: float, prompt: str, path: Path) -> str:
    if not prompt.lstrip().startswith("/no_think"):
        prompt = f"/no_think\n{prompt}"
    response = client.generate(prompt=prompt, model=model, temperature=temperature, num_predict=8000)
    text = (response.get("response") or "").strip()
    path.write_text(text + "\n", encoding="utf-8")
    return text


def render_final_document(draft: str, review: str) -> str:
    return f"{draft.rstrip()}\n\n---\n\n## 多轮复核摘要\n\n{review.strip()}\n"


def load_study_context(run_dir: Path) -> dict[str, Any]:
    guide = read_json_if_exists(run_dir / "study_guide.json") or {}
    gaps = read_json_if_exists(run_dir / "evidence_gaps.json") or {}
    decision = read_json_if_exists(run_dir / "publish_decision.json") or {}
    chapter_dir = run_dir / "study_chapters"
    chapter_packets = []
    if chapter_dir.is_dir():
        for path in sorted(chapter_dir.glob("chapter_*.json")):
            payload = read_json_if_exists(path)
            if payload:
                chapter_packets.append(payload)
    return {
        "study_guide": guide,
        "evidence_gaps": gaps,
        "publish_decision": decision,
        "chapter_packets": chapter_packets,
    }


def parse_study_chapters(study_context: dict[str, Any]) -> list[dict[str, Any]]:
    guide = study_context.get("study_guide") or {}
    chapters = guide.get("chapters") or study_context.get("chapter_packets") or []
    parsed = []
    for index, chapter in enumerate(chapters, start=1):
        start = chapter.get("start") or chapter.get("start_time") or "00:00:00"
        end = chapter.get("end") or chapter.get("end_time") or ""
        title = chapter.get("title") or chapter.get("summary") or f"学习章节 {index:02d}"
        parsed.append(
            {
                "start": format_timestamp(timestamp_to_seconds(start)),
                "end": format_timestamp(timestamp_to_seconds(end)) if end else "",
                "title": str(title).strip(),
            }
        )
    return parsed


def normalize_transcript(transcript: Any) -> dict[str, Any]:
    if isinstance(transcript, list):
        transcript = {"segments": transcript}
    if not isinstance(transcript, dict):
        return {}
    normalized = dict(transcript)
    segments = []
    for segment in transcript.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        item = dict(segment)
        item["start"] = first_present(segment, ("start", "Start", "start_time", "StartTime", "begin", "Begin"))
        item["end"] = first_present(segment, ("end", "End", "end_time", "EndTime", "finish", "Finish"))
        item["text"] = first_present(segment, ("text", "Text", "content", "Content", "transcript", "Transcript"))
        segments.append(item)
    normalized["segments"] = segments
    return normalized


def first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def summarize_study_context(study_context: dict[str, Any]) -> str:
    guide = study_context.get("study_guide") or {}
    gaps = study_context.get("evidence_gaps") or {}
    decision = study_context.get("publish_decision") or {}
    chapter_packets = study_context.get("chapter_packets") or guide.get("chapters") or []
    compact_chapters = []
    for chapter in chapter_packets[:24]:
        compact_chapters.append(
            {
                "chapter_id": chapter.get("chapter_id"),
                "title": chapter.get("title"),
                "time": f"{chapter.get('start', '')} - {chapter.get('end', '')}".strip(),
                "summary": chapter.get("summary"),
                "key_points": chapter.get("key_points") or [],
                "evidence_ids": (chapter.get("evidence_ids") or [])[:12],
            }
        )
    compact_gaps = {
        "summary": gaps.get("summary") or {},
        "items": (gaps.get("items") or [])[:40],
    }
    payload = {
        "available": bool(guide),
        "overview": guide.get("overview") or {},
        "chapters": compact_chapters,
        "evidence_gaps": compact_gaps,
        "publish_decision": decision,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_chapters(page_context: str, transcript: dict[str, Any]) -> list[dict[str, Any]]:
    chapters = []
    for line in page_context.splitlines():
        match = re.match(r"-\s+(\d\d:\d\d:\d\d)\s+-\s+(\d\d:\d\d:\d\d):\s+(.+)", line.strip())
        if match:
            chapters.append({"start": match.group(1), "end": match.group(2), "title": match.group(3).strip()})
    if chapters:
        return chapters
    segments = (transcript or {}).get("segments") or []
    timed_segments = [segment for segment in segments if segment_seconds(segment, "end") > segment_seconds(segment, "start")]
    if not timed_segments:
        return [{"start": "00:00:00", "end": "", "title": "全片"}]
    return build_fallback_chapters(timed_segments)


def build_fallback_chapters(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first = min(segment_seconds(segment, "start") for segment in segments)
    last = max(segment_seconds(segment, "end") for segment in segments)
    duration = max(last - first, 1.0)
    chapter_count = max(1, min(10, round(duration / 240.0)))
    if duration >= 900:
        chapter_count = max(6, chapter_count)
    if chapter_count == 1:
        return [{"start": format_timestamp(first), "end": format_timestamp(last), "title": "全片"}]

    boundaries = [first + (duration * index / chapter_count) for index in range(chapter_count + 1)]
    chapters = []
    for index in range(chapter_count):
        start = boundaries[index]
        end = boundaries[index + 1]
        chapter_segments = [
            segment
            for segment in segments
            if start <= segment_seconds(segment, "start") < end
        ]
        chapters.append(
            {
                "start": format_timestamp(start),
                "end": format_timestamp(end),
                "title": fallback_chapter_title(chapter_segments, index + 1),
            }
        )
    return chapters


def fallback_chapter_title(segments: list[dict[str, Any]], index: int) -> str:
    text = " ".join(str(segment.get("text") or "").replace("\n", " ").strip() for segment in segments if segment.get("text"))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return f"章节 {index:02d}"
    title = concise_topic_from_text(text)
    return title or f"章节 {index:02d}"


def concise_topic_from_text(text: str) -> str:
    text = clean_topic_text(text)
    if not text:
        return ""
    clauses = [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*|[。！？!?；;]\s*|\n+", text) if part.strip()]
    candidate = next((part for part in clauses if len(part) >= 8), clauses[0] if clauses else text)
    candidate = clean_topic_text(candidate)
    if re.search(r"[\u4e00-\u9fff]", candidate):
        return trim_chinese_title(candidate)
    return trim_english_title(candidate)


def clean_topic_text(text: str) -> str:
    text = re.sub(r"\[[0-9:.]+\s*-\s*[0-9:.]+\]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip(" ，,；;。.!！?？:-")
    for _ in range(4):
        updated = re.sub(
            r"^(today|now|so|and|okay|alright|right|basically|actually|you know|let me|let's|you can see here|you can see|we're going to|we are going to|i'm going to)\b[\s,.:;-]*",
            "",
            text,
            flags=re.I,
        ).strip(" ，,；;。.!！?？:-")
        updated = re.sub(r"^(we're|we are|i'm|i am)\s+going\s+to\s+", "", updated, flags=re.I)
        updated = re.sub(r"^(be|to)\s+", "", updated, flags=re.I).strip(" ，,；;。.!！?？:-")
        if updated == text:
            break
        text = updated
    return text


def trim_chinese_title(text: str) -> str:
    candidate = re.split(r"[，,。；;：:]", text, maxsplit=1)[0].strip()
    return candidate[:18].strip() if candidate else text[:18].strip()


def trim_english_title(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)
    stop_prefix = {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "we",
        "i",
        "you",
        "they",
        "it",
        "can",
        "could",
        "will",
        "would",
        "should",
        "just",
        "really",
    }
    while words and words[0].lower() in stop_prefix:
        words.pop(0)
    if not words:
        return ""
    title_words = words[:7]
    suffix_stop = {"to", "of", "in", "on", "at", "for", "with", "here", "there", "right"}
    while len(title_words) > 3 and title_words[-1].lower() in suffix_stop:
        title_words.pop()
    return " ".join(word if word.isupper() else word.capitalize() for word in title_words)


def build_chapter_transcript_digest(
    chapters: list[dict[str, Any]],
    transcript: dict[str, Any],
    max_chars_per_chapter: int = 1800,
) -> str:
    segments = (transcript or {}).get("segments") or []
    if not segments:
        return ""
    blocks = []
    for chapter in chapters:
        start = timestamp_to_seconds(chapter.get("start"))
        end = timestamp_to_seconds(chapter.get("end"))
        if end <= start:
            end = float("inf")
        chapter_segments = [
            segment
            for segment in segments
            if start <= segment_seconds(segment, "start") < end
        ]
        lines = [
            f"## {chapter.get('start', '')} - {chapter.get('end', '')} {chapter.get('title', '')}",
            f"- Segments: {len(chapter_segments)}",
        ]
        text_lines = [format_segment_line(segment) for segment in chapter_segments]
        lines.append(trim_preserving_ends("\n".join(text_lines), max_chars_per_chapter))
        blocks.append("\n".join(line for line in lines if line))
    return "\n\n".join(blocks)


def format_segment_line(segment: dict[str, Any]) -> str:
    start = format_timestamp(segment_seconds(segment, "start"))
    end = format_timestamp(segment_seconds(segment, "end"))
    text = str(segment.get("text") or "").replace("\n", " ").strip()
    return f"[{start}-{end}] {text}"


def segment_seconds(segment: dict[str, Any], key: str) -> float:
    value = segment.get(key)
    if value is None:
        value = segment.get(f"{key}_time")
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def timestamp_to_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(text)
    except ValueError:
        return 0.0


def summarize_frame_evidence(evidence: dict[str, Any]) -> str:
    ocr_events = evidence.get("ocr_events") or []
    frame_analyses = evidence.get("frame_analyses") or []
    lines = []
    for index, analysis in enumerate(frame_analyses[:40]):
        ocr = ocr_events[index] if index < len(ocr_events) else {}
        lines.extend(
            [
                f"Frame {index}",
                f"Visual: {str((analysis or {}).get('response') or '')[:800]}",
                f"OCR: {str((ocr or {}).get('text') or '')[:800]}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def format_timestamp(value: Any) -> str:
    try:
        seconds = int(float(value or 0))
    except (TypeError, ValueError):
        seconds = 0
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def trim(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[truncated]"


def trim_preserving_ends(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n\n[chapter middle truncated]\n\n" + text[-half:].lstrip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return read_json(path)


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
