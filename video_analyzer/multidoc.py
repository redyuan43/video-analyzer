#!/usr/bin/env python3
"""Run multi-round document analysis from an existing operation-manual run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
CHAPTER_ANALYSIS_DIRNAME = "chapters"
CHAPTER_ANALYSIS_VERSION = 1
EVIDENCE_SOURCE_PRIORITY = {
    "ocr": 6,
    "vl": 5,
    "subtitle": 4,
    "asr": 4,
    "manual": 3,
    "page_context": 2,
    "comment": 1,
}
DEFAULT_CHAPTER_EVIDENCE_CHARS = 9000
MAX_CHAPTER_EVIDENCE_CHARS = 18000
DEFAULT_CHAPTER_OUTPUT_TOKENS = 900
MAX_CHAPTER_OUTPUT_TOKENS = 1800
DEFAULT_OVERVIEW_OUTPUT_TOKENS = 1000


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
    parser.add_argument("--chapter-concurrency", type=int, help="Concurrent chapter requests; default from profile or 1")
    parser.add_argument("--refresh", action="store_true", help="Regenerate completed chapter checkpoints")
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
        refresh=args.refresh,
        chapter_concurrency=args.chapter_concurrency or int(profile.get("multidoc_chapter_concurrency") or 1),
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
    refresh: bool = False,
    chapter_concurrency: int = 1,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    validate_run_dir(run_dir)
    output_dir = (output_dir or (run_dir / "docs_analysis")).expanduser().resolve()
    orin_dir = output_dir / "orin"
    output_dir.mkdir(parents=True, exist_ok=True)
    orin_dir.mkdir(parents=True, exist_ok=True)
    chapter_dir = orin_dir / CHAPTER_ANALYSIS_DIRNAME
    chapter_dir.mkdir(parents=True, exist_ok=True)

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
    chapter_packets = build_generation_chapter_packets(evidence)
    if not chapter_packets:
        raise ValueError("No chapter evidence packets available for document generation")

    generation_metrics = {
        "model_calls": 0,
        "checkpoint_hits": 0,
        "chapter_checkpoint_hits": 0,
        "overview_checkpoint_hit": False,
        "prompt_chars_total": 0,
        "prompt_chars_max": 0,
        "output_token_budget_total": 0,
        "planned_prompt_chars_total": 0,
        "planned_output_token_budget_total": 0,
    }
    chapter_concurrency = max(1, min(int(chapter_concurrency), len(chapter_packets)))
    chapter_results_by_id: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], Path, Path, str, int]] = []
    for packet in chapter_packets:
        checkpoint = chapter_dir / f"{packet['chapter_id']}.json"
        raw_path = chapter_dir / f"{packet['chapter_id']}.raw.md"
        prompt = build_chapter_generation_prompt(packet, language)
        output_budget = chapter_output_budget(packet)
        generation_metrics["planned_prompt_chars_total"] += len(prompt)
        generation_metrics["planned_output_token_budget_total"] += output_budget
        cached = read_json_if_exists(checkpoint) if not refresh else None
        if is_valid_chapter_result(cached, packet["chapter_id"]):
            result = repair_cached_chapter_result(cached, raw_path, packet)
            if result != cached:
                write_json(checkpoint, result)
            chapter_results_by_id[packet["chapter_id"]] = result
            generation_metrics["checkpoint_hits"] += 1
            generation_metrics["chapter_checkpoint_hits"] += 1
            continue
        pending.append((packet, checkpoint, raw_path, prompt, output_budget))

    if pending:
        def generate_chapter(item: tuple[dict[str, Any], Path, Path, str, int]) -> tuple[str, dict[str, Any], int, int]:
            packet, checkpoint, raw_path, prompt, output_budget = item
            response = client.generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                num_predict=output_budget,
            )
            raw_text = str(response.get("response") or "").strip()
            raw_path.write_text(raw_text + "\n", encoding="utf-8")
            result = normalize_chapter_result(raw_text, packet)
            write_json(checkpoint, result)
            return packet["chapter_id"], result, len(prompt), output_budget

        with ThreadPoolExecutor(max_workers=chapter_concurrency) as executor:
            futures = [executor.submit(generate_chapter, item) for item in pending]
            for future in as_completed(futures):
                chapter_id, result, prompt_chars, output_budget = future.result()
                chapter_results_by_id[chapter_id] = result
                generation_metrics["model_calls"] += 1
                generation_metrics["prompt_chars_total"] += prompt_chars
                generation_metrics["prompt_chars_max"] = max(generation_metrics["prompt_chars_max"], prompt_chars)
                generation_metrics["output_token_budget_total"] += output_budget

    chapter_results = [chapter_results_by_id[packet["chapter_id"]] for packet in chapter_packets]

    overview_path = orin_dir / "overview.json"
    overview_prompt = build_cross_chapter_overview_prompt(chapter_results, language)
    overview_budget = overview_output_budget(chapter_results)
    generation_metrics["planned_prompt_chars_total"] += len(overview_prompt)
    generation_metrics["planned_output_token_budget_total"] += overview_budget
    overview_cached = read_json_if_exists(overview_path) if not refresh else None
    if is_valid_overview(overview_cached):
        overview = repair_cached_overview(overview_cached, chapter_results)
        if overview != overview_cached:
            write_json(overview_path, overview)
        generation_metrics["checkpoint_hits"] += 1
        generation_metrics["overview_checkpoint_hit"] = True
    else:
        generation_metrics["model_calls"] += 1
        generation_metrics["prompt_chars_total"] += len(overview_prompt)
        generation_metrics["prompt_chars_max"] = max(generation_metrics["prompt_chars_max"], len(overview_prompt))
        generation_metrics["output_token_budget_total"] += overview_budget
        overview_response = client.generate(
            prompt=overview_prompt,
            model=model,
            temperature=temperature,
            num_predict=overview_budget,
        )
        overview = normalize_overview(str(overview_response.get("response") or ""), chapter_results)
        write_json(overview_path, overview)

    review = build_deterministic_review(chapter_results, overview)
    write_json(orin_dir / "round_04_review.json", review)

    rendered_docs = {
        "knowledge_notes": render_knowledge_notes(chapter_results, overview),
        "deep_report": render_deep_report(chapter_results, overview),
        "operation_manual_review": render_operation_manual_review(chapter_results, overview),
    }
    final_docs: dict[str, str] = {}
    for doc_type in doc_types:
        path = output_dir / DOC_FILENAMES[doc_type]
        path.write_text(rendered_docs[doc_type], encoding="utf-8")
        final_docs[doc_type] = str(path)
    chapter_output_dir = run_dir / "docs_analysis_chapters"
    chapter_output_dir.mkdir(parents=True, exist_ok=True)
    chapter_outputs = {
        "knowledge_notes_v2": chapter_output_dir / "knowledge_notes_v2.md",
        "deep_report_v2": chapter_output_dir / "deep_report_v2.md",
    }
    chapter_outputs["knowledge_notes_v2"].write_text(rendered_docs["knowledge_notes"], encoding="utf-8")
    chapter_outputs["deep_report_v2"].write_text(rendered_docs["deep_report"], encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "orin_dir": str(orin_dir),
        "language": language,
        "doc_types": doc_types,
        "llm_base_url": base_url,
        "text_model": model,
        "generation": {
            "version": CHAPTER_ANALYSIS_VERSION,
            "chapter_count": len(chapter_results),
            "chapter_checkpoints": str(chapter_dir),
            "overview": str(overview_path),
            "review": str(orin_dir / "round_04_review.json"),
            "resumable": True,
            "chapter_concurrency": chapter_concurrency,
            "metrics": generation_metrics,
        },
        "outputs": final_docs,
        "chapter_outputs": {name: str(path) for name, path in chapter_outputs.items()},
    }
    write_json(output_dir / "analysis.json", summary)
    return summary


def build_generation_chapter_packets(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    study_context = evidence.get("study_context") or {}
    packets = study_context.get("chapter_packets") or []
    if not packets:
        packets = [
            {
                "chapter_id": f"chapter_{index:02d}",
                "index": index,
                **chapter,
                "summary": "",
                "key_points": [],
                "evidence": [],
            }
            for index, chapter in enumerate(evidence.get("chapters") or [], start=1)
        ]

    result = []
    for index, chapter in enumerate(packets, start=1):
        chapter_id = str(chapter.get("chapter_id") or f"chapter_{index:02d}")
        selected_evidence = select_chapter_evidence(chapter.get("evidence") or [])
        result.append(
            {
                "chapter_id": chapter_id,
                "index": int(chapter.get("index") or index),
                "title": str(chapter.get("title") or f"章节 {index:02d}").strip(),
                "start": str(chapter.get("start") or "00:00:00"),
                "end": str(chapter.get("end") or ""),
                "summary": str(chapter.get("summary") or "").strip(),
                "key_points": [str(item).strip() for item in chapter.get("key_points") or [] if str(item).strip()][:10],
                "review_flags": [str(item).strip() for item in chapter.get("review_flags") or [] if str(item).strip()][:8],
                "evidence": selected_evidence,
            }
        )
    return result


def select_chapter_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source_type = str(item.get("source_type") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if not text or source_type == "comment":
            continue
        if source_type == "vl" and text.lower().startswith("vl analysis skipped"):
            continue
        normalized.append(
            {
                "id": str(item.get("id") or ""),
                "source_type": source_type or "unknown",
                "timestamp_label": str(item.get("timestamp_label") or ""),
                "timestamp_sec": float(item.get("timestamp_sec") or 0),
                "confidence": float(item.get("confidence") or 0),
                "text": text,
            }
        )

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for source_type in ("ocr", "vl", "subtitle", "asr", "manual", "page_context"):
        candidate = next((item for item in normalized if item["source_type"] == source_type), None)
        if candidate:
            selected.append(candidate)
            used_ids.add(candidate["id"])

    ranked = sorted(
        (item for item in normalized if item["id"] not in used_ids),
        key=lambda item: (
            EVIDENCE_SOURCE_PRIORITY.get(item["source_type"], 0),
            item["confidence"],
            -item["timestamp_sec"],
        ),
        reverse=True,
    )
    selected.extend(ranked)

    budget = DEFAULT_CHAPTER_EVIDENCE_CHARS
    used = 0
    packed = []
    for item in selected:
        source_limit = 1200 if item["source_type"] in {"manual", "page_context"} else 900
        text = trim(item["text"], source_limit)
        size = len(text)
        if packed and used + size > budget:
            continue
        packed.append({**item, "text": text})
        used += size
        if used >= budget:
            break
    return packed


def chapter_output_budget(packet: dict[str, Any]) -> int:
    evidence_chars = sum(len(str(item.get("text") or "")) for item in packet.get("evidence") or [])
    key_point_count = len(packet.get("key_points") or [])
    return min(
        MAX_CHAPTER_OUTPUT_TOKENS,
        max(DEFAULT_CHAPTER_OUTPUT_TOKENS, 700 + evidence_chars // 45 + key_point_count * 25),
    )


def overview_output_budget(chapter_results: list[dict[str, Any]]) -> int:
    return min(1600, max(DEFAULT_OVERVIEW_OUTPUT_TOKENS, 700 + len(chapter_results) * 60))


def build_chapter_generation_prompt(packet: dict[str, Any], language: str) -> str:
    evidence_lines = [
        (
            f"- [{item['id']}] {item['timestamp_label']} {item['source_type']} "
            f"(confidence={item['confidence']:.2f}): {item['text']}"
        )
        for item in packet.get("evidence") or []
    ]
    return f"""
/no_think
你是视频证据编辑。只分析一个章节，所有事实必须来自给定证据。

章节：{packet['index']}. {packet['title']}（{packet['start']} - {packet['end']}）
章节摘要：{packet['summary']}
已有要点：{json.dumps(packet['key_points'], ensure_ascii=False)}
复核标记：{json.dumps(packet['review_flags'], ensure_ascii=False)}

证据：
{chr(10).join(evidence_lines)}

请用 {language} 且只输出 JSON：
{{
  "chapter_summary": "本章完整但紧凑的总结",
  "key_facts": [{{"claim": "...", "evidence_ids": ["..."]}}],
  "analysis": ["解释、对比、适用条件或方法论"],
  "manual_review": ["手册需补充、修正或明确标注的不确定项"],
  "cautions": ["证据不足、ASR/OCR 可能错误或不应确定化的内容"],
  "citations": ["本章最关键的 evidence id"]
}}

规则：
- 覆盖本章主题，不引入其他章节结论。
- 可见文字与界面以 OCR/VL 为优先依据；口播事实以转写为依据。
- 评论或没有给出的信息不得成为结论。
- 引用只能使用上方 evidence id。
""".strip()


def normalize_chapter_result(raw_text: str, packet: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_json_object(raw_text)
    allowed_ids = {item["id"] for item in packet.get("evidence") or [] if item.get("id")}
    citations = [
        str(item)
        for item in (parsed.get("citations") or [])
        if str(item) in allowed_ids
    ] if isinstance(parsed, dict) else []
    if not citations:
        citations = [item["id"] for item in packet.get("evidence")[:4] if item.get("id")]
    key_facts = parsed.get("key_facts") if isinstance(parsed, dict) else []
    if not isinstance(key_facts, list):
        key_facts = []
    normalized_facts = []
    for fact in key_facts[:12]:
        if isinstance(fact, dict):
            evidence_ids = [str(item) for item in fact.get("evidence_ids") or [] if str(item) in allowed_ids]
            normalized_facts.append(
                {
                    "claim": str(fact.get("claim") or "").strip(),
                    "evidence_ids": evidence_ids or citations[:2],
                }
            )
        elif str(fact).strip():
            normalized_facts.append({"claim": str(fact).strip(), "evidence_ids": citations[:2]})

    def values(name: str) -> list[str]:
        source = parsed.get(name) if isinstance(parsed, dict) else []
        if isinstance(source, str):
            source = [source]
        return [str(item).strip() for item in source or [] if str(item).strip()][:12]

    summary = str(parsed.get("chapter_summary") or "").strip() if isinstance(parsed, dict) else ""
    if not summary:
        summary = extract_json_string_field(raw_text, "chapter_summary")
    if not summary:
        summary = packet.get("summary") or packet.get("title") or "本章未生成有效摘要。"
    if not normalized_facts:
        normalized_facts = [
            {"claim": point, "evidence_ids": citations[:2]}
            for point in packet.get("key_points") or []
            if point
        ][:8]
    return {
        "version": CHAPTER_ANALYSIS_VERSION,
        "chapter_id": packet["chapter_id"],
        "index": packet["index"],
        "title": packet["title"],
        "start": packet["start"],
        "end": packet["end"],
        "chapter_summary": summary,
        "key_facts": normalized_facts,
        "analysis": values("analysis"),
        "manual_review": values("manual_review"),
        "cautions": values("cautions"),
        "citations": citations,
        "evidence_count": len(packet.get("evidence") or []),
    }


def repair_cached_chapter_result(
    cached: dict[str, Any],
    raw_path: Path,
    packet: dict[str, Any],
) -> dict[str, Any]:
    summary = str(cached.get("chapter_summary") or "").lstrip()
    if not summary.startswith(("{", "[")):
        return cached
    raw_text = read_text_if_exists(raw_path)
    return normalize_chapter_result(raw_text, packet)


def extract_json_string_field(text: str, field: str) -> str:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*', text or "")
    if not match:
        return ""
    try:
        value, _ = json.JSONDecoder().raw_decode((text or "")[match.end() :].lstrip())
    except json.JSONDecodeError:
        return ""
    return str(value).strip() if isinstance(value, str) else ""


def is_valid_chapter_result(payload: Any, chapter_id: str) -> bool:
    return bool(
        isinstance(payload, dict)
        and payload.get("version") == CHAPTER_ANALYSIS_VERSION
        and payload.get("chapter_id") == chapter_id
        and str(payload.get("chapter_summary") or "").strip()
    )


def build_cross_chapter_overview_prompt(chapters: list[dict[str, Any]], language: str) -> str:
    compact = [
        {
            "chapter_id": chapter["chapter_id"],
            "title": chapter["title"],
            "time": f"{chapter['start']} - {chapter['end']}".strip(),
            "summary": chapter["chapter_summary"],
            "cautions": chapter["cautions"][:3],
        }
        for chapter in chapters
    ]
    return f"""
/no_think
你是视频文档总编辑。根据已完成的章节分析，用 {language} 输出 JSON：
{{
  "overview": "覆盖全片的紧凑总览",
  "cross_chapter_conclusions": ["跨章节结论"],
  "limitations": ["需要保留的不确定性或限制"]
}}

章节分析：
{json.dumps(compact, ensure_ascii=False)}

不得引入章节分析中没有支持的新事实。
""".strip()


def normalize_overview(raw_text: str, chapters: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = parse_json_object(raw_text)
    overview = str(parsed.get("overview") or "").strip() if isinstance(parsed, dict) else ""
    if not overview:
        overview = extract_json_string_field(raw_text, "overview")
    if not overview:
        overview = "\n".join(chapter["chapter_summary"] for chapter in chapters[:3])
    def values(name: str) -> list[str]:
        source = parsed.get(name) if isinstance(parsed, dict) else []
        if isinstance(source, str):
            source = [source]
        return [str(item).strip() for item in source or [] if str(item).strip()][:12]
    return {
        "version": CHAPTER_ANALYSIS_VERSION,
        "overview": overview,
        "cross_chapter_conclusions": values("cross_chapter_conclusions"),
        "limitations": values("limitations"),
    }


def repair_cached_overview(cached: dict[str, Any], chapters: list[dict[str, Any]]) -> dict[str, Any]:
    overview = str(cached.get("overview") or "").lstrip()
    if not overview.startswith(("{", "[")):
        return cached
    return normalize_overview(overview, chapters)


def is_valid_overview(payload: Any) -> bool:
    return bool(
        isinstance(payload, dict)
        and payload.get("version") == CHAPTER_ANALYSIS_VERSION
        and str(payload.get("overview") or "").strip()
    )


def build_deterministic_review(chapters: list[dict[str, Any]], overview: dict[str, Any]) -> dict[str, Any]:
    cautions = [
        {"chapter_id": chapter["chapter_id"], "title": chapter["title"], "items": chapter["cautions"]}
        for chapter in chapters
        if chapter["cautions"]
    ]
    return {
        "version": CHAPTER_ANALYSIS_VERSION,
        "status": "completed",
        "chapter_count": len(chapters),
        "chapters_with_cautions": len(cautions),
        "cautions": cautions,
        "overview_limitations": overview.get("limitations") or [],
    }


def render_knowledge_notes(chapters: list[dict[str, Any]], overview: dict[str, Any]) -> str:
    lines = ["# 知识笔记", "", overview["overview"], ""]
    for chapter in chapters:
        lines.extend([f"## {chapter['index']:02d}. {chapter['title']}", f"`{chapter['start']} - {chapter['end']}`", "", chapter["chapter_summary"], ""])
        if chapter["key_facts"]:
            lines.append("### 关键事实")
            lines.extend(
                f"- {fact['claim']}（证据：{', '.join(fact['evidence_ids'])}）"
                for fact in chapter["key_facts"] if fact["claim"]
            )
            lines.append("")
        if chapter["analysis"]:
            lines.append("### 理解与应用")
            lines.extend(f"- {item}" for item in chapter["analysis"])
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_deep_report(chapters: list[dict[str, Any]], overview: dict[str, Any]) -> str:
    lines = ["# 深度报告", "", "## 总览", overview["overview"], ""]
    if overview["cross_chapter_conclusions"]:
        lines.append("## 跨章节结论")
        lines.extend(f"- {item}" for item in overview["cross_chapter_conclusions"])
        lines.append("")
    for chapter in chapters:
        lines.extend([f"## {chapter['index']:02d}. {chapter['title']}", f"`{chapter['start']} - {chapter['end']}`", "", chapter["chapter_summary"], ""])
        if chapter["analysis"]:
            lines.extend(f"- {item}" for item in chapter["analysis"])
            lines.append("")
        if chapter["cautions"]:
            lines.append("### 限制与待复核")
            lines.extend(f"- {item}" for item in chapter["cautions"])
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_operation_manual_review(chapters: list[dict[str, Any]], overview: dict[str, Any]) -> str:
    lines = ["# 操作手册复核", "", "本复核按视频章节和已保留证据生成；未被证据支持的内容不应写成确定步骤。", ""]
    for chapter in chapters:
        items = chapter["manual_review"] or chapter["cautions"]
        if not items:
            continue
        lines.extend([f"## {chapter['index']:02d}. {chapter['title']}（{chapter['start']} - {chapter['end']}）"])
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    if overview["limitations"]:
        lines.append("## 全局限制")
        lines.extend(f"- {item}" for item in overview["limitations"])
    return "\n".join(lines).strip() + "\n"


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


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
