#!/usr/bin/env python3
"""Generate chapter-by-chapter long-form notes from an existing run."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config
from video_analyzer.multidoc import (
    build_chapter_transcript_digest,
    load_evidence,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detailed chapter notes from an operation-manual run")
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile", default="ivan_minicpm_v100")
    parser.add_argument("--output", help="Output directory; default RUN_DIR/docs_analysis_chapters")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--text-model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-chars-per-chapter", type=int, default=5200)
    parser.add_argument("--max-tokens", type=int, default=2600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve() if args.output else run_dir / "docs_analysis_chapters"
    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    profile = Config(args.config).get_runtime_profile(args.profile)
    analysis = read_json(run_dir / "analysis.json")
    evidence = load_evidence(run_dir, analysis)
    metadata = analysis.get("metadata") or {}
    model = args.text_model or profile.get("text_model") or metadata.get("text_model")
    base_url = args.llm_base_url or profile.get("llm_base_url") or metadata.get("llm_base_url")
    client = GenericOpenAIAPIClient("0", base_url)

    chapter_outputs = []
    started = time.perf_counter()
    for index, chapter in enumerate(evidence["chapters"], start=1):
        chapter_path = chapters_dir / f"chapter_{index:02d}.md"
        if chapter_path.exists() and chapter_path.stat().st_size > 200:
            text = chapter_path.read_text(encoding="utf-8")
            chapter_outputs.append((index, chapter, text, True))
            print(f"[skip] chapter {index:02d}: {chapter_path}", flush=True)
            continue
        digest = build_chapter_transcript_digest(
            [chapter],
            evidence["transcript"],
            max_chars_per_chapter=args.max_chars_per_chapter,
        )
        prompt = build_chapter_prompt(index, len(evidence["chapters"]), chapter, digest)
        print(f"[run] chapter {index:02d}/{len(evidence['chapters'])}: {chapter.get('title')}", flush=True)
        response = client.generate(
            prompt=prompt,
            model=model,
            temperature=args.temperature,
            num_predict=args.max_tokens,
        )
        text = clean_chapter_output(response.get("response") or "")
        chapter_path.write_text(text + "\n", encoding="utf-8")
        chapter_outputs.append((index, chapter, text, False))

    knowledge_notes = render_knowledge_notes(chapter_outputs)
    deep_report = render_deep_report(chapter_outputs)
    (output_dir / "knowledge_notes.md").write_text(knowledge_notes, encoding="utf-8")
    (output_dir / "deep_report.md").write_text(deep_report, encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "chapters": len(chapter_outputs),
        "llm_base_url": base_url,
        "text_model": model,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "outputs": {
            "knowledge_notes": str(output_dir / "knowledge_notes.md"),
            "deep_report": str(output_dir / "deep_report.md"),
        },
    }
    (output_dir / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_chapter_prompt(index: int, total: int, chapter: dict, digest: str) -> str:
    return f"""/no_think
你是长访谈内容分析员。请基于下面这一章的字幕证据，生成详细中文章节笔记。

章节：{index}/{total}
标题：{chapter.get('start')} - {chapter.get('end')} {chapter.get('title')}

要求：
- 只输出最终章节笔记，不要输出 thinking process、分析过程、草稿计划或英文自述。
- 不要只概括一句话；要解释这章的论点、推理链、例子和隐含判断。
- 保留关键时间戳。
- 如果本章涉及人物、组织、产品或技术判断，请拆成清晰小节。
- 输出结构固定为：
  1. 本章主旨
  2. 关键观点
  3. 重要例子与证据
  4. 可复用洞察
  5. 需复核或容易误读的点

章节字幕证据：
{digest}
""".strip()


def clean_chapter_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    markers = [
        r"\*\*1\.\s*本章主旨\*\*",
        r"^1[\.、]\s*本章主旨",
        r"^#*\s*本章主旨",
    ]
    best_start = None
    for pattern in markers:
        matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
        if matches:
            best_start = max(best_start or 0, matches[-1].start())
    if best_start is not None:
        text = text[best_start:].strip()
    text = re.sub(r"(?is)^here'?s a thinking process:.*?(?=\*\*1\.\s*本章主旨\*\*|^1[\.、]\s*本章主旨|^#*\s*本章主旨)", "", text).strip()
    text = re.sub(r"(?is)\n\s*(?:\d+\.\s+)?\*\*Final (?:Review|Check).*?$", "", text).strip()
    text = re.sub(r"(?is)\n\s*(?:\d+\.\s+)?Final (?:Review|Check).*?$", "", text).strip()
    text = re.sub(r"(?is)\n\s*Self-Correction/Refinement.*?$", "", text).strip()
    return text


def render_knowledge_notes(chapter_outputs: list[tuple[int, dict, str, bool]]) -> str:
    lines = [
        "# 逐章知识笔记",
        "",
        "本文档按原视频章节逐段展开，避免把长访谈压缩成少数概览章节。",
        "",
        "## 章节目录",
        "",
    ]
    for index, chapter, _text, _cached in chapter_outputs:
        lines.append(f"- {index:02d}. {chapter.get('start')} - {chapter.get('end')} {chapter.get('title')}")
    lines.append("")
    for index, chapter, text, _cached in chapter_outputs:
        lines.extend(
            [
                f"## {index:02d}. {chapter.get('start')} - {chapter.get('end')} {chapter.get('title')}",
                "",
                text.strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_deep_report(chapter_outputs: list[tuple[int, dict, str, bool]]) -> str:
    lines = [
        "# 深度分析报告",
        "",
        "这份报告采用逐章展开方式生成，重点保留长访谈中的中后段信息密度。",
        "",
        "## 总体结构",
        "",
    ]
    for index, chapter, _text, _cached in chapter_outputs:
        lines.append(f"- {index:02d}. {chapter.get('start')} - {chapter.get('end')} {chapter.get('title')}")
    lines.extend(["", "## 逐章分析", ""])
    for index, chapter, text, _cached in chapter_outputs:
        lines.extend(
            [
                f"### {index:02d}. {chapter.get('title')}",
                "",
                f"时间范围：{chapter.get('start')} - {chapter.get('end')}",
                "",
                text.strip(),
                "",
            ]
        )
    lines.extend(
        [
            "## 阅读提示",
            "",
            "- 本报告基于作者字幕和页面章节生成，评论只应作为社区反馈参考。",
            "- 每章均保留时间范围，方便回到原视频核查。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
