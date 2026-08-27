#!/usr/bin/env python3
"""Prepare Baoyu-style image generation prompts from a video analysis run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DOCS = [
    (
        "02-infographic-knowledge-notes.md",
        "baoyu-infographic",
        "docs_analysis_chapters/knowledge_notes_v2.md",
        "02-knowledge-notes-chapter-",
        "生成一张中文知识框架信息图，突出分析方法、概念关系和可迁移的判断模型。",
        "中心辐射或六模块布局；出版级信息图；中文标签简短；层级分明；避免大段正文。",
    ),
    (
        "03-infographic-deep-report.md",
        "baoyu-infographic",
        "docs_analysis_chapters/deep_report_v2.md",
        "03-deep-report-chapter-",
        "生成一张中文深度分析信息图，呈现核心论点、因果链、风险点和结论。",
        "高密度模块化布局；适合长文配图；包含时间线、风险矩阵或因果箭头；中文清晰可读。",
    ),
]

DEPRECATED_PROMPTS = ("01-image-cards-operation-manual.md", "04-infographic-manual-evidence.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="operation-manual run directory")
    parser.add_argument("--output-dir", help="default: RUN_DIR/baoyu_images/prompts")
    parser.add_argument("--max-chars", type=int, default=3600)
    parser.add_argument("--max-chapter-prompts-per-doc", type=int, default=4)
    return parser.parse_args()


def read_excerpt(path: Path, max_chars: int) -> str:
    if not path.exists():
        return "[missing]"
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].rstrip()
    tail = text[-max_chars // 2 :].lstrip()
    return f"{head}\n\n[...中间内容已压缩，生成图片时保留主题和结构...]\n\n{tail}"


def write_prompt(
    output_path: Path,
    skill_name: str,
    source_rel: str,
    goal: str,
    visual_direction: str,
    excerpt: str,
) -> None:
    output_path.write_text(
        f"""# {skill_name} prompt

Source Markdown: `{source_rel}`

请使用 `{skill_name}` 的思路生成图片。

目标：
{goal}

视觉方向：
{visual_direction}

硬性要求：
- 输出中文图片。
- 画面必须适合直接分享，不要像软件说明书截图。
- 只保留短标题、短标签和关键信息，不要把 Markdown 全文塞进图片。
- 如果事实细节不确定，用概念化表达，不要编造视频外的新事实。
- 使用内容的结构和结论，不要出现链接、文件路径、模型名或内部流水线术语。

内容摘录：

```markdown
{excerpt}
```
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "baoyu_images" / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in DEPRECATED_PROMPTS:
        (output_dir / filename).unlink(missing_ok=True)

    for pattern in ("02-knowledge-notes-chapter-*.md", "03-deep-report-chapter-*.md"):
        for path in output_dir.glob(pattern):
            path.unlink()

    image_plan = []
    for filename, skill_name, source_rel, chapter_prefix, goal, visual_direction in DOCS:
        source_path = run_dir / source_rel
        excerpt = read_excerpt(source_path, args.max_chars)
        write_prompt(output_dir / filename, skill_name, source_rel, goal, visual_direction, excerpt)
        image_plan.append(
            {
                "prompt": filename,
                "image": f"{Path(filename).stem}.png",
                "document": source_rel,
                "kind": "overview",
            }
        )
        chapters = parse_chapter_sections(source_path)
        for chapter in select_evenly(chapters, max(0, args.max_chapter_prompts_per_doc)):
            chapter_filename = f"{chapter_prefix}{chapter['index']:02d}.md"
            chapter_goal = (
                f"为“{chapter['title']}”生成一张章节插图，突出本章最重要的概念、关系、流程或判断。"
            )
            chapter_direction = (
                "单章主题插图；一眼能看懂；使用流程、对比、结构示意或场景化信息图；"
                "少量中文短标签；不要复刻整页文字。"
            )
            write_prompt(
                output_dir / chapter_filename,
                skill_name,
                source_rel,
                chapter_goal,
                chapter_direction,
                chapter["text"][: args.max_chars],
            )
            image_plan.append(
                {
                    "prompt": chapter_filename,
                    "image": f"{Path(chapter_filename).stem}.png",
                    "document": source_rel,
                    "kind": "chapter",
                    "chapter_index": chapter["index"],
                    "chapter_title": chapter["title"],
                }
            )

    (output_dir / "image_plan.json").write_text(
        json.dumps({"images": image_plan}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    final_dir = run_dir / "baoyu_images" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"PROMPTS_DIR={output_dir}")
    print(f"FINAL_IMAGES_DIR={final_dir}")
    for path in sorted(output_dir.glob("*.md")):
        print(path)
    return 0


def parse_chapter_sections(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(re.finditer(r"^##\s+(?P<index>\d{2})\.\s+(?P<title>.+?)\s*$", text, flags=re.MULTILINE))
    chapters = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        chapters.append(
            {
                "index": int(match.group("index")),
                "title": match.group("title").strip(),
                "text": text[match.start() : end].strip(),
            }
        )
    return chapters


def select_evenly(items: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (limit - 1)
    selected = []
    seen = set()
    for index in range(limit):
        item = items[round(index * step)]
        if item["index"] not in seen:
            selected.append(item)
            seen.add(item["index"])
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
