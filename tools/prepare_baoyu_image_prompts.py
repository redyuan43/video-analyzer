#!/usr/bin/env python3
"""Prepare Baoyu-style image generation prompts from a video analysis run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DOCS = [
    (
        "01-image-cards-operation-manual.md",
        "baoyu-image-cards",
        "operation_manual.md",
        "生成一组适合微信图文/小红书的中文图文卡片，强调视频中可复用的操作路径、判断框架和关键结论。",
        "四张纵向卡片堆叠在一张海报里；每张卡片一个主题；手绘教育风；清晰中文标题；不要出现真实人物肖像。",
    ),
    (
        "02-infographic-knowledge-notes.md",
        "baoyu-infographic",
        "docs_analysis/knowledge_notes.md",
        "生成一张中文知识框架信息图，突出分析方法、概念关系和可迁移的判断模型。",
        "中心辐射或六模块布局；出版级信息图；中文标签简短；层级分明；避免大段正文。",
    ),
    (
        "03-infographic-deep-report.md",
        "baoyu-infographic",
        "docs_analysis/deep_report.md",
        "生成一张中文深度分析信息图，呈现核心论点、因果链、风险点和结论。",
        "高密度模块化布局；适合长文配图；包含时间线、风险矩阵或因果箭头；中文清晰可读。",
    ),
    (
        "04-infographic-manual-evidence.md",
        "baoyu-infographic",
        "manual_evidence.md",
        "生成一张证据地图/审稿仪表盘，展示视频证据来源、帧画面、OCR/VL/ASR支撑和复核路径。",
        "技术仪表盘风格；分区展示输入来源、证据强度、时间线覆盖、风险提醒；不要堆满小字。",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="operation-manual run directory")
    parser.add_argument("--output-dir", help="default: RUN_DIR/baoyu_images/prompts")
    parser.add_argument("--max-chars", type=int, default=3600)
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

    for filename, skill_name, source_rel, goal, visual_direction in DOCS:
        source_path = run_dir / source_rel
        excerpt = read_excerpt(source_path, args.max_chars)
        write_prompt(output_dir / filename, skill_name, source_rel, goal, visual_direction, excerpt)

    final_dir = run_dir / "baoyu_images" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"PROMPTS_DIR={output_dir}")
    print(f"FINAL_IMAGES_DIR={final_dir}")
    for path in sorted(output_dir.glob("*.md")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
