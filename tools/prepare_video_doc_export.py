#!/usr/bin/env python3
"""Prepare a Markdown document for stable PDF/long-PNG export."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


IMAGE_LINK_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")
SKIP_PREFIXES = ("http://", "https://", "data:", "mailto:", "#", "file://")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite local image paths for export from the run root")
    parser.add_argument("run_dir", help="Operation-manual run directory")
    parser.add_argument("input_md", help="Source Markdown file")
    parser.add_argument("output_md", help="Prepared Markdown file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    input_md = Path(args.input_md).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()

    text = input_md.read_text(encoding="utf-8")
    if input_md.name == "manual_evidence.md":
        text = simplify_manual_evidence_tables(text)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(rewrite_image_paths(text, input_md, output_md.parent, run_dir), encoding="utf-8")
    print(output_md)
    return 0


def simplify_manual_evidence_tables(text: str) -> str:
    """Create a PDF-friendly summary while preserving the source Markdown file."""
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if is_frame_evidence_header(lines[index]) and index + 1 < len(lines) and is_table_separator(lines[index + 1]):
            output.extend(render_frame_evidence_cards(lines[index], lines[index + 2 :]))
            output.extend(
                [
                    "## 完整证据",
                    "",
                    "完整逐帧 OCR 原文、视觉分析文本和证据明细保留在原始 `manual_evidence.md`、`analysis.json`、`orin/ocr_events.json`、`orin/frame_analyses.json` 中。PDF 版仅保留摘要索引，避免窄版页面渲染超长表格和原始 HTML 证据时卡住。",
                    "",
                ]
            )
            return "\n".join(output) + "\n"
        output.append(lines[index])
        index += 1
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def render_frame_evidence_cards(header_line: str, rows: list[str]) -> list[str]:
    headers = parse_table_row(header_line)
    rendered = ["### 逐帧证据摘要", ""]
    for row in rows:
        if not is_table_row(row):
            break
        cells = parse_table_row(row)
        if len(cells) != len(headers):
            continue
        item = dict(zip(headers, cells))
        title = " / ".join(part for part in (item.get("时间"), item.get("帧")) if part)
        rendered.extend(
            [
                f"#### {title or '证据帧'}",
                "",
                f"- OCR 状态：{item.get('OCR 状态', '')}",
                f"- OCR 摘要：{pdf_summary_text(item.get('OCR 摘要', ''))}",
                f"- 视觉摘要：{pdf_summary_text(item.get('视觉摘要', ''))}",
                "",
            ]
        )
    return rendered


def pdf_summary_text(value: str, limit: int = 180) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[_*]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text or "无"
    return text[:limit].rstrip() + "..."


def is_frame_evidence_header(line: str) -> bool:
    cells = parse_table_row(line)
    return cells == ["时间", "帧", "OCR 状态", "OCR 摘要", "视觉摘要"]


def is_table_row(line: str) -> bool:
    value = line.strip()
    return value.startswith("|") and value.endswith("|")


def is_table_separator(line: str) -> bool:
    cells = parse_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def parse_table_row(line: str) -> list[str]:
    value = line.strip()
    if not value.startswith("|"):
        return []
    value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return [cell.strip() for cell in value.split("|")]


def rewrite_image_paths(text: str, input_md: Path, output_parent: Path, run_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_target = match.group(2).strip()
        image_target, suffix = split_target(raw_target)
        if should_skip(image_target):
            return match.group(0)

        asset_path = resolve_asset(input_md.parent, image_target)
        if asset_path is None:
            return match.group(0)

        try:
            asset_path.relative_to(run_dir)
        except ValueError:
            return match.group(0)

        rewritten = os.path.relpath(asset_path, output_parent).replace(os.sep, "/")
        return f"{match.group(1)}{rewritten}{suffix}{match.group(3)}"

    return IMAGE_LINK_RE.sub(replace, text)


def split_target(raw_target: str) -> tuple[str, str]:
    if raw_target.startswith("<"):
        end = raw_target.find(">")
        if end != -1:
            return raw_target[1:end], raw_target[end + 1 :]

    for marker in (' "', " '"):
        index = raw_target.find(marker)
        if index != -1:
            return raw_target[:index], raw_target[index:]
    return raw_target, ""


def should_skip(target: str) -> bool:
    return target == "" or target.startswith(SKIP_PREFIXES)


def resolve_asset(base_dir: Path, target: str) -> Path | None:
    candidate = (base_dir / target).resolve()
    if candidate.exists():
        return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())
