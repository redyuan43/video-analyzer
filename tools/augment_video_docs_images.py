#!/usr/bin/env python3
"""Insert final visual assets into video-link Markdown deliverables."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


DOCS = [
    (
        "operation_manual.md",
        "02-infographic-knowledge-notes.png",
        "操作手册视觉摘要",
        r"^## 1\. 概览\s*$",
        ("01-image-cards-operation-manual.png",),
    ),
    ("docs_analysis_chapters/knowledge_notes_v2.md", "02-infographic-knowledge-notes.png", "逐章知识笔记视觉摘要", r"^## 01\. .+$", ()),
    ("docs_analysis_chapters/deep_report_v2.md", "03-infographic-deep-report.png", "逐章深度报告视觉摘要", r"^## 逐章分析\s*$", ()),
    ("manual_evidence.md", "04-infographic-manual-evidence.png", "证据索引视觉摘要", r"^## 0\.00s / Frame 0\s*$", ()),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add generated images and representative frames to final Markdown docs")
    parser.add_argument("run_dir", help="Operation-manual run directory")
    parser.add_argument("--max-frame-images", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    final_dir = run_dir / "baoyu_images" / "final"
    chapter_assets = load_chapter_assets(run_dir)
    frame_assets = load_frame_assets(run_dir, args.max_frame_images)
    changed = []
    for rel, final_name, title, target_heading, deprecated_image_names in DOCS:
        path = run_dir / rel
        if not path.exists():
            print(f"[skip missing] {rel}")
            continue
        final_image = final_dir / final_name
        text = path.read_text(encoding="utf-8")
        text = ensure_final_image(text, path, final_image, title, target_heading, deprecated_image_names)
        text = ensure_representative_images(text, path, chapter_assets, frame_assets)
        path.write_text(normalize_spacing(text), encoding="utf-8")
        changed.append(rel)
        print(f"[augmented] {rel}")

    print(json.dumps({"run_dir": str(run_dir), "documents": changed}, ensure_ascii=False, indent=2))
    return 0


def load_chapter_assets(run_dir: Path) -> list[Path]:
    manifest = run_dir / "docs_analysis_chapters" / "chapter_assets_manifest.json"
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assets = []
    for item in payload.get("assets") or []:
        path = Path(str(item.get("path") or ""))
        if path.exists():
            assets.append(path.resolve())
    return assets


def load_frame_assets(run_dir: Path, limit: int) -> list[Path]:
    frames = sorted((run_dir / "manual_assets").glob("frame_*.jpg"))
    if len(frames) <= limit:
        return frames
    if limit <= 1:
        return frames[:1]
    step = (len(frames) - 1) / (limit - 1)
    selected = []
    for index in range(limit):
        selected.append(frames[round(index * step)])
    return selected


def ensure_final_image(
    text: str,
    md_path: Path,
    image_path: Path,
    title: str,
    target_heading: str,
    deprecated_image_names: tuple[str, ...] = (),
) -> str:
    text = remove_deprecated_image_refs(text, deprecated_image_names)
    if not image_path.exists():
        print(f"[warn] final image missing: {image_path}")
        return text
    rel = markdown_relpath(md_path, image_path)
    text = remove_final_image_block(text, title)
    if rel in text:
        return text
    block = f"![{title}]({rel})\n\n"
    return insert_after_heading(text, target_heading, block)


def remove_final_image_block(text: str, title: str) -> str:
    block_re = re.compile(
        rf"(?:^|\n)## {re.escape(title)}\s*\n+\s*!\[{re.escape(title)}\]\([^)]+\)\s*\n+",
        flags=re.MULTILINE,
    )
    text = block_re.sub("\n", text)
    image_re = re.compile(
        rf"(?:^|\n)\s*!\[{re.escape(title)}\]\([^)]+\)\s*\n*",
        flags=re.MULTILINE,
    )
    text = image_re.sub("\n", text)
    return text


def remove_deprecated_image_refs(text: str, deprecated_image_names: tuple[str, ...]) -> str:
    for name in deprecated_image_names:
        deprecated_re = re.compile(
            rf"(?:^|\n)\s*!\[[^\]]*\]\([^)]*{re.escape(name)}\)\s*\n*",
            flags=re.MULTILINE,
        )
        text = deprecated_re.sub("\n", text)
    return text


def insert_after_heading(text: str, heading_pattern: str, block: str) -> str:
    match = re.search(heading_pattern, text, flags=re.MULTILINE)
    if not match:
        return insert_after_first_heading(text, block)
    insert_at = match.end()
    return text[:insert_at].rstrip() + "\n\n" + block + text[insert_at:].lstrip()


def ensure_representative_images(
    text: str,
    md_path: Path,
    chapter_assets: list[Path],
    frame_assets: list[Path],
) -> str:
    if has_local_image(text):
        return text
    candidates = chapter_assets or frame_assets
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return text
    images = [f"![代表帧 {index:02d}]({markdown_relpath(md_path, path)})" for index, path in enumerate(existing, start=1)]
    rows = ["| 图 1 | 图 2 | 图 3 |", "|---|---|---|"]
    for offset in range(0, len(images), 3):
        row = images[offset : offset + 3] + [""] * (3 - len(images[offset : offset + 3]))
        rows.append("| " + " | ".join(row) + " |")
    block = "## 视频代表帧\n\n" + "\n".join(rows) + "\n\n"
    return insert_after_visual_summary(text, block)


def has_local_image(text: str) -> bool:
    return bool(re.search(r"!\[[^\]]*\]\((?:\.\./)*?(?:manual_assets|chapter_assets|docs_analysis_chapters/chapter_assets)/", text))


def insert_after_first_heading(text: str, block: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("# "):
            return "\n".join(lines[: index + 1] + ["", block.rstrip(), ""] + lines[index + 1 :])
    return block + text


def insert_after_visual_summary(text: str, block: str) -> str:
    marker = re.search(r"^## .*视觉摘要\s*$", text, flags=re.MULTILINE)
    if not marker:
        return insert_after_first_heading(text, block)
    next_heading = re.search(r"^## (?!.*视觉摘要).+$", text[marker.end() :], flags=re.MULTILINE)
    if not next_heading:
        return text.rstrip() + "\n\n" + block
    insert_at = marker.end() + next_heading.start()
    return text[:insert_at].rstrip() + "\n\n" + block + text[insert_at:].lstrip()


def markdown_relpath(md_path: Path, asset_path: Path) -> str:
    return os.path.relpath(asset_path, md_path.parent).replace(os.sep, "/")


def normalize_spacing(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
