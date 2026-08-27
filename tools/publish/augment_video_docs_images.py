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
        "",
    ),
    (
        "docs_analysis_chapters/knowledge_notes_v2.md",
        "02-infographic-knowledge-notes.png",
        "逐章知识笔记视觉摘要",
        r"^# 知识笔记\s*$",
        (),
        "02-knowledge-notes-chapter-",
    ),
    (
        "docs_analysis_chapters/deep_report_v2.md",
        "03-infographic-deep-report.png",
        "逐章深度报告视觉摘要",
        r"^## 总览\s*$",
        (),
        "03-deep-report-chapter-",
    ),
]

CLEANUP_IMAGE_REFS = [
    ("manual_evidence.md", ("04-infographic-manual-evidence.png",)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add generated images and representative frames to final Markdown docs")
    parser.add_argument("run_dir", help="Operation-manual run directory")
    parser.add_argument("--max-frame-images", type=int, default=6)
    parser.add_argument("--skip-final-images", action="store_true", help="remove generated final-image blocks and keep real representative frames")
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
    for rel, final_name, title, target_heading, deprecated_image_names, chapter_image_prefix in DOCS:
        path = run_dir / rel
        if not path.exists():
            print(f"[skip missing] {rel}")
            continue
        final_image = final_dir / final_name
        text = path.read_text(encoding="utf-8")
        if args.skip_final_images:
            text = remove_final_image_block(text, title)
            text = remove_deprecated_image_refs(text, deprecated_image_names + (final_name,))
        else:
            text = ensure_final_image(text, path, final_image, title, target_heading, deprecated_image_names)
        if chapter_image_prefix:
            text = ensure_chapter_images(
                text,
                path,
                chapter_assets,
                frame_assets,
                {} if args.skip_final_images else load_generated_chapter_assets(final_dir, chapter_image_prefix),
                chapter_image_prefix,
            )
        else:
            text = ensure_representative_images(text, path, list(chapter_assets.values()), frame_assets)
        path.write_text(normalize_spacing(text), encoding="utf-8")
        changed.append(rel)
        print(f"[augmented] {rel}")
    for rel, deprecated_image_names in CLEANUP_IMAGE_REFS:
        path = run_dir / rel
        if not path.exists():
            continue
        text = remove_deprecated_image_refs(path.read_text(encoding="utf-8"), deprecated_image_names)
        path.write_text(normalize_spacing(text), encoding="utf-8")
        changed.append(rel)
        print(f"[cleaned] {rel}")

    print(json.dumps({"run_dir": str(run_dir), "documents": changed}, ensure_ascii=False, indent=2))
    return 0


def load_chapter_assets(run_dir: Path) -> dict[int, Path]:
    manifest = run_dir / "docs_analysis_chapters" / "chapter_assets_manifest.json"
    if not manifest.exists():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assets = {}
    for item in payload.get("assets") or []:
        path = Path(str(item.get("path") or ""))
        chapter_index = int(item.get("chapter_index") or 0)
        if chapter_index > 0 and path.exists():
            assets[chapter_index] = path.resolve()
    return assets


def load_generated_chapter_assets(final_dir: Path, prefix: str) -> dict[int, Path]:
    assets = {}
    pattern = re.compile(rf"^{re.escape(prefix)}(?P<index>\d{{2}})\.png$")
    for path in final_dir.glob(f"{prefix}*.png"):
        match = pattern.match(path.name)
        if match and path.stat().st_size > 0:
            assets[int(match.group("index"))] = path.resolve()
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


def ensure_chapter_images(
    text: str,
    md_path: Path,
    chapter_assets: dict[int, Path],
    frame_assets: list[Path],
    generated_assets: dict[int, Path],
    generated_prefix: str,
) -> str:
    text = remove_representative_frame_section(text)
    text = remove_chapter_image_refs(text, generated_prefix)
    headings = list(re.finditer(r"^##\s+(?P<index>\d{2})\.\s+(?P<title>.+?)\s*$", text, flags=re.MULTILINE))
    if not headings:
        return ensure_representative_images(text, md_path, list(chapter_assets.values()), frame_assets)

    fallback_assets = spread_assets(frame_assets, len(headings))
    insertions = []
    for position, match in enumerate(headings):
        chapter_index = int(match.group("index"))
        title = match.group("title").strip()
        images = []
        generated = generated_assets.get(chapter_index)
        if generated:
            images.append(
                f"![第 {chapter_index:02d} 章概念图：{title}]({markdown_relpath(md_path, generated)})"
            )
        representative = chapter_assets.get(chapter_index)
        if representative is None and position < len(fallback_assets):
            representative = fallback_assets[position]
        if representative and representative.exists():
            images.append(
                f"![第 {chapter_index:02d} 章视频画面：{title}]({markdown_relpath(md_path, representative)})"
            )
        if images:
            insertions.append((match.end(), "\n\n".join(images)))

    for insert_at, block in reversed(insertions):
        text = text[:insert_at].rstrip() + "\n\n" + block + "\n\n" + text[insert_at:].lstrip()
    return text


def remove_representative_frame_section(text: str) -> str:
    return re.sub(
        r"(?:^|\n)## 视频代表帧\s*\n.*?(?=\n## |\Z)",
        "\n",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )


def remove_chapter_image_refs(text: str, generated_prefix: str) -> str:
    patterns = (
        r"(?:^|\n)\s*!\[第 \d{2} 章(?:概念图|视频画面)[^\]]*\]\([^)]+\)\s*\n*",
        rf"(?:^|\n)\s*!\[[^\]]*\]\([^)]*{re.escape(generated_prefix)}\d{{2}}\.png\)\s*\n*",
        r"(?:^|\n)\s*!\[[^\]]*\]\([^)]*chapter_assets/chapter_\d{2}\.jpg\)\s*\n*",
    )
    for pattern in patterns:
        text = re.sub(pattern, "\n", text, flags=re.MULTILINE)
    return text


def spread_assets(assets: list[Path], count: int) -> list[Path]:
    if not assets or count <= 0:
        return []
    if len(assets) <= count:
        return assets
    if count == 1:
        return [assets[len(assets) // 2]]
    step = (len(assets) - 1) / (count - 1)
    return [assets[round(index * step)] for index in range(count)]


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
