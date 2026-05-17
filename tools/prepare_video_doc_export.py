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
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(rewrite_image_paths(text, input_md, output_md.parent, run_dir), encoding="utf-8")
    print(output_md)
    return 0


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
