#!/usr/bin/env python3
"""Convert a PDF into one vertically stitched, mobile-friendly PNG."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF pages to a tightly stitched PNG")
    parser.add_argument("input_pdf", help="Input PDF path")
    parser.add_argument("output_png", help="Output long PNG path")
    parser.add_argument("--dpi", type=int, default=180, help="Rasterization DPI")
    parser.add_argument("--padding", type=int, default=24, help="Vertical padding kept around page content")
    parser.add_argument("--threshold", type=int, default=248, help="Background threshold for trimming")
    return parser.parse_args()


def page_vertical_bounds(image: Image.Image, threshold: int, padding: int) -> tuple[int, int]:
    rgb = image.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg).convert("L")
    mask = diff.point(lambda value: 255 if value > 255 - threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return 0, rgb.height
    top = max(0, bbox[1] - padding)
    bottom = min(rgb.height, bbox[3] + padding)
    return top, max(top + 1, bottom)


def rasterize_pdf(input_pdf: Path, work_dir: Path, dpi: int) -> list[Path]:
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm is required to generate long PNG exports")
    prefix = work_dir / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(input_pdf), str(prefix)],
        check=True,
    )
    return sorted(work_dir.glob("page-*.png"))


def stitch_pages(page_paths: list[Path], output_png: Path, threshold: int, padding: int) -> None:
    if not page_paths:
        raise RuntimeError("No rasterized PDF pages were produced")

    cropped_pages: list[Image.Image] = []
    target_width = 0
    for page_path in page_paths:
        image = Image.open(page_path).convert("RGB")
        top, bottom = page_vertical_bounds(image, threshold=threshold, padding=padding)
        cropped = image.crop((0, top, image.width, bottom))
        cropped_pages.append(cropped)
        target_width = max(target_width, cropped.width)

    total_height = sum(page.height for page in cropped_pages)
    output = Image.new("RGB", (target_width, total_height), (255, 255, 255))
    y = 0
    for page in cropped_pages:
        x = (target_width - page.width) // 2
        output.paste(page, (x, y))
        y += page.height

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_png, "PNG", compress_level=3)


def main() -> int:
    args = parse_args()
    input_pdf = Path(args.input_pdf).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()
    if not input_pdf.is_file():
        raise SystemExit(f"PDF file not found: {input_pdf}")
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")
    if args.padding < 0:
        raise SystemExit("--padding must be non-negative")

    Image.MAX_IMAGE_PIXELS = None
    with tempfile.TemporaryDirectory(prefix="video-doc-long-png-") as tmp:
        pages = rasterize_pdf(input_pdf, Path(tmp), args.dpi)
        stitch_pages(pages, output_png, threshold=args.threshold, padding=args.padding)
    if not output_png.is_file() or output_png.stat().st_size == 0:
        raise SystemExit(f"PNG output not written: {output_png}")
    print(output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
