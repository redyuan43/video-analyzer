#!/usr/bin/env python3
"""Generate the Video Analyzer app icons from code.

Keeping the icons generated rather than committed as opaque binaries means the
mark can be tweaked in one place and re-rendered for every target size.

Outputs:
  video-analyzer-ui/video_analyzer_ui/static/icons/apple-touch-icon.png  (180px)
  video-analyzer-ui/video_analyzer_ui/static/icons/icon-192.png
  video-analyzer-ui/video_analyzer_ui/static/icons/icon-512.png
  ios/VideoAnalyzer/Assets.xcassets/AppIcon.appiconset/AppIcon-1024.png

Usage:
  .venv/bin/python tools/publish/generate_app_icons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ICON_DIR = REPO_ROOT / 'video-analyzer-ui' / 'video_analyzer_ui' / 'static' / 'icons'
IOS_ICON_DIR = (
    REPO_ROOT / 'ios' / 'VideoAnalyzer' / 'Assets.xcassets' / 'AppIcon.appiconset'
)

# Render at 4x the largest target, then downsample, so the edges stay clean
# without hand-written antialiasing.
SUPERSAMPLE = 4
BASE = 1024

BG_TOP = (15, 23, 42)
BG_BOTTOM = (30, 58, 95)
ACCENT = (73, 174, 255)
WHITE = (255, 255, 255)

# Relative bar heights for the "analysis" waveform under the play mark.
BAR_HEIGHTS = (0.32, 0.62, 0.44, 0.86, 0.52, 0.70, 0.36)


def render(size: int) -> Image.Image:
    canvas = size * SUPERSAMPLE
    image = Image.new('RGB', (canvas, canvas), BG_TOP)
    draw = ImageDraw.Draw(image)

    # Vertical gradient background. No alpha: iOS app icons must be opaque.
    for y in range(canvas):
        blend = y / max(canvas - 1, 1)
        draw.line(
            [(0, y), (canvas, y)],
            fill=tuple(
                round(top + (bottom - top) * blend)
                for top, bottom in zip(BG_TOP, BG_BOTTOM)
            ),
        )

    # Play triangle, optically centred (a centroid-centred triangle reads as
    # sitting too far left, so nudge it right).
    tri_half = canvas * 0.17
    cx = canvas * 0.5 + tri_half * 0.18
    cy = canvas * 0.40
    draw.polygon(
        [
            (cx - tri_half, cy - tri_half * 1.12),
            (cx - tri_half, cy + tri_half * 1.12),
            (cx + tri_half * 1.05, cy),
        ],
        fill=WHITE,
    )

    # Waveform bars.
    span = canvas * 0.52
    left = (canvas - span) / 2
    slot = span / len(BAR_HEIGHTS)
    bar_w = slot * 0.46
    baseline = canvas * 0.79
    max_h = canvas * 0.20

    for index, ratio in enumerate(BAR_HEIGHTS):
        x0 = left + slot * index + (slot - bar_w) / 2
        height = max_h * ratio
        draw.rounded_rectangle(
            [x0, baseline - height, x0 + bar_w, baseline],
            radius=bar_w / 2,
            fill=ACCENT,
        )

    return image.resize((size, size), Image.LANCZOS)


def main() -> int:
    WEB_ICON_DIR.mkdir(parents=True, exist_ok=True)
    IOS_ICON_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        (WEB_ICON_DIR / 'apple-touch-icon.png', 180),
        (WEB_ICON_DIR / 'icon-192.png', 192),
        (WEB_ICON_DIR / 'icon-512.png', 512),
        (IOS_ICON_DIR / 'AppIcon-1024.png', 1024),
    ]

    for path, size in targets:
        render(size).save(path, 'PNG')
        print(f'wrote {path.relative_to(REPO_ROOT)} ({size}x{size})')

    return 0


if __name__ == '__main__':
    sys.exit(main())
