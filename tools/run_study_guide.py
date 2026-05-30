#!/usr/bin/env python3
"""Thin CLI wrapper for study-guide artifact generation."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analyzer.study_guide import main


if __name__ == "__main__":
    raise SystemExit(main())
