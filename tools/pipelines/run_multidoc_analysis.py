#!/usr/bin/env python3
"""Thin CLI wrapper for multi-round video document analysis."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analyzer.multidoc import *  # re-exported for legacy tests/imports
from video_analyzer.multidoc import main


if __name__ == "__main__":
    raise SystemExit(main())
