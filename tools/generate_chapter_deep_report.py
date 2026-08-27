#!/usr/bin/env python3
"""Backward-compatible shim for tools/publish/generate_chapter_deep_report.py.

Both `python tools/generate_chapter_deep_report.py ...` and
`from tools.generate_chapter_deep_report import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.publish.generate_chapter_deep_report"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "publish" / "generate_chapter_deep_report.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
