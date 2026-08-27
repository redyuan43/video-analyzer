#!/usr/bin/env python3
"""Backward-compatible shim for tools/publish/prepare_video_doc_export.py.

Both `python tools/prepare_video_doc_export.py ...` and
`from tools.prepare_video_doc_export import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.publish.prepare_video_doc_export"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "publish" / "prepare_video_doc_export.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
