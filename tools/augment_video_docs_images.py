#!/usr/bin/env python3
"""Backward-compatible shim for tools/publish/augment_video_docs_images.py.

Both `python tools/augment_video_docs_images.py ...` and
`from tools.augment_video_docs_images import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.publish.augment_video_docs_images"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "publish" / "augment_video_docs_images.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
