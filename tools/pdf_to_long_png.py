#!/usr/bin/env python3
"""Backward-compatible shim for tools/publish/pdf_to_long_png.py.

Both `python tools/pdf_to_long_png.py ...` and
`from tools.pdf_to_long_png import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.publish.pdf_to_long_png"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "publish" / "pdf_to_long_png.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
