#!/usr/bin/env python3
"""Backward-compatible shim for tools/publish/md_to_mobile_pdf.py.

Both `python tools/md_to_mobile_pdf.py ...` and
`from tools.md_to_mobile_pdf import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.publish.md_to_mobile_pdf"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "publish" / "md_to_mobile_pdf.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
