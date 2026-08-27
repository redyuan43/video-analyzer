#!/usr/bin/env python3
"""Backward-compatible shim for tools/ocr_servers/minicpm_p40_proxy.py.

Both `python tools/minicpm_p40_proxy.py ...` and
`from tools.minicpm_p40_proxy import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.ocr_servers.minicpm_p40_proxy"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "ocr_servers" / "minicpm_p40_proxy.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
