#!/usr/bin/env python3
"""Backward-compatible shim for tools/pipelines/regenerate_operation_manual.py.

Both `python tools/regenerate_operation_manual.py ...` and
`from tools.regenerate_operation_manual import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.pipelines.regenerate_operation_manual"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "pipelines" / "regenerate_operation_manual.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
