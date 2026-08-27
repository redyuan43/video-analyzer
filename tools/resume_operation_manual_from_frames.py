#!/usr/bin/env python3
"""Backward-compatible shim for tools/pipelines/resume_operation_manual_from_frames.py.

Both `python tools/resume_operation_manual_from_frames.py ...` and
`from tools.resume_operation_manual_from_frames import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.pipelines.resume_operation_manual_from_frames"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "pipelines" / "resume_operation_manual_from_frames.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
