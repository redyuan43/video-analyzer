#!/usr/bin/env python3
"""Backward-compatible shim for tools/pipelines/run_skill_project_worker.py.

Both `python tools/run_skill_project_worker.py ...` and
`from tools.run_skill_project_worker import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.pipelines.run_skill_project_worker"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "pipelines" / "run_skill_project_worker.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
