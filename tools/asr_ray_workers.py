#!/usr/bin/env python3
"""Backward-compatible shim for tools/asr_servers/asr_ray_workers.py.

Both `python tools/asr_ray_workers.py ...` and
`from tools.asr_ray_workers import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.asr_servers.asr_ray_workers"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "asr_servers" / "asr_ray_workers.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
