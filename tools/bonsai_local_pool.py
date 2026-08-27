#!/usr/bin/env python3
"""Backward-compatible shim for tools/ops/bonsai_local_pool.py."""
import importlib
import sys
from pathlib import Path

_REAL = "tools.ops.bonsai_local_pool"
if __name__ == "__main__":
    from runpy import run_path

    sys.exit(
        run_path(
            str(Path(__file__).resolve().parent / "ops" / "bonsai_local_pool.py"),
            run_name="__main__",
        )
    )
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
