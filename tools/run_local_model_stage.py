#!/usr/bin/env python3
"""Backward-compatible shim for tools/ops/run_local_model_stage.py."""
import importlib
import sys
from pathlib import Path

_REAL = "tools.ops.run_local_model_stage"
if __name__ == "__main__":
    from runpy import run_path

    sys.exit(
        run_path(
            str(Path(__file__).resolve().parent / "ops" / "run_local_model_stage.py"),
            run_name="__main__",
        )
    )
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
