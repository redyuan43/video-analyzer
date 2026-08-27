#!/usr/bin/env python3
"""Backward-compatible shim for tools/video_link/video_link_status_supervisor.py."""
import importlib
import sys
from pathlib import Path

_REAL = "tools.video_link.video_link_status_supervisor"
if __name__ == "__main__":
    from runpy import run_path

    sys.exit(
        run_path(
            str(Path(__file__).resolve().parent / "video_link" / "video_link_status_supervisor.py"),
            run_name="__main__",
        )
    )
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
