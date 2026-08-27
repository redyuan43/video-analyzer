#!/usr/bin/env python3
"""Backward-compatible shim for tools/pipelines/generate_audio_narration.py.

Both `python tools/generate_audio_narration.py ...` and
`from tools.generate_audio_narration import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.pipelines.generate_audio_narration"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "pipelines" / "generate_audio_narration.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
