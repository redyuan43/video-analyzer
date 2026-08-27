#!/usr/bin/env python3
"""Backward-compatible shim for tools/pipelines/run_audio_transcription.py.

Both `python tools/run_audio_transcription.py ...` and
`from tools.run_audio_transcription import ...` keep working after the Phase 3b move.
"""

import importlib
import sys
from pathlib import Path

_REAL = "tools.pipelines.run_audio_transcription"

if __name__ == "__main__":
    from runpy import run_path

    sys.exit(run_path(str(Path(__file__).resolve().parent / "pipelines" / "run_audio_transcription.py"), run_name="__main__"))
else:
    sys.modules[__name__] = importlib.import_module(_REAL)
