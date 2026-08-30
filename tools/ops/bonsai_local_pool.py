#!/usr/bin/env python3
"""Compatibility entrypoint for the Ray-coordinated Qwen3.8 Q4 pool."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ops.qwen38_ray_pool import *  # noqa: F401,F403,E402
from tools.ops.qwen38_ray_pool import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
