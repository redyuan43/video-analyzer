#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

DEEPSEEK_ENV="${VIDEO_ANALYZER_DEEPSEEK_ENV:-$HOME/.config/video-analyzer/deepseek.env}"
if [[ -f "$DEEPSEEK_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEEPSEEK_ENV"
  set +a
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" tools/run_multidoc_analysis.py "$@"
