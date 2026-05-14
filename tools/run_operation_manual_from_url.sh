#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
if [[ -d "$ROOT_DIR/.venv/bin" ]]; then
  export PATH="$ROOT_DIR/.venv/bin:$PATH"
fi

source "$ROOT_DIR/tools/operation_manual_no_proxy_env.sh"

ARGS=("$@")
has_ytdlp_proxy=0
for arg in "${ARGS[@]}"; do
  if [[ "$arg" == "--ytdlp-proxy" || "$arg" == --ytdlp-proxy=* ]]; then
    has_ytdlp_proxy=1
    break
  fi
done
if [[ "$has_ytdlp_proxy" -eq 0 ]] && timeout 1 bash -c '</dev/tcp/127.0.0.1/10808' 2>/dev/null; then
  ARGS+=("--ytdlp-proxy" "http://127.0.0.1:10808")
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" tools/run_operation_manual_from_url.py --python "$PYTHON_BIN" "${ARGS[@]}"
