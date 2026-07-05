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

DEEPSEEK_ENV="${VIDEO_ANALYZER_DEEPSEEK_ENV:-$HOME/.config/video-analyzer/deepseek.env}"
if [[ -f "$DEEPSEEK_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEEPSEEK_ENV"
  set +a
fi

ARGS=("$@")
has_ytdlp_proxy=0
has_ytdlp_extractor_args=0
has_youtube_url=0
for arg in "${ARGS[@]}"; do
  if [[ "$arg" == "--ytdlp-proxy" || "$arg" == --ytdlp-proxy=* ]]; then
    has_ytdlp_proxy=1
  fi
  if [[ "$arg" == "--ytdlp-extractor-args" || "$arg" == --ytdlp-extractor-args=* ]]; then
    has_ytdlp_extractor_args=1
  fi
  if [[ "$arg" == *"youtube.com"* || "$arg" == *"youtu.be"* ]]; then
    has_youtube_url=1
  fi
done
if [[ "$has_ytdlp_proxy" -eq 0 && "$has_youtube_url" -eq 1 ]] && timeout 1 bash -c '</dev/tcp/127.0.0.1/10808' 2>/dev/null; then
  ARGS+=("--ytdlp-proxy" "http://127.0.0.1:10808")
fi
if [[ "$has_youtube_url" -eq 1 && "$has_ytdlp_extractor_args" -eq 0 ]]; then
  ARGS+=("--ytdlp-extractor-args" "youtube:player_client=mweb,web")
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" tools/run_operation_manual_from_url.py --python "$PYTHON_BIN" "${ARGS[@]}"
