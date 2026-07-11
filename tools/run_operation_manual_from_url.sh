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

node_major_version() {
  local node_bin="$1"
  local node_version
  node_version="$("$node_bin" --version 2>/dev/null || true)"
  node_version="${node_version#v}"
  node_version="${node_version%%.*}"
  if [[ "$node_version" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$node_version"
  fi
}

ensure_supported_node() {
  local node_bin
  local node_major
  node_bin="$(command -v node || true)"
  node_major="$(node_major_version "$node_bin")"
  if [[ -n "$node_bin" && "$node_major" =~ ^[0-9]+$ && "$node_major" -ge 20 ]]; then
    return 0
  fi

  local candidate
  local candidate_major
  local selected=""
  for candidate in "$HOME"/.nvm/versions/node/*/bin/node; do
    [[ -x "$candidate" ]] || continue
    candidate_major="$(node_major_version "$candidate")"
    if [[ "$candidate_major" =~ ^[0-9]+$ && "$candidate_major" -ge 20 ]]; then
      selected="$candidate"
    fi
  done
  if [[ -n "$selected" ]]; then
    export PATH="$(dirname "$selected"):$PATH"
  fi
}

ensure_supported_node

source "$ROOT_DIR/tools/operation_manual_no_proxy_env.sh"

YTDLP_RUNTIME_LOCK="${VIDEO_ANALYZER_YTDLP_RUNTIME_LOCK:-$ROOT_DIR/tmp/video-link-status/ytdlp-runtime.lock}"
if command -v flock >/dev/null 2>&1; then
  mkdir -p "$(dirname "$YTDLP_RUNTIME_LOCK")"
  exec 9>"$YTDLP_RUNTIME_LOCK"
  flock -s 9
fi

DEEPSEEK_ENV="${VIDEO_ANALYZER_DEEPSEEK_ENV:-$HOME/.config/video-analyzer/deepseek.env}"
if [[ -f "$DEEPSEEK_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEEPSEEK_ENV"
  set +a
fi

ARGS=("$@")
has_ytdlp_proxy=0
has_youtube_url=0
for arg in "${ARGS[@]}"; do
  if [[ "$arg" == "--ytdlp-proxy" || "$arg" == --ytdlp-proxy=* ]]; then
    has_ytdlp_proxy=1
  fi
  if [[ "$arg" == *"youtube.com"* || "$arg" == *"youtu.be"* ]]; then
    has_youtube_url=1
  fi
done
if [[ "$has_ytdlp_proxy" -eq 0 && "$has_youtube_url" -eq 1 ]] && timeout 1 bash -c '</dev/tcp/127.0.0.1/10808' 2>/dev/null; then
  ARGS+=("--ytdlp-proxy" "http://127.0.0.1:10808")
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" tools/run_operation_manual_from_url.py --python "$PYTHON_BIN" "${ARGS[@]}"
