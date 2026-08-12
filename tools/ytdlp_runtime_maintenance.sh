#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${VIDEO_ANALYZER_YTDLP_PYTHON:-$ROOT_DIR/.venv/bin/python}"
PIP_BIN="$(dirname "$PYTHON_BIN")/pip"
YTDLP_BIN="$(dirname "$PYTHON_BIN")/yt-dlp"
LOCK_FILE="${VIDEO_ANALYZER_YTDLP_RUNTIME_LOCK:-$ROOT_DIR/tmp/video-link-status/ytdlp-runtime.lock}"
STATE_FILE="${VIDEO_ANALYZER_YTDLP_STATE_FILE:-$ROOT_DIR/tmp/video-link-status/ytdlp-runtime-maintenance.json}"

usage() {
  echo "Usage: $0 check|update" >&2
}

write_state() {
  local action="$1"
  local status="$2"
  local detail="$3"
  "$PYTHON_BIN" - "$STATE_FILE" "$action" "$status" "$detail" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "action": sys.argv[2],
    "status": sys.argv[3],
    "detail": sys.argv[4],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
tmp_path = path.with_name(f".{path.name}.tmp")
tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp_path.replace(path)
PY
}

require_runtime() {
  if [[ ! -x "$PYTHON_BIN" || ! -x "$PIP_BIN" ]]; then
    echo "Project virtual environment is unavailable: $PYTHON_BIN" >&2
    return 1
  fi
}

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

find_supported_node() {
  local node_bin
  local node_major
  node_bin="$(command -v node || true)"
  node_major="$(node_major_version "$node_bin")"
  if [[ -n "$node_bin" && "$node_major" =~ ^[0-9]+$ && "$node_major" -ge 20 ]]; then
    printf '%s\n' "$node_bin"
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
    printf '%s\n' "$selected"
    return 0
  fi
  return 1
}

check_runtime() {
  require_runtime
  "$PYTHON_BIN" - <<'PY'
from importlib.metadata import version

print(f"yt-dlp={version('yt-dlp')}")
print(f"yt-dlp-ejs={version('yt-dlp-ejs')}")
PY
  if [[ ! -x "$YTDLP_BIN" ]]; then
    echo "yt-dlp console script is missing from the project virtual environment: $YTDLP_BIN" >&2
    return 1
  fi
  "$YTDLP_BIN" --version
  local node_bin
  node_bin="$(find_supported_node || true)"
  if [[ -z "$node_bin" ]]; then
    echo "Node.js 20 or newer is required for YouTube JS challenges" >&2
    return 1
  fi
  export PATH="$(dirname "$node_bin"):$PATH"
  local node_version
  node_version="$("$node_bin" --version)"
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required for media merging" >&2
    return 1
  fi
  local ffmpeg_version
  ffmpeg_version="$(ffmpeg -version 2>&1)"
  echo "node=$node_version"
  echo "ffmpeg=${ffmpeg_version%%$'\n'*}"
}

package_version() {
  "$PYTHON_BIN" - "$1" <<'PY'
from importlib.metadata import version
import sys

print(version(sys.argv[1]))
PY
}

update_runtime() {
  require_runtime
  local previous_ytdlp
  local previous_ejs
  previous_ytdlp="$(package_version "yt-dlp")"
  previous_ejs="$(package_version "yt-dlp-ejs")"
  if ! "$PIP_BIN" install --upgrade "yt-dlp[default]"; then
    write_state "update" "failed" "yt-dlp upgrade command failed"
    return 1
  fi
  if check_runtime; then
    write_state "update" "succeeded" "yt-dlp upgraded from $previous_ytdlp; yt-dlp-ejs upgraded from $previous_ejs"
    return 0
  fi
  echo "Runtime verification failed; restoring yt-dlp=$previous_ytdlp and yt-dlp-ejs=$previous_ejs" >&2
  "$PIP_BIN" install "yt-dlp==$previous_ytdlp" "yt-dlp-ejs==$previous_ejs"
  check_runtime
  write_state "update" "rolled_back" "post-upgrade verification failed; restored yt-dlp=$previous_ytdlp and yt-dlp-ejs=$previous_ejs"
  return 1
}

main() {
  case "${1:-}" in
    check)
      if check_runtime; then
        write_state "check" "succeeded" "runtime verification passed"
      else
        write_state "check" "failed" "runtime verification failed"
        return 1
      fi
      ;;
    update)
      mkdir -p "$(dirname "$LOCK_FILE")"
      exec 9>"$LOCK_FILE"
      if ! flock -n -x 9; then
        write_state "update" "deferred" "a URL operation currently holds the yt-dlp runtime lock"
        exit 0
      fi
      update_runtime
      ;;
    *)
      usage
      return 2
      ;;
  esac
}

main "$@"
