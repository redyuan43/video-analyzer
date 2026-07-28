#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/tmp/video-link-status"
PID_FILE="$RUNTIME_DIR/server.pid"
LOG_FILE="$RUNTIME_DIR/server.log"
STATUS_FILE="$RUNTIME_DIR/supervisor.json"
default_host() {
  if command -v tailscale >/dev/null 2>&1; then
    tailscale ip -4 2>/dev/null | head -n 1
    return 0
  fi
  printf '127.0.0.1\n'
}

PUBLIC_HOST="${VIDEO_LINK_STATUS_HOST:-$(default_host)}"
PUBLIC_HOST="${PUBLIC_HOST:-127.0.0.1}"
BIND_HOST="${VIDEO_LINK_STATUS_BIND_HOST:-0.0.0.0}"
PORT="${VIDEO_LINK_STATUS_PORT:-5000}"
PYTHON_BIN="${VIDEO_LINK_STATUS_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

detect_agx_lan_host() {
  if [[ -n "${JETSON_AGX_LAN_HOST:-}" ]]; then
    printf '%s\n' "$JETSON_AGX_LAN_HOST"
    return 0
  fi

  local candidates="${VIDEO_LINK_AGX_LAN_HOST_CANDIDATES:-agx-lan,agx.local,ubuntu.local}"
  local candidate
  local -a candidate_list
  IFS=, read -r -a candidate_list <<<"$candidates"
  for candidate in "${candidate_list[@]}"; do
    candidate="${candidate//[[:space:]]/}"
    [[ -n "$candidate" ]] || continue
    timeout 2 getent ahostsv4 "$candidate" >/dev/null 2>&1 || continue
    timeout 4 ssh -o BatchMode=yes -o ConnectTimeout=2 -o ConnectionAttempts=1 \
      -o HostKeyAlias=agx-lan "agx@$candidate" true >/dev/null 2>&1 || continue
    printf '%s\n' "$candidate"
    return 0
  done
}

mkdir -p "$RUNTIME_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

clear_stale_pid() {
  if [[ -f "$PID_FILE" ]] && ! is_running; then
    rm -f "$PID_FILE"
  fi
}

start_server() {
  clear_stale_pid
  if is_running; then
    echo "video-link status server already running: pid=$(cat "$PID_FILE")"
    return 0
  fi
  cd "$ROOT_DIR"
  AGX_LAN_HOST="$(detect_agx_lan_host || true)"
  if [[ -n "$AGX_LAN_HOST" ]]; then
    export JETSON_AGX_LAN_HOST="$AGX_LAN_HOST"
  fi
  PYTHONPATH="$ROOT_DIR/video-analyzer-ui:$ROOT_DIR:${PYTHONPATH:-}" \
    setsid env JETSON_AGX_LAN_HOST="${JETSON_AGX_LAN_HOST:-}" "$PYTHON_BIN" tools/video_link_status_supervisor.py \
      --repo-root "$ROOT_DIR" \
      --python "$PYTHON_BIN" \
      --host "$BIND_HOST" \
      --port "$PORT" \
      --jobs-dir "$RUNTIME_DIR/jobs" \
      --status-file "$STATUS_FILE" \
    >"$LOG_FILE" 2>&1 < /dev/null &
  echo "$!" >"$PID_FILE"
  echo "video-link status server started: http://$PUBLIC_HOST:$PORT/ (bind: $BIND_HOST)"
  if [[ -n "${JETSON_AGX_LAN_HOST:-}" ]]; then
    echo "AGX LAN host: $JETSON_AGX_LAN_HOST"
  else
    echo "AGX LAN host: not detected; set JETSON_AGX_LAN_HOST or VIDEO_LINK_AGX_LAN_HOST_CANDIDATES"
  fi
  echo "log: $LOG_FILE"
}

stop_server() {
  if ! is_running; then
    echo "video-link status server is not running"
    rm -f "$PID_FILE"
    return 0
  fi
  kill "$(cat "$PID_FILE")"
  for _ in $(seq 1 50); do
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null || break
    sleep 0.1
  done
  rm -f "$PID_FILE"
  rm -f "$STATUS_FILE"
  echo "video-link status server stopped"
}

case "${1:-status}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  status)
    clear_stale_pid
    if is_running; then
      if [[ -f "$STATUS_FILE" ]]; then
        "$PYTHON_BIN" - "$STATUS_FILE" "$PUBLIC_HOST" "$PORT" "$BIND_HOST" <<'PY'
import json
import sys

path, public_host, port, bind_host = sys.argv[1:]
payload = json.load(open(path, encoding="utf-8"))
print(
    f"running: supervisor_pid={payload.get('supervisor_pid')} "
    f"server_pid={payload.get('server_pid')} runtime_id={payload.get('runtime_id')} "
    f"stale={payload.get('source_stale')} http://{public_host}:{port}/ (bind: {bind_host})"
)
PY
      else
        echo "running: supervisor_pid=$(cat "$PID_FILE") http://$PUBLIC_HOST:$PORT/ (bind: $BIND_HOST)"
      fi
    else
      echo "not running"
      exit 1
    fi
    ;;
  logs)
    tail -n "${2:-120}" "$LOG_FILE"
    ;;
  *)
    echo "Usage: $0 start|stop|restart|status|logs [lines]" >&2
    exit 2
    ;;
esac
