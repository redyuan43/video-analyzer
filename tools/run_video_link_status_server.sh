#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/tmp/video-link-status"
PID_FILE="$RUNTIME_DIR/server.pid"
LOG_FILE="$RUNTIME_DIR/server.log"
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
  PYTHONPATH="$ROOT_DIR/video-analyzer-ui:$ROOT_DIR:${PYTHONPATH:-}" \
    setsid "$PYTHON_BIN" -m video_analyzer_ui.server --host "$BIND_HOST" --port "$PORT" --jobs-dir "$RUNTIME_DIR/jobs" \
    >"$LOG_FILE" 2>&1 < /dev/null &
  echo "$!" >"$PID_FILE"
  echo "video-link status server started: http://$PUBLIC_HOST:$PORT/ (bind: $BIND_HOST)"
  echo "log: $LOG_FILE"
}

stop_server() {
  if ! is_running; then
    echo "video-link status server is not running"
    rm -f "$PID_FILE"
    return 0
  fi
  kill "$(cat "$PID_FILE")"
  rm -f "$PID_FILE"
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
      echo "running: pid=$(cat "$PID_FILE") http://$PUBLIC_HOST:$PORT/ (bind: $BIND_HOST)"
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
