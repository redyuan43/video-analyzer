#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${MINICPM_RUNTIME_DIR:-${ROOT_DIR}/tmp/minicpm-p40}"
LOG_DIR="${MINICPM_LOG_DIR:-${RUNTIME_DIR}/logs}"
PID_FILE="${MINICPM_PID_FILE:-${RUNTIME_DIR}/proxy.pid}"
PROXY_HOST="${MINICPM_PROXY_HOST:-0.0.0.0}"
PROXY_PORT="${MINICPM_PROXY_PORT:-18082}"
BASE_BACKEND_PORT="${MINICPM_BASE_BACKEND_PORT:-18182}"
WORKER_COUNT="${MINICPM_WORKER_COUNT:-5}"
PYTHON_BIN="${MINICPM_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

usage() {
  echo "Usage: $0 start|stop|restart|status [worker-count]" >&2
}

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

clear_stale_pid() {
  if [[ -f "${PID_FILE}" ]] && ! is_running; then
    rm -f "${PID_FILE}"
  fi
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

worker_spec() {
  local count="$1"
  local specs=()
  for ((gpu = 0; gpu < count; gpu++)); do
    specs+=("${gpu}:$((BASE_BACKEND_PORT + gpu))")
  done
  join_by_comma "${specs[@]}"
}

stop_pattern() {
  local pattern="$1"
  local pids
  pids="$(ps -eo pid=,args= | awk -v pattern="${pattern}" 'index($0, pattern) && $1 != PROCINFO["pid"] {print $1}')"
  if [[ -n "${pids}" ]]; then
    xargs -r kill <<<"${pids}" || true
    for _ in $(seq 1 "${MINICPM_STOP_TIMEOUT:-30}"); do
      local remaining=()
      while read -r pid; do
        [[ -z "${pid}" ]] && continue
        kill -0 "${pid}" 2>/dev/null && remaining+=("${pid}")
      done <<<"${pids}"
      if (( ${#remaining[@]} == 0 )); then
        return 0
      fi
      sleep 1
    done
    xargs -r kill -9 <<<"${pids}" || true
  fi
}

stop_conflicting_gpu_services() {
  systemctl --user stop dots-mocr-p40.service >/dev/null 2>&1 || true
  systemctl --user stop vibevoice-p40-asr.service >/dev/null 2>&1 || true
  pkill -f "[d]ots_mocr_p40_proxy.py" || true
  pkill -f "[v]ibevoice_vllm_p40_http_server.py" || true
  pkill -f "[v]llm.entrypoints.openai.api_server --host 127.0.0.1 --port 1800[0-4]" || true
}

stop_minicpm() {
  clear_stale_pid
  if is_running; then
    kill "$(cat "${PID_FILE}")" || true
    rm -f "${PID_FILE}"
  fi
  stop_pattern "tools/minicpm_p40_proxy.py"
  stop_pattern "MiniCPM-V-4_5-gguf/ggml-model-Q4_K_M.gguf"
}

start_minicpm() {
  local count="${1:-${WORKER_COUNT}}"
  if ! [[ "${count}" =~ ^[1-5]$ ]]; then
    echo "worker-count must be an integer from 1 to 5" >&2
    exit 2
  fi
  clear_stale_pid
  if is_running; then
    echo "MiniCPM proxy already running: pid=$(cat "${PID_FILE}") http://127.0.0.1:${PROXY_PORT}/v1"
    return 0
  fi

  mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"
  stop_conflicting_gpu_services
  stop_minicpm

  cd "${ROOT_DIR}"
  local workers
  workers="$(worker_spec "${count}")"
  echo "Starting MiniCPM P40 proxy with ${count} worker(s): ${workers}"
  echo "Proxy URL: http://127.0.0.1:${PROXY_PORT}/v1"

  NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}" \
  no_proxy="${no_proxy:-127.0.0.1,localhost}" \
  setsid "${PYTHON_BIN}" tools/minicpm_p40_proxy.py \
    --host "${PROXY_HOST}" \
    --port "${PROXY_PORT}" \
    --workers "${workers}" \
    --log-dir "${LOG_DIR}" \
    >"${LOG_DIR}/launcher.log" 2>&1 < /dev/null &
  echo "$!" >"${PID_FILE}"

  for _ in $(seq 1 30); do
    if curl --noproxy "*" -fsS "http://127.0.0.1:${PROXY_PORT}/api/health" >/dev/null 2>&1; then
      echo "MiniCPM proxy ready: http://127.0.0.1:${PROXY_PORT}/v1"
      curl --noproxy "*" -sS "http://127.0.0.1:${PROXY_PORT}/api/health"
      echo
      return 0
    fi
    sleep 1
  done

  echo "MiniCPM proxy did not become ready. Check ${LOG_DIR}/launcher.log and ${LOG_DIR}/proxy.log" >&2
  exit 1
}

status_minicpm() {
  clear_stale_pid
  if is_running; then
    echo "running: pid=$(cat "${PID_FILE}") http://127.0.0.1:${PROXY_PORT}/v1"
  else
    echo "not running"
  fi
  curl --noproxy "*" -fsS "http://127.0.0.1:${PROXY_PORT}/api/health" || true
  echo
}

case "${1:-}" in
  start)
    start_minicpm "${2:-${WORKER_COUNT}}"
    ;;
  stop)
    stop_minicpm
    echo "MiniCPM proxy stopped"
    ;;
  restart)
    stop_minicpm
    start_minicpm "${2:-${WORKER_COUNT}}"
    ;;
  status)
    status_minicpm
    ;;
  *)
    usage
    exit 2
    ;;
esac
