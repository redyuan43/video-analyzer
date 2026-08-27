#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${QWEN3_ASR_RUNTIME_DIR:-${ROOT_DIR}/tmp/qwen3-asr-p40}"
LOG_DIR="${QWEN3_ASR_LOG_DIR:-${RUNTIME_DIR}/logs}"
PID_FILE="${QWEN3_ASR_PID_FILE:-${RUNTIME_DIR}/proxy.pid}"
GPU_IDS="${QWEN3_ASR_GPU_IDS:-0,1,2,4,5}"
WORKER_COUNT="${QWEN3_ASR_WORKER_COUNT:-5}"
BASE_PORT="${QWEN3_ASR_BASE_WORKER_PORT:-18300}"
PROXY_PORT="${QWEN3_ASR_PROXY_PORT:-18013}"
PROXY_PYTHON="${QWEN3_ASR_PROXY_PYTHON:-/home/ai/vllm-p40-nightly-test/bin/python}"

stop_python_script() {
  local script="$1"
  local pids
  pids="$(
    ps -eo pid=,comm=,args= \
      | awk -v script="${script}" '$2 ~ /^python/ && index($0, script) {print $1}'
  )"
  [[ -z "${pids}" ]] && return 0
  xargs -r kill <<<"${pids}" || true
  for _ in $(seq 1 50); do
    local remaining=""
    while read -r pid; do
      [[ -z "${pid}" ]] && continue
      kill -0 "${pid}" 2>/dev/null && remaining+="${pid}"$'\n'
    done <<<"${pids}"
    [[ -z "${remaining}" ]] && return 0
    sleep 0.1
  done
  xargs -r kill -9 <<<"${pids}" || true
}

stop_service() {
  if [[ -f "${PID_FILE}" ]]; then
    kill "$(cat "${PID_FILE}")" >/dev/null 2>&1 || true
    rm -f "${PID_FILE}"
  fi
  stop_python_script "tools/asr_servers/qwen3_asr_p40_proxy.py"
  stop_python_script "http_api_server.py --qwen3-asr-worker"
}

start_service() {
  local count="${1:-${WORKER_COUNT}}"
  local ids=()
  local gpu_name
  IFS=, read -r -a ids <<<"${GPU_IDS}"
  if (( count < 1 || count > ${#ids[@]} )); then
    echo "worker-count ${count} exceeds QWEN3_ASR_GPU_IDS (${GPU_IDS})" >&2
    exit 2
  fi
  for ((index = 0; index < count; index++)); do
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${ids[index]}" 2>/dev/null || true)"
    [[ "${gpu_name}" == *"Tesla P40"* ]] || {
      echo "Qwen3-ASR requires Tesla P40; GPU ${ids[index]} is ${gpu_name:-unknown}" >&2
      exit 2
    }
  done
  local specs=()
  for ((index = 0; index < count; index++)); do
    specs+=("${ids[index]}:$((BASE_PORT + index))")
  done
  local worker_spec
  worker_spec="$(IFS=,; echo "${specs[*]}")"
  mkdir -p "${LOG_DIR}"
  stop_service
  QWEN3_ASR_WORKERS="${worker_spec}" \
  QWEN3_ASR_LOG_DIR="${LOG_DIR}" \
  QWEN3_ASR_PROXY_PORT="${PROXY_PORT}" \
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}" \
  no_proxy="${no_proxy:-127.0.0.1,localhost}" \
    setsid "${PROXY_PYTHON}" "${ROOT_DIR}/tools/asr_servers/qwen3_asr_p40_proxy.py" \
      >"${LOG_DIR}/proxy.log" 2>&1 < /dev/null &
  echo "$!" >"${PID_FILE}"
  for _ in $(seq 1 90); do
    if curl --noproxy "*" -fsS "http://127.0.0.1:${PROXY_PORT}/api/health" >/dev/null 2>&1; then
      echo "Qwen3-ASR proxy ready: http://127.0.0.1:${PROXY_PORT}/api/asr/transcribe"
      return 0
    fi
    sleep 1
  done
  echo "Qwen3-ASR proxy failed to start; check ${LOG_DIR}/proxy.log" >&2
  return 1
}

case "${1:-start}" in
  start) start_service "${2:-${WORKER_COUNT}}" ;;
  restart) stop_service; start_service "${2:-${WORKER_COUNT}}" ;;
  stop) stop_service ;;
  status) curl --noproxy "*" -fsS "http://127.0.0.1:${PROXY_PORT}/api/health" ;;
  *) echo "Usage: $0 start|restart|stop|status [worker-count]" >&2; exit 2 ;;
esac
