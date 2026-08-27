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
GPU_IDS="${MINICPM_GPU_IDS:-0,1,2,4,5}"
PYTHON_BIN="${MINICPM_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
STOP_CONFLICTS="${MINICPM_STOP_CONFLICTS:-1}"
VISION_ENGINE="${VISION_ENGINE:-minicpm_v45}"
MODEL_PATH=""
MMPROJ_PATH=""
MODEL_ALIAS=""

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

usage() {
  echo "Usage: $0 start|stop|restart|status [worker-count]" >&2
  echo "Set MINICPM_GPU_IDS to choose physical GPUs; default: ${GPU_IDS}" >&2
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

configure_model() {
  case "${VISION_ENGINE}" in
    minicpm_v45|minicpm)
      MODEL_PATH="${MINICPM_MODEL_PATH:-/home/ai/.lmstudio/models/openbmb/MiniCPM-V-4_5-gguf/ggml-model-Q4_K_M.gguf}"
      MMPROJ_PATH="${MINICPM_MMPROJ_PATH:-/home/ai/.lmstudio/models/openbmb/MiniCPM-V-4_5-gguf/mmproj-model-f16.gguf}"
      MODEL_ALIAS="${MINICPM_MODEL_ALIAS:-minicpm-v-4.5-v100}"
      ;;
    qwen3_vl_4b|qwen3-vl-4b)
      MODEL_PATH="${QWEN3_VL_MODEL_PATH:-/home/ai/.lmstudio/models/Qwen/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q4_K_M.gguf}"
      MMPROJ_PATH="${QWEN3_VL_MMPROJ_PATH:-/home/ai/.lmstudio/models/Qwen/Qwen3-VL-4B-Instruct-GGUF/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf}"
      MODEL_ALIAS="${QWEN3_VL_MODEL_ALIAS:-qwen3-vl-4b-instruct}"
      ;;
    *)
      echo "Unknown VISION_ENGINE=${VISION_ENGINE}" >&2
      exit 2
      ;;
  esac
  if [[ ! -f "${MODEL_PATH}" || ! -f "${MMPROJ_PATH}" ]]; then
    echo "Vision model files are missing for ${VISION_ENGINE}:" >&2
    echo "  model=${MODEL_PATH}" >&2
    echo "  mmproj=${MMPROJ_PATH}" >&2
    exit 1
  fi
}

worker_spec() {
  local count="$1"
  local specs=()
  local gpu_ids=()
  IFS=, read -r -a gpu_ids <<<"${GPU_IDS}"
  if (( count > ${#gpu_ids[@]} )); then
    echo "worker-count ${count} exceeds MINICPM_GPU_IDS entries (${GPU_IDS})" >&2
    exit 2
  fi
  for ((index = 0; index < count; index++)); do
    local gpu="${gpu_ids[index]}"
    if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
      echo "invalid GPU id in MINICPM_GPU_IDS: ${gpu}" >&2
      exit 2
    fi
    if [[ "${gpu}" == "3" ]]; then
      echo "GPU 3 is reserved for the Foundation-Sec security model" >&2
      exit 2
    fi
    specs+=("${gpu}:$((BASE_BACKEND_PORT + index))")
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
  pkill -f "[v]llm.entrypoints.openai.api_server --host 127.0.0.1 --port 1800[0-9]" || true
}

stop_minicpm() {
  clear_stale_pid
  if is_running; then
    kill "$(cat "${PID_FILE}")" || true
    rm -f "${PID_FILE}"
  fi
  stop_pattern "tools/ocr_servers/minicpm_p40_proxy.py"
  stop_pattern "MiniCPM-V-4_5-gguf/ggml-model-Q4_K_M.gguf"
}

start_minicpm() {
  local count="${1:-${WORKER_COUNT}}"
  if ! [[ "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "worker-count must be a positive integer" >&2
    exit 2
  fi
  clear_stale_pid
  if is_running; then
    echo "MiniCPM proxy already running: pid=$(cat "${PID_FILE}") http://127.0.0.1:${PROXY_PORT}/v1"
    return 0
  fi

  configure_model
  mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"
  if [[ "${STOP_CONFLICTS}" != "0" ]]; then
    stop_conflicting_gpu_services
  fi
  stop_minicpm

  cd "${ROOT_DIR}"
  local workers
  workers="$(worker_spec "${count}")"
  echo "Starting vision proxy (${VISION_ENGINE}) with ${count} worker(s)."
  echo "Configured GPUs: ${GPU_IDS}"
  echo "Worker mapping: ${workers}"
  echo "Proxy URL: http://127.0.0.1:${PROXY_PORT}/v1"

  NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}" \
  no_proxy="${no_proxy:-127.0.0.1,localhost}" \
  CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
  setsid "${PYTHON_BIN}" tools/ocr_servers/minicpm_p40_proxy.py \
    --host "${PROXY_HOST}" \
    --port "${PROXY_PORT}" \
    --workers "${workers}" \
    --log-dir "${LOG_DIR}" \
    --model-path "${MODEL_PATH}" \
    --mmproj-path "${MMPROJ_PATH}" \
    --model-alias "${MODEL_ALIAS}" \
    >"${LOG_DIR}/launcher.log" 2>&1 < /dev/null &
  echo "$!" >"${PID_FILE}"

  for _ in $(seq 1 30); do
    if curl --noproxy "*" -fsS "http://127.0.0.1:${PROXY_PORT}/api/health" >/dev/null 2>&1; then
      echo "Vision proxy ready (${VISION_ENGINE}): http://127.0.0.1:${PROXY_PORT}/v1"
      curl --noproxy "*" -sS "http://127.0.0.1:${PROXY_PORT}/api/health"
      echo
      return 0
    fi
    sleep 1
  done

  echo "Vision proxy (${VISION_ENGINE}) did not become ready. Check ${LOG_DIR}/launcher.log and ${LOG_DIR}/proxy.log" >&2
  exit 1
}

status_minicpm() {
  clear_stale_pid
  if is_running; then
    echo "running: engine=${VISION_ENGINE} pid=$(cat "${PID_FILE}") http://127.0.0.1:${PROXY_PORT}/v1"
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
    echo "Vision proxy stopped"
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
