#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${FIRERED_ASR2_RUNTIME_DIR:-${ROOT_DIR}/tmp/firered-asr2-p40}"
LOG_DIR="${FIRERED_ASR2_LOG_DIR:-${RUNTIME_DIR}/logs}"
GPU_IDS="${FIRERED_ASR2_GPU_IDS:-auto}"
GPU_SELECTION="${FIRERED_ASR2_GPU_SELECTION:-auto}"
WORKER_COUNT="${FIRERED_ASR2_WORKER_COUNT:-5}"
BASE_PORT="${FIRERED_ASR2_BASE_WORKER_PORT:-18400}"
PROXY_PORT="${FIRERED_ASR2_PROXY_PORT:-18014}"
PYTHON="${FIRERED_ASR2_PYTHON:-/home/ai/diarization-ab-venv/bin/python}"
PROXY_PYTHON="${FIRERED_ASR2_PROXY_PYTHON:-${PYTHON}}"
SOURCE_ROOT="${FIRERED_ASR2_SOURCE_ROOT:-/home/ai/src/firered-asr2s}"

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
  stop_python_script "tools/asr_servers/firered_asr2_p40_proxy.py"
  stop_python_script "tools/asr_servers/firered_asr2_worker.py"
}

start_service() {
  local count="${1:-${WORKER_COUNT}}"
  if [[ "${count}" != "auto" ]] && ! [[ "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "worker-count must be auto or a positive integer" >&2
    exit 2
  fi
  stop_service
  if [[ "${GPU_SELECTION}" != "manual" ]]; then
    GPU_IDS="$(
      "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/tools/ops/discover_idle_gpus.py" \
        --allowed-name "Tesla P40" \
        --min-total-mib "${FIRERED_ASR2_MIN_TOTAL_MIB:-22000}" \
        --min-free-mib "${FIRERED_ASR2_MIN_FREE_MIB:-10000}" \
        --max-count "${count}" \
        --format csv
    )"
  fi
  local ids=()
  IFS=, read -r -a ids <<<"${GPU_IDS}"
  if (( ${#ids[@]} == 0 )); then
    echo "No compatible idle Tesla P40 is available for FireRedASR2" >&2
    exit 1
  fi
  if [[ "${count}" == "auto" ]]; then
    count="${#ids[@]}"
  elif (( count > ${#ids[@]} )); then
    echo "Requested ${count} FireRedASR2 worker(s); reducing to ${#ids[@]} based on usable P40 GPUs." >&2
    count="${#ids[@]}"
  fi
  (( count >= 1 )) || { echo "invalid worker count" >&2; exit 2; }
  for ((index = 0; index < count; index++)); do
    local gpu_name
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${ids[index]}" 2>/dev/null || true)"
    [[ "${gpu_name}" == *"Tesla P40"* ]] || {
      echo "FireRedASR2 requires Tesla P40; GPU ${ids[index]} is ${gpu_name:-unknown}" >&2
      exit 2
    }
  done
  mkdir -p "${LOG_DIR}"
  local ports=()
  for ((index = 0; index < count; index++)); do
    local gpu="${ids[index]}"
    local port="$((BASE_PORT + index))"
    ports+=("${port}")
    CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${SOURCE_ROOT}:${ROOT_DIR}" \
    FIRERED_ASR2_WORKER_PORT="${port}" \
    NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" \
      setsid "${PYTHON}" "${ROOT_DIR}/tools/asr_servers/firered_asr2_worker.py" \
        >"${LOG_DIR}/worker-gpu${gpu}.log" 2>&1 < /dev/null &
  done
  local port_list
  port_list="$(IFS=,; echo "${ports[*]}")"
  local ready=0
  for _ in $(seq 1 60); do
    ready=0
    for port in "${ports[@]}"; do
      curl --noproxy "*" -fsS "http://127.0.0.1:${port}/api/health" >/dev/null 2>&1 && ready=$((ready + 1))
    done
    (( ready == count )) && break
    sleep 1
  done
  if (( ready != count )); then
    echo "FireRedASR2 workers failed to start (${ready}/${count}). Check ${LOG_DIR}/worker-gpu*.log" >&2
    stop_service
    return 1
  fi
  PYTHONPATH="${SOURCE_ROOT}:${ROOT_DIR}" FIRERED_ASR2_WORKER_PORTS="${port_list}" FIRERED_ASR2_PROXY_PORT="${PROXY_PORT}" \
  NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" \
    setsid "${PROXY_PYTHON}" "${ROOT_DIR}/tools/asr_servers/firered_asr2_p40_proxy.py" \
      >"${LOG_DIR}/proxy.log" 2>&1 < /dev/null &
  for _ in $(seq 1 30); do
    curl --noproxy "*" -fsS "http://127.0.0.1:${PROXY_PORT}/api/health" >/dev/null 2>&1 && {
      echo "FireRedASR2 proxy ready: http://127.0.0.1:${PROXY_PORT}/api/asr/transcribe"
      return 0
    }
    sleep 1
  done
  echo "FireRedASR2 proxy failed to start" >&2
  stop_service
  return 1
}

case "${1:-start}" in
  start) start_service "${2:-${WORKER_COUNT}}" ;;
  restart) stop_service; start_service "${2:-${WORKER_COUNT}}" ;;
  stop) stop_service ;;
  status) curl --noproxy "*" -fsS "http://127.0.0.1:${PROXY_PORT}/api/health" ;;
  *) echo "Usage: $0 start|restart|stop|status [worker-count]" >&2; exit 2 ;;
esac
