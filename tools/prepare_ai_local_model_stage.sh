#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-}"

usage() {
  echo "Usage: $0 asr|ocr|vl|stop" >&2
}

stop_pids() {
  local pids="$1"
  [[ -z "${pids}" ]] && return 0
  xargs -r kill <<<"${pids}" || true
  for _ in $(seq 1 30); do
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
}

stop_matching() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "${pattern}" || true)"
  stop_pids "${pids}"
}

stop_ocr() {
  systemctl --user stop dots-mocr-p40.service >/dev/null 2>&1 || true
  ps -eo pid=,comm=,args= \
    | awk '/python/ && /\/home\/ai\/ocr-deploy\/scripts\/dots_mocr_p40_proxy.py|\/home\/ai\/ocr-deploy\/dots\.mocr\/weights\/DotsMOCR|\/home\/ai\/ocr-deploy\/scripts\/unlimited_ocr_p40_proxy.py|\/home\/ai\/ocr-deploy\/scripts\/unlimited_ocr_transformers_worker.py/ {print $1}' \
    | {
        pids="$(cat)"
        stop_pids "${pids}"
      }
}

stop_vibevoice() {
  systemctl --user stop vibevoice-p40-asr.service >/dev/null 2>&1 || true
  stop_matching "[v]ibevoice_vllm_p40_http_server.py"
  stop_matching "[v]llm.entrypoints.openai.api_server --host 127.0.0.1 --port 1800[0-4]"
}

stop_qwen3_asr() {
  "${ROOT_DIR}/tools/start_qwen3_asr_p40_service.sh" stop >/dev/null 2>&1 || true
}

stop_firered_asr2() {
  "${ROOT_DIR}/tools/start_firered_asr2_p40_service.sh" stop >/dev/null 2>&1 || true
}

stop_minicpm() {
  "${ROOT_DIR}/tools/start_minicpm_p40_service.sh" stop >/dev/null 2>&1 || true
}

start_vibevoice() {
  local workers="${VIBEVOICE_WORKER_COUNT:-5}"
  stop_vibevoice
  if ! "/home/ai/github/VibeVoice-bench/start_vibevoice_p40_workers.sh" "${workers}"; then
    stop_vibevoice
    return 1
  fi
}

start_asr() {
  local engine="${ASR_ENGINE:-vibevoice}"
  stop_vibevoice
  stop_qwen3_asr
  stop_firered_asr2
  case "${engine}" in
    vibevoice)
      start_vibevoice
      ;;
    qwen3_asr|qwen3-asr|capswriter)
      "${ROOT_DIR}/tools/start_qwen3_asr_p40_service.sh" start "${QWEN3_ASR_WORKER_COUNT:-5}"
      ;;
    firered_asr2)
      "${ROOT_DIR}/tools/start_firered_asr2_p40_service.sh" start "${FIRERED_ASR2_WORKER_COUNT:-5}"
      ;;
    firered_3dspeaker|none)
      ;;
    *)
      echo "Unknown ASR_ENGINE=${engine}" >&2
      return 2
      ;;
  esac
}

start_ocr() {
  local engine="${OCR_ENGINE:-unlimited}"
  case "${engine}" in
    unlimited|unlimited-ocr)
      local workers="${UNLIMITED_OCR_WORKER_COUNT:-5}"
      local model="${UNLIMITED_OCR_MODEL:-/home/ai/ocr-deploy/models/unlimited-ocr-f799-p40-runtime}"
      UNLIMITED_OCR_MODEL="${model}" \
      UNLIMITED_OCR_GPU_IDS="${UNLIMITED_OCR_GPU_IDS:-0,1,2,4,5}" \
      UNLIMITED_OCR_PROXY_PORT="${UNLIMITED_OCR_PROXY_PORT:-18088}" \
        "/home/ai/ocr-deploy/start_unlimited_ocr_p40_service.sh" "${workers}"
      ;;
    dots|dotsmocr|dots-mocr)
      local workers="${DOTS_MOCR_WORKER_COUNT:-5}"
      DOTS_MOCR_PROXY_PORT="${DOTS_MOCR_PROXY_PORT:-18088}" \
        "/home/ai/ocr-deploy/start_dots_mocr_p40_service.sh" "${workers}"
      ;;
    *)
      echo "Unknown OCR_ENGINE=${engine}; expected unlimited or dots" >&2
      return 2
      ;;
  esac
}

start_minicpm() {
  "${ROOT_DIR}/tools/start_minicpm_p40_service.sh" start "${MINICPM_WORKER_COUNT:-5}"
}

case "${STAGE}" in
  asr)
    stop_minicpm
    stop_ocr
    start_asr
    ;;
  ocr)
    stop_minicpm
    stop_vibevoice
    stop_qwen3_asr
    stop_firered_asr2
    start_ocr
    ;;
  vl)
    stop_ocr
    stop_vibevoice
    stop_qwen3_asr
    stop_firered_asr2
    start_minicpm
    ;;
  stop)
    stop_ocr
    stop_vibevoice
    stop_qwen3_asr
    stop_firered_asr2
    stop_minicpm
    ;;
  *)
    usage
    exit 2
    ;;
esac
