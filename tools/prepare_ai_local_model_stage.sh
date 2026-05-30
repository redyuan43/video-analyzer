#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-}"

usage() {
  echo "Usage: $0 asr|ocr|vl" >&2
}

stop_ocr() {
  systemctl --user stop dots-mocr-p40.service >/dev/null 2>&1 || true
  ps -eo pid=,comm=,args= \
    | awk '/python/ && /\/home\/ai\/ocr-deploy\/scripts\/dots_mocr_p40_proxy.py|\/home\/ai\/ocr-deploy\/dots\.mocr\/weights\/DotsMOCR/ {print $1}' \
    | xargs -r kill || true
}

stop_vibevoice() {
  systemctl --user stop vibevoice-p40-asr.service >/dev/null 2>&1 || true
  pkill -f "[v]ibevoice_vllm_p40_http_server.py" || true
  pkill -f "[v]llm.entrypoints.openai.api_server --host 127.0.0.1 --port 1800[0-4]" || true
}

stop_minicpm() {
  "${ROOT_DIR}/tools/start_minicpm_p40_service.sh" stop >/dev/null 2>&1 || true
}

start_vibevoice() {
  local workers="${VIBEVOICE_WORKER_COUNT:-5}"
  "/home/ai/github/VibeVoice-bench/start_vibevoice_p40_workers.sh" "${workers}"
}

start_ocr() {
  local workers="${DOTS_MOCR_WORKER_COUNT:-5}"
  DOTS_MOCR_PROXY_PORT="${DOTS_MOCR_PROXY_PORT:-18088}" \
    "/home/ai/ocr-deploy/start_dots_mocr_p40_service.sh" "${workers}"
}

start_minicpm() {
  "${ROOT_DIR}/tools/start_minicpm_p40_service.sh" start "${MINICPM_WORKER_COUNT:-5}"
}

case "${STAGE}" in
  asr)
    stop_minicpm
    stop_ocr
    start_vibevoice
    ;;
  ocr)
    stop_minicpm
    stop_vibevoice
    start_ocr
    ;;
  vl)
    stop_ocr
    stop_vibevoice
    start_minicpm
    ;;
  *)
    usage
    exit 2
    ;;
esac
