#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-}"

usage() {
  echo "Usage: $0 asr|ocr|vl|text|tts|stop" >&2
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
  "${ROOT_DIR}/tools/ops/start_qwen3_asr_p40_service.sh" stop >/dev/null 2>&1 || true
}

stop_firered_asr2() {
  "${ROOT_DIR}/tools/ops/start_firered_asr2_p40_service.sh" stop >/dev/null 2>&1 || true
}

stop_minicpm() {
  "${ROOT_DIR}/tools/ops/start_minicpm_p40_service.sh" stop >/dev/null 2>&1 || true
}

stop_indextts() {
  local endpoint="http://127.0.0.1:${INDEXTTS_PORT:-8092}/internal/backend/unload"
  local status
  for _ in $(seq 1 900); do
    status="$(
      curl --noproxy "*" -sS -o /dev/null -w '%{http_code}' \
        -X POST --max-time 125 "${endpoint}" 2>/dev/null || true
    )"
    case "${status}" in
      2??)
        return 0
        ;;
      409)
        sleep 1
        ;;
      *)
        if ! fuser -n tcp "${INDEXTTS_PORT:-8092}" >/dev/null 2>&1; then
          return 0
        fi
        echo "IndexTTS backend did not unload cleanly (HTTP ${status:-000})" >&2
        return 1
        ;;
    esac
  done
  echo "IndexTTS backend stayed busy for 900 seconds" >&2
  return 1
}

stop_bonsai() {
  systemctl --user stop bonsai-local-pool.service >/dev/null
  for _ in $(seq 1 60); do
    if ! systemctl --user is-active --quiet bonsai-local-pool.service \
      && ! fuser -n tcp 18103 >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "BONSAI local pool did not stop within 60 seconds" >&2
  return 1
}

start_bonsai() {
  local config_changed
  local health_payload
  config_changed="$(write_bonsai_runtime_config)"
  if systemctl --user is-active --quiet bonsai-local-pool.service; then
    if [[ "${config_changed}" == "1" ]]; then
      systemctl --user restart bonsai-local-pool.service
    fi
  else
    systemctl --user start bonsai-local-pool.service
  fi
  for _ in $(seq 1 900); do
    health_payload="$(
      curl --noproxy "*" -fsS "http://127.0.0.1:${BONSAI_LOCAL_PORT:-18103}/api/health" \
        2>/dev/null || true
    )"
    if [[ -n "${health_payload}" ]] && printf '%s' "${health_payload}" \
      | "${ROOT_DIR}/.venv/bin/python" -c '
import json
import sys

payload = json.load(sys.stdin)
raise SystemExit(0 if payload.get("ok") else 1)
'; then
      return 0
    fi
    if ! systemctl --user is-active --quiet bonsai-local-pool.service; then
      systemctl --user status bonsai-local-pool.service --no-pager >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "BONSAI local pool did not become ready within 900 seconds" >&2
  return 1
}

write_bonsai_runtime_config() {
  local runtime_dir="${BONSAI_LOCAL_RUNTIME_DIR:-${ROOT_DIR}/tmp/bonsai-local-pool}"
  local config_path="${BONSAI_LOCAL_CONFIG:-${runtime_dir}/config.json}"
  mkdir -p "${runtime_dir}"
  BONSAI_LOCAL_CONFIG="${config_path}" \
    "${ROOT_DIR}/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["BONSAI_LOCAL_CONFIG"])
keys = (
    "BONSAI_LOCAL_HOST",
    "BONSAI_LOCAL_PORT",
    "BONSAI_LOCAL_BACKEND_BASE_PORT",
    "BONSAI_LOCAL_GPU_IDS",
    "BONSAI_LOCAL_WORKER_COUNT",
    "BONSAI_LOCAL_CONTEXT_SIZE",
)
payload = {key: os.environ[key] for key in keys if os.environ.get(key)}
current = None
try:
    current = json.loads(path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    pass
if current == payload:
    print("0")
    raise SystemExit
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(path)
print("1")
PY
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
      "${ROOT_DIR}/tools/ops/start_qwen3_asr_p40_service.sh" start "${QWEN3_ASR_WORKER_COUNT:-5}"
      ;;
    firered_asr2)
      "${ROOT_DIR}/tools/ops/start_firered_asr2_p40_service.sh" start "${FIRERED_ASR2_WORKER_COUNT:-5}"
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
  "${ROOT_DIR}/tools/ops/start_minicpm_p40_service.sh" start "${MINICPM_WORKER_COUNT:-5}"
}

case "${STAGE}" in
  asr)
    stop_indextts
    stop_bonsai
    stop_minicpm
    stop_ocr
    start_asr
    ;;
  ocr)
    stop_indextts
    stop_bonsai
    stop_minicpm
    stop_vibevoice
    stop_qwen3_asr
    stop_firered_asr2
    start_ocr
    ;;
  vl)
    stop_indextts
    stop_bonsai
    stop_ocr
    stop_vibevoice
    stop_qwen3_asr
    stop_firered_asr2
    start_minicpm
    ;;
  text)
    stop_indextts
    stop_ocr
    stop_vibevoice
    stop_qwen3_asr
    stop_firered_asr2
    stop_minicpm
    start_bonsai
    ;;
  tts)
    stop_bonsai
    stop_ocr
    stop_vibevoice
    stop_qwen3_asr
    stop_firered_asr2
    stop_minicpm
    ;;
  stop)
    stop_indextts
    stop_bonsai
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
