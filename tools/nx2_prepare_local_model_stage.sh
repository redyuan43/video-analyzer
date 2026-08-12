#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
if [[ "$stage" != "asr" && "$stage" != "ocr" && "$stage" != "vl" && "$stage" != "text" && "$stage" != "stop" ]]; then
  echo "Usage: $0 asr|ocr|vl|text|stop" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${NX2_MODEL_RUNTIME_DIR:-$root_dir/tmp/nx2-local-models}"
model_root="${NX2_MODEL_ROOT:-$(cd "$root_dir/.." && pwd)/models}"
llama_server="${NX2_LLAMA_SERVER:-$HOME/github/llama.cpp-latest-mtp/build-vlm-bench/bin/llama-server}"
asr_python="${NX2_ASR_PYTHON:-$(cd "$root_dir/.." && pwd)/asr/.venv/bin/python}"
ocr_python="${NX2_OCR_PYTHON:-$(cd "$root_dir/.." && pwd)/ocr/.venv/bin/python}"
vl_model="${NX2_VL_MODEL:-/data/models/video-analyzer-vl-modelscope/Qwen3-VL-4B-Instruct/Qwen3VL-4B-Instruct-Q4_K_M.gguf}"
vl_mmproj="${NX2_VL_MMPROJ:-/data/models/video-analyzer-vl-modelscope/Qwen3-VL-4B-Instruct/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf}"
vl_alias="${NX2_VL_ALIAS:-qwen3-vl-4b-nx2}"
text_model="${NX2_TEXT_MODEL:-/data/models/video-analyzer-text/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf}"
text_alias="${NX2_TEXT_ALIAS:-qwythos}"

if [[ ! -x "$asr_python" ]]; then
  asr_python="$(command -v python3)"
fi
if [[ ! -x "$ocr_python" ]]; then
  ocr_python="$(command -v python3)"
fi

asr_port="${NX2_ASR_PORT:-18013}"
ocr_port="${NX2_OCR_PORT:-18089}"
vl_port="${NX2_VL_PORT:-18082}"
text_port="${NX2_TEXT_PORT:-18081}"

mkdir -p "$runtime_dir"

stop_pid_file() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 0

  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "Timed out stopping pid $pid from $pid_file" >&2
      exit 1
    fi
  fi
  rm -f "$pid_file"
}

stop_all_models() {
  stop_pid_file "$runtime_dir/funasr.pid"
  stop_pid_file "$runtime_dir/easyocr.pid"
  stop_pid_file "$runtime_dir/minicpm.pid"
  stop_pid_file "$runtime_dir/qwythos.pid"
}

wait_for_url() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 180); do
    if curl --noproxy "*" -fsS "$url" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "$name did not become ready: $url" >&2
  return 1
}

wait_for_text_generation() {
  local url="http://127.0.0.1:${text_port}/v1/chat/completions"
  local payload='{"model":"'"${text_alias}"'","messages":[{"role":"user","content":"/no_think ready"}],"max_tokens":1,"temperature":0}'
  for _ in $(seq 1 180); do
    if curl --noproxy "*" -fsS --max-time 30 \
      -X POST "$url" \
      -H "Content-Type: application/json" \
      --data-binary "$payload" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "Qwythos did not accept chat requests: $url" >&2
  return 1
}

start_background() {
  local pid_file="$1"
  local log_file="$2"
  shift 2
  nohup "$@" >"$log_file" 2>&1 < /dev/null &
  echo "$!" >"$pid_file"
}

case "$stage" in
  asr)
    stop_all_models
    start_background \
      "$runtime_dir/funasr.pid" \
      "$runtime_dir/funasr.log" \
      "$asr_python" "$root_dir/tools/nx2_sensevoice_http_server.py" \
      --host 127.0.0.1 \
      --port "$asr_port" \
      --model "$model_root/asr/modelscope/models/iic--SenseVoiceSmall/snapshots/master" \
      --vad-model "$model_root/asr/modelscope/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/master" \
      --punc-model "$model_root/asr/modelscope/models/iic--punc_ct-transformer_zh-cn-common-vocab272727-pytorch/snapshots/master" \
      --idle-unload-seconds 60
    wait_for_url "http://127.0.0.1:${asr_port}/api/health" "FunASR"
    ;;
  ocr)
    stop_all_models
    start_background \
      "$runtime_dir/easyocr.pid" \
      "$runtime_dir/easyocr.log" \
      "$ocr_python" "$root_dir/tools/nx2_easyocr_openai_server.py" \
      --host 127.0.0.1 \
      --port "$ocr_port" \
      --model-storage-directory "$model_root/ocr/easyocr" \
      --languages ch_sim,en \
      --served-model-name easyocr-ch-sim-en
    wait_for_url "http://127.0.0.1:${ocr_port}/api/health" "EasyOCR"
    ;;
  vl)
    stop_all_models
    start_background \
      "$runtime_dir/minicpm.pid" \
      "$runtime_dir/minicpm.log" \
      "$llama_server" \
      --model "$vl_model" \
      --mmproj "$vl_mmproj" \
      --alias "$vl_alias" \
      --host 127.0.0.1 \
      --port "$vl_port" \
      --ctx-size 8192 \
      --parallel 1 \
      --gpu-layers 999 \
      --image-min-tokens 1024 \
      --no-cache-prompt
    wait_for_url "http://127.0.0.1:${vl_port}/health" "VL"
    ;;
  text)
    stop_all_models
    if [[ ! -f "$text_model" ]]; then
      echo "Qwythos model missing: $text_model" >&2
      exit 1
    fi
    start_background \
      "$runtime_dir/qwythos.pid" \
      "$runtime_dir/qwythos.log" \
      "$llama_server" \
      --model "$text_model" \
      --alias "$text_alias" \
      --host 127.0.0.1 \
      --port "$text_port" \
      --ctx-size 65536 \
      --parallel 1 \
      --gpu-layers 999 \
      --cache-ram 0 \
      --no-cache-prompt \
      --reasoning off \
      --reasoning-budget 0
    wait_for_url "http://127.0.0.1:${text_port}/health" "Qwythos"
    wait_for_text_generation
    sleep 2
    ;;
  stop)
    stop_all_models
    ;;
esac

echo "NX2 local model stage ready: $stage"
