#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
if [[ "$stage" != "asr" && "$stage" != "ocr" && "$stage" != "vl" ]]; then
  echo "Usage: $0 asr|ocr|vl" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${NX2_MODEL_PYTHON:-$root_dir/.venv/bin/python}"
runtime_dir="${NX2_MODEL_RUNTIME_DIR:-$root_dir/tmp/nx2-local-models}"
model_root="${NX2_MODEL_ROOT:-$(cd "$root_dir/.." && pwd)/models}"
llama_server="${NX2_LLAMA_SERVER:-$HOME/github/llama.cpp-latest-mtp/build/bin/llama-server}"

if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3)"
fi

asr_port="${NX2_ASR_PORT:-18013}"
ocr_port="${NX2_OCR_PORT:-18089}"
vl_port="${NX2_VL_PORT:-18082}"

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
      "$python_bin" "$root_dir/tools/nx2_sensevoice_http_server.py" \
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
      "$python_bin" "$root_dir/tools/nx2_easyocr_openai_server.py" \
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
      --model "$model_root/vl/MiniCPM-V-4_5/ggml-model-Q4_K_M.gguf" \
      --mmproj "$model_root/vl/MiniCPM-V-4_5/mmproj-model-f16.gguf" \
      --alias minicpm-v-4.5-nx2 \
      --host 127.0.0.1 \
      --port "$vl_port" \
      --ctx-size 8192 \
      --parallel 1 \
      --gpu-layers 999
    wait_for_url "http://127.0.0.1:${vl_port}/health" "MiniCPM"
    ;;
esac

echo "NX2 local model stage ready: $stage"
