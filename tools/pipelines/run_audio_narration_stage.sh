#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${1:-}" ]]; then
  echo "Usage: tools/pipelines/run_audio_narration_stage.sh RUN_DIR [--profile PROFILE] [--config CONFIG_DIR]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$(realpath "$1")"
shift
PROFILE="deepseek_v4_flash"
CONFIG_DIR="config"

while (( $# )); do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG_DIR="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

restore_text_pool() {
  local restore_status
  trap - EXIT
  set +e
  "$PYTHON_BIN" tools/ops/run_local_model_stage.py \
    --stage text \
    --config "$CONFIG_DIR" \
    --profile "$PROFILE" \
    --prepare-only
  restore_status=$?
  set -e
  if (( restore_status != 0 )); then
    echo "Warning: failed to restore the local text model pool after audio narration." >&2
  fi
}

cd "$ROOT_DIR"
trap restore_text_pool EXIT
"$PYTHON_BIN" tools/ops/run_local_model_stage.py \
  --stage text \
  --config "$CONFIG_DIR" \
  --profile "$PROFILE" \
  -- "$PYTHON_BIN" tools/pipelines/generate_audio_narration.py "$RUN_DIR" \
    --profile "$PROFILE" \
    --config "$CONFIG_DIR" \
    --skip-tts

if [[ "${VIDEO_ANALYZER_TTS_ROUTE:-local}" != "cloud_fallback" ]]; then
  "$PYTHON_BIN" tools/ops/run_local_model_stage.py \
    --stage tts \
    --config "$CONFIG_DIR" \
    --profile "$PROFILE" \
    --prepare-only
else
  echo "[audio-narration] using cloud TTS fallback; skipping local TTS preparation"
fi

"$PYTHON_BIN" tools/pipelines/generate_audio_narration.py "$RUN_DIR" \
  --profile "$PROFILE" \
  --config "$CONFIG_DIR" \
  --render-only
