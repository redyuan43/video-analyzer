#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: tools/run_long_talk_fast_from_url.sh URL [extra run_operation_manual_from_url args...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

tools/start_jetson_frame_ray.sh

exec tools/run_operation_manual_from_url.sh "$1" \
  --profile ivan_minicpm_v100 \
  --pipeline-mode fast \
  --candidate-frames auto \
  --max-frames 48 \
  --frame-extractor jetson \
  --jetson-frame-hosts nx1,nx2,nx3,nx4,agx \
  --jetson-frame-backend ray \
  --jetson-sample-fps 0.5 \
  --jetson-require-hwdec \
  --vl-frame-policy none \
  --prefer-subtitle-transcript \
  --refresh-context \
  --include-subtitles \
  --include-comments \
  --subtitle-langs en-GB,en-US,en,zh-CN,zh-Hans,zh \
  --max-comments 30 \
  "${@:2}"
