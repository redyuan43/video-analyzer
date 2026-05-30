#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: tools/run_long_talk_fast_from_url.sh URL [extra run_operation_manual_from_url args...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

tools/start_jetson_frame_ray.sh
ACTIVE_HOSTS_FILE="${JETSON_RAY_ACTIVE_HOSTS_FILE:-tmp/video-link-status/jetson-ray-active-hosts}"
JETSON_FRAME_HOSTS="${JETSON_FRAME_HOSTS:-agx,agx}"
if [[ -s "$ACTIVE_HOSTS_FILE" ]]; then
  JETSON_FRAME_HOSTS="$(<"$ACTIVE_HOSTS_FILE")"
fi
echo "[jetson-ray] using frame hosts: $JETSON_FRAME_HOSTS"

exec tools/run_operation_manual_from_url.sh "$1" \
  --pipeline-mode fast \
  --candidate-frames auto \
  --ocr-keyframe-strategy scan-text \
  --ocr-keyframe-budget auto \
  --ocr-scan-sample-fps 0.5 \
  --frame-extractor jetson \
  --jetson-frame-hosts "$JETSON_FRAME_HOSTS" \
  --jetson-frame-backend ray \
  --jetson-sample-fps 0.5 \
  --jetson-require-hwdec \
  --vl-frame-policy none \
  --prefer-subtitle-transcript \
  --refresh-context \
  --include-subtitles \
  --include-comments \
  --subtitle-langs en-GB,en-US,en,zh-CN,zh-Hans,zh \
  --max-comments 3000 \
  "${@:2}"
