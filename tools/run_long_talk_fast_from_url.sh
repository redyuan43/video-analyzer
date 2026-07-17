#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: tools/run_long_talk_fast_from_url.sh URL [extra run_operation_manual_from_url args...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTIVE_HOSTS_FILE="${JETSON_RAY_ACTIVE_HOSTS_FILE:-tmp/video-link-status/jetson-ray-active-hosts}"
JETSON_FRAME_HOSTS="${JETSON_FRAME_HOSTS:-agx,agx}"
JETSON_FRAME_BACKEND="${JETSON_FRAME_BACKEND:-ray}"
extra_args=("${@:2}")
for ((index = 0; index < ${#extra_args[@]}; index++)); do
  case "${extra_args[$index]}" in
    --jetson-frame-hosts)
      JETSON_FRAME_HOSTS="${extra_args[$((index + 1))]:-$JETSON_FRAME_HOSTS}"
      ;;
    --jetson-frame-hosts=*)
      JETSON_FRAME_HOSTS="${extra_args[$index]#*=}"
      ;;
    --jetson-frame-backend)
      JETSON_FRAME_BACKEND="${extra_args[$((index + 1))]:-$JETSON_FRAME_BACKEND}"
      ;;
    --jetson-frame-backend=*)
      JETSON_FRAME_BACKEND="${extra_args[$index]#*=}"
      ;;
  esac
done

if [[ "$JETSON_FRAME_BACKEND" == "ray" ]]; then
  tools/start_jetson_frame_ray.sh
fi
if [[ "$JETSON_FRAME_BACKEND" == "ray" && -s "$ACTIVE_HOSTS_FILE" ]]; then
  JETSON_FRAME_HOSTS="$(<"$ACTIVE_HOSTS_FILE")"
fi
echo "[jetson-frame] backend=$JETSON_FRAME_BACKEND hosts=$JETSON_FRAME_HOSTS"

exec tools/run_operation_manual_from_url.sh "$1" \
  --pipeline-mode fast \
  --candidate-frames auto \
  --ocr-keyframe-strategy scan-text \
  --ocr-keyframe-budget auto \
  --ocr-scan-sample-fps 0.5 \
  --frame-extractor jetson \
  --jetson-frame-hosts "$JETSON_FRAME_HOSTS" \
  --jetson-frame-backend "$JETSON_FRAME_BACKEND" \
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
