#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VIDEO_PATH="${VIDEO_PATH:-downloads/url-videos/S36ri23-l60/video.mp4}"
CONTEXT_FILE="${CONTEXT_FILE:-downloads/url-videos/S36ri23-l60/page_context.md}"
RUN_NAME="${RUN_NAME:-operation-manual-fast-full-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-downloads/url-videos/S36ri23-l60/${RUN_NAME}}"
OCR_CACHE="${OCR_CACHE:-on}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video not found: $VIDEO_PATH" >&2
  exit 1
fi

if [[ ! -f "$CONTEXT_FILE" ]]; then
  echo "Context file not found: $CONTEXT_FILE" >&2
  exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  echo "Set RUN_NAME or OUTPUT_DIR to a new path." >&2
  exit 1
fi

echo "[run] video: $VIDEO_PATH"
echo "[run] output: $OUTPUT_DIR"
echo "[run] mode: fast"
echo "[run] OCR cache: $OCR_CACHE"
echo "[run] frame extractor: jetson nx2,nx3"

".venv/bin/python" -m video_analyzer.cli "$VIDEO_PATH" \
  --task operation_manual \
  --output "$OUTPUT_DIR" \
  --context-file "$CONTEXT_FILE" \
  --pipeline-mode fast \
  --candidate-frames auto \
  --frame-extractor jetson \
  --jetson-frame-hosts "nx2,nx3" \
  --jetson-frame-backend auto \
  --jetson-sample-fps auto \
  --jetson-chunk-overlap-seconds 2 \
  --asr-provider vibevoice \
  --vibevoice-url "http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe" \
  --ocr-provider auto \
  --ocr-base-url "http://spark-31d6.taild500c8.ts.net:8000/v1" \
  --ocr-base-url "http://edgexpert-4353.taild500c8.ts.net:8000/v1" \
  --ocr-concurrency auto \
  --ocr-cache "$OCR_CACHE" \
  --llm-base-url "http://100.90.114.26:18081/v1" \
  --vision-base-url "http://100.96.79.21:18082/v1" \
  --text-base-url "http://100.90.114.26:18081/v1" \
  --vision-model "minicpm-v-4.5-v100" \
  --text-model "hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive" \
  --manual-language zh-CN \
  --keep-frames \
  --log-level INFO

echo
echo "[done] analysis: $OUTPUT_DIR/analysis.json"
echo "[done] manual: $OUTPUT_DIR/operation_manual.md"
echo
echo "[timings]"
jq '.metadata.timings, .metadata.ocr, .metadata.frame_selection' "$OUTPUT_DIR/analysis.json"
