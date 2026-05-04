#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR=""
PROFILE="${PROFILE:-spark}"
AGX_HOST="${AGX_HOST:-agx}"
SPEAKER="${SPEAKER:-vivian}"
TARGET_DURATION="${TARGET_DURATION:-30}"
MIN_DURATION="${MIN_DURATION:-27}"
MAX_DURATION="${MAX_DURATION:-33}"
OUT_BASENAME="${OUT_BASENAME:-video_30s_summary_agx}"
TEXT_TIMEOUT="${TEXT_TIMEOUT:-180}"
TTS_TIMEOUT="${TTS_TIMEOUT:-240}"

usage() {
  cat >&2 <<'EOF'
Usage:
  tools/generate_30s_agx_tts.sh RUN_DIR [--profile PROFILE]

Environment:
  AGX_HOST=agx              SSH alias for the AGX host
  SPEAKER=vivian            CapsWriter/Qwen3-TTS speaker id
  TARGET_DURATION=30        Desired final duration in seconds
  MIN_DURATION=27           Lower acceptable duration before adjustment
  MAX_DURATION=33           Upper acceptable duration before adjustment
  OUT_BASENAME=video_30s_summary_agx
  TEXT_TIMEOUT=180        Max seconds to wait for LLM recap text
  TTS_TIMEOUT=240         Max seconds to wait for AGX TTS synthesis

Outputs:
  RUN_DIR/$OUT_BASENAME.txt
  RUN_DIR/$OUT_BASENAME.raw.wav
  RUN_DIR/$OUT_BASENAME.wav
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:?missing value for --profile}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ -z "$RUN_DIR" ]]; then
        RUN_DIR="$1"
      else
        echo "Unexpected positional argument: $1" >&2
        usage
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  usage
  exit 2
fi

cd "$ROOT_DIR"
RUN_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$RUN_DIR")"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd ssh
require_cmd scp
require_cmd python3
require_cmd timeout
require_cmd ffmpeg
require_cmd ffprobe

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory does not exist: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
TEXT_PATH="$RUN_DIR/$OUT_BASENAME.txt"
RAW_WAV="$RUN_DIR/$OUT_BASENAME.raw.wav"
FINAL_WAV="$RUN_DIR/$OUT_BASENAME.wav"

QUESTION="请基于这些视频分析文档，生成一段可以直接给人听的中文30秒口播稿。只输出口播正文，不要标题、项目符号、引用、Markdown 或解释。语气自然，长度控制在约80到110个汉字。"

echo "[agx-tts] generating spoken recap text"
if timeout "$TEXT_TIMEOUT" tools/ask_video_docs.sh "$RUN_DIR" "$QUESTION" --profile "$PROFILE" >"$TEXT_PATH.tmp" 2>"$TEXT_PATH.ask.log"; then
  python3 - "$TEXT_PATH.tmp" "$TEXT_PATH" <<'PY'
from pathlib import Path
import re
import sys

raw = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
text = raw.strip()
text = re.sub(r"^```(?:\w+)?\s*", "", text)
text = re.sub(r"\s*```$", "", text)
text = re.sub(r"^(口播稿|口播正文|30秒口播稿|摘要)[:：]\s*", "", text.strip())
text = re.sub(r"\s+", " ", text).strip(" \n\t\"'“”")
Path(sys.argv[2]).write_text(text + "\n", encoding="utf-8")
PY
else
  echo "[agx-tts] ask_video_docs failed; using deterministic fallback text" >&2
  python3 - "$RUN_DIR" "$TEXT_PATH" <<'PY'
from pathlib import Path
import re
import sys

run_dir = Path(sys.argv[1])
manual = ""
for name in ("docs_analysis/deep_report.md", "operation_manual.md", "transcript.md"):
    path = run_dir / name
    if path.exists():
        manual = path.read_text(encoding="utf-8", errors="replace")
        break

clean = re.sub(r"```.*?```", "", manual, flags=re.S)
clean = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", clean)
clean = re.sub(r"`([^`]+)`", r"\1", clean)
clean = re.sub(r"[#>*_\-|]+", " ", clean)
sentences = re.split(r"(?<=[。！？!?])\s*", clean)
picked = [s.strip() for s in sentences if 18 <= len(s.strip()) <= 90][:3]
if picked:
    text = " ".join(picked)
else:
    text = "这段视频主要介绍一个技术主题的核心背景、关键参数和使用价值。重点不是表面的展示，而是帮助听众快速理解它解决了什么问题、适合什么场景，以及实际使用前需要复核哪些限制。"
if len(text) > 120:
    text = text[:118].rstrip("，,；;、 ") + "。"
Path(sys.argv[2]).write_text(text + "\n", encoding="utf-8")
PY
fi

TEXT_CHARS="$(python3 -c 'from pathlib import Path; import sys; print(len(Path(sys.argv[1]).read_text(encoding="utf-8").strip()))' "$TEXT_PATH")"
if [[ "$TEXT_CHARS" -lt 20 ]]; then
  echo "Generated recap text is too short; see $TEXT_PATH" >&2
  exit 1
fi

echo "[agx-tts] checking AGX TTS health on $AGX_HOST"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$AGX_HOST" \
  'curl -sS --max-time 10 http://127.0.0.1:8002/api/health >/tmp/agx_tts_health.json'

echo "[agx-tts] ensuring AGX TTS model is loaded"
ssh -o BatchMode=yes "$AGX_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
health="$(cat /tmp/agx_tts_health.json 2>/dev/null || curl -sS http://127.0.0.1:8002/api/health)"
loaded="$(printf "%s" "$health" | grep -o "\"tts_model_loaded\":[^,}]*" | grep -o "true\|false" || true)"
workers="$(printf "%s" "$health" | grep -o "\"tts_parallel_workers_ready\":[0-9]*" | grep -o "[0-9]*$" || true)"
if [[ "$loaded" != "true" || "${workers:-0}" -lt 1 ]]; then
  curl -sS -X POST http://127.0.0.1:8002/api/tts/load >/tmp/agx_tts_load.json
fi
for _ in $(seq 1 40); do
  h="$(curl -sS http://127.0.0.1:8002/api/health)"
  loaded="$(printf "%s" "$h" | grep -o "\"tts_model_loaded\":[^,}]*" | grep -o "true\|false" || true)"
  workers="$(printf "%s" "$h" | grep -o "\"tts_parallel_workers_ready\":[0-9]*" | grep -o "[0-9]*$" || true)"
  if [[ "$loaded" = "true" && "${workers:-0}" -gt 0 ]]; then
    printf "%s\n" "$h" >/tmp/agx_tts_ready.json
    exit 0
  fi
  sleep 3
done
cat /tmp/agx_tts_ready.json /tmp/agx_tts_load.json /tmp/agx_tts_health.json 2>/dev/null || true
exit 1
REMOTE

REMOTE_TEXT="/tmp/$OUT_BASENAME.txt"
REMOTE_WAV="/tmp/$OUT_BASENAME.wav"

echo "[agx-tts] synthesizing on AGX"
scp -q "$TEXT_PATH" "$AGX_HOST:$REMOTE_TEXT"
ssh -o BatchMode=yes "$AGX_HOST" \
  "REMOTE_TEXT='$REMOTE_TEXT' REMOTE_WAV='$REMOTE_WAV' SPEAKER='$SPEAKER' TTS_TIMEOUT='$TTS_TIMEOUT' bash -s" <<'REMOTE'
set -euo pipefail
payload="$(python3 -c 'import json, os, pathlib; text = pathlib.Path(os.environ["REMOTE_TEXT"]).read_text(encoding="utf-8"); print(json.dumps({"text": text, "speaker": os.environ["SPEAKER"], "speed": 1.0}, ensure_ascii=False))')"
printf "%s" "$payload" | curl -sS --max-time "$TTS_TIMEOUT" -X POST http://127.0.0.1:8002/api/tts/speak \
  -H "Content-Type: application/json" \
  --data-binary @- \
  -o "$REMOTE_WAV"
file "$REMOTE_WAV" | grep -q 'WAVE audio'
REMOTE
scp -q "$AGX_HOST:$REMOTE_WAV" "$RAW_WAV"
cp "$RAW_WAV" "$FINAL_WAV"

duration_of() {
  ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$1"
}

RAW_DURATION="$(duration_of "$RAW_WAV")"
echo "[agx-tts] raw duration: ${RAW_DURATION}s"

NEEDS_ADJUST="$(python3 - "$RAW_DURATION" "$MIN_DURATION" "$MAX_DURATION" <<'PY'
import sys
d, lo, hi = map(float, sys.argv[1:4])
print("1" if d < lo or d > hi else "0")
PY
)"

if [[ "$NEEDS_ADJUST" = "1" ]]; then
  ATEMPO="$(python3 - "$RAW_DURATION" "$TARGET_DURATION" <<'PY'
import sys
d = float(sys.argv[1])
target = float(sys.argv[2])
ratio = d / target if target else 1.0
ratio = max(0.5, min(2.0, ratio))
print(f"{ratio:.6f}")
PY
)"
  echo "[agx-tts] adjusting duration with atempo=$ATEMPO"
  ffmpeg -y -hide_banner -loglevel error -i "$RAW_WAV" -filter:a "atempo=$ATEMPO" "$FINAL_WAV"
fi

FINAL_DURATION="$(duration_of "$FINAL_WAV")"
echo "[agx-tts] final duration: ${FINAL_DURATION}s"

python3 - "$FINAL_DURATION" "$MIN_DURATION" "$MAX_DURATION" "$FINAL_WAV" <<'PY'
import sys
d, lo, hi = map(float, sys.argv[1:4])
path = sys.argv[4]
if not (lo <= d <= hi):
    raise SystemExit(f"Final duration {d:.3f}s outside expected range {lo:.0f}-{hi:.0f}s: {path}")
PY

echo "[agx-tts] text: $TEXT_PATH"
echo "[agx-tts] raw wav: $RAW_WAV"
echo "[agx-tts] final wav: $FINAL_WAV"
