#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-spark}"
RAW_URL="${1:-https://www.bilibili.com/video/BV1prXyYMEjL/}"
LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT

URL="$(
  python3 - "$RAW_URL" <<'PY'
import re
import sys
from urllib.parse import urlsplit

raw_url = sys.argv[1].strip()
parsed = urlsplit(raw_url)
host = parsed.netloc.lower()
match = re.search(r"/video/(BV[0-9A-Za-z]+)", parsed.path)
if host.endswith("bilibili.com") and match:
    print(f"{parsed.scheme or 'https'}://{parsed.netloc}/video/{match.group(1)}/")
else:
    print(raw_url)
PY
)"

if [[ "$URL" != "$RAW_URL" ]]; then
  echo "Using canonical Bilibili URL: $URL" >&2
  echo "Tip: quote full share URLs that contain &: ./start_example.sh 'https://...?a=1&b=2'" >&2
fi

# Full Spark operation-manual pipeline: download, page context, ASR, OCR, VL, manual.
tools/run_operation_manual_from_url.sh \
  "$URL" \
  --profile "$PROFILE" \
  --cookies-from-browser chrome | tee "$LOG_FILE"

RUN_DIR="$(
  {
    python3 - "$LOG_FILE" <<'PY'
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
    if line.startswith("[done] run_dir: "):
        print(line.split(": ", 1)[1].strip())
        break
else:
    raise SystemExit(1)
PY
  } || true
)"

if [[ -z "$RUN_DIR" ]]; then
  echo "Operation-manual run completed, but no [done] run_dir marker was found." >&2
  exit 1
fi

# Follow-up multi-round document analysis from the generated operation-manual run.
tools/run_multidoc_analysis.sh "$RUN_DIR" --profile "$PROFILE"

# Final full narration script and WAV synthesized by the Ivan Qwen3-TTS gateway.
tools/generate_audio_narration.sh "$RUN_DIR" --profile "$PROFILE"
