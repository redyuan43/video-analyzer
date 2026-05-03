#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-local_lan}"
URL="${1:-https://www.bilibili.com/video/BV1prXyYMEjL/?spm_id_from=333.788.recommend_more_video.1&trackid=web_related_0.router-related-2479604-ptszh.1777734943162.1017&vd_source=70e95bad7ca28ab5623ab4b95161d8c2}"
LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT

# Full local-lan operation-manual pipeline: download, page context, ASR, OCR, VL, manual.
tools/run_operation_manual_from_url.sh \
  "$URL" \
  --profile "$PROFILE" \
  --cookies-from-browser chrome | tee "$LOG_FILE"

RUN_DIR="$(
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
  || true
)"

if [[ -z "$RUN_DIR" ]]; then
  echo "Operation-manual run completed, but no [done] run_dir marker was found." >&2
  exit 1
fi

# Follow-up multi-round document analysis from the generated operation-manual run.
tools/run_multidoc_analysis.sh "$RUN_DIR" --profile "$PROFILE"

# Final 30-second spoken recap synthesized by the AGX local TTS service.
tools/generate_30s_agx_tts.sh "$RUN_DIR" --profile "$PROFILE"
