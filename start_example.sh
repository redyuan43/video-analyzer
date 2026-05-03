#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-local_lan}"
RUN_DIR="downloads/url-videos/BV1EGdrBQEVN/operation-manual"

# Full local-lan operation-manual pipeline: download, page context, ASR, OCR, VL, manual.
tools/run_operation_manual_from_url.sh \
  'https://www.bilibili.com/video/BV1prXyYMEjL/?spm_id_from=333.788.recommend_more_video.1&trackid=web_related_0.router-related-2479604-ptszh.1777734943162.1017&vd_source=70e95bad7ca28ab5623ab4b95161d8c2' \
  --profile "$PROFILE" \
  --cookies-from-browser chrome

# Follow-up multi-round document analysis from the generated operation-manual run.
tools/run_multidoc_analysis.sh "$RUN_DIR" --profile "$PROFILE"
