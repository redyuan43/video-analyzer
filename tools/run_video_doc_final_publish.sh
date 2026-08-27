#!/usr/bin/env bash
# Backward-compatible shim for tools/publish/run_video_doc_final_publish.sh.
# Kept so existing callers using tools/run_video_doc_final_publish.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/publish/run_video_doc_final_publish.sh" "$@"
