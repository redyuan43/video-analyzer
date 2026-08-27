#!/usr/bin/env bash
# Backward-compatible shim for tools/publish/export_video_docs.sh.
# Kept so existing callers using tools/export_video_docs.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/publish/export_video_docs.sh" "$@"
