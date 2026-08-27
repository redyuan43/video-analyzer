#!/usr/bin/env bash
# Backward-compatible shim for tools/pipelines/chat_with_video_docs.sh.
# Kept so existing callers using tools/chat_with_video_docs.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipelines/chat_with_video_docs.sh" "$@"
