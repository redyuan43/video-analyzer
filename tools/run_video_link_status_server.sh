#!/usr/bin/env bash
# Backward-compatible shim for tools/video_link/run_video_link_status_server.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/video_link/run_video_link_status_server.sh" "$@"
