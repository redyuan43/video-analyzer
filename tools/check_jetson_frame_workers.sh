#!/usr/bin/env bash
# Backward-compatible shim for tools/ops/check_jetson_frame_workers.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ops/check_jetson_frame_workers.sh" "$@"
