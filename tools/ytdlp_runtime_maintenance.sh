#!/usr/bin/env bash
# Backward-compatible shim for tools/ops/ytdlp_runtime_maintenance.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ops/ytdlp_runtime_maintenance.sh" "$@"
