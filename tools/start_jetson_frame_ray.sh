#!/usr/bin/env bash
# Backward-compatible shim for tools/ops/start_jetson_frame_ray.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ops/start_jetson_frame_ray.sh" "$@"
