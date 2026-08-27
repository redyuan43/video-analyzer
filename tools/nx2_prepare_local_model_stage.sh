#!/usr/bin/env bash
# Backward-compatible shim for tools/ops/nx2_prepare_local_model_stage.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ops/nx2_prepare_local_model_stage.sh" "$@"
