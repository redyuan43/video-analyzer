#!/usr/bin/env bash
# Backward-compatible shim for tools/ops/prepare_ai_local_model_stage.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ops/prepare_ai_local_model_stage.sh" "$@"
