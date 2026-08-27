#!/usr/bin/env bash
# Backward-compatible shim for tools/ops/start_qwen3_asr_p40_service.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ops/start_qwen3_asr_p40_service.sh" "$@"
