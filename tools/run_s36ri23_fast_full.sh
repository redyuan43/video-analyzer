#!/usr/bin/env bash
# Backward-compatible shim for tools/pipelines/run_s36ri23_fast_full.sh.
# Kept so existing callers using tools/run_s36ri23_fast_full.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipelines/run_s36ri23_fast_full.sh" "$@"
