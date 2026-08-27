#!/usr/bin/env bash
# Backward-compatible shim for tools/pipelines/run_operation_manual_from_url.sh.
# Kept so existing callers using tools/run_operation_manual_from_url.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipelines/run_operation_manual_from_url.sh" "$@"
