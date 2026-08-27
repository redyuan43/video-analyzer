#!/usr/bin/env bash
# Backward-compatible shim for tools/pipelines/run_multidoc_analysis.sh.
# Kept so existing callers using tools/run_multidoc_analysis.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipelines/run_multidoc_analysis.sh" "$@"
