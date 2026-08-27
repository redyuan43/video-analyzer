#!/usr/bin/env bash
# Backward-compatible shim for tools/pipelines/run_long_talk_fast_from_url.sh.
# Kept so existing callers using tools/run_long_talk_fast_from_url.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipelines/run_long_talk_fast_from_url.sh" "$@"
