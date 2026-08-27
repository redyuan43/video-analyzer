#!/usr/bin/env bash
# Backward-compatible shim for tools/asr_servers/generate_30s_agx_tts.sh.
# Kept so existing callers using tools/generate_30s_agx_tts.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_servers/generate_30s_agx_tts.sh" "$@"
