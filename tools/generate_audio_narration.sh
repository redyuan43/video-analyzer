#!/usr/bin/env bash
# Backward-compatible shim for tools/pipelines/generate_audio_narration.sh.
# Kept so existing callers using tools/generate_audio_narration.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipelines/generate_audio_narration.sh" "$@"
