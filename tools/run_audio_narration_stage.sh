#!/usr/bin/env bash
# Backward-compatible shim for tools/pipelines/run_audio_narration_stage.sh.
# Kept so existing callers using tools/run_audio_narration_stage.sh keep working.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipelines/run_audio_narration_stage.sh" "$@"
