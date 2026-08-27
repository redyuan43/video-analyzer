#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cat >&2 <<'EOF'
[deprecated] tools/generate_30s_agx_tts.sh no longer generates a 30-second AGX recap.
[deprecated] Forwarding to tools/pipelines/generate_audio_narration.sh for full Markdown narration + Ivan TTS WAV.
EOF

exec "$ROOT_DIR/tools/pipelines/generate_audio_narration.sh" "$@"
