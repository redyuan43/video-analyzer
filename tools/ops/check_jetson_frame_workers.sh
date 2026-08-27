#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -gt 0 ]]; then
  HOSTS=("$@")
else
  HOSTS=(agx)
fi

check_host() {
  local host="$1"
  local ssh_args=("$host")
  if [[ "$host" == "nx3" ]]; then
    ssh_args=(-o ProxyCommand=none nx@nx3.taild500c8.ts.net)
  fi

  echo "== $host =="
  ssh "${ssh_args[@]}" 'set -e
    echo "hostname=$(hostname)"
    printf "ffmpeg="; command -v ffmpeg || true
    printf "ffprobe="; command -v ffprobe || true
    printf "rsync="; command -v rsync || true
    python3 - << "PY"
import importlib.util
for name in ["cv2", "numpy", "PIL", "ray"]:
    print(f"{name}={bool(importlib.util.find_spec(name))}")
PY
    if command -v gst-inspect-1.0 >/dev/null 2>&1; then
      for plugin in nvv4l2decoder nvvidconv nvjpegenc h264parse avdec_h264 jpegenc multifilesink; do
        gst-inspect-1.0 "$plugin" >/dev/null 2>&1 && echo "$plugin=true" || echo "$plugin=false"
      done
    else
      echo "gst-inspect-1.0=false"
    fi'
}

for host in "${HOSTS[@]}"; do
  check_host "$host"
done
