#!/usr/bin/env bash
set -euo pipefail

HEAD_HOST="${JETSON_RAY_HEAD_HOST:-agx}"
HEAD_IP="${JETSON_RAY_HEAD_IP:-192.168.31.201}"
RAY_PORT="${JETSON_RAY_PORT:-6379}"
ADDRESS="${HEAD_IP}:${RAY_PORT}"

NO_PROXY_ENV=(
  env
  -u HTTP_PROXY
  -u HTTPS_PROXY
  -u ALL_PROXY
  -u http_proxy
  -u https_proxy
  -u all_proxy
  RAY_memory_usage_threshold=0.99
)

WORKERS=(
  "nx1:192.168.31.40:1"
  "nx2:192.168.31.68:1"
  "nx3:192.168.31.35:1"
  "nx4:192.168.31.10:1"
)

required_resources=(
  "host_agx"
  "host_nx1"
  "host_nx2"
  "host_nx3"
  "host_nx4"
  "frame_worker"
)

ray_status() {
  ssh -o BatchMode=yes "$HEAD_HOST" "PATH=\$HOME/.local/bin:\$PATH ray status" 2>/dev/null || true
}

cluster_ready() {
  local status="$1"
  [[ -n "$status" ]] || return 1
  for resource in "${required_resources[@]}"; do
    [[ "$status" == *"$resource"* ]] || return 1
  done
}

start_head() {
  ssh -o BatchMode=yes "$HEAD_HOST" \
    "PATH=\$HOME/.local/bin:\$PATH ${NO_PROXY_ENV[*]} ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT --dashboard-host=0.0.0.0 --resources='{\\\"host_agx\\\":1,\\\"frame_worker\\\":2}' --disable-usage-stats"
}

start_worker() {
  local host="$1"
  local ip="$2"
  local frame_workers="$3"
  ssh -o BatchMode=yes "$host" \
    "PATH=\$HOME/.local/bin:\$PATH ${NO_PROXY_ENV[*]} ray start --address=$ADDRESS --node-ip-address=$ip --resources='{\\\"host_$host\\\":1,\\\"frame_worker\\\":$frame_workers}' --disable-usage-stats"
}

stop_all() {
  local item host
  for item in "${WORKERS[@]}" "$HEAD_HOST:$HEAD_IP:2"; do
    host="${item%%:*}"
    ssh -o BatchMode=yes "$host" "PATH=\$HOME/.local/bin:\$PATH ray stop -f >/dev/null 2>&1 || true" &
  done
  wait
}

status="$(ray_status)"
if cluster_ready "$status"; then
  echo "[jetson-ray] existing cluster is ready"
  exit 0
fi

echo "[jetson-ray] starting AGX Ray head and NX workers"
stop_all
start_head
for item in "${WORKERS[@]}"; do
  IFS=: read -r host ip frame_workers <<<"$item"
  start_worker "$host" "$ip" "$frame_workers"
done

status="$(ray_status)"
if ! cluster_ready "$status"; then
  echo "$status" >&2
  echo "[jetson-ray] cluster did not expose all required resources" >&2
  exit 1
fi

echo "$status" | sed -n '1,120p'
