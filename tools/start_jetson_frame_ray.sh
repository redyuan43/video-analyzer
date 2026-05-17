#!/usr/bin/env bash
set -euo pipefail

NETWORK="${JETSON_RAY_NETWORK:-tailscale}"
HEAD_HOST="${JETSON_RAY_HEAD_HOST:-agx}"
RAY_PORT="${JETSON_RAY_PORT:-6379}"
READY_TIMEOUT="${JETSON_RAY_READY_TIMEOUT:-90}"
SSH_TIMEOUT="${JETSON_RAY_SSH_TIMEOUT:-5}"
ACTIVE_HOSTS_FILE="${JETSON_RAY_ACTIVE_HOSTS_FILE:-tmp/video-link-status/jetson-ray-active-hosts}"

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

case "$NETWORK" in
  tailscale)
    HEAD_IP="${JETSON_RAY_HEAD_IP:-100.103.199.121}"
    WORKERS=(
      "nx1:100.119.5.57:1"
      "nx2:100.123.222.45:1"
      "nx3:100.127.71.86:1"
      "nx4:100.82.227.71:1"
    )
    ;;
  lan)
    HEAD_IP="${JETSON_RAY_HEAD_IP:-192.168.31.201}"
    WORKERS=(
      "nx1:192.168.31.40:1"
      "nx2:192.168.31.68:1"
      "nx3:192.168.31.35:1"
      "nx4:192.168.31.10:1"
    )
    ;;
  *)
    echo "JETSON_RAY_NETWORK must be tailscale or lan, got: $NETWORK" >&2
    exit 2
    ;;
esac

if [[ -n "${JETSON_RAY_WORKERS:-}" ]]; then
  IFS=, read -r -a WORKERS <<<"$JETSON_RAY_WORKERS"
fi

ADDRESS="${HEAD_IP}:${RAY_PORT}"
ACTIVE_WORKERS=()

required_resources=(
  "host_agx"
  "frame_worker"
)
for item in "${ACTIVE_WORKERS[@]}"; do
  host="${item%%:*}"
  required_resources+=("host_$host")
done

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

build_required_resources() {
  required_resources=(
    "host_agx"
    "frame_worker"
  )
  local item host
  for item in "${ACTIVE_WORKERS[@]}"; do
    host="${item%%:*}"
    required_resources+=("host_$host")
  done
}

host_reachable() {
  local host="$1"
  ssh -o BatchMode=yes -o ConnectTimeout="$SSH_TIMEOUT" "$host" true >/dev/null 2>&1
}

select_active_workers() {
  ACTIVE_WORKERS=()
  local item host
  for item in "${WORKERS[@]}"; do
    host="${item%%:*}"
    if host_reachable "$host"; then
      ACTIVE_WORKERS+=("$item")
    else
      echo "[jetson-ray] skipping offline worker: $host" >&2
    fi
  done
  build_required_resources
}

active_hosts_csv() {
  local hosts=("$HEAD_HOST")
  local item host
  for item in "${ACTIVE_WORKERS[@]}"; do
    host="${item%%:*}"
    hosts+=("$host")
  done
  local IFS=,
  printf '%s\n' "${hosts[*]}"
}

write_active_hosts() {
  mkdir -p "$(dirname "$ACTIVE_HOSTS_FILE")"
  active_hosts_csv >"$ACTIVE_HOSTS_FILE"
  echo "[jetson-ray] active hosts: $(cat "$ACTIVE_HOSTS_FILE")"
}

wait_cluster_ready() {
  local status
  local deadline=$((SECONDS + READY_TIMEOUT))
  while true; do
    status="$(ray_status)"
    if cluster_ready "$status"; then
      printf '%s\n' "$status"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      printf '%s\n' "$status"
      return 1
    fi
    sleep 3
  done
}

start_head() {
  local resources
  resources="$(resource_json "$HEAD_HOST" 2)"
  ssh -o BatchMode=yes "$HEAD_HOST" \
    "PATH=\$HOME/.local/bin:\$PATH ${NO_PROXY_ENV[*]} ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT --dashboard-host=0.0.0.0 --resources='$resources' --disable-usage-stats"
}

start_worker() {
  local host="$1"
  local ip="$2"
  local frame_workers="$3"
  local resources
  resources="$(resource_json "$host" "$frame_workers")"
  ssh -o BatchMode=yes "$host" \
    "PATH=\$HOME/.local/bin:\$PATH ${NO_PROXY_ENV[*]} ray start --address=$ADDRESS --node-ip-address=$ip --resources='$resources' --disable-usage-stats"
}

resource_json() {
  local host="$1"
  local frame_workers="$2"
  printf '{"host_%s":1,"frame_worker":%s}' "$host" "$frame_workers"
}

stop_all() {
  local item host
  for item in "${ACTIVE_WORKERS[@]}" "$HEAD_HOST:$HEAD_IP:2"; do
    host="${item%%:*}"
    ssh -o BatchMode=yes "$host" "PATH=\$HOME/.local/bin:\$PATH ray stop -f >/dev/null 2>&1 || true" &
  done
  wait
}

if ! host_reachable "$HEAD_HOST"; then
  echo "[jetson-ray] Ray head is offline: $HEAD_HOST" >&2
  exit 1
fi

select_active_workers
status="$(ray_status)"
if cluster_ready "$status"; then
  echo "[jetson-ray] existing cluster is ready"
  write_active_hosts
  exit 0
fi

echo "[jetson-ray] starting AGX Ray head and NX workers over $NETWORK ($ADDRESS)"
stop_all
start_head
for item in "${ACTIVE_WORKERS[@]}"; do
  IFS=: read -r host ip frame_workers <<<"$item"
  start_worker "$host" "$ip" "$frame_workers"
done

if ! status="$(wait_cluster_ready)"; then
  echo "$status" >&2
  echo "[jetson-ray] cluster did not expose all required resources" >&2
  exit 1
fi

write_active_hosts
echo "$status" | sed -n '1,120p'
