#!/usr/bin/env bash
set -euo pipefail

NETWORK="${JETSON_RAY_NETWORK:-lan}"
HEAD_HOST="${JETSON_RAY_HEAD_HOST:-agx}"
RAY_PORT="${JETSON_RAY_PORT:-6379}"
READY_TIMEOUT="${JETSON_RAY_READY_TIMEOUT:-90}"
SSH_TIMEOUT="${JETSON_RAY_SSH_TIMEOUT:-5}"
ACTIVE_HOSTS_FILE="${JETSON_RAY_ACTIVE_HOSTS_FILE:-tmp/video-link-status/jetson-ray-active-hosts}"
HEAD_FRAME_WORKERS="${JETSON_RAY_HEAD_FRAME_WORKERS:-2}"

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
    DEFAULT_HEAD_SSH="$HEAD_HOST"
    HEAD_IP="${JETSON_RAY_HEAD_IP:-100.103.199.121}"
    WORKERS=()
    ;;
  lan)
    DEFAULT_HEAD_SSH="agx@192.168.2.110"
    HEAD_IP="${JETSON_RAY_HEAD_IP:-192.168.2.110}"
    WORKERS=()
    ;;
  *)
    echo "JETSON_RAY_NETWORK must be tailscale or lan, got: $NETWORK" >&2
    exit 2
    ;;
esac

HEAD_SSH_TARGET="${JETSON_RAY_HEAD_SSH:-$DEFAULT_HEAD_SSH}"
HEAD_SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout="$SSH_TIMEOUT" -o ConnectionAttempts=1)
if [[ "$NETWORK" == "lan" ]]; then
  HEAD_SSH_OPTIONS+=(-o HostKeyAlias=agx-lan)
fi

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
  ssh "${HEAD_SSH_OPTIONS[@]}" "$HEAD_SSH_TARGET" "PATH=\$HOME/.local/bin:\$PATH ray status" 2>/dev/null || true
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
  local hosts=()
  local index
  for ((index = 0; index < HEAD_FRAME_WORKERS; index++)); do
    hosts+=("$HEAD_HOST")
  done
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
  resources="$(resource_json "$HEAD_HOST" "$HEAD_FRAME_WORKERS")"
  ssh "${HEAD_SSH_OPTIONS[@]}" "$HEAD_SSH_TARGET" \
    "PATH=\$HOME/.local/bin:\$PATH ${NO_PROXY_ENV[*]} ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT --dashboard-host=0.0.0.0 --resources='$resources' --disable-usage-stats"
}

start_worker() {
  local host="$1"
  local ip="$2"
  local frame_workers="$3"
  local resources
  resources="$(resource_json "$host" "$frame_workers")"
  ssh -o BatchMode=yes -o ConnectTimeout="$SSH_TIMEOUT" -o ConnectionAttempts=1 "$host" \
    "PATH=\$HOME/.local/bin:\$PATH ${NO_PROXY_ENV[*]} ray start --address=$ADDRESS --node-ip-address=$ip --resources='$resources' --disable-usage-stats"
}

resource_json() {
  local host="$1"
  local frame_workers="$2"
  printf '{"host_%s":1,"frame_worker":%s}' "$host" "$frame_workers"
}

stop_all() {
  local item host
  for item in "${ACTIVE_WORKERS[@]}"; do
    host="${item%%:*}"
    ssh -o BatchMode=yes -o ConnectTimeout="$SSH_TIMEOUT" -o ConnectionAttempts=1 "$host" "PATH=\$HOME/.local/bin:\$PATH ray stop -f >/dev/null 2>&1 || true" &
  done
  ssh "${HEAD_SSH_OPTIONS[@]}" "$HEAD_SSH_TARGET" "PATH=\$HOME/.local/bin:\$PATH ray stop -f >/dev/null 2>&1 || true" &
  wait
}

if ! ssh "${HEAD_SSH_OPTIONS[@]}" "$HEAD_SSH_TARGET" true >/dev/null 2>&1; then
  echo "[jetson-ray] Ray head is offline: $HEAD_SSH_TARGET" >&2
  exit 1
fi

select_active_workers
status="$(ray_status)"
if cluster_ready "$status"; then
  echo "[jetson-ray] existing cluster is ready"
  write_active_hosts
  exit 0
fi

echo "[jetson-ray] starting AGX Ray head and optional workers over $NETWORK ($ADDRESS)"
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
