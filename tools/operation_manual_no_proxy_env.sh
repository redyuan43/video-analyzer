#!/usr/bin/env bash
# Operation-manual runtime talks to LAN/Tailscale services directly.

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

export NO_PROXY="*"
export no_proxy="*"

