#!/usr/bin/env bash
# Generate VideoAnalyzer.xcodeproj from project.yml.
#
# Run this on the Mac after cloning, and again any time project.yml changes or
# a Swift file is added or removed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "xcodegen not found."
  if command -v brew >/dev/null 2>&1; then
    echo "Installing with Homebrew..."
    brew install xcodegen
  else
    echo "Install Homebrew first (https://brew.sh), then: brew install xcodegen" >&2
    exit 1
  fi
fi

if grep -qE '^DEVELOPMENT_TEAM[[:space:]]*=[[:space:]]*$' Config/App.xcconfig; then
  echo "warning: DEVELOPMENT_TEAM is empty in Config/App.xcconfig." >&2
  echo "         Signing will fail until you set it (and PRODUCT_BUNDLE_PREFIX)." >&2
fi

xcodegen generate

echo
echo "Generated VideoAnalyzer.xcodeproj"
echo "Next: open VideoAnalyzer.xcodeproj, pick your iPhone, and Run."
