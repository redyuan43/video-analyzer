#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: tools/export_video_docs.sh <operation-manual-run-dir> [export-dir]" >&2
  exit 2
fi

RUN_DIR="$(realpath "$1")"
EXPORT_DIR="${2:-$RUN_DIR/exports}"
EXPORT_DIR="$(realpath -m "$EXPORT_DIR")"

PDF_SCRIPT="${PDF_SCRIPT:-/home/ivan/github/my-skills-repo/markdown-to-pdf-cli/scripts/md_to_pdf.sh}"
PNG_SCRIPT="${PNG_SCRIPT:-/home/ivan/github/my-skills-repo/markdown-to-longpng/scripts/md_to_longpng.sh}"

if [ ! -d "$RUN_DIR" ]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi
if [ ! -x "$PDF_SCRIPT" ]; then
  echo "PDF converter not executable: $PDF_SCRIPT" >&2
  exit 1
fi
if [ ! -x "$PNG_SCRIPT" ]; then
  echo "Long PNG converter not executable: $PNG_SCRIPT" >&2
  exit 1
fi

mkdir -p "$EXPORT_DIR"

docs=(
  "operation_manual.md"
  "docs_analysis/knowledge_notes.md"
  "docs_analysis/deep_report.md"
  "manual_evidence.md"
)

for rel in "${docs[@]}"; do
  input="$RUN_DIR/$rel"
  if [ ! -f "$input" ]; then
    echo "[skip missing] $rel" >&2
    continue
  fi
  name="$(basename "$input" .md)"
  echo "[pdf] $rel"
  "$PDF_SCRIPT" "$input" "$EXPORT_DIR/$name.pdf"
  echo "[longpng] $rel"
  "$PNG_SCRIPT" "$input" "$EXPORT_DIR/$name.long.png"
done

find "$EXPORT_DIR" -maxdepth 1 -type f -printf '%s %p\n' | sort -n
