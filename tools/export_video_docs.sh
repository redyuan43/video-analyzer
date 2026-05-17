#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: tools/export_video_docs.sh <operation-manual-run-dir> [export-dir] [--final-only] [--jobs N]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

RUN_DIR="$(realpath "$1")"
shift
EXPORT_DIR="$RUN_DIR/exports"
FINAL_ONLY=0
JOBS=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --final-only)
      FINAL_ONLY=1
      shift
      ;;
    --jobs)
      JOBS="${2:-}"
      if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
        echo "--jobs requires a positive integer" >&2
        exit 2
      fi
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#--jobs=}"
      if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
        echo "--jobs requires a positive integer" >&2
        exit 2
      fi
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      EXPORT_DIR="$1"
      shift
      ;;
  esac
done
EXPORT_DIR="$(realpath -m "$EXPORT_DIR")"

PDF_SCRIPT="${PDF_SCRIPT:-/home/ivan/github/my-skills-repo/markdown-to-pdf-cli/scripts/md_to_pdf.sh}"
PNG_SCRIPT="${PNG_SCRIPT:-/home/ivan/github/my-skills-repo/markdown-to-longpng/scripts/md_to_longpng.sh}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-$ROOT_DIR/tools/prepare_video_doc_export.py}"
PREPARE_SCRIPT="$(realpath -m "$PREPARE_SCRIPT")"

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
if [ ! -f "$PREPARE_SCRIPT" ]; then
  echo "Export prepare script not found: $PREPARE_SCRIPT" >&2
  exit 1
fi

mkdir -p "$EXPORT_DIR"
find "$EXPORT_DIR" -maxdepth 1 -type f \( -name '*.pdf' -o -name '*.long.png' \) -delete

final_docs=(
  "operation_manual.md"
  "docs_analysis_chapters/knowledge_notes_v2.md"
  "docs_analysis_chapters/deep_report_v2.md"
  "manual_evidence.md"
)

docs=("${final_docs[@]}")
if [ "$FINAL_ONLY" -ne 1 ]; then
  docs=("${final_docs[@]}")
fi

export_one() {
  local rel="$1"
  local input="$RUN_DIR/$rel"
  if [ ! -f "$input" ]; then
    echo "[skip missing] $rel" >&2
    return 0
  fi
  local name
  name="$(basename "$input" .md)"
  local prepared="$RUN_DIR/.$name.export.md"
  cleanup_prepared() {
    rm -f "$RUN_DIR/.$name.export.md" "$RUN_DIR/.$name.export."*.md "$RUN_DIR/..$name.export."*.md
  }
  "$PYTHON_BIN" "$PREPARE_SCRIPT" "$RUN_DIR" "$input" "$prepared" >/dev/null
  trap cleanup_prepared RETURN
  echo "[pdf] $rel"
  "$PDF_SCRIPT" "$prepared" "$EXPORT_DIR/$name.pdf"
  echo "[longpng] $rel"
    LONGPNG_VIEWPORT_SIZE="${LONGPNG_VIEWPORT_SIZE:-1600,1000}" \
    LONGPNG_NO_MARGIN="${LONGPNG_NO_MARGIN:-1}" \
    LONGPNG_CONTENT_PADDING="${LONGPNG_CONTENT_PADDING:-5}" \
    "$PNG_SCRIPT" "$prepared" "$EXPORT_DIR/$name.long.png"
  cleanup_prepared
  trap - RETURN
}

active=0
for rel in "${docs[@]}"; do
  if [ "$JOBS" -le 1 ]; then
    export_one "$rel"
    continue
  fi
  export_one "$rel" &
  active=$((active + 1))
  if [ "$active" -ge "$JOBS" ]; then
    wait -n
    active=$((active - 1))
  fi
done
wait

find "$EXPORT_DIR" -maxdepth 1 -type f -printf '%s %p\n' | sort -n
