#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: tools/export_video_docs.sh <operation-manual-run-dir> [export-dir] [--final-only] [--jobs N] [--long-png]" >&2
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
LONG_PNG=0
PNG_DPI=180
PNG_PADDING=24

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
    --long-png)
      LONG_PNG=1
      shift
      ;;
    --png-dpi)
      PNG_DPI="${2:-}"
      if ! [[ "$PNG_DPI" =~ ^[0-9]+$ ]] || [ "$PNG_DPI" -lt 72 ]; then
        echo "--png-dpi requires an integer >= 72" >&2
        exit 2
      fi
      shift 2
      ;;
    --png-dpi=*)
      PNG_DPI="${1#--png-dpi=}"
      if ! [[ "$PNG_DPI" =~ ^[0-9]+$ ]] || [ "$PNG_DPI" -lt 72 ]; then
        echo "--png-dpi requires an integer >= 72" >&2
        exit 2
      fi
      shift
      ;;
    --png-padding)
      PNG_PADDING="${2:-}"
      if ! [[ "$PNG_PADDING" =~ ^[0-9]+$ ]]; then
        echo "--png-padding requires a non-negative integer" >&2
        exit 2
      fi
      shift 2
      ;;
    --png-padding=*)
      PNG_PADDING="${1#--png-padding=}"
      if ! [[ "$PNG_PADDING" =~ ^[0-9]+$ ]]; then
        echo "--png-padding requires a non-negative integer" >&2
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

PDF_SCRIPT="${PDF_SCRIPT:-$ROOT_DIR/tools/md_to_mobile_pdf.py}"
LONG_PNG_SCRIPT="${LONG_PNG_SCRIPT:-$ROOT_DIR/tools/pdf_to_long_png.py}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-$ROOT_DIR/tools/prepare_video_doc_export.py}"
PREPARE_SCRIPT="$(realpath -m "$PREPARE_SCRIPT")"
PDF_SCRIPT="$(realpath -m "$PDF_SCRIPT")"
LONG_PNG_SCRIPT="$(realpath -m "$LONG_PNG_SCRIPT")"

if [ ! -d "$RUN_DIR" ]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi
if [ ! -f "$PDF_SCRIPT" ]; then
  echo "PDF converter not found: $PDF_SCRIPT" >&2
  exit 1
fi
if [ ! -f "$PREPARE_SCRIPT" ]; then
  echo "Export prepare script not found: $PREPARE_SCRIPT" >&2
  exit 1
fi
if [ "$LONG_PNG" -eq 1 ] && [ ! -f "$LONG_PNG_SCRIPT" ]; then
  echo "Long PNG converter not found: $LONG_PNG_SCRIPT" >&2
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
  local name
  name="$(basename "$rel" .md)"
  if [ ! -f "$input" ]; then
    case "$rel" in
      docs_analysis_chapters/knowledge_notes_v2.md)
        input="$RUN_DIR/docs_analysis/knowledge_notes.md"
        ;;
      docs_analysis_chapters/deep_report_v2.md)
        input="$RUN_DIR/docs_analysis/deep_report.md"
        ;;
    esac
    if [ ! -f "$input" ]; then
      echo "[skip missing] $rel" >&2
      return 0
    fi
    echo "[fallback] $rel <- ${input#$RUN_DIR/}" >&2
  fi
  local prepared="$RUN_DIR/.$name.export.md"
  cleanup_prepared() {
    rm -f "$RUN_DIR/.$name.export.md" "$RUN_DIR/.$name.export."*.md "$RUN_DIR/..$name.export."*.md
  }
  "$PYTHON_BIN" "$PREPARE_SCRIPT" "$RUN_DIR" "$input" "$prepared" >/dev/null
  trap cleanup_prepared RETURN
  echo "[pdf] $rel"
  "$PYTHON_BIN" "$PDF_SCRIPT" "$prepared" "$EXPORT_DIR/$name.pdf" --title "$name"
  if [ "$LONG_PNG" -eq 1 ]; then
    echo "[long-png] $rel"
    "$PYTHON_BIN" "$LONG_PNG_SCRIPT" "$EXPORT_DIR/$name.pdf" "$EXPORT_DIR/$name.long.png" \
      --dpi "$PNG_DPI" \
      --padding "$PNG_PADDING"
  fi
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
