#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: tools/run_video_doc_final_publish.sh RUN_DIR [--profile PROFILE] [--jobs N] [--finalize-only] [--skip-images] [--skip-pdf] [--skip-send] [--to WECHAT_ID] [--long-png]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$(realpath "$1")"
shift

PROFILE="deepseek_v4_pro"
JOBS=3
FINALIZE_ONLY=0
SKIP_IMAGES=0
SKIP_PDF=0
SKIP_SEND=0
LONG_PNG=0
WECHAT_TO="${WECLAW_TO:-}"
WECLAW_API_URL="${WECLAW_API_URL:-http://127.0.0.1:18011}"
IMAGE_CODEX_SANDBOX="${VIDEO_DOC_IMAGE_CODEX_SANDBOX:-danger-full-access}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --finalize-only)
      FINALIZE_ONLY=1
      shift
      ;;
    --skip-images)
      SKIP_IMAGES=1
      shift
      ;;
    --skip-pdf)
      SKIP_PDF=1
      shift
      ;;
    --skip-send)
      SKIP_SEND=1
      shift
      ;;
    --long-png)
      LONG_PNG=1
      shift
      ;;
    --to)
      WECHAT_TO="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [ ! -d "$RUN_DIR" ]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [ -x "$HOME/.nvm/versions/node/v20.19.5/bin/node" ]; then
  export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:$PATH"
fi

DEEPSEEK_ENV="${VIDEO_ANALYZER_DEEPSEEK_ENV:-$HOME/.config/video-analyzer/deepseek.env}"
if [[ -f "$DEEPSEEK_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEEPSEEK_ENV"
  set +a
fi

FINAL_DIR="$RUN_DIR/baoyu_images/final"
PROMPT_DIR="$RUN_DIR/baoyu_images/prompts"
IMAGE_LOG_DIR="$RUN_DIR/baoyu_images/logs"
EXPORT_DIR="$RUN_DIR/exports"
mkdir -p "$FINAL_DIR" "$IMAGE_LOG_DIR" "$EXPORT_DIR"

declare -a FINAL_IMAGES=(
  "02-infographic-knowledge-notes.png"
  "03-infographic-deep-report.png"
)

declare -a PROMPTS=(
  "02-infographic-knowledge-notes.md"
  "03-infographic-deep-report.md"
)

declare -a DEPRECATED_FINAL_IMAGES=(
  "01-image-cards-operation-manual.png"
  "04-infographic-manual-evidence.png"
)

declare -a DEPRECATED_PROMPTS=(
  "01-image-cards-operation-manual.md"
  "04-infographic-manual-evidence.md"
)

declare -a DOCS=(
  "operation_manual"
  "knowledge_notes_v2"
  "deep_report_v2"
  "manual_evidence"
)

cleanup_deprecated_image_outputs() {
  for name in "${DEPRECATED_FINAL_IMAGES[@]}"; do
    rm -f "$FINAL_DIR/$name"
  done
  for name in "${DEPRECATED_PROMPTS[@]}"; do
    rm -f "$PROMPT_DIR/$name" "$IMAGE_LOG_DIR/${name%.md}.codex.log"
  done
}

generate_final_images() {
  if [ ! -d "$PROMPT_DIR" ]; then
    echo "[images] prompt directory missing: $PROMPT_DIR" >&2
    return 1
  fi
  for index in "${!PROMPTS[@]}"; do
    generate_one_final_image "$index"
  done
}

generate_one_final_image() {
    local index="$1"
    local prompt_file="$PROMPT_DIR/${PROMPTS[$index]}"
    local target="$FINAL_DIR/${FINAL_IMAGES[$index]}"
    local prompt_base="${PROMPTS[$index]%.md}"
    if [ -s "$target" ]; then
      echo "[images] exists: $target"
      return 0
    fi
    if [ ! -f "$prompt_file" ]; then
      echo "[images] missing prompt: $prompt_file" >&2
      return 1
    fi
    local marker
    marker="$(mktemp)"
    touch "$marker"
    echo "[images] codex exec image_gen: $(basename "$prompt_file") sandbox=$IMAGE_CODEX_SANDBOX"
    local log_file
    log_file="$IMAGE_LOG_DIR/$prompt_base.codex.log"
    : > "$log_file"
    local session_id
    local codex_env=()
    if timeout 1 bash -c '</dev/tcp/127.0.0.1/10808' 2>/dev/null; then
      codex_env=(
        HTTP_PROXY="http://127.0.0.1:10808/"
        HTTPS_PROXY="http://127.0.0.1:10808/"
        ALL_PROXY="socks5://127.0.0.1:10808"
        http_proxy="http://127.0.0.1:10808/"
        https_proxy="http://127.0.0.1:10808/"
        all_proxy="socks5://127.0.0.1:10808"
        NO_PROXY="localhost,127.0.0.0/8,::1"
        no_proxy="localhost,127.0.0.0/8,::1"
      )
    fi
    local codex_cmd=(
      codex exec --cd "$RUN_DIR" --skip-git-repo-check --sandbox "$IMAGE_CODEX_SANDBOX" "$(cat <<EOF
Use the \$imagegen skill with the built-in image_gen tool to generate exactly one PNG image from this prompt file:
$prompt_file

Read the prompt file and generate the raster PNG image. Do not create or edit Markdown. Do not modify repository files. The wrapper script will copy the generated PNG from the default generated_images location.
EOF
)"
    )
    if ! env "${codex_env[@]}" "${codex_cmd[@]}" 2>&1 | tee "$log_file"; then
      rm -f "$marker"
      return 1
    fi
    session_id="$(awk 'tolower($0) ~ /session id:/ {print $3}' "$log_file" | tail -1)"
    local generated
    if [ -n "$session_id" ] && [ -d "$HOME/.codex/generated_images/$session_id" ]; then
      generated="$(find "$HOME/.codex/generated_images/$session_id" -type f -name '*.png' -newer "$marker" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    else
      generated=""
    fi
    if [ -z "$generated" ]; then
      generated="$(find "$HOME/.codex/generated_images" -type f -name '*.png' -newer "$marker" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    fi
    rm -f "$marker"
    if [ -z "$generated" ] || [ ! -s "$generated" ]; then
      echo "[images] no generated PNG found for $prompt_file" >&2
      return 1
    fi
    cp "$generated" "$target"
    echo "[images] copied: $target"
}

regenerate_docs() {
  echo "[docs] multidoc"
  "$ROOT_DIR/tools/run_multidoc_analysis.sh" "$RUN_DIR" --profile "$PROFILE"
  echo "[docs] deep-v2"
  "$PYTHON_BIN" "$ROOT_DIR/tools/generate_chapter_deep_report.py" "$RUN_DIR" \
    --profile "$PROFILE" \
    --deep-v2 \
    --no-final-synthesis \
    --no-format-markdown-final \
    --refresh-chapters \
    --chapter-concurrency "$JOBS"
}

verify_counts() {
  if [ "$SKIP_IMAGES" -eq 0 ]; then
    for name in "${FINAL_IMAGES[@]}"; do
      test -s "$FINAL_DIR/$name"
    done
  fi
  if [ "$SKIP_PDF" -eq 1 ]; then
    echo "[verify] pdf=skipped"
    return 0
  fi
  local pdf_count
  pdf_count="$(find "$EXPORT_DIR" -maxdepth 1 -type f -name '*.pdf' | wc -l)"
  echo "[verify] pdf=$pdf_count"
  [ "$pdf_count" -eq 4 ]
  for name in "${DOCS[@]}"; do
    test -s "$EXPORT_DIR/$name.pdf"
  done
  if [ "$LONG_PNG" -eq 1 ]; then
    local png_count
    png_count="$(find "$EXPORT_DIR" -maxdepth 1 -type f -name '*.long.png' | wc -l)"
    echo "[verify] long_png=$png_count"
    [ "$png_count" -eq 4 ]
    for name in "${DOCS[@]}"; do
      test -s "$EXPORT_DIR/$name.long.png"
    done
  fi
}

write_summary() {
  "$PYTHON_BIN" - "$RUN_DIR" "$SKIP_SEND" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
skip_send = bool(int(sys.argv[2]))
exports = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "exports").glob("*") if path.is_file())
final_images = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "baoyu_images" / "final").glob("*.png"))
summary = {
    "final_documents": [
        "operation_manual",
        "knowledge_notes_v2",
        "deep_report_v2",
        "manual_evidence",
    ],
    "export_files": exports,
    "final_images": final_images,
    "send_skipped": skip_send,
}
(run_dir / "final_publish_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("[summary] " + str(run_dir / "final_publish_summary.json"))
PY
}

resolve_wechat_to() {
  if [ -n "$WECHAT_TO" ]; then
    return 0
  fi
  if [ -f "$HOME/.weclaw/weclaw.log" ]; then
    WECHAT_TO="$(tail -n 500 "$HOME/.weclaw/weclaw.log" | rg -o '[A-Za-z0-9._-]+@im\.wechat' | tail -1 || true)"
  fi
  if [ -z "$WECHAT_TO" ]; then
    echo "No WECLAW_TO and no recent @im.wechat ID found" >&2
    return 1
  fi
}

send_file() {
  local path="$1"
  test -f "$path"
  "$PYTHON_BIN" - "$WECLAW_API_URL" "$WECHAT_TO" "$path" <<'PY'
import json
import sys
import urllib.request

api, to, path = sys.argv[1], sys.argv[2], sys.argv[3]
payload = json.dumps({"to": to, "media_path": path}).encode("utf-8")
request = urllib.request.Request(
    api.rstrip("/") + "/api/send",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=180) as response:
    print(path + "\t" + response.read().decode("utf-8", errors="replace"))
PY
}

send_outputs() {
  curl -fsS "$WECLAW_API_URL/health" >/dev/null
  resolve_wechat_to
  echo "[send] to=$WECHAT_TO"
  for name in "${DOCS[@]}"; do
    send_file "$EXPORT_DIR/$name.pdf"
  done
}

cd "$ROOT_DIR"
cleanup_deprecated_image_outputs

if [ "$SKIP_IMAGES" -eq 0 ]; then
  generate_final_images &
  images_pid=$!
fi
if [ "$FINALIZE_ONLY" -eq 0 ]; then
  regenerate_docs &
  docs_pid=$!
fi
if [ "$SKIP_IMAGES" -eq 0 ]; then
  wait "$images_pid"
fi
if [ "$FINALIZE_ONLY" -eq 0 ]; then
  wait "$docs_pid"
fi

augment_args=()
if [ "$SKIP_IMAGES" -eq 1 ]; then
  augment_args+=(--skip-final-images)
fi
"$PYTHON_BIN" "$ROOT_DIR/tools/augment_video_docs_images.py" "$RUN_DIR" "${augment_args[@]}"
export_args=(--final-only --jobs "$JOBS")
if [ "$SKIP_PDF" -eq 1 ]; then
  echo "[export] skipped pdf"
elif [ "$LONG_PNG" -eq 1 ]; then
  export_args+=(--long-png)
fi
if [ "$SKIP_PDF" -eq 0 ]; then
  "$ROOT_DIR/tools/export_video_docs.sh" "$RUN_DIR" "${export_args[@]}"
fi
verify_counts
write_summary

if [ "$SKIP_SEND" -eq 0 ] && [ "$SKIP_PDF" -eq 0 ]; then
  send_outputs
elif [ "$SKIP_SEND" -eq 0 ]; then
  echo "[send] skipped: pdf export disabled"
else
  echo "[send] skipped"
fi
