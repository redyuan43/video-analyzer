#!/usr/bin/env python3
"""Download an online video and generate an illustrated operation manual.

The default runtime policy matches the current tested setup:
- video download through yt-dlp
- VibeVoice ASR on edge, single remote worker
- DotsMOCR OCR on spark
- LM Studio local VL/text models
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_VIBEVOICE_URL = "http://192.168.100.236:8003/api/asr/transcribe"
DEFAULT_OCR_URL = "http://192.168.100.169:8000/v1"
DEFAULT_OUTPUT_ROOT = Path("downloads/url-videos")
DEFAULT_RUN_NAME = "operation-manual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a YouTube/Bilibili video and generate operation_manual.md")
    parser.add_argument("url", help="YouTube, Bilibili, or other yt-dlp supported URL")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for downloaded videos and runs")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Output run directory name under the video folder")
    parser.add_argument("--max-frames", type=int, default=24, help="Keyframe budget for operation_manual")
    parser.add_argument("--duration", type=float, help="Optional duration in seconds to process")
    parser.add_argument("--manual-language", default="zh-CN")
    parser.add_argument("--vibevoice-url", default=DEFAULT_VIBEVOICE_URL, help="Remote GPU VibeVoice ASR endpoint")
    parser.add_argument("--ocr-base-url", default=DEFAULT_OCR_URL, help="DotsMOCR OpenAI-compatible base URL")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--vision-model", default="qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive@?")
    parser.add_argument("--text-model", default="redhatai_qwen3.6-35b-a3b-nvfp4")
    parser.add_argument("--cookies-from-browser", help="Forward to yt-dlp, e.g. chrome, chromium, firefox, brave")
    parser.add_argument("--cookies", help="Forward cookies.txt to yt-dlp")
    parser.add_argument("--download-only", action="store_true", help="Only download video and page context")
    parser.add_argument("--keep-existing", action="store_true", help="Reuse existing video/context if present")
    parser.add_argument("--no-keep-frames", action="store_true", help="Do not keep extracted frames after analysis")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run video_analyzer.cli")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_tool("yt-dlp")
    ensure_tool("ffmpeg")

    output_root = Path(args.output_root)
    info = fetch_metadata(args.url, args)
    video_id = safe_slug(str(info.get("id") or info.get("display_id") or info.get("title") or "video"))
    video_dir = output_root / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / "video.mp4"
    info_path = video_dir / "info.json"
    context_path = video_dir / "description.md"
    if args.keep_existing and video_path.exists() and context_path.exists():
        print(f"[download] reusing {video_path}")
    else:
        download_video(args.url, video_dir, args)
        materialize_download(video_dir, video_path)
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        context_path.write_text(build_context_markdown(info, args.url), encoding="utf-8")
        print(f"[download] video: {video_path}")
        print(f"[download] context: {context_path}")

    if args.download_only:
        return 0

    run_dir = safe_child_dir(video_dir, args.run_name)
    if run_dir.exists():
        shutil.rmtree(run_dir)

    command = build_analyzer_command(args, video_path, context_path, run_dir)
    print("[analyze] " + " ".join(shell_quote(part) for part in command))
    subprocess.run(command, check=True)
    analysis_path = run_dir / "analysis.json"
    manual_path = read_manual_path(analysis_path) or first_existing_manual(run_dir)
    if manual_path:
        print(f"[done] manual: {manual_path}")
    else:
        print(f"[done] manual: not found under {run_dir}")
    print(f"[done] analysis: {analysis_path}")
    return 0


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required command not found: {name}. Install it first, e.g. `pip install yt-dlp` for yt-dlp.")


def fetch_metadata(url: str, args: argparse.Namespace) -> dict[str, Any]:
    command = ["yt-dlp", "--dump-single-json", "--no-warnings", "--skip-download", url]
    add_cookie_args(command, args)
    raw = subprocess.check_output(command, text=True)
    return json.loads(raw)


def download_video(url: str, video_dir: Path, args: argparse.Namespace) -> None:
    command = [
        "yt-dlp",
        "--no-playlist",
        "--write-info-json",
        "--write-description",
        "--write-auto-subs",
        "--sub-langs",
        "zh.*,en.*",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*[vcodec^=avc1]+ba/b[vcodec^=avc1]/bv*+ba/b",
        "-o",
        str(video_dir / "download.%(ext)s"),
        url,
    ]
    add_cookie_args(command, args)
    subprocess.run(command, check=True)


def add_cookie_args(command: list[str], args: argparse.Namespace) -> None:
    if args.cookies_from_browser:
        command.extend(["--cookies-from-browser", args.cookies_from_browser])
    if args.cookies:
        command.extend(["--cookies", args.cookies])


def materialize_download(video_dir: Path, video_path: Path) -> None:
    candidates = sorted(video_dir.glob("download.*"))
    media_candidates = [
        path
        for path in candidates
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".flv"} and path.name != video_path.name
    ]
    if not media_candidates:
        raise RuntimeError(f"yt-dlp did not produce a video file in {video_dir}")
    source = media_candidates[0]
    if video_path.exists():
        video_path.unlink()
    source.rename(video_path)


def build_context_markdown(info: dict[str, Any], url: str) -> str:
    lines = [
        f"# {info.get('title') or 'Video'}",
        "",
        f"- URL: {url}",
        f"- Platform ID: {info.get('id') or info.get('display_id') or ''}",
        f"- Uploader: {info.get('uploader') or info.get('channel') or ''}",
        f"- Upload date: {info.get('upload_date') or ''}",
        f"- Duration: {info.get('duration') or ''}",
        "",
    ]
    description = (info.get("description") or "").strip()
    if description:
        lines.extend(["## Page Description", "", description, ""])
    chapters = info.get("chapters") or []
    if chapters:
        lines.extend(["## Chapters", ""])
        for chapter in chapters:
            title = chapter.get("title") or ""
            start = chapter.get("start_time")
            end = chapter.get("end_time")
            lines.append(f"- {format_seconds(start)} - {format_seconds(end)}: {title}")
        lines.append("")
    tags = info.get("tags") or []
    if tags:
        lines.extend(["## Tags", "", ", ".join(str(tag) for tag in tags), ""])
    return "\n".join(lines).strip() + "\n"


def build_analyzer_command(args: argparse.Namespace, video_path: Path, context_path: Path, run_dir: Path) -> list[str]:
    command = [
        args.python,
        "-m",
        "video_analyzer.cli",
        str(video_path),
        "--task",
        "operation_manual",
        "--output",
        str(run_dir),
        "--context-file",
        str(context_path),
        "--asr-provider",
        "vibevoice",
        "--vibevoice-url",
        args.vibevoice_url,
        "--ocr-provider",
        "auto",
        "--ocr-base-url",
        args.ocr_base_url,
        "--llm-base-url",
        args.llm_base_url,
        "--vision-model",
        args.vision_model,
        "--text-model",
        args.text_model,
        "--manual-language",
        args.manual_language,
        "--max-frames",
        str(args.max_frames),
        "--log-level",
        args.log_level,
    ]
    if args.duration is not None:
        command.extend(["--duration", str(args.duration)])
    if not args.no_keep_frames:
        command.append("--keep-frames")
    return command


def safe_slug(value: str) -> str:
    value = value.strip() or "video"
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = value.strip(".-")
    return value[:120] or "video"


def safe_child_dir(parent: Path, name: str) -> Path:
    if Path(name).is_absolute():
        raise ValueError("--run-name must be a directory name, not an absolute path")
    slug = safe_slug(name)
    if not slug or slug in {".", ".."}:
        raise ValueError("--run-name must contain at least one safe character")
    child = (parent / slug).resolve()
    parent_resolved = parent.resolve()
    if child == parent_resolved or parent_resolved not in child.parents:
        raise ValueError("--run-name must stay inside the video output directory")
    return child


def read_manual_path(analysis_path: Path) -> Path | None:
    try:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    manual_path = (((payload.get("operation_manual") or {}).get("manual_path")) or "").strip()
    return Path(manual_path) if manual_path else None


def first_existing_manual(run_dir: Path) -> Path | None:
    for name in ("operation_manual.md", "operation_manual.quality_failed.md"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def format_seconds(value: Any) -> str:
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return ""
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=@?+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
