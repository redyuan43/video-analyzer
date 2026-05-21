#!/usr/bin/env python3
"""Download an online video and generate an illustrated operation manual.

The default runtime policy matches the current tested setup:
- video download through yt-dlp
- VibeVoice ASR on spark, single remote worker
- DotsMOCR OCR on spark
- LM Studio VL/text models on spark
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

from video_analyzer.config import Config


FALLBACK_OUTPUT_ROOT = "downloads/url-videos"
FALLBACK_RUN_NAME = "operation-manual"
FALLBACK_SUBTITLE_LANGS = "zh-CN,zh-Hans,zh,en"
FALLBACK_LLM_BASE_URL = "http://spark-31d6.taild500c8.ts.net:1234/v1"
FALLBACK_VISION_MODEL = "qwen/qwen3-vl-30b"
FALLBACK_TEXT_MODEL = "redhatai_qwen3.6-35b-a3b-nvfp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a YouTube/Bilibili video and generate operation_manual.md")
    parser.add_argument("url", help="YouTube, Bilibili, or other yt-dlp supported URL")
    parser.add_argument("--config", default="config", help="Configuration directory containing optional config.json")
    parser.add_argument("--profile", help="Runtime profile from config/default_config.json or config.json")
    parser.add_argument("--output-root", help="Directory for downloaded videos and runs")
    parser.add_argument("--run-name", help="Output run directory name under the video folder")
    parser.add_argument("--max-frames", type=int, help="Explicit upper limit for the operation-manual candidate frame pool")
    parser.add_argument("--pipeline-mode", choices=["fast", "balanced", "deep"])
    parser.add_argument("--candidate-frames", help="auto or explicit candidate frame pool size")
    parser.add_argument("--min-vl-frames", help="auto or minimum frames sent to VL")
    parser.add_argument("--max-vl-frames", help="auto or maximum frames sent to VL")
    parser.add_argument("--vl-frame-policy", choices=["auto", "all", "none"])
    parser.add_argument("--vl-concurrency", type=int)
    parser.add_argument("--vl-context-before", type=int)
    parser.add_argument("--vl-context-after", type=int)
    parser.add_argument("--vl-context-max-gap")
    parser.add_argument("--duration", type=float, help="Optional duration in seconds to process")
    parser.add_argument("--manual-language")
    parser.add_argument("--asr-provider", choices=["none", "vibevoice"], help="Analyzer ASR provider when no subtitle transcript is used")
    parser.add_argument("--vibevoice-url", action="append", help="Remote GPU VibeVoice ASR endpoint; can be provided multiple times")
    parser.add_argument("--ocr-base-url", action="append", help="DotsMOCR OpenAI-compatible base URL; can be provided multiple times")
    parser.add_argument("--ocr-concurrency", help="OCR concurrency per endpoint, or auto")
    parser.add_argument("--ocr-cache", choices=["on", "off", "refresh"], help="OCR cache mode")
    parser.add_argument("--ocr-cache-dir", help="OCR cache directory")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--vision-base-url")
    parser.add_argument("--text-base-url")
    parser.add_argument("--vision-model")
    parser.add_argument("--text-model")
    parser.add_argument("--cookies-from-browser", help="Forward to yt-dlp, e.g. chrome, chromium, firefox, brave")
    parser.add_argument("--cookies", help="Forward cookies.txt to yt-dlp")
    parser.add_argument("--ytdlp-proxy", help="Proxy URL used only by yt-dlp download/metadata requests")
    parser.add_argument(
        "--ytdlp-js-runtimes",
        help="Forward to yt-dlp --js-runtimes. Use auto/node for YouTube JS challenges, or none to disable.",
    )
    parser.add_argument("--refresh-context", action="store_true", help="Refresh page context, subtitles, and comments without redownloading an existing video")
    parser.add_argument(
        "--prefer-subtitle-transcript",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use downloaded author/automatic subtitles as transcript and skip audio ASR when available",
    )
    parser.add_argument("--transcript-file", help="Existing transcript markdown file passed through to the analyzer")
    parser.add_argument("--frame-extractor", choices=["local", "jetson", "auto"])
    parser.add_argument("--jetson-frame-hosts")
    parser.add_argument("--jetson-frame-backend", choices=["auto", "ssh", "ray"])
    parser.add_argument("--jetson-sample-fps")
    parser.add_argument("--jetson-chunk-overlap-seconds", type=float)
    parser.add_argument("--jetson-frame-weights")
    parser.add_argument("--jetson-require-hwdec", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--download-only", action="store_true", help="Only download video and page context")
    parser.add_argument("--keep-existing", action="store_true", help="Reuse existing video/context if present")
    parser.add_argument("--no-keep-frames", action="store_true", help="Do not keep extracted frames after analysis")
    parser.add_argument(
        "--include-subtitles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Download and include author/automatic subtitles in page_context.md",
    )
    parser.add_argument(
        "--include-comments",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Download and include selected low-trust comments in page_context.md",
    )
    parser.add_argument("--max-comments", type=int, help="Maximum selected comments to include")
    parser.add_argument("--subtitle-langs", help="Comma-separated subtitle language priority")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run video_analyzer.cli")
    args = parser.parse_args()
    apply_runtime_profile(args)
    return args


def apply_runtime_profile(args: argparse.Namespace) -> argparse.Namespace:
    profile = Config(args.config).get_runtime_profile(args.profile)
    defaults = {
        "output_root": profile.get("output_root", FALLBACK_OUTPUT_ROOT),
        "run_name": profile.get("run_name", FALLBACK_RUN_NAME),
        "pipeline_mode": profile.get("pipeline_mode", "balanced"),
        "candidate_frames": profile.get("candidate_frames", "auto"),
        "min_vl_frames": profile.get("min_vl_frames", "auto"),
        "max_vl_frames": profile.get("max_vl_frames", "auto"),
        "vl_frame_policy": profile.get("vl_frame_policy", "auto"),
        "vl_concurrency": profile.get("vl_concurrency", 3),
        "vl_context_before": profile.get("vl_context_before", 0),
        "vl_context_after": profile.get("vl_context_after", 0),
        "vl_context_max_gap": profile.get("vl_context_max_gap", "auto"),
        "manual_language": profile.get("manual_language", "zh-CN"),
        "asr_provider": profile.get("asr_provider", "vibevoice"),
        "vibevoice_url": profile.get("vibevoice_urls") or profile.get("vibevoice_url"),
        "ocr_base_url": profile.get("ocr_base_urls") or profile.get("ocr_base_url"),
        "ocr_concurrency": profile.get("ocr_concurrency", "auto"),
        "ocr_cache": profile.get("ocr_cache", "on"),
        "ocr_cache_dir": profile.get("ocr_cache_dir", ".cache/video-analyzer/ocr"),
        "llm_base_url": profile.get("llm_base_url", FALLBACK_LLM_BASE_URL),
        "vision_base_url": profile.get("vision_base_url"),
        "text_base_url": profile.get("text_base_url"),
        "vision_model": profile.get("vision_model", FALLBACK_VISION_MODEL),
        "text_model": profile.get("text_model", FALLBACK_TEXT_MODEL),
        "include_subtitles": profile.get("include_subtitles", True),
        "include_comments": profile.get("include_comments", True),
        "max_comments": profile.get("max_comments", 30),
        "subtitle_langs": profile.get("subtitle_langs", FALLBACK_SUBTITLE_LANGS),
        "ytdlp_js_runtimes": profile.get("ytdlp_js_runtimes", "auto"),
        "prefer_subtitle_transcript": profile.get("prefer_subtitle_transcript", False),
        "frame_extractor": profile.get("frame_extractor", "local"),
        "jetson_frame_hosts": profile.get("jetson_frame_hosts", "nx2,nx3"),
        "jetson_frame_backend": profile.get("jetson_frame_backend", "auto"),
        "jetson_sample_fps": profile.get("jetson_sample_fps", "auto"),
        "jetson_chunk_overlap_seconds": profile.get("jetson_chunk_overlap_seconds", 2.0),
        "jetson_frame_weights": profile.get("jetson_frame_weights", ""),
        "jetson_require_hwdec": bool(profile.get("jetson_require_hwdec", False)),
    }
    for key, value in defaults.items():
        if getattr(args, key, None) is None and value is not None:
            setattr(args, key, value)
    if not args.vibevoice_url:
        raise ValueError("Runtime profile must provide vibevoice_url, or pass --vibevoice-url")
    if not args.ocr_base_url:
        raise ValueError("Runtime profile must provide ocr_base_url, or pass --ocr-base-url")
    return args


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
    description_path = video_dir / "description.md"
    page_context_path = video_dir / "page_context.md"
    if args.keep_existing and video_path.exists() and page_context_path.exists() and not args.refresh_context:
        print(f"[download] reusing {video_path}")
        print(f"[download] video: {video_path}")
        print(f"[download] description: {description_path}")
        print(f"[download] context: {page_context_path}")
        info = load_downloaded_info(video_dir) or info
    else:
        if args.keep_existing and video_path.exists():
            download_context_assets(args.url, video_dir, args)
        else:
            download_video(args.url, video_dir, args)
            materialize_download(video_dir, video_path)
        info = load_downloaded_info(video_dir) or info
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        description_text = build_context_markdown(info, args.url)
        description_path.write_text(description_text, encoding="utf-8")
        page_context = build_page_context_bundle(info, args.url, description_text, video_dir, args)
        page_context_path.write_text(page_context["markdown"], encoding="utf-8")
        (video_dir / "page_context.json").write_text(
            json.dumps(page_context["metadata"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[download] video: {video_path}")
        print(f"[download] description: {description_path}")
        print(f"[download] context: {page_context_path}")

    if args.prefer_subtitle_transcript and not args.transcript_file:
        transcript_path = materialize_subtitle_transcript(video_dir, info)
        if transcript_path:
            args.transcript_file = str(transcript_path)
            args.asr_provider = "none"
            print(f"[download] subtitle transcript: {transcript_path}")
        else:
            print("[download] subtitle transcript: not available; analyzer will use configured ASR")

    if args.download_only:
        return 0

    run_dir = safe_child_dir(video_dir, args.run_name)
    if run_dir.exists():
        shutil.rmtree(run_dir)

    command = build_analyzer_command(args, video_path, page_context_path, run_dir)
    print("[analyze] " + " ".join(shell_quote(part) for part in command))
    subprocess.run(command, check=True)
    analysis_path = run_dir / "analysis.json"
    manual_path = read_manual_path(analysis_path) or first_existing_manual(run_dir)
    if manual_path:
        print(f"[done] manual: {manual_path}")
    else:
        print(f"[done] manual: not found under {run_dir}")
    print(f"[done] analysis: {analysis_path}")
    print(f"[done] run_dir: {run_dir.resolve()}")
    return 0


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required command not found: {name}. Install it first, e.g. `pip install yt-dlp` for yt-dlp.")


def fetch_metadata(url: str, args: argparse.Namespace) -> dict[str, Any]:
    command = ["yt-dlp", "--dump-single-json", "--no-warnings", "--skip-download", url]
    add_ytdlp_runtime_args(command, args)
    add_ytdlp_network_args(command, args)
    add_cookie_args(command, args)
    raw = subprocess.check_output(command, text=True)
    return json.loads(raw)


def download_video(url: str, video_dir: Path, args: argparse.Namespace) -> None:
    command = [
        "yt-dlp",
        "--no-playlist",
        "--write-info-json",
        "--write-description",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*[vcodec^=avc1]+ba/b[vcodec^=avc1]/bv*+ba/b",
        "-o",
        str(video_dir / "download.%(ext)s"),
        url,
    ]
    if args.include_subtitles:
        command.extend(["--write-subs", "--write-auto-subs", "--sub-langs", subtitle_langs_for_ytdlp(args.subtitle_langs)])
    if args.include_comments:
        command.append("--write-comments")
    add_ytdlp_runtime_args(command, args)
    add_ytdlp_network_args(command, args)
    add_cookie_args(command, args)
    subprocess.run(command, check=True)


def download_context_assets(url: str, video_dir: Path, args: argparse.Namespace) -> None:
    command = [
        "yt-dlp",
        "--no-playlist",
        "--skip-download",
        "--write-info-json",
        "--write-description",
        "-o",
        str(video_dir / "download.%(ext)s"),
        url,
    ]
    if args.include_subtitles:
        command.extend(["--write-subs", "--write-auto-subs", "--sub-langs", subtitle_langs_for_ytdlp(args.subtitle_langs)])
    if args.include_comments:
        command.append("--write-comments")
    add_ytdlp_runtime_args(command, args)
    add_ytdlp_network_args(command, args)
    add_cookie_args(command, args)
    subprocess.run(command, check=True)


def load_downloaded_info(video_dir: Path) -> dict[str, Any] | None:
    for path in sorted(video_dir.glob("download*.info.json")):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def add_cookie_args(command: list[str], args: argparse.Namespace) -> None:
    if args.cookies_from_browser:
        command.extend(["--cookies-from-browser", args.cookies_from_browser])
    if args.cookies:
        command.extend(["--cookies", args.cookies])


def add_ytdlp_network_args(command: list[str], args: argparse.Namespace) -> None:
    if getattr(args, "ytdlp_proxy", None):
        command.extend(["--proxy", args.ytdlp_proxy])


def add_ytdlp_runtime_args(command: list[str], args: argparse.Namespace) -> None:
    value = str(getattr(args, "ytdlp_js_runtimes", None) or "auto").strip()
    if value.lower() == "auto":
        if not shutil.which("node"):
            return
        value = "node"
    if not value or value.lower() in {"none", "no", "off", "false", "disabled"}:
        return
    if "--js-runtimes" not in command:
        command.extend(["--js-runtimes", value])


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


def build_page_context_bundle(
    info: dict[str, Any],
    url: str,
    description_text: str,
    video_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    subtitles = collect_subtitles(video_dir, info, args)
    comments = collect_comments(video_dir, info, args)
    markdown = build_page_context_markdown(info, url, description_text, subtitles, comments)
    metadata = {
        "context_file": str(video_dir / "page_context.md"),
        "description_file": str(video_dir / "description.md"),
        "subtitles": subtitles["metadata"],
        "comments": comments["metadata"],
    }
    return {"markdown": markdown, "metadata": metadata}


def collect_subtitles(video_dir: Path, info: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metadata = {"enabled": bool(args.include_subtitles), "success": False, "diagnostics": []}
    if not args.include_subtitles:
        metadata["diagnostics"].append("subtitle collection disabled by CLI")
        return {"markdown": "", "metadata": metadata}

    files = [path for path in video_dir.glob("download.*") if path.suffix.lower() in {".vtt", ".srt", ".json3"}]
    if not files:
        metadata["diagnostics"].append("yt-dlp did not produce subtitle files")
        return {"markdown": "", "metadata": metadata}

    preferred = parse_csv(args.subtitle_langs)
    chosen = choose_subtitle_file(files, preferred, info)
    text = clean_subtitle_file(chosen)
    subtitles_dir = video_dir / "subtitles"
    subtitles_dir.mkdir(exist_ok=True)
    raw_path = subtitles_dir / chosen.name
    text_path = subtitles_dir / f"{chosen.stem}.cleaned.txt"
    shutil.copy2(chosen, raw_path)
    text_path.write_text(text, encoding="utf-8")

    language = infer_subtitle_lang(chosen)
    source = infer_subtitle_source(language, info)
    metadata.update(
        {
            "success": bool(text.strip()),
            "language": language,
            "source": source,
            "raw_file": str(raw_path),
            "text_file": str(text_path),
            "available_files": [path.name for path in files],
            "text_length": len(text),
        }
    )
    if not text.strip():
        metadata["diagnostics"].append(f"subtitle file was empty after cleanup: {chosen.name}")
    markdown = "\n".join(
        [
            "## Subtitles",
            "",
            f"- Evidence weight: {'author subtitles' if source == 'author' else 'automatic subtitles'}",
            f"- Language: {language}",
            f"- Source file: {raw_path.name}",
            "",
            trim_text(text, 12000) if text.strip() else "_No usable subtitle text._",
            "",
        ]
    )
    return {"markdown": markdown, "metadata": metadata}


def materialize_subtitle_transcript(video_dir: Path, info: dict[str, Any]) -> Path | None:
    files = [path for path in video_dir.glob("download.*") if path.suffix.lower() in {".vtt", ".srt", ".json3"}]
    if not files:
        return None
    context_metadata_path = video_dir / "page_context.json"
    preferred: list[str] = []
    if context_metadata_path.exists():
        try:
            metadata = json.loads(context_metadata_path.read_text(encoding="utf-8"))
            language = (((metadata.get("subtitles") or {}).get("language")) or "").strip()
            if language:
                preferred.append(language)
        except Exception:
            preferred = []
    chosen = choose_subtitle_file(files, preferred, info)
    cleaned = clean_subtitle_file(chosen)
    segments = parse_cleaned_subtitle_segments(cleaned)
    if not segments:
        return None

    transcript_path = video_dir / "subtitle_transcript.md"
    language = infer_subtitle_lang(chosen)
    lines = [
        "# Transcript",
        "",
        f"- Language: {language}",
        f"- Segments: {len(segments)}",
        "",
        "- Source: subtitle via yt-dlp",
        f"- Raw file: {chosen}",
        "",
    ]
    lines.extend(f"- [{start} - {end}] {text}" for start, end, text in segments)
    transcript_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return transcript_path


def parse_cleaned_subtitle_segments(text: str) -> list[tuple[str, str, str]]:
    segments = []
    pattern = re.compile(
        r"^\[(?P<start>\d\d:\d\d:\d\d)(?:\.\d+)?\s+(?:-->|-)\s+(?P<end>\d\d:\d\d:\d\d)(?:\.\d+)?\]\s+(?P<text>.*)$"
    )
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        body = match.group("text").strip()
        if body:
            segments.append((match.group("start"), match.group("end"), body))
    return segments


def collect_comments(video_dir: Path, info: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metadata = {"enabled": bool(args.include_comments), "success": False, "diagnostics": [], "selected_count": 0}
    if not args.include_comments:
        metadata["diagnostics"].append("comment collection disabled by CLI")
        return {"markdown": "", "metadata": metadata}

    raw_comments = info.get("comments") or []
    if not isinstance(raw_comments, list) or not raw_comments:
        metadata["diagnostics"].append("yt-dlp did not return comments; the platform may block comments or require login")
        return {"markdown": "", "metadata": metadata}

    selected = select_comments(raw_comments, max(0, args.max_comments), info)
    comments_json = video_dir / "comments.json"
    selected_comments_json = video_dir / "selected_comments.json"
    comments_md = video_dir / "comments.md"
    comments_json.write_text(json.dumps(raw_comments, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_comments_json.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = build_comments_markdown(selected)
    comments_md.write_text(markdown, encoding="utf-8")
    metadata.update(
        {
            "success": bool(selected),
            "available_count": len(raw_comments),
            "selected_count": len(selected),
            "json_file": str(comments_json),
            "selected_json_file": str(selected_comments_json),
            "markdown_file": str(comments_md),
        }
    )
    if not selected:
        metadata["diagnostics"].append("comments were present but max-comments or filtering selected none")
    return {"markdown": markdown, "metadata": metadata}


def build_page_context_markdown(
    info: dict[str, Any],
    url: str,
    description_text: str,
    subtitles: dict[str, Any],
    comments: dict[str, Any],
) -> str:
    diagnostics = []
    diagnostics.extend(subtitles["metadata"].get("diagnostics", []))
    diagnostics.extend(comments["metadata"].get("diagnostics", []))
    lines = [
        f"# Page Context Evidence: {info.get('title') or 'Video'}",
        "",
        "Evidence weights for manual generation:",
        "- OCR/VL frame evidence is highest confidence for visible operations.",
        "- Author subtitles are high-confidence timeline/speech evidence.",
        "- VibeVoice ASR is high-confidence semantic audio evidence.",
        "- Automatic subtitles are medium-confidence and can contain recognition errors.",
        "- Page description and metadata are contextual evidence.",
        "- Pinned/uploader comments are low-confidence supplemental evidence.",
        "- Ordinary comments are lowest-confidence and must stay in community notes or FAQ unless confirmed elsewhere.",
        "",
        "## Metadata Summary",
        "",
        f"- URL: {url}",
        f"- Platform ID: {info.get('id') or info.get('display_id') or ''}",
        f"- Uploader: {info.get('uploader') or info.get('channel') or ''}",
        f"- Upload date: {info.get('upload_date') or ''}",
        f"- Duration: {info.get('duration') or ''}",
        "",
        "## Original Description",
        "",
        description_text.strip(),
        "",
    ]
    if subtitles["markdown"]:
        lines.extend([subtitles["markdown"].strip(), ""])
    else:
        lines.extend(["## Subtitles", "", "_No usable subtitles collected._", ""])
    if comments["markdown"]:
        lines.extend([comments["markdown"].strip(), ""])
    else:
        lines.extend(["## Comments", "", "_No usable comments collected._", ""])
    if diagnostics:
        lines.extend(["## Diagnostics", ""])
        lines.extend(f"- {item}" for item in diagnostics)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def choose_subtitle_file(files: list[Path], preferred: list[str], info: dict[str, Any]) -> Path:
    def score(path: Path) -> tuple[int, int, str]:
        lang = infer_subtitle_lang(path)
        source_rank = 0 if infer_subtitle_source(lang, info) == "author" else 1
        lang_rank = next((idx for idx, wanted in enumerate(preferred) if lang == wanted or lang.startswith(wanted)), len(preferred))
        return (source_rank, lang_rank, path.name)

    return sorted(files, key=score)[0]


def infer_subtitle_lang(path: Path) -> str:
    parts = path.name.split(".")
    if len(parts) >= 3 and parts[0] == "download":
        return ".".join(parts[1:-1])
    return path.stem


def infer_subtitle_source(language: str, info: dict[str, Any]) -> str:
    subtitles = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}
    if language in subtitles:
        return "author"
    if language in automatic:
        return "automatic"
    return "unknown"


def clean_subtitle_file(path: Path) -> str:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json3":
        return clean_json3_subtitles(text)
    return clean_text_subtitles(text)


def clean_json3_subtitles(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    lines = []
    for event in payload.get("events", []):
        body = "".join(seg.get("utf8", "") for seg in event.get("segs", [])).strip()
        if not body:
            continue
        start_ms = event.get("tStartMs", 0)
        end_ms = start_ms + event.get("dDurationMs", 0)
        lines.append(f"[{format_seconds_ms(start_ms)} - {format_seconds_ms(end_ms)}] {body}")
    return "\n".join(lines).strip() + "\n"


def clean_text_subtitles(text: str) -> str:
    lines = []
    pending_time = ""
    pending_text = []
    for raw_line in text.splitlines():
        line = strip_subtitle_tags(raw_line.strip())
        if not line or line.isdigit() or line in {"WEBVTT", "Kind: captions", "Language: en"}:
            if pending_time and pending_text:
                lines.append(f"[{pending_time}] {' '.join(pending_text)}")
            pending_time = ""
            pending_text = []
            continue
        if "-->" in line:
            if pending_time and pending_text:
                lines.append(f"[{pending_time}] {' '.join(pending_text)}")
            pending_time = line
            pending_text = []
            continue
        if pending_time:
            pending_text.append(line)
    if pending_time and pending_text:
        lines.append(f"[{pending_time}] {' '.join(pending_text)}")
    return "\n".join(lines).strip() + "\n"


def strip_subtitle_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def select_comments(comments: list[dict[str, Any]], max_comments: int, info: dict[str, Any]) -> list[dict[str, Any]]:
    uploader = str(info.get("uploader") or info.get("channel") or "").strip()

    def enrich(comment: dict[str, Any]) -> dict[str, Any]:
        author = str(comment.get("author") or "").strip()
        category = "ordinary"
        if comment.get("is_pinned") or comment.get("pinned"):
            category = "pinned"
        elif comment.get("author_is_uploader") or (uploader and author == uploader):
            category = "uploader"
        return {
            "category": category,
            "author": author,
            "text": str(comment.get("text") or "").strip(),
            "like_count": comment.get("like_count") or 0,
            "timestamp": comment.get("timestamp"),
            "id": comment.get("id"),
            "raw": comment,
        }

    enriched = [item for item in (enrich(comment) for comment in comments) if item["text"]]
    category_rank = {"pinned": 0, "uploader": 1, "ordinary": 2}
    enriched.sort(key=lambda item: (category_rank.get(item["category"], 9), -int(item.get("like_count") or 0)))
    return enriched[:max_comments]


def build_comments_markdown(comments: list[dict[str, Any]]) -> str:
    lines = [
        "## Comments",
        "",
        "Evidence weight: low. Use comments only for community supplements, FAQ, pinned/uploader clarifications, or version-change clues.",
        "",
    ]
    for comment in comments:
        lines.extend(
            [
                f"### {comment['category']} comment by {comment.get('author') or 'unknown'}",
                "",
                f"- Likes: {comment.get('like_count') or 0}",
                f"- Timestamp: {comment.get('timestamp') or ''}",
                "",
                trim_text(comment.get("text") or "", 1000),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_cli_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = [str(value)]
    normalized = []
    for item in values:
        cleaned = str(item).strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def subtitle_langs_for_ytdlp(value: str) -> str:
    langs = parse_csv(value)
    return ",".join(langs) if langs else FALLBACK_SUBTITLE_LANGS


def trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[truncated]"


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
        "--ocr-provider",
        "auto",
        "--llm-base-url",
        args.llm_base_url,
        "--vision-base-url",
        getattr(args, "vision_base_url", None) or args.llm_base_url,
        "--text-base-url",
        getattr(args, "text_base_url", None) or args.llm_base_url,
        "--vision-model",
        args.vision_model,
        "--text-model",
        args.text_model,
        "--manual-language",
        args.manual_language,
        "--log-level",
        args.log_level,
        "--pipeline-mode",
        args.pipeline_mode,
        "--candidate-frames",
        str(args.candidate_frames),
        "--min-vl-frames",
        str(args.min_vl_frames),
        "--max-vl-frames",
        str(args.max_vl_frames),
        "--vl-frame-policy",
        args.vl_frame_policy,
        "--vl-concurrency",
        str(args.vl_concurrency),
        "--vl-context-before",
        str(args.vl_context_before),
        "--vl-context-after",
        str(args.vl_context_after),
        "--vl-context-max-gap",
        str(args.vl_context_max_gap),
    ]
    if getattr(args, "transcript_file", None):
        command.extend(["--transcript-file", args.transcript_file, "--asr-provider", "none"])
    else:
        asr_provider = getattr(args, "asr_provider", None) or "vibevoice"
        command.extend(["--asr-provider", asr_provider])
        if asr_provider == "vibevoice":
            for vibevoice_url in _as_list(args.vibevoice_url):
                command.extend(["--vibevoice-url", vibevoice_url])
    command.extend(
        [
            "--frame-extractor",
            getattr(args, "frame_extractor", "local"),
            "--jetson-frame-hosts",
            getattr(args, "jetson_frame_hosts", "nx2,nx3"),
            "--jetson-frame-backend",
            getattr(args, "jetson_frame_backend", "auto"),
            "--jetson-sample-fps",
            str(getattr(args, "jetson_sample_fps", "auto")),
            "--jetson-chunk-overlap-seconds",
            str(getattr(args, "jetson_chunk_overlap_seconds", 2.0)),
        ]
    )
    if getattr(args, "jetson_frame_weights", ""):
        command.extend(["--jetson-frame-weights", args.jetson_frame_weights])
    if getattr(args, "jetson_require_hwdec", False):
        command.append("--jetson-require-hwdec")
    for endpoint in normalize_cli_list(args.ocr_base_url):
        command.extend(["--ocr-base-url", endpoint])
    command.extend(
        [
            "--ocr-concurrency",
            str(getattr(args, "ocr_concurrency", "auto")),
            "--ocr-cache",
            getattr(args, "ocr_cache", "on"),
            "--ocr-cache-dir",
            getattr(args, "ocr_cache_dir", ".cache/video-analyzer/ocr"),
        ]
    )
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
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


def format_seconds_ms(value: Any) -> str:
    try:
        seconds = int(float(value) / 1000)
    except (TypeError, ValueError):
        seconds = 0
    return format_seconds(seconds)


def shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=@?+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


if __name__ == "__main__":
    raise SystemExit(main())
