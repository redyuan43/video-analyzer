#!/usr/bin/env python3
"""Generate chapter-by-chapter long-form notes from an existing run."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature
from video_analyzer.multidoc import (
    build_chapter_transcript_digest,
    format_timestamp,
    load_evidence,
    read_json,
    timestamp_to_seconds,
    trim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detailed chapter notes from an operation-manual run")
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile")
    parser.add_argument("--output", help="Output directory; default RUN_DIR/docs_analysis_chapters")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--text-model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-chars-per-chapter", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--deep-v2", action="store_true", help="Use larger chapter context and write deep_report_v2.md")
    parser.add_argument("--include-chapter-images", action="store_true", help="Extract and insert one representative frame per chapter")
    parser.add_argument("--video-file", type=Path, help="Video file used for chapter image extraction")
    parser.add_argument("--final-synthesis", action="store_true", help="Run a final report synthesis pass over chapter notes")
    parser.add_argument("--no-final-synthesis", action="store_true", help="Skip final report synthesis even when --deep-v2 is enabled")
    parser.add_argument("--final-max-tokens", type=int, default=10000)
    parser.add_argument("--format-markdown-final", action="store_true", help="Run final Markdown formatting with a small model")
    parser.add_argument("--no-format-markdown-final", action="store_true", help="Skip final Markdown formatting even when --deep-v2 is enabled")
    parser.add_argument("--formatter-base-url")
    parser.add_argument("--formatter-model")
    parser.add_argument("--formatter-max-tokens", type=int, default=6000)
    parser.add_argument("--review-report", action="store_true", help="Write review markdown/json for the final report")
    parser.add_argument("--no-review-report", action="store_true", help="Skip report review even when --deep-v2 is enabled")
    parser.add_argument("--review-base-url")
    parser.add_argument("--review-model", help="Optional model for semantic AI review; deterministic checks always run")
    parser.add_argument("--review-max-tokens", type=int, default=3000)
    parser.add_argument("--refresh-chapters", action="store_true", help="Regenerate chapter files even when cached files exist")
    parser.add_argument("--chapter-concurrency", type=int, default=1, help="Concurrent LLM chapter generation requests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve() if args.output else run_dir / "docs_analysis_chapters"
    chapters_dir = output_dir / ("chapters_v2" if args.deep_v2 else "chapters")
    chapters_dir.mkdir(parents=True, exist_ok=True)
    max_chars_per_chapter = args.max_chars_per_chapter or (22000 if args.deep_v2 else 5200)
    max_tokens = args.max_tokens or (6000 if args.deep_v2 else 2600)

    profile = Config(args.config).get_runtime_profile(args.profile)
    analysis = read_json(run_dir / "analysis.json")
    evidence = load_evidence(run_dir, analysis)
    metadata = analysis.get("metadata") or {}
    model = args.text_model or profile.get("text_model") or metadata.get("text_model")
    review_model = args.review_model or profile.get("review_model")
    base_url = args.llm_base_url or profile.get("llm_base_url") or metadata.get("llm_base_url")
    temperature = args.temperature if args.temperature is not None else resolve_temperature(profile, 0.2)
    extra_body = build_openai_extra_body(profile, base_url)
    review_extra_body = build_openai_extra_body(profile, args.review_base_url or base_url, prefix="review_")
    api_key = resolve_api_key(
        profile.get("api_key"),
        profile.get("text_api_key_env") or profile.get("api_key_env"),
        base_url,
    )
    client = GenericOpenAIAPIClient(api_key, base_url, extra_body=extra_body)
    include_chapter_images = args.include_chapter_images or args.deep_v2
    final_synthesis = (args.final_synthesis or args.deep_v2) and not args.no_final_synthesis
    format_markdown_final = (args.format_markdown_final or args.deep_v2) and not args.no_format_markdown_final
    review_report = (args.review_report or args.deep_v2) and not args.no_review_report
    chapter_assets = []
    if include_chapter_images:
        chapter_assets = prepare_chapter_assets(
            run_dir=run_dir,
            output_dir=output_dir,
            chapters=evidence["chapters"],
            video_file=args.video_file,
        )

    started = time.perf_counter()
    chapter_jobs = list(enumerate(evidence["chapters"], start=1))
    concurrency = max(1, min(args.chapter_concurrency, len(chapter_jobs) or 1))

    def generate_chapter(index: int, chapter: dict) -> tuple[int, dict, str, bool]:
        chapter_path = chapters_dir / f"chapter_{index:02d}.md"
        if not args.refresh_chapters and chapter_path.exists() and chapter_path.stat().st_size > 200:
            text = normalize_chapter_section_titles(chapter_path.read_text(encoding="utf-8"))
            print(f"[skip] chapter {index:02d}: {chapter_path}", flush=True)
            return index, chapter, text, True
        digest = build_chapter_transcript_digest(
            [chapter],
            evidence["transcript"],
            max_chars_per_chapter=max_chars_per_chapter,
        )
        prompt = build_chapter_prompt(
            index,
            len(evidence["chapters"]),
            chapter,
            digest,
            chapter_assets[index - 1] if index - 1 < len(chapter_assets) else None,
            deep_v2=args.deep_v2,
        )
        print(f"[run] chapter {index:02d}/{len(evidence['chapters'])}: {chapter.get('title')}", flush=True)
        client = GenericOpenAIAPIClient(api_key, base_url, extra_body=extra_body)
        response = client.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
        )
        text = normalize_chapter_section_titles(clean_chapter_output(response.get("response") or ""))
        chapter_path.write_text(text + "\n", encoding="utf-8")
        return index, chapter, text, False

    if concurrency == 1:
        chapter_outputs = [generate_chapter(index, chapter) for index, chapter in chapter_jobs]
    else:
        print(f"[run] chapter concurrency={concurrency}", flush=True)
        chapter_outputs = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(generate_chapter, index, chapter): index for index, chapter in chapter_jobs}
            for future in as_completed(futures):
                chapter_outputs.append(future.result())
        chapter_outputs.sort(key=lambda item: item[0])

    knowledge_notes = render_knowledge_notes(chapter_outputs)
    deep_report = render_deep_report(chapter_outputs, chapter_assets=chapter_assets, deep_v2=args.deep_v2)
    if final_synthesis:
        structured_deep_report = deep_report
        print("[run] final synthesis", flush=True)
        response = client.generate(
            prompt=build_final_synthesis_prompt(evidence, deep_report, chapter_assets),
            model=model,
            temperature=temperature,
            num_predict=args.final_max_tokens,
        )
        candidate_report = clean_report_output(response.get("response") or deep_report)
        candidate_report = ensure_chapter_images(candidate_report, chapter_assets)
        try:
            validate_deep_report(candidate_report, output_dir, chapter_assets)
            deep_report = candidate_report
        except Exception as exc:
            rejected_name = "deep_report_v2.synthesis_rejected.md" if args.deep_v2 else "deep_report.synthesis_rejected.md"
            (output_dir / rejected_name).write_text(candidate_report, encoding="utf-8")
            print(f"[warn] final synthesis rejected, using structured report: {exc}", flush=True)
            deep_report = structured_deep_report
    pre_format_name = "deep_report_v2.pre_format.md" if args.deep_v2 else "deep_report.pre_format.md"
    if format_markdown_final:
        (output_dir / pre_format_name).write_text(deep_report, encoding="utf-8")
    if format_markdown_final:
        print("[run] markdown format", flush=True)
        formatter_base_url = args.formatter_base_url or base_url
        formatter_client = GenericOpenAIAPIClient(
            api_key,
            formatter_base_url,
            extra_body=build_openai_extra_body(profile, formatter_base_url),
        )
        pre_format_report = deep_report
        formatted_report = format_markdown_document(
            formatter_client,
            args.formatter_model or model,
            temperature,
            args.formatter_max_tokens,
            deep_report,
        )
        formatted_report = ensure_chapter_images(formatted_report, chapter_assets)
        try:
            validate_deep_report(formatted_report, output_dir, chapter_assets)
            deep_report = formatted_report
        except Exception as exc:
            rejected_name = "deep_report_v2.format_rejected.md" if args.deep_v2 else "deep_report.format_rejected.md"
            (output_dir / rejected_name).write_text(formatted_report, encoding="utf-8")
            print(f"[warn] markdown format rejected, using deterministic report: {exc}", flush=True)
            deep_report = normalize_markdown_spacing(pre_format_report) + "\n"
    validate_deep_report(deep_report, output_dir, chapter_assets)

    knowledge_notes_name = "knowledge_notes_v2.md" if args.deep_v2 else "knowledge_notes.md"
    deep_report_name = "deep_report_v2.md" if args.deep_v2 else "deep_report.md"
    analysis_name = "analysis_v2.json" if args.deep_v2 else "analysis.json"
    (output_dir / knowledge_notes_name).write_text(knowledge_notes, encoding="utf-8")
    (output_dir / deep_report_name).write_text(deep_report, encoding="utf-8")
    review_outputs = {}
    if review_report:
        review_outputs = write_report_review(
            output_dir=output_dir,
            report_name=deep_report_name,
            report_text=deep_report,
            evidence=evidence,
            chapter_assets=chapter_assets,
            base_url=args.review_base_url or base_url,
            api_key=api_key,
            extra_body=review_extra_body,
            model=review_model,
            temperature=temperature,
            max_tokens=args.review_max_tokens,
        )
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "chapters": len(chapter_outputs),
        "llm_base_url": base_url,
        "text_model": model,
        "text_temperature": temperature,
        "deep_v2": args.deep_v2,
        "max_chars_per_chapter": max_chars_per_chapter,
        "max_tokens": max_tokens,
        "include_chapter_images": include_chapter_images,
        "final_synthesis": final_synthesis,
        "format_markdown_final": format_markdown_final,
        "formatter_model": args.formatter_model if format_markdown_final else None,
        "review_report": review_report,
        "review_model": review_model,
        "chapter_concurrency": concurrency,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "outputs": {
            "knowledge_notes": str(output_dir / knowledge_notes_name),
            "deep_report": str(output_dir / deep_report_name),
        },
    }
    if chapter_assets:
        summary["outputs"]["chapter_assets_manifest"] = str(output_dir / "chapter_assets_manifest.json")
    if format_markdown_final:
        summary["outputs"]["pre_format_deep_report"] = str(output_dir / pre_format_name)
    if review_outputs:
        summary["outputs"].update(review_outputs)
    (output_dir / analysis_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_chapter_prompt(
    index: int,
    total: int,
    chapter: dict,
    digest: str,
    chapter_asset: dict | None = None,
    deep_v2: bool = False,
) -> str:
    asset_note = ""
    if chapter_asset:
        asset_note = f"""

本章代表帧：
- 时间点：{chapter_asset['timestamp_label']}
- Markdown 路径：{chapter_asset['markdown_path']}
- 用途：最终报告会在本章时间范围后插入这张图。章节笔记里不要重复输出图片 Markdown，但分析时可以把它视为本章视觉锚点。
"""
    depth_rules = """
- 这是一份长视频深度报告的章节素材，不要压缩成泛泛摘要。
- 尽量覆盖本章里的主要问答转折、重要例子、技术判断、价值判断和主持人追问。
- 如果同一观点在本章内多次被推进，请写出推进关系，而不是只保留结论。
""" if deep_v2 else ""
    return f"""/no_think
你是长访谈内容分析员。请基于下面这一章的字幕证据，生成详细中文章节笔记。

章节：{index}/{total}
标题：{chapter.get('start')} - {chapter.get('end')} {chapter.get('title')}
{asset_note}

要求：
- 只输出最终章节笔记，不要输出 thinking process、分析过程、草稿计划或英文自述。
- 不要只概括一句话；要解释这章的论点、推理链、例子和隐含判断。
- 保留关键时间戳。
- 如果本章涉及人物、组织、产品或技术判断，请拆成清晰小节。
{depth_rules}
- 输出结构固定为：
  1. 本章主旨
  2. 关键观点
  3. 重要例子与证据
  4. 可复用洞察
  5. 证据边界与易误读点

章节字幕证据：
{digest}
""".strip()


def clean_chapter_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    markers = [
        r"\*\*1\.\s*本章主旨\*\*",
        r"^1[\.、]\s*本章主旨",
        r"^#*\s*本章主旨",
    ]
    best_start = None
    for pattern in markers:
        matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
        if matches:
            best_start = max(best_start or 0, matches[-1].start())
    if best_start is not None:
        text = text[best_start:].strip()
    text = re.sub(r"(?is)^here'?s a thinking process:.*?(?=\*\*1\.\s*本章主旨\*\*|^1[\.、]\s*本章主旨|^#*\s*本章主旨)", "", text).strip()
    text = re.sub(r"(?is)\n\s*(?:\d+\.\s+)?\*\*Final (?:Review|Check).*?$", "", text).strip()
    text = re.sub(r"(?is)\n\s*(?:\d+\.\s+)?Final (?:Review|Check).*?$", "", text).strip()
    text = re.sub(r"(?is)\n\s*Self-Correction/Refinement.*?$", "", text).strip()
    return text


def normalize_chapter_section_titles(text: str) -> str:
    text = re.sub(r"(?m)^(\s*(?:#+\s*)?5[\.、]\s*)需复核或容易误读的点\s*$", r"\1证据边界与易误读点", text)
    text = re.sub(r"(?m)^(\s*\*\*5[\.、]\s*)需复核或容易误读的点(\*\*)\s*$", r"\1证据边界与易误读点\2", text)
    return text


def clean_report_output(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:markdown|md)?\s*\n(?P<body>.*)\n```\s*$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group("body").strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"(?m)^#\s+", text)
    if match:
        text = text[match.start():].strip()
    return normalize_markdown_spacing(normalize_chapter_section_titles(text))


def render_knowledge_notes(chapter_outputs: list[tuple[int, dict, str, bool]]) -> str:
    lines = [
        "# 逐章知识笔记",
        "",
        "本文档按原视频章节逐段展开，避免把长访谈压缩成少数概览章节。",
        "",
        "## 章节目录",
        "",
    ]
    for index, chapter, _text, _cached in chapter_outputs:
        lines.append(f"- {index:02d}. {chapter.get('start')} - {chapter.get('end')} {chapter.get('title')}")
    lines.append("")
    for index, chapter, text, _cached in chapter_outputs:
        lines.extend(
            [
                f"## {index:02d}. {chapter.get('start')} - {chapter.get('end')} {chapter.get('title')}",
                "",
                text.strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_deep_report(
    chapter_outputs: list[tuple[int, dict, str, bool]],
    chapter_assets: list[dict] | None = None,
    deep_v2: bool = False,
) -> str:
    lines = [
        "# 深度分析报告 v2" if deep_v2 else "# 深度分析报告",
        "",
        "这份报告采用逐章展开方式生成，重点保留长访谈中的中后段信息密度。"
        if not deep_v2
        else "这份报告采用大上下文逐章分析和全局整合生成，并为每章配有视频代表帧，方便回到原片核查。",
        "",
        "## 总体结构",
        "",
    ]
    for index, chapter, _text, _cached in chapter_outputs:
        lines.append(f"- {index:02d}. {chapter.get('start')} - {chapter.get('end')} {chapter.get('title')}")
    lines.extend(["", "## 逐章分析", ""])
    for index, chapter, text, _cached in chapter_outputs:
        asset = chapter_assets[index - 1] if chapter_assets and index - 1 < len(chapter_assets) else None
        lines.extend(
            [
                f"### {index:02d}. {chapter.get('title')}",
                "",
                f"时间范围：{chapter.get('start')} - {chapter.get('end')}",
                "",
            ]
        )
        if asset:
            lines.extend(
                [
                    f"![第{index:02d}章插图：{chapter.get('title')}（{asset['timestamp_label']}）]({asset['markdown_path']})",
                    "",
                    f"*图：视频 {asset['timestamp_label']} 处的章节代表帧。*",
                    "",
                ]
            )
        lines.extend([text.strip(), ""])
    lines.extend(
        [
            "## 阅读提示",
            "",
            "- 本报告基于作者字幕和页面章节生成，评论只应作为社区反馈参考。",
            "- 每章均保留时间范围，方便回到原视频核查。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def prepare_chapter_assets(
    run_dir: Path,
    output_dir: Path,
    chapters: list[dict],
    video_file: Path | None = None,
) -> list[dict]:
    assets_dir = output_dir / "chapter_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    resolved_video = video_file.expanduser().resolve() if video_file else find_video_file(run_dir)
    fallback_frames = load_existing_frames(run_dir)
    assets = []
    for index, chapter in enumerate(chapters, start=1):
        timestamp = representative_timestamp(chapter)
        target = assets_dir / f"chapter_{index:02d}.jpg"
        source = "ffmpeg_extract"
        if resolved_video:
            try:
                extract_frame(resolved_video, timestamp, target)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"[warn] ffmpeg extract failed for chapter {index:02d}: {exc}", flush=True)
                source = copy_nearest_existing_frame(fallback_frames, timestamp, target)
        else:
            source = copy_nearest_existing_frame(fallback_frames, timestamp, target)
        if not target.exists() or target.stat().st_size == 0:
            source = copy_nearest_existing_frame(fallback_frames, timestamp, target)
        markdown_path = target.relative_to(output_dir).as_posix()
        assets.append(
            {
                "chapter_index": index,
                "title": chapter.get("title") or "",
                "start": chapter.get("start") or "",
                "end": chapter.get("end") or "",
                "timestamp": round(timestamp, 3),
                "timestamp_label": format_timestamp(timestamp),
                "path": str(target),
                "markdown_path": markdown_path,
                "source": source,
            }
        )
    (output_dir / "chapter_assets_manifest.json").write_text(
        json.dumps({"assets": assets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return assets


def find_video_file(run_dir: Path) -> Path | None:
    candidates = [
        run_dir.parent / "video.mp4",
        run_dir / "video.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    for pattern in ("*.mp4", "*.webm", "*.mkv", "*.mov"):
        matches = sorted(run_dir.parent.glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def representative_timestamp(chapter: dict) -> float:
    start = timestamp_to_seconds(chapter.get("start"))
    end = timestamp_to_seconds(chapter.get("end"))
    if end <= start:
        return max(start, 0.0)
    duration = end - start
    if duration <= 6:
        return start + duration / 2
    return start + min(max(duration * 0.2, 3.0), duration - 2.0)


def extract_frame(video_file: Path, timestamp: float, target: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_file),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(target),
    ]
    subprocess.run(command, check=True)


def load_existing_frames(run_dir: Path) -> list[dict]:
    manifest = run_dir / "frames_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        frames = payload.get("frames") if isinstance(payload, dict) else payload
        if isinstance(frames, list):
            return frames
    return []


def copy_nearest_existing_frame(frames: list[dict], timestamp: float, target: Path) -> str:
    if not frames:
        raise FileNotFoundError("No video file and no existing frames_manifest.json fallback available")
    nearest = min(frames, key=lambda item: abs(float(item.get("timestamp") or 0) - timestamp))
    source = Path(str(nearest.get("path") or ""))
    if not source.is_absolute():
        source = (target.parents[2] / source).resolve()
    shutil.copy2(source, target)
    return "nearest_existing_frame"


def build_final_synthesis_prompt(evidence: dict, draft_report: str, chapter_assets: list[dict]) -> str:
    return f"""/no_think
你是长访谈深度报告编辑。请基于下面的逐章草稿和证据，输出最终版中文 Markdown 深度报告。

硬性要求：
- 必须覆盖全部原始章节，不要合并成少数大章。
- 每章都保留“### XX. 标题”和“时间范围：开始 - 结束”。
- 每章时间范围后必须保留且只保留一张图片 Markdown，图片路径必须使用给定 manifest 中的 markdown_path。
- 不要删除时间戳、图片链接、章节标题。
- 不要输出 thinking process、草稿计划或英文自述。
- 评论只能进入“社区反馈/需复核/FAQ”性质内容，不能当作视频事实。
- 增强跨章节主线、关键概念解释、争议点和可回看索引，但不要编造证据。

章节图片 manifest：
{json.dumps(chapter_assets, ensure_ascii=False, indent=2)}

页面上下文：
{trim(evidence.get('page_context') or '', 8000)}

手册证据：
{trim(evidence.get('manual_evidence') or '', 8000)}

评论补充：
{trim(evidence.get('comments') or '', 4000)}

逐章草稿：
{trim(draft_report, 120000)}
""".strip()


def ensure_chapter_images(text: str, chapter_assets: list[dict]) -> str:
    if not chapter_assets:
        return text
    asset_paths = [asset["markdown_path"] for asset in chapter_assets]
    lines = [
        line
        for line in text.splitlines()
        if not any(path in line for path in asset_paths)
    ]
    output = []
    current_index = None
    current_block = []
    inserted_indices = set()

    def flush_block() -> None:
        if current_index is None:
            output.extend(current_block)
            return
        if current_index in inserted_indices:
            output.extend(current_block)
            return
        asset = chapter_assets[current_index - 1] if current_index - 1 < len(chapter_assets) else None
        output.extend(ensure_image_in_block(current_block, current_index, asset))
        if asset:
            inserted_indices.add(current_index)

    for line in lines:
        match = re.match(r"^###\s+(\d{2})\.", line)
        if match:
            flush_block()
            current_index = int(match.group(1))
            current_block = [line]
        else:
            current_block.append(line)
    flush_block()
    return normalize_markdown_spacing("\n".join(output)) + "\n"


def ensure_image_in_block(block: list[str], index: int, asset: dict | None) -> list[str]:
    if not asset:
        return block
    image_pattern = re.compile(rf"^!\[第{index:02d}章插图：.*?\]\(.+?\)\s*$")
    path_pattern = re.compile(rf"^.*{re.escape(asset['markdown_path'])}.*$")
    caption_pattern = re.compile(r"^\*图：视频\s+\d\d:\d\d:\d\d\s+处的章节代表帧。\*\s*$")
    filtered = [
        line
        for line in block
        if not image_pattern.match(line)
        and not path_pattern.match(line)
        and not caption_pattern.match(line)
    ]
    time_line_index = next((i for i, line in enumerate(filtered) if line.startswith("时间范围：")), None)
    image_lines = [
        "",
        f"![第{index:02d}章插图：{asset['title']}（{asset['timestamp_label']}）]({asset['markdown_path']})",
        "",
        f"*图：视频 {asset['timestamp_label']} 处的章节代表帧。*",
        "",
    ]
    if time_line_index is None:
        return filtered + image_lines
    return filtered[: time_line_index + 1] + image_lines + filtered[time_line_index + 1 :]


def format_markdown_document(
    client: GenericOpenAIAPIClient,
    model: str,
    temperature: float,
    max_tokens: int,
    text: str,
) -> str:
    normalized = normalize_markdown_spacing(text)
    blocks = split_markdown_blocks(normalized, max_chars=2600)
    formatted_blocks = []
    for index, block in enumerate(blocks, start=1):
        try:
            response = client.generate(
                prompt=build_markdown_format_prompt(block),
                model=model,
                temperature=temperature,
                num_predict=min(max_tokens, 1200),
            )
            formatted = clean_report_output(response.get("response") or block)
        except Exception as exc:
            print(f"[warn] markdown format block {index}/{len(blocks)} failed: {exc}", flush=True)
            formatted = normalize_markdown_spacing(block)
        formatted_blocks.append(formatted)
        print(f"[format] block {index}/{len(blocks)}", flush=True)
    return normalize_markdown_spacing("\n\n".join(formatted_blocks)) + "\n"


def split_markdown_blocks(text: str, max_chars: int = 2600) -> list[str]:
    lines = text.splitlines()
    chapter_blocks = []
    current = []
    for line in lines:
        if re.match(r"^###\s+\d{2}\.", line) and current:
            chapter_blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chapter_blocks.append("\n".join(current).strip())
    blocks = []
    for block in chapter_blocks:
        if len(block) <= max_chars:
            blocks.append(block)
            continue
        blocks.extend(split_long_markdown_block(block, max_chars))
    return [block for block in blocks if block]


def split_long_markdown_block(block: str, max_chars: int) -> list[str]:
    paragraphs = re.split(r"(\n\s*\n)", block)
    chunks = []
    current = ""
    for part in paragraphs:
        candidate = current + part
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            current = part
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_markdown_format_prompt(markdown: str) -> str:
    return f"""/no_think
你是 Markdown 排版整理器。只做格式规范化，不做内容创作。

硬性规则：
- 不改变事实、观点、章节标题、时间戳、图片链接和引用路径。
- 不新增段落，不删除段落，不合并章节。
- 保留所有 Markdown 图片语法。
- 只修复空行、列表缩进、标题层级间距、尾随空格和明显的 Markdown 排版问题。
- 只输出整理后的 Markdown 正文。

待整理 Markdown：
{markdown}
""".strip()


def normalize_markdown_spacing(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)(^#{1,6}\s+.+)\n(?!\n)", r"\1\n\n", text)
    text = re.sub(r"(?m)(^!\[.+?\]\(.+?\))\n(?!\n)", r"\1\n\n", text)
    return text.strip()


def validate_deep_report(text: str, output_dir: Path, chapter_assets: list[dict]) -> None:
    if not chapter_assets:
        return
    if re.search(r"</?think>|thinking process|草稿计划", text, flags=re.IGNORECASE):
        raise ValueError("Report contains thinking or draft-process leakage")
    for asset in chapter_assets:
        chapter_index = int(asset["chapter_index"])
        link_pattern = re.compile(rf"!\[[^\]]*\]\({re.escape(asset['markdown_path'])}\)")
        link_count = len(link_pattern.findall(text))
        if link_count != 1:
            raise ValueError(
                f"Expected one image link for chapter {chapter_index:02d}, found {link_count}: {asset['markdown_path']}"
            )
        image_path = output_dir / asset["markdown_path"]
        if not image_path.exists() or image_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing chapter image file: {image_path}")


def write_report_review(
    output_dir: Path,
    report_name: str,
    report_text: str,
    evidence: dict,
    chapter_assets: list[dict],
    base_url: str,
    api_key: str,
    extra_body: dict,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> dict[str, str]:
    review = build_deterministic_review(report_text, output_dir, evidence, chapter_assets)
    if model:
        try:
            client = GenericOpenAIAPIClient(api_key, base_url, extra_body=extra_body)
            response = client.generate(
                prompt=build_ai_review_prompt(report_text, evidence, review),
                model=model,
                temperature=temperature,
                num_predict=max_tokens,
            )
            review["ai_semantic_review"] = clean_report_output(response.get("response") or "")
            review["ai_semantic_review_status"] = "completed"
        except Exception as exc:
            review["ai_semantic_review"] = ""
            review["ai_semantic_review_status"] = f"failed: {exc}"
    else:
        review["ai_semantic_review"] = ""
        review["ai_semantic_review_status"] = "skipped; pass --review-model to enable"

    stem = Path(report_name).stem
    json_path = output_dir / f"{stem}.review.json"
    markdown_path = output_dir / f"{stem}.review.md"
    json_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_review_markdown(review), encoding="utf-8")
    return {
        "deep_report_review": str(markdown_path),
        "deep_report_review_json": str(json_path),
    }


def build_deterministic_review(
    report_text: str,
    output_dir: Path,
    evidence: dict,
    chapter_assets: list[dict],
) -> dict:
    expected_chapters = len(evidence.get("chapters") or [])
    chapter_headings = re.findall(r"^###\s+(\d{2})\.", report_text, flags=re.MULTILINE)
    time_ranges = re.findall(r"^时间范围：", report_text, flags=re.MULTILINE)
    image_links = re.findall(r"^!\[第\d{2}章插图：", report_text, flags=re.MULTILINE)
    issues = []

    def add_issue(severity: str, code: str, message: str, location: str = "") -> None:
        issues.append({"severity": severity, "code": code, "message": message, "location": location})

    if expected_chapters and len(chapter_headings) != expected_chapters:
        add_issue("error", "chapter_count", f"章节数应为 {expected_chapters}，实际为 {len(chapter_headings)}")
    if expected_chapters and len(time_ranges) != expected_chapters:
        add_issue("error", "time_range_count", f"时间范围数量应为 {expected_chapters}，实际为 {len(time_ranges)}")
    if chapter_assets and len(image_links) != len(chapter_assets):
        add_issue("error", "image_count", f"章节插图数量应为 {len(chapter_assets)}，实际为 {len(image_links)}")
    if re.search(r"</?think>|thinking process|草稿计划", report_text, flags=re.IGNORECASE):
        add_issue("error", "thinking_leak", "报告包含 thinking 或草稿过程泄漏")

    for asset in chapter_assets:
        link_pattern = re.compile(rf"!\[[^\]]*\]\({re.escape(asset['markdown_path'])}\)")
        link_count = len(link_pattern.findall(report_text))
        if link_count != 1:
            add_issue("error", "image_link_count", f"图片链接出现次数应为 1，实际为 {link_count}", asset["markdown_path"])
        image_path = output_dir / asset["markdown_path"]
        if not image_path.exists() or image_path.stat().st_size == 0:
            add_issue("error", "image_file_missing", "图片文件不存在或为空", str(image_path))

    human_required = collect_human_review_candidates(report_text)
    for item in human_required:
        add_issue("human_required", item["code"], item["message"], item["location"])

    evidence_markers = collect_evidence_boundary_markers(report_text)
    status = "pass"
    if any(issue["severity"] == "error" for issue in issues):
        status = "fail"
    elif any(issue["severity"] == "human_required" for issue in issues):
        status = "needs_human_review"

    return {
        "status": status,
        "summary": {
            "expected_chapters": expected_chapters,
            "chapter_headings": len(chapter_headings),
            "time_ranges": len(time_ranges),
            "chapter_images": len(image_links),
            "asset_files": sum(1 for asset in chapter_assets if (output_dir / asset["markdown_path"]).exists()),
        },
        "issues": issues,
        "evidence_boundary_markers": evidence_markers,
        "review_policy": {
            "ai": "自动复核结构、来源边界和高风险外部事实候选。",
            "human": "只需要复核 human_required 项，以及业务上不能接受错误的外部事实。",
        },
    }


def collect_human_review_candidates(report_text: str) -> list[dict]:
    company_pattern = re.compile(r"OpenAI|Anthropic|Google|DeepMind|Meta|SpaceX|xAI|Cursor|Manus|腾讯|字节|豆包", re.I)
    benchmark_pattern = re.compile(r"SWE-bench|AIME|benchmark|Benchmark|基准|分数|突破")
    temporal_or_risk_pattern = re.compile(r"需复核|容易误读|需确认|证据不足|可能|当前|最新|如今|今年|去年|202\d|录制于|收购|撤销|合并|现状|公开资料|动态")
    placeholder_pattern = re.compile(r"XXX|某[一-龥]{0,8}(?:大佬|人物|公司|实验室)")
    candidates = []
    lines = report_text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        compact = line.strip()
        if not compact:
            continue
        if re.match(r"^-\s+\d{2}\.\s+", compact):
            continue
        code = ""
        if placeholder_pattern.search(compact):
            code = "placeholder_or_masked_name"
        elif company_pattern.search(compact) and temporal_or_risk_pattern.search(compact):
            code = "external_company_fact"
        elif benchmark_pattern.search(compact) and temporal_or_risk_pattern.search(compact):
            code = "benchmark_or_score"
        elif re.search(r"202\d|录制于|收购|撤销|合并|最新|当前现状", compact):
            code = "time_sensitive_claim"
        if code:
            candidates.append(
                {
                    "code": code,
                    "message": compact[:220],
                    "location": f"line {line_number}",
                }
            )
        if len(candidates) >= 15:
            break
    return candidates


def collect_evidence_boundary_markers(report_text: str) -> list[dict]:
    markers = []
    pattern = re.compile(r"证据边界与易误读点|需复核|容易误读|需确认|证据不足|可能|不确定")
    for line_number, line in enumerate(report_text.splitlines(), start=1):
        compact = line.strip()
        if compact and pattern.search(compact):
            markers.append({"location": f"line {line_number}", "text": compact[:220]})
        if len(markers) >= 50:
            break
    return markers


def build_ai_review_prompt(report_text: str, evidence: dict, deterministic_review: dict) -> str:
    return f"""/no_think
你是视频深度报告复核员。请基于报告正文和证据摘要，输出中文复核结论。

要求：
- 不重写报告。
- 重点检查：章节是否遗漏、结论是否超出字幕/page context/评论证据、评论是否污染主结论、外部事实是否需要人工复核。
- 明确列出 human_required 项；如果没有，写“无”。
- 只输出 Markdown 复核报告。

确定性结构复核：
{json.dumps(deterministic_review, ensure_ascii=False, indent=2)}

页面上下文：
{trim(evidence.get('page_context') or '', 8000)}

手册证据：
{trim(evidence.get('manual_evidence') or '', 8000)}

报告正文：
{trim(report_text, 30000)}
""".strip()


def render_review_markdown(review: dict) -> str:
    lines = [
        "# 深度报告复核报告",
        "",
        f"- 状态：{review.get('status')}",
        f"- 章节数：{review['summary']['chapter_headings']}/{review['summary']['expected_chapters']}",
        f"- 时间范围：{review['summary']['time_ranges']}/{review['summary']['expected_chapters']}",
        f"- 章节插图：{review['summary']['chapter_images']}/{review['summary']['expected_chapters']}",
        f"- 图片文件：{review['summary']['asset_files']}/{review['summary']['expected_chapters']}",
        "",
        "## 复核分工",
        "",
        f"- AI/脚本：{review['review_policy']['ai']}",
        f"- 人工：{review['review_policy']['human']}",
        "",
        "## 问题清单",
        "",
    ]
    if review["issues"]:
        for issue in review["issues"]:
            location = f"（{issue['location']}）" if issue.get("location") else ""
            lines.append(f"- `{issue['severity']}` `{issue['code']}`{location}: {issue['message']}")
    else:
        lines.append("- 未发现结构性错误或必须人工复核项。")

    lines.extend(["", "## 证据边界提示", ""])
    markers = review.get("evidence_boundary_markers") or []
    if markers:
        for marker in markers[:20]:
            lines.append(f"- {marker['location']}: {marker['text']}")
    else:
        lines.append("- 未发现显式证据边界提示。")

    lines.extend(["", "## AI 语义复核", ""])
    if review.get("ai_semantic_review"):
        lines.append(review["ai_semantic_review"].strip())
    else:
        lines.append(f"- {review.get('ai_semantic_review_status')}")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
