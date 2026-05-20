#!/usr/bin/env python3
"""Generate a full Chinese narration script and WAV for video-analysis docs."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature

DEFAULT_RENDERER = Path.home() / "github/my-skills-repo/audio-narration-script/scripts/render_with_ivan_tts.py"
DEFAULT_MAX_SOURCE_CHARS = 90000
DEFAULT_TEMPERATURE = 0.2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="operation-manual run directory")
    parser.add_argument("--source", help="Markdown/PDF source path or basename. Defaults to operation_manual.md")
    parser.add_argument("--profile", help="Runtime profile from config/default_config.json or config.json")
    parser.add_argument("--config", default="config", help="Configuration directory containing optional config.json")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--text-model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output-dir", help="Output directory. Defaults to RUN_DIR/audio_narration")
    parser.add_argument("--speaker", default=os.getenv("IVAN_TTS_SPEAKER", "serena"))
    parser.add_argument(
        "--tts-concurrency",
        type=int,
        default=int(os.getenv("NARRATION_TTS_CONCURRENCY", "2")),
        help="Ivan TTS chunk render concurrency. Defaults to 2 to use both gateway workers.",
    )
    parser.add_argument("--renderer", type=Path, default=Path(os.getenv("AUDIO_NARRATION_RENDERER", DEFAULT_RENDERER)))
    parser.add_argument("--skip-tts", action="store_true", help="Write narration files without rendering WAV")
    parser.add_argument("--max-source-chars", type=int, default=DEFAULT_MAX_SOURCE_CHARS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    source_path = resolve_source(run_dir, args.source)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "audio_narration"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_text = read_source(source_path, args.max_source_chars)
    client, model, temperature = create_text_client(args)
    prompt = build_narration_prompt(source_path, source_text)

    print(f"[audio-narration] source: {source_path}", file=sys.stderr)
    script_md = generate_text(client, model, prompt, temperature)
    script_md = normalize_model_markdown(script_md)
    if len(script_md) < 120:
        raise RuntimeError("Generated narration script is unexpectedly short")

    script_path = output_dir / "narration_script.md"
    text_path = output_dir / "narration_script.txt"
    outline_path = output_dir / "narration_outline.md"
    script_path.write_text(script_md.rstrip() + "\n", encoding="utf-8")
    text_path.write_text(markdown_to_spoken_text(script_md), encoding="utf-8")
    outline_path.write_text(build_outline(script_md), encoding="utf-8")

    if args.skip_tts:
        print(f"[audio-narration] script: {script_path}")
        print(f"[audio-narration] text: {text_path}")
        return 0

    renderer = args.renderer.expanduser().resolve()
    if not renderer.exists():
        raise FileNotFoundError(
            f"Audio narration renderer not found: {renderer}. "
            "Set AUDIO_NARRATION_RENDERER to the audio-narration-script render_with_ivan_tts.py path."
        )

    audio_dir = output_dir / "audio_output"
    cmd = [
        sys.executable,
        str(renderer),
        "--input",
        str(text_path),
        "--output-dir",
        str(audio_dir),
        "--speaker",
        args.speaker,
        "--concurrency",
        str(max(1, args.tts_concurrency)),
    ]
    print(f"[audio-narration] rendering WAV with {renderer}", file=sys.stderr)
    subprocess.run(cmd, check=True)

    full_wav = audio_dir / "narration_full.wav"
    if not full_wav.exists() or full_wav.stat().st_size <= 44:
        raise RuntimeError(f"Narration WAV was not created: {full_wav}")

    print(f"[audio-narration] outline: {outline_path}")
    print(f"[audio-narration] script: {script_path}")
    print(f"[audio-narration] text: {text_path}")
    print(f"[audio-narration] wav: {full_wav}")
    return 0


def create_text_client(args: argparse.Namespace) -> tuple[Any, str, float]:
    profile = Config(args.config).get_runtime_profile(args.profile)
    base_url = args.llm_base_url or profile.get("llm_base_url")
    model = args.text_model or profile.get("text_model")
    if not base_url:
        raise ValueError("Runtime profile must provide llm_base_url, or pass --llm-base-url")
    if not model:
        raise ValueError("Runtime profile must provide text_model, or pass --text-model")
    client = GenericOpenAIAPIClient(
        resolve_api_key(
            api_key_env=profile.get("text_api_key_env") or profile.get("api_key_env"),
            api_url=base_url,
        ),
        base_url,
        extra_body=build_openai_extra_body(profile, base_url),
    )
    temperature = args.temperature if args.temperature is not None else resolve_temperature(profile, DEFAULT_TEMPERATURE)
    return client, model, temperature


def resolve_source(run_dir: Path, source: str | None) -> Path:
    if source:
        source_path = Path(source).expanduser()
        candidates: list[Path] = []
        if source_path.is_absolute():
            candidates.append(source_path)
        candidates.extend(
            [
                run_dir / source_path,
                run_dir / source_path.name,
                run_dir / "docs_analysis_chapters" / source_path.name,
                run_dir / "docs_analysis" / source_path.name,
            ]
        )
        if source_path.suffix.lower() == ".pdf":
            md_name = f"{source_path.stem}.md"
            candidates.extend(
                [
                    run_dir / md_name,
                    run_dir / "docs_analysis_chapters" / md_name,
                    run_dir / "docs_analysis" / md_name,
                ]
            )
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                if candidate.suffix.lower() == ".pdf":
                    mapped = candidate.with_suffix(".md")
                    if mapped.exists():
                        return mapped
                    mapped = first_named_markdown(run_dir, candidate.stem)
                    if mapped:
                        return mapped
                    raise FileNotFoundError(f"PDF source needs a matching Markdown file: {candidate}")
                return candidate
        mapped = first_named_markdown(run_dir, source_path.stem or source)
        if mapped:
            return mapped
        raise FileNotFoundError(f"Could not resolve source under {run_dir}: {source}")

    for rel_path in (
        "operation_manual.md",
        "docs_analysis_chapters/deep_report_v2.md",
        "docs_analysis/deep_report.md",
        "docs_analysis_chapters/knowledge_notes_v2.md",
        "docs_analysis/knowledge_notes.md",
    ):
        candidate = run_dir / rel_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No Markdown source found under {run_dir}")


def first_named_markdown(run_dir: Path, stem: str) -> Path | None:
    target = f"{stem}.md"
    for root in (run_dir, run_dir / "docs_analysis_chapters", run_dir / "docs_analysis"):
        candidate = root / target
        if candidate.exists():
            return candidate
    for candidate in run_dir.rglob("*.md"):
        if candidate.stem == stem:
            return candidate
    return None


def read_source(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"Source is empty: {path}")
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)].rstrip()
    tail = text[-int(max_chars * 0.3) :].lstrip()
    return f"{head}\n\n[中间内容因长度被截断]\n\n{tail}"


def build_narration_prompt(source_path: Path, source_text: str) -> str:
    return f"""
你是中文音频讲解稿作者。请把下面 Markdown 文档改写成适合直接朗读的中文长音频讲解稿。

要求：
- 输出 Markdown，标题为“# 音频讲解稿：{{标题}}”。
- 保留原文核心观点、结论、关键证据和必要数字，不要编造原文没有的事实。
- 按“讲给人听”的顺序组织，不要机械朗读原文。
- 开场用 2 到 4 句话说明这篇内容在解决什么问题。
- 主体拆成若干部分，每部分先给结论，再解释原因、例子和影响。
- 视觉依赖、图片路径、链接、脚注、代码和表格要转成听觉友好的说法；不重要的噪声可以省略。
- 句子短一些，段落控制在 2 到 5 句。
- 结尾用 3 到 6 句话复盘最重要的结论。
- 末尾追加“## 术语与读法”“## TTS 分段建议”“## 时长估算”。
- 不要加入背景音乐、音效或停顿指令。

来源文件：{source_path}

原文：
{source_text}
""".strip()


def generate_text(client: Any, model: str, prompt: str, temperature: float) -> str:
    response = client.generate(prompt=prompt, model=model, temperature=temperature, num_predict=12000)
    text = (response.get("response") or "").strip()
    if not text:
        raise RuntimeError("LLM returned empty narration script")
    return text


def normalize_model_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def markdown_to_spoken_text(markdown: str) -> str:
    lines: list[str] = []
    in_code = False
    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line:
            continue
        if re.match(r"^#{1,6}\s+", line):
            line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"^[>*\-\s]+", "", line)
        line = re.sub(r"^\d+[.)、]\s*", "", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        if line:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def build_outline(markdown: str) -> str:
    headings = [line.strip() for line in markdown.splitlines() if re.match(r"^#{1,3}\s+", line.strip())]
    if not headings:
        return "# 音频讲解稿大纲\n\n- 见 narration_script.md\n"
    body = "\n".join(f"- {strip_heading_prefix(heading)}" for heading in headings)
    return f"# 音频讲解稿大纲\n\n{body}\n"


def strip_heading_prefix(heading: str) -> str:
    return re.sub(r"^#{1,3}\s+", "", heading)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
