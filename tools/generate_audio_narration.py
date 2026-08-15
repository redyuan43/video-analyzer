#!/usr/bin/env python3
"""Generate a full Chinese narration script and WAV for video-analysis docs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import requests

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature

DEFAULT_MAX_SOURCE_CHARS = 90000
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TTS_TIMEOUT_SECONDS = 1800
MAX_TTS_INPUT_CHARS = 50000


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
    parser.add_argument("--tts-base-url")
    parser.add_argument("--tts-model")
    parser.add_argument("--voice")
    parser.add_argument("--tts-speed", type=float)
    parser.add_argument("--tts-timeout", type=int)
    parser.add_argument("--skip-tts", action="store_true", help="Write narration files without rendering WAV")
    parser.add_argument("--render-only", action="store_true", help="Render an existing narration_script.txt without calling the text model")
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

    script_path = output_dir / "narration_script.md"
    text_path = output_dir / "narration_script.txt"
    outline_path = output_dir / "narration_outline.md"
    if args.render_only:
        if not text_path.is_file() or text_path.stat().st_size == 0:
            raise FileNotFoundError(f"Narration text does not exist: {text_path}")
    else:
        source_text = read_source(source_path, args.max_source_chars)
        client, model, temperature = create_text_client(args)
        prompt = build_narration_prompt(source_path, source_text)

        print(f"[audio-narration] source: {source_path}", file=sys.stderr)
        script_md = normalize_model_markdown(generate_text(client, model, prompt, temperature))
        if len(script_md) < 120:
            raise RuntimeError("Generated narration script is unexpectedly short")
        script_path.write_text(script_md.rstrip() + "\n", encoding="utf-8")
        text_path.write_text(markdown_to_spoken_text(script_md), encoding="utf-8")
        outline_path.write_text(build_outline(script_md), encoding="utf-8")

    if args.skip_tts:
        print(f"[audio-narration] script: {script_path}")
        print(f"[audio-narration] text: {text_path}")
        return 0

    audio_dir = output_dir / "audio_output"
    audio_dir.mkdir(parents=True, exist_ok=True)
    full_wav = audio_dir / "narration_full.wav"
    tts = create_tts_config(args)
    spoken_text = text_path.read_text(encoding="utf-8").strip()
    timeline = render_tts(spoken_text, full_wav, tts)
    if not full_wav.exists() or full_wav.stat().st_size <= 44:
        raise RuntimeError(f"Narration WAV was not created: {full_wav}")
    duration_seconds = wav_duration_seconds(full_wav)
    timeline_path = output_dir / "narration_timeline.json"
    if timeline:
        timeline["audio"] = str(full_wav)
        timeline["duration_seconds"] = round(duration_seconds, 3)
        timeline_path.write_text(
            json.dumps(timeline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    metadata_path = output_dir / "narration_metadata.json"
    metadata = {
        "source": str(source_path),
        "script": str(script_path),
        "text": str(text_path),
        "audio": str(full_wav),
        "text_chars": len(spoken_text),
        "duration_seconds": round(duration_seconds, 3),
        "timeline": str(timeline_path) if timeline else "",
        "timeline_segments": len(timeline.get("segments") or []) if timeline else 0,
        "tts": {
            "base_url": tts["base_url"],
            "model": tts["model"],
            "voice": tts["voice"],
            "speed": tts["speed"],
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[audio-narration] outline: {outline_path}")
    print(f"[audio-narration] script: {script_path}")
    print(f"[audio-narration] text: {text_path}")
    print(f"[audio-narration] wav: {full_wav}")
    print(f"[audio-narration] duration_seconds: {duration_seconds:.3f}")
    if timeline:
        print(f"[audio-narration] timeline: {timeline_path}")
    print(f"[audio-narration] metadata: {metadata_path}")
    return 0


def create_tts_config(args: argparse.Namespace) -> dict[str, Any]:
    profile = Config(args.config).get_runtime_profile(args.profile)
    base_url = str(args.tts_base_url or profile.get("tts_base_url") or "").rstrip("/")
    model = str(args.tts_model or profile.get("tts_model") or "")
    voice = str(args.voice or profile.get("tts_voice") or "check_boards_sweet")
    speed = float(args.tts_speed if args.tts_speed is not None else profile.get("tts_speed") or 0.9)
    timeout = int(args.tts_timeout or profile.get("tts_timeout_seconds") or DEFAULT_TTS_TIMEOUT_SECONDS)
    if not base_url:
        raise ValueError("Runtime profile must provide tts_base_url")
    if not model:
        raise ValueError("Runtime profile must provide tts_model")
    if not 0.5 <= speed <= 2.0:
        raise ValueError("TTS speed must be between 0.5 and 2.0")
    return {
        "base_url": base_url,
        "model": model,
        "voice": voice,
        "speed": speed,
        "timeout": max(30, timeout),
        "api_key_env": str(profile.get("tts_api_key_env") or ""),
        "extra_params": dict(profile.get("tts_extra_params") or {"lang": "zh"}),
    }


def render_tts(text: str, output: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    if not text:
        raise ValueError("Narration text is empty")
    if len(text) > MAX_TTS_INPUT_CHARS:
        raise ValueError(f"Narration text exceeds {MAX_TTS_INPUT_CHARS} characters")
    base_url = str(config["base_url"]).rstrip("/")
    endpoint = f"{base_url}/audio/speech" if base_url.endswith("/v1") else f"{base_url}/v1/audio/speech"
    headers = {"Content-Type": "application/json"}
    api_key_env = str(config.get("api_key_env") or "")
    if api_key_env:
        token = os.environ.get(api_key_env)
        if not token:
            raise ValueError(f"Missing TTS API key environment variable: {api_key_env}")
        headers["Authorization"] = f"Bearer {token}"
    session = requests.Session()
    session.trust_env = False
    print(
        f"[audio-narration] rendering with {config['model']} voice={config['voice']} chars={len(text)}",
        file=sys.stderr,
    )
    payload = {
        "model": config["model"],
        "input": text,
        "voice": config["voice"],
        "response_format": "wav",
        "speed": config["speed"],
        "extra_params": config.get("extra_params") or {},
    }
    response = None
    for attempt in range(1, 4):
        response = session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=int(config["timeout"]),
        )
        if response.status_code not in {429, 503} or attempt == 3:
            break
        retry_after = response.headers.get("Retry-After") or ""
        delay = int(retry_after) if retry_after.isdigit() else 10 * attempt
        print(
            f"[audio-narration] TTS busy (HTTP {response.status_code}); retrying in {delay}s "
            f"({attempt}/3)",
            file=sys.stderr,
        )
        time.sleep(min(60, max(1, delay)))
    assert response is not None
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(f"TTS request failed with HTTP {response.status_code}: {detail}")
    if not response.content.startswith(b"RIFF"):
        raise RuntimeError("TTS service returned a non-WAV response")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(response.content)
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    request_id = str(response.headers.get("X-IndexTTS-Request-ID") or "").strip()
    if not request_id:
        return None
    return fetch_indextts_timeline(
        session,
        base_url,
        request_id,
        int(response.headers.get("X-IndexTTS-Segment-Count") or 0),
    )


def fetch_indextts_timeline(
    session: requests.Session,
    base_url: str,
    request_id: str,
    expected_segments: int,
) -> dict[str, Any]:
    jobs_base = (
        f"{base_url}/audio/speech/jobs"
        if base_url.endswith("/v1")
        else f"{base_url}/v1/audio/speech/jobs"
    )
    response = session.get(f"{jobs_base}/{request_id}", timeout=30)
    response.raise_for_status()
    payload = response.json()
    raw_segments = payload.get("segment_results") or []
    if payload.get("status") != "succeeded":
        raise RuntimeError(f"IndexTTS timeline job is {payload.get('status')}")
    if expected_segments and len(raw_segments) != expected_segments:
        raise RuntimeError(
            f"IndexTTS timeline segment count mismatch: {len(raw_segments)} != {expected_segments}"
        )
    segments = []
    for expected_index, item in enumerate(
        sorted(raw_segments, key=lambda value: int(value.get("index") or 0))
    ):
        index = int(item.get("index") or 0)
        start = item.get("start_seconds")
        end = item.get("end_seconds")
        duration = item.get("duration_seconds")
        if index != expected_index or start is None or end is None or duration is None:
            raise RuntimeError("IndexTTS timeline is incomplete")
        start_value = float(start)
        end_value = float(end)
        duration_value = float(duration)
        if start_value < 0 or end_value <= start_value or duration_value <= 0:
            raise RuntimeError("IndexTTS timeline contains invalid timestamps")
        segments.append(
            {
                "index": index,
                "text": str(item.get("text") or "").strip(),
                "start_seconds": round(start_value, 6),
                "end_seconds": round(end_value, 6),
                "duration_seconds": round(duration_value, 6),
            }
        )
    if not segments:
        raise RuntimeError("IndexTTS timeline contains no segments")
    return {
        "version": 1,
        "provider": "indextts",
        "request_id": request_id,
        "segment_count": len(segments),
        "segments": segments,
    }


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


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
