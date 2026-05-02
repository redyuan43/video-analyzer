import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audio_processor import AudioTranscript
from .clients.llm_client import LLMClient
from .frame import Frame
from .ocr import OCREvent

logger = logging.getLogger(__name__)


def read_context_file(path: Optional[str]) -> str:
    if not path:
        return ""
    context_path = Path(path).expanduser()
    if not context_path.exists():
        raise FileNotFoundError(f"Context file not found: {context_path}")
    return context_path.read_text(encoding="utf-8")


def build_operation_manual_prompt(
    frame_analyses: List[Dict[str, Any]],
    frames: List[Frame],
    transcript: Optional[AudioTranscript],
    ocr_events: List[OCREvent],
    page_context: str,
    language: str,
    frame_assets: Optional[Dict[int, str]] = None,
) -> str:
    frame_notes = []
    frame_assets = frame_assets or {}
    ocr_by_frame = {event.frame_number: event for event in ocr_events}
    for frame, analysis in zip(frames, frame_analyses):
        ocr = ocr_by_frame.get(frame.number)
        ocr_text = ocr.text if ocr and ocr.text else ""
        ocr_status = ocr.status if ocr else "not_run"
        image_path = frame_assets.get(frame.number, "")
        frame_notes.append(
            "\n".join(
                [
                    f"Frame {frame.number} at {frame.timestamp:.2f}s",
                    f"Markdown image path: {image_path}",
                    f"Visual analysis: {analysis.get('response', '')}",
                    f"OCR status: {ocr_status}",
                    f"OCR text: {ocr_text}",
                ]
            )
        )

    transcript_text = transcript.text if transcript else ""
    transcript_segments = json.dumps(transcript.segments, ensure_ascii=False, indent=2) if transcript else "[]"

    return f"""
You are converting an installation or operation video into a precise operating manual.
Write the final answer in {language}.

Rules:
- Produce a practical manual that a user can follow.
- Use a user-facing "总-分-总" structure: first explain the video's overall structure, then present illustrated steps, then end with checks and caveats.
- Put screenshots directly inside the relevant steps. Do not append a large screenshot gallery at the end.
- For each major step, choose 1 to 4 representative frame images from the provided Markdown image paths. Use adjacent images when they clarify before/after or multiple UI states.
- If several screenshots belong together, use a compact Markdown table with images side by side.
- Add a small Mermaid flowchart near the overview when the video has a clear workflow.
- Include timestamps as evidence for important steps.
- Separate facts observed in video/OCR/ASR from page-description context.
- If OCR, visual analysis, and ASR disagree, mark the item as "需复核" and do not invent details.
- Preserve exact commands, parameters, file names, URLs, UI labels, and model names.

Page description/context:
{page_context}

Transcript:
{transcript_text}

Transcript segments:
{transcript_segments}

Frame evidence:
{chr(10).join(frame_notes)}

Return Markdown with these sections:
1. 概览
2. 视频结构与流程图
3. 准备条件
4. 图文操作步骤
5. 关键参数和命令
6. 常见分支/错误/验证方式
7. 需复核项
""".strip()


def generate_operation_manual(
    client: LLMClient,
    text_model: str,
    frame_analyses: List[Dict[str, Any]],
    frames: List[Frame],
    transcript: Optional[AudioTranscript],
    ocr_events: List[OCREvent],
    page_context: str,
    language: str,
    temperature: float,
    frame_assets: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    prompt = build_operation_manual_prompt(
        frame_analyses=frame_analyses,
        frames=frames,
        transcript=transcript,
        ocr_events=ocr_events,
        page_context=page_context,
        language=language,
        frame_assets=frame_assets,
    )
    try:
        response = client.generate(
            prompt=prompt,
            model=text_model,
            temperature=temperature,
            num_predict=8000,
        )
        return {k: v for k, v in response.items() if k != "context"}
    except Exception as exc:
        logger.error("Error generating operation manual: %s", exc)
        return {"response": f"Error generating operation manual: {exc}"}


def prepare_frame_assets(frames: List[Frame], output_dir: Path) -> Dict[int, str]:
    evidence_dir = output_dir / "manual_assets"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    assets: Dict[int, str] = {}
    for frame in frames:
        target = evidence_dir / f"frame_{frame.number:03d}{frame.path.suffix or '.jpg'}"
        if frame.path.exists():
            shutil.copy2(frame.path, target)
            assets[frame.number] = target.relative_to(output_dir).as_posix()
    return assets


def write_frame_evidence_index(
    frames: List[Frame],
    output_dir: Path,
    ocr_events: List[OCREvent],
    frame_analyses: List[Dict[str, Any]],
    frame_assets: Dict[int, str],
) -> Path:
    ocr_by_frame = {event.frame_number: event for event in ocr_events}
    lines = [
        "# 帧证据索引",
        "",
        "这个文件用于复核与调试。面向用户的手册应优先阅读 `operation_manual.md`。",
        "",
    ]
    for frame, analysis in zip(frames, frame_analyses):
        image_path = frame_assets.get(frame.number)
        ocr = ocr_by_frame.get(frame.number)
        lines.extend(
            [
                f"## {frame.timestamp:.2f}s / Frame {frame.number}",
                "",
                f"![{frame.timestamp:.2f}s frame {frame.number}]({image_path})" if image_path else "",
                "",
                f"- OCR status: `{ocr.status if ocr else 'not_run'}`",
                f"- OCR provider: `{ocr.provider if ocr else ''}`",
                "",
                "### OCR 文本",
                "",
                (ocr.text if ocr and ocr.text else "_无_"),
                "",
                "### 视觉分析",
                "",
                analysis.get("response", "_无_"),
                "",
            ]
        )
    evidence_path = output_dir / "manual_evidence.md"
    evidence_path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")
    return evidence_path


def embed_step_images(manual_text: str, frames: List[Frame], frame_assets: Dict[int, str]) -> str:
    """Insert compact screenshot strips into step sections based on nearby timestamps."""
    if not frames or not frame_assets:
        return manual_text

    lines = manual_text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        result.append(line)
        if line.startswith("### ") and "步骤" in line:
            section_lines = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith("### ") and not lines[j].startswith("## "):
                section_lines.append(lines[j])
                j += 1
            section_text = "\n".join([line, *section_lines])
            if "manual_assets/" not in section_text:
                strip = _build_step_image_strip(section_text, frames, frame_assets)
                if strip:
                    result.extend(["", *strip, ""])
        i += 1
    return "\n".join(result).rstrip() + "\n"


def _build_step_image_strip(section_text: str, frames: List[Frame], frame_assets: Dict[int, str]) -> List[str]:
    selected = _select_section_frames(section_text, frames)
    selected = [frame for frame in selected if frame.number in frame_assets][:4]
    if not selected:
        return []

    headers = [f"{frame.timestamp:.0f}s" for frame in selected]
    separators = ["---" for _ in selected]
    images = [
        f"![{frame.timestamp:.0f}s / Frame {frame.number}]({frame_assets[frame.number]})"
        for frame in selected
    ]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separators) + " |",
        "| " + " | ".join(images) + " |",
    ]


def _select_section_frames(section_text: str, frames: List[Frame]) -> List[Frame]:
    timestamps = [int(minutes) * 60 + int(seconds) for minutes, seconds in _TIMESTAMP_RE.findall(section_text)]
    if timestamps:
        start = max(min(timestamps) - 3, 0)
        end = max(timestamps) + 3
        matches = [frame for frame in frames if start <= frame.timestamp <= end]
        if matches:
            return _spread_frames(matches, max_count=4)

    step_match = re.search(r"步骤\s*(\d+)", section_text)
    if not step_match:
        return []
    step_index = max(int(step_match.group(1)) - 1, 0)
    bucket_size = max(len(frames) // 4, 1)
    start_index = min(step_index * bucket_size, len(frames) - 1)
    end_index = len(frames) if step_index >= 3 else min(start_index + bucket_size, len(frames))
    return _spread_frames(frames[start_index:end_index], max_count=4)


def _spread_frames(frames: List[Frame], max_count: int) -> List[Frame]:
    if len(frames) <= max_count:
        return frames
    if max_count <= 1:
        return [frames[0]]
    step = (len(frames) - 1) / (max_count - 1)
    indexes = sorted({round(i * step) for i in range(max_count)})
    return [frames[index] for index in indexes]


_TIMESTAMP_RE = re.compile(r"\[(\d{1,2}):(\d{2})\]")
