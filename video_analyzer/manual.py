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
    asr_metadata: Optional[Dict[str, Any]],
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
    asr_metadata_text = json.dumps(asr_metadata or {}, ensure_ascii=False, indent=2)
    asr_rules = _build_asr_rules(asr_metadata or {})

    return f"""
You are converting an installation or operation video into a precise operating manual.
Write the final answer in {language}.

Rules:
- Produce a practical manual that a user can follow.
- Use a user-facing "总-分-总" structure: first explain the video's overall structure, then present illustrated steps, then end with checks and caveats.
- Put screenshots directly inside the relevant steps. Do not append a large screenshot gallery at the end.
- For each major step, choose 1 to 4 representative frame images from the provided Markdown image paths. Use adjacent images when they clarify before/after or multiple UI states.
- Screenshots must be real Markdown images, for example `![12s screenshot](manual_assets/frame_003.jpg)`.
- Never write screenshot paths as plain text, code spans, or table text such as `manual_assets/frame_003.jpg`.
- If several screenshots belong together, use a compact Markdown table with images side by side.
- Add a small Mermaid flowchart near the overview when the video has a clear workflow.
- Include timestamps as evidence for important steps.
- Separate facts observed in video/OCR/ASR from page context.
- Evidence priority: OCR/VL frame evidence > author subtitles > VibeVoice ASR > automatic subtitles > page description/metadata > pinned or uploader comments > ordinary comments.
- If subtitles and ASR disagree, mark the item as "需复核" unless OCR/VL evidence clearly resolves it.
- Treat comments as low-confidence supplemental context. Put comment-only material in "社区补充/常见问题"; do not use comments alone to create deterministic operation steps, commands, or parameters.
- If OCR, visual analysis, ASR, subtitles, and page context disagree, mark the item as "需复核" and do not invent details.
- Preserve exact commands, parameters, file names, URLs, UI labels, and model names.
{asr_rules}

Page context evidence package:
{page_context}

Transcript:
{transcript_text}

Transcript segments:
{transcript_segments}

ASR strategy evidence:
{asr_metadata_text}

Frame evidence:
{chr(10).join(frame_notes)}

Return Markdown with these sections:
1. 概览
2. 视频结构与流程图
3. 准备条件
4. 图文操作步骤
5. 关键参数和命令
6. 常见分支/错误/验证方式
7. 社区补充/常见问题
8. 需复核项
""".strip()


def _build_asr_rules(asr_metadata: Dict[str, Any]) -> str:
    fast_summary = asr_metadata.get("fast_transcript") or {}
    deep_summary = asr_metadata.get("deep_transcript") or {}
    has_deep = bool(deep_summary.get("text_length"))
    has_fast = bool(fast_summary.get("text_length"))
    if has_deep and has_fast:
        return "\n".join(
            [
                "- Use merged ASR as the main transcript. When fast ASR and VibeVoice disagree,",
                "  prefer VibeVoice for terminology and chapter meaning, but prefer fast ASR",
                "  segments for timestamps.",
                "- Treat ASR disagreements as uncertainty candidates first; resolve them only",
                "  when OCR, visual evidence, page context, or nearby frames support a clear answer.",
            ]
        )
    if has_deep:
        return "- Use VibeVoice ASR as long-context audio evidence, but avoid inventing timestamps it did not provide."
    return "- Use the transcript as ASR evidence and rely on frame timestamps for visual evidence."


def generate_operation_manual(
    client: LLMClient,
    text_model: str,
    frame_analyses: List[Dict[str, Any]],
    frames: List[Frame],
    transcript: Optional[AudioTranscript],
    asr_metadata: Optional[Dict[str, Any]],
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
        asr_metadata=asr_metadata,
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
        return _render_asset_references(manual_text)

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
            if not _has_rendered_asset_image(section_text):
                strip = _build_step_image_strip(section_text, frames, frame_assets)
                if strip:
                    result.extend(["", *strip, ""])
        i += 1
    return _render_asset_references("\n".join(result)).rstrip() + "\n"


def review_operation_manual_markdown(manual_text: str) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    code_span_images = re.findall(r"`\s*!\[[^\]]*\]\(manual_assets/[^)]+\)\s*`", manual_text)
    if code_span_images:
        issues.append(
            {
                "severity": "error",
                "code": "image_in_code_span",
                "message": "Manual contains Markdown images wrapped in code spans, so screenshots will not render.",
            }
        )

    raw_assets = []
    for match in re.finditer(r"manual_assets/frame_\d+\.(?:jpg|jpeg|png|webp)", manual_text):
        preceding_line = manual_text[manual_text.rfind("\n", 0, match.start()) + 1:match.start()]
        if not re.search(r"(^|[^`])!\[[^\]]*\]\($", preceding_line):
            raw_assets.append(match.group(0))
    if raw_assets:
        issues.append(
            {
                "severity": "error",
                "code": "raw_asset_path",
                "message": "Manual contains screenshot asset paths that are not rendered as Markdown images.",
            }
        )

    step_sections = re.findall(r"(?ms)^### .*?步骤.*?(?=^### |^## |\Z)", manual_text)
    for index, section in enumerate(step_sections, start=1):
        if "manual_assets/" in section and not _has_rendered_asset_image(section):
            issues.append(
                {
                    "severity": "error",
                    "code": "step_asset_not_rendered",
                    "message": f"Step section {index} references screenshots but does not render them as Markdown images.",
                }
            )
        elif "manual_assets/" not in section:
            issues.append(
                {
                    "severity": "warning",
                    "code": "step_missing_screenshot",
                    "message": f"Step section {index} has no screenshot evidence.",
                }
            )
    return issues


def _has_rendered_asset_image(text: str) -> bool:
    return bool(re.search(r"(^|[^`])!\[[^\]]*\]\(manual_assets/[^)]+\)", text))


def _render_asset_references(manual_text: str) -> str:
    """Convert model-emitted asset paths into real Markdown images."""
    manual_text = re.sub(
        r"`\s*(!\[[^\]]*\]\(manual_assets/[^)]+\))\s*`",
        r"\1",
        manual_text,
    )
    pattern = re.compile(r"`?(manual_assets/frame_\d+\.(?:jpg|jpeg|png|webp))`?")

    def replace(match: re.Match[str]) -> str:
        start = match.start()
        path = match.group(1)
        prefix = manual_text[max(0, start - 3):start]
        if prefix.endswith("]("):
            return match.group(0)
        label = Path(path).stem
        return f"![{label}]({path})"

    return pattern.sub(replace, manual_text)


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
