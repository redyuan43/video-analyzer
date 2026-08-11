import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .audio_processor import AudioTranscript
from .clients.llm_client import LLMClient
from .frame import Frame
from .ocr import OCREvent

logger = logging.getLogger(__name__)

DEFAULT_MAX_FRAME_EVIDENCE_CHARS = 70_000
DEFAULT_MAX_TRANSCRIPT_TEXT_CHARS = 90_000
DEFAULT_MAX_TRANSCRIPT_SEGMENTS_CHARS = 50_000
DEFAULT_MAX_PAGE_CONTEXT_CHARS = 30_000
DEFAULT_MANUAL_RESPONSE_TOKENS = 8_000
DEFAULT_MANUAL_CONTEXT_RESERVE_TOKENS = 2_048
MIN_MANUAL_PROMPT_CHARS = 6_000

KEY_VISUAL_TERMS = (
    "workflow",
    "flowchart",
    "diagram",
    "architecture",
    "structure",
    "framework",
    "brainstorming",
    "goal:",
    "agent",
    "流程",
    "流程图",
    "架构",
    "结构",
    "框架",
    "工作流",
    "技术架构",
    "人工验证",
    "任务完成",
    "目标",
)


def _truncate_evidence_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    value = value.strip()
    if len(value) <= max_chars:
        return value
    marker = f"\n[已截断，原始长度 {len(value)} 字符；完整证据见 manual_evidence.md / analysis.json]"
    return value[: max(0, max_chars - len(marker))].rstrip() + marker


def _truncate_balanced_text(value: str, max_chars: int, label: str) -> str:
    if max_chars <= 0:
        return ""
    value = value.strip()
    if len(value) <= max_chars:
        return value
    marker = f"\n[{label}已压缩，原始长度 {len(value)} 字符；保留开头、中段、结尾，完整内容见 transcript.md / analysis.json]\n"
    remaining = max_chars - len(marker)
    if remaining <= 0:
        return marker.strip()
    head = remaining // 3
    middle = remaining // 3
    tail = remaining - head - middle
    mid_start = max((len(value) // 2) - (middle // 2), head)
    return (
        value[:head].rstrip()
        + marker
        + value[mid_start : mid_start + middle].strip()
        + marker
        + value[-tail:].lstrip()
    )


def _format_transcript_segments(segments: List[Dict[str, Any]], max_chars: int) -> str:
    if not segments or max_chars <= 0:
        return "[]"

    def line_for(segment: Dict[str, Any]) -> str:
        start = segment.get("start", "")
        end = segment.get("end", "")
        text = str(segment.get("text") or "").replace("\n", " ").strip()
        return f"[{start}-{end}] {text}"

    lines = [line_for(segment) for segment in segments]
    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text

    marker = (
        f"[Transcript segments sampled: {len(segments)} segments, "
        f"original {len(full_text)} chars; full data in transcript.md / analysis.json]"
    )
    budget = max(max_chars - len(marker) - 2, 0)
    if budget <= 0:
        return marker

    selected: List[str] = []
    used_indices = set()
    step = max(len(lines) // 240, 1)
    for index in list(range(0, len(lines), step)) + [len(lines) - 1]:
        if index in used_indices:
            continue
        candidate = lines[index]
        projected = len("\n".join(selected + [candidate]))
        if projected > budget:
            break
        selected.append(candidate)
        used_indices.add(index)
    return marker + "\n" + "\n".join(selected)


def resolve_manual_prompt_char_budget(
    context_length: Any,
    configured_max_chars: Any = None,
) -> Optional[int]:
    if configured_max_chars not in (None, "", 0, "0"):
        return max(MIN_MANUAL_PROMPT_CHARS, int(configured_max_chars))
    if context_length in (None, "", 0, "0"):
        return None
    available_tokens = max(
        int(context_length)
        - DEFAULT_MANUAL_RESPONSE_TOKENS
        - DEFAULT_MANUAL_CONTEXT_RESERVE_TOKENS,
        MIN_MANUAL_PROMPT_CHARS,
    )
    return max(MIN_MANUAL_PROMPT_CHARS, int(available_tokens * 0.85))


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
    max_frame_evidence_chars: int = DEFAULT_MAX_FRAME_EVIDENCE_CHARS,
    max_prompt_chars: Optional[int] = None,
) -> str:
    if max_prompt_chars:
        evidence_budget = max(max_prompt_chars - 7_000, 2_000)
        max_frame_evidence_chars = min(
            max_frame_evidence_chars,
            max(600, int(evidence_budget * 0.28)),
        )
        max_transcript_text_chars = max(900, int(evidence_budget * 0.42))
        max_transcript_segments_chars = max(600, int(evidence_budget * 0.16))
        max_page_context_chars = max(500, int(evidence_budget * 0.08))
        max_asr_metadata_chars = max(400, int(evidence_budget * 0.06))
    else:
        max_transcript_text_chars = DEFAULT_MAX_TRANSCRIPT_TEXT_CHARS
        max_transcript_segments_chars = DEFAULT_MAX_TRANSCRIPT_SEGMENTS_CHARS
        max_page_context_chars = DEFAULT_MAX_PAGE_CONTEXT_CHARS
        max_asr_metadata_chars = 8_000

    frame_notes = []
    frame_assets = frame_assets or {}
    ocr_by_frame = {event.frame_number: event for event in ocr_events}
    per_frame_budget = max(max_frame_evidence_chars // max(len(frames), 1), 180)
    visual_budget = max(int(per_frame_budget * 0.62), 100)
    ocr_budget = max(int(per_frame_budget * 0.30), 60)
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
                    f"Visual analysis: {_truncate_evidence_text(analysis.get('response', ''), visual_budget)}",
                    f"OCR status: {ocr_status}",
                    f"OCR text: {_truncate_evidence_text(ocr_text, ocr_budget)}",
                ]
            )
        )
    frame_evidence = _truncate_evidence_text("\n\n".join(frame_notes), max_frame_evidence_chars)
    key_visual_anchors = render_key_visual_anchors(
        frames=frames,
        frame_analyses=frame_analyses,
        ocr_by_frame=ocr_by_frame,
        frame_assets=frame_assets,
    )

    transcript_text = _truncate_balanced_text(
        transcript.text if transcript else "",
        max_transcript_text_chars,
        "Transcript",
    )
    transcript_segments = (
        _format_transcript_segments(transcript.segments or [], max_transcript_segments_chars)
        if transcript
        else "[]"
    )
    asr_metadata_text = _truncate_balanced_text(
        json.dumps(asr_metadata or {}, ensure_ascii=False, indent=2),
        max_asr_metadata_chars,
        "ASR metadata",
    )
    asr_rules = _build_asr_rules(asr_metadata or {})
    page_context = _truncate_balanced_text(page_context, max_page_context_chars, "Page context")

    prompt = f"""
You are converting an installation or operation video into a precise operating manual.
Write the final answer in {language}.

Rules:
- Produce a practical manual that a user can follow.
- Use a user-facing "总-分-总" structure: first explain the video's overall structure, then present illustrated steps, then end with checks and caveats.
- Put screenshots directly inside the relevant steps. Do not append a large screenshot gallery at the end.
- For each major step, choose 1 to 4 representative frame images from the provided Markdown image paths. Use adjacent images when they clarify before/after or multiple UI states.
- If key visual anchors include workflow, structure, or architecture diagrams, show those screenshots near the overview or "视频结构与流程图" section.
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
{frame_evidence}

Key visual anchors:
{key_visual_anchors}

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
    if max_prompt_chars and len(prompt) > max_prompt_chars:
        reduced_budget = max(
            MIN_MANUAL_PROMPT_CHARS,
            max_prompt_chars - (len(prompt) - max_prompt_chars) - 512,
        )
        if reduced_budget < max_prompt_chars:
            return build_operation_manual_prompt(
                frame_analyses=frame_analyses,
                frames=frames,
                transcript=transcript,
                asr_metadata=asr_metadata,
                ocr_events=ocr_events,
                page_context=page_context,
                language=language,
                frame_assets=frame_assets,
                max_frame_evidence_chars=max_frame_evidence_chars,
                max_prompt_chars=reduced_budget,
            )
    return prompt


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
    no_think: bool = False,
    max_frame_evidence_chars: int = DEFAULT_MAX_FRAME_EVIDENCE_CHARS,
    max_prompt_chars: Optional[int] = None,
    fallback_client: Optional[LLMClient] = None,
    fallback_model: str = "",
    fallback_temperature: Optional[float] = None,
    fallback_status_callback: Optional[Callable[[str, str], None]] = None,
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
        max_frame_evidence_chars=max_frame_evidence_chars,
        max_prompt_chars=max_prompt_chars,
    )
    if no_think:
        prompt = f"/no_think\n{prompt}"
    try:
        response = client.generate(
            prompt=prompt,
            model=text_model,
            temperature=temperature,
            num_predict=8000,
        )
        return {
            **{k: v for k, v in response.items() if k != "context"},
            "prompt_chars": len(prompt),
            "fallback_used": False,
        }
    except Exception as exc:
        logger.error("Error generating operation manual: %s", exc)
        if fallback_client is None or not fallback_model:
            return {
                "response": f"Error generating operation manual: {exc}",
                "prompt_chars": len(prompt),
                "fallback_used": False,
                "primary_error": str(exc),
            }
        if fallback_status_callback:
            fallback_status_callback("running", f"primary text model failed: {exc}")
        logger.warning(
            "Primary text model failed; retrying operation manual with fallback model %s",
            fallback_model,
        )
        try:
            response = fallback_client.generate(
                prompt=prompt,
                model=fallback_model,
                temperature=(
                    temperature
                    if fallback_temperature is None
                    else fallback_temperature
                ),
                num_predict=8000,
            )
            if fallback_status_callback:
                fallback_status_callback(
                    "succeeded",
                    f"fallback model {fallback_model} completed",
                )
            return {
                **{k: v for k, v in response.items() if k != "context"},
                "prompt_chars": len(prompt),
                "fallback_used": True,
                "fallback_model": fallback_model,
                "primary_error": str(exc),
            }
        except Exception as fallback_exc:
            logger.error(
                "Fallback operation-manual model %s also failed: %s",
                fallback_model,
                fallback_exc,
            )
            if fallback_status_callback:
                fallback_status_callback(
                    "failed",
                    f"fallback model failed: {fallback_exc}",
                )
            return {
                "response": (
                    "Error generating operation manual: "
                    f"primary={exc}; fallback={fallback_exc}"
                ),
                "prompt_chars": len(prompt),
                "fallback_used": True,
                "fallback_model": fallback_model,
                "primary_error": str(exc),
                "fallback_error": str(fallback_exc),
            }


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
        *render_text_evidence_map(frames, frame_analyses, ocr_by_frame),
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
                render_raw_evidence_text(ocr.text if ocr else ""),
                "",
                "### 视觉分析",
                "",
                render_raw_evidence_text(analysis.get("response", "")),
                "",
            ]
        )
    evidence_path = output_dir / "manual_evidence.md"
    evidence_path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")
    return evidence_path


def render_text_evidence_map(frames: List[Frame], frame_analyses: List[Dict[str, Any]], ocr_by_frame: Dict[int, OCREvent]) -> List[str]:
    total = len(frames)
    ocr_ok = 0
    ocr_with_text = 0
    vl_with_text = 0
    rows = []
    for frame, analysis in zip(frames, frame_analyses):
        ocr = ocr_by_frame.get(frame.number)
        ocr_status = ocr.status if ocr else "not_run"
        ocr_text = clean_evidence_cell(ocr.text if ocr else "")
        vl_text = clean_evidence_cell(analysis.get("response", ""))
        if ocr_status in {"ok", "succeeded", "success"}:
            ocr_ok += 1
        if ocr_text != "_无_":
            ocr_with_text += 1
        if vl_text != "_无_":
            vl_with_text += 1
        rows.append(
            "| "
            + " | ".join([f"{frame.timestamp:.2f}s", f"Frame {frame.number}", f"`{ocr_status}`", ocr_text, vl_text])
            + " |"
        )
    return [
        "## 文字证据地图",
        "",
        f"- 覆盖帧数：{total}",
        f"- OCR 成功：{ocr_ok}/{total}",
        f"- OCR 有文本：{ocr_with_text}/{total}",
        f"- 视觉分析有内容：{vl_with_text}/{total}",
        "",
        "| 时间 | 帧 | OCR 状态 | OCR 摘要 | 视觉摘要 |",
        "| --- | --- | --- | --- | --- |",
        *(rows or ["| - | - | - | _无_ | _无_ |"]),
    ]


def clean_evidence_cell(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = strip_generated_markdown_images(text)
    text = re.sub(r"<img\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "_无_"
    text = text.replace("|", "\\|")
    if len(text) <= 90:
        return text
    return text[:87].rstrip() + "..."


def render_raw_evidence_text(value: str) -> str:
    text = strip_generated_markdown_images(str(value or "")).strip()
    if not text:
        return "_无_"
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}\n{text}\n{fence}"


def strip_generated_markdown_images(value: str) -> str:
    return re.sub(r"!\[[^\]]*\]\(\s*images/[^)]+\)", " ", str(value or ""))


def render_key_visual_anchors(
    frames: List[Frame],
    frame_analyses: List[Dict[str, Any]],
    ocr_by_frame: Dict[int, OCREvent],
    frame_assets: Dict[int, str],
    max_count: int = 6,
) -> str:
    selected = select_key_visual_frames(frames, frame_analyses, ocr_by_frame, frame_assets, max_count=max_count)
    if not selected:
        return "_无_"
    lines = []
    for frame, score, reason in selected:
        lines.append(
            f"- Frame {frame.number} at {frame.timestamp:.2f}s, score={score:.1f}, "
            f"path={frame_assets.get(frame.number, '')}, reason={reason}"
        )
    return "\n".join(lines)


def select_key_visual_frames(
    frames: List[Frame],
    frame_analyses: List[Dict[str, Any]],
    ocr_by_frame: Dict[int, OCREvent],
    frame_assets: Dict[int, str],
    max_count: int = 4,
) -> List[tuple[Frame, float, str]]:
    analyses_by_number = {
        frame.number: frame_analyses[index] if index < len(frame_analyses) else {}
        for index, frame in enumerate(frames)
    }
    candidates: List[tuple[Frame, float, str]] = []
    for frame in frames:
        if frame.number not in frame_assets:
            continue
        analysis_text = str((analyses_by_number.get(frame.number) or {}).get("response") or "")
        ocr_text = (ocr_by_frame.get(frame.number).text if ocr_by_frame.get(frame.number) else "") or ""
        score, reason = score_key_visual_text(f"{analysis_text}\n{ocr_text}")
        if score >= 2.5:
            candidates.append((frame, score, reason))

    candidates.sort(key=lambda item: (-item[1], item[0].timestamp))
    selected: List[tuple[Frame, float, str]] = []
    for candidate in candidates:
        frame = candidate[0]
        if any(abs(frame.timestamp - existing[0].timestamp) < 8 for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_count:
            break
    return sorted(selected, key=lambda item: item[0].timestamp)


def score_key_visual_text(value: str) -> tuple[float, str]:
    text = strip_generated_markdown_images(value).lower()
    score = 0.0
    matched: List[str] = []
    for term in KEY_VISUAL_TERMS:
        if term.lower() in text:
            score += 1.0
            matched.append(term)
    if "flowchart" in text or "流程图" in text:
        score += 2.0
    if "diagram" in text or "图" in text:
        score += 1.0
    if "goal:" in text and ("人工验证" in text or "任务完成" in text):
        score += 2.0
    if "architecture" in text or "技术架构" in text:
        score += 1.5
    reason = ", ".join(dict.fromkeys(matched[:6])) or "visual/ocr diagram signal"
    return score, reason


def _is_step_heading(line: str) -> bool:
    match = re.match(r"^#{3,5}\s+(.+)$", line.strip())
    if not match:
        return False
    title = match.group(1).strip()
    if "图文操作步骤" in title:
        return False
    return bool(re.search(r"步骤\s*(?:\d+|[一二三四五六七八九十]+)", title))


def _is_section_boundary(line: str) -> bool:
    return bool(re.match(r"^#{2,5}\s+", line))


def embed_step_images(
    manual_text: str,
    frames: List[Frame],
    frame_assets: Dict[int, str],
    frame_analyses: Optional[List[Dict[str, Any]]] = None,
    ocr_events: Optional[List[OCREvent]] = None,
) -> str:
    """Insert compact screenshot strips into step sections based on nearby timestamps."""
    if not frames or not frame_assets:
        return _render_asset_references(manual_text)

    lines = manual_text.splitlines()
    step_count = sum(1 for line in lines if _is_step_heading(line))
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_step_heading(line):
            section_lines = []
            j = i + 1
            while j < len(lines) and not _is_section_boundary(lines[j]):
                section_lines.append(lines[j])
                j += 1
            result.extend(_remap_step_section_images([line, *section_lines], frames, frame_assets, step_count))
            i = j
            continue
        result.append(line)
        i += 1
    rendered = _render_asset_references("\n".join(result))
    rendered = _remove_broad_step_heading_image_strips(rendered)
    rendered = _insert_key_visual_overview_images(
        rendered,
        frames,
        frame_assets,
        frame_analyses or [],
        ocr_events or [],
    )
    return rendered.rstrip() + "\n"


def review_operation_manual_markdown(manual_text: str) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    if manual_text.strip().startswith("Error generating operation manual:"):
        issues.append(
            {
                "severity": "error",
                "code": "manual_generation_error",
                "message": "Manual generation returned an error placeholder instead of a usable manual.",
            }
        )

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
        if not re.search(r"(^|[^`])!?\[[^\]]*\]\($", preceding_line):
            raw_assets.append(match.group(0))
    if raw_assets:
        issues.append(
            {
                "severity": "error",
                "code": "raw_asset_path",
                "message": "Manual contains screenshot asset paths that are not rendered as Markdown images.",
            }
        )

    step_sections = re.findall(r"(?ms)^### [^\n]*步骤[^\n]*.*?(?=^### |^## |\Z)", manual_text)
    for index, section in enumerate(step_sections, start=1):
        if "manual_assets/" in section and not _has_rendered_asset_image(section):
            issues.append(
                {
                    "severity": "error",
                    "code": "step_asset_not_rendered",
                    "message": f"Step section {index} references screenshots but does not render them as Markdown images.",
                }
            )
        elif _has_rendered_asset_image(section) and _section_image_time_mismatch(section):
            issues.append(
                {
                    "severity": "warning",
                    "code": "step_image_time_mismatch",
                    "message": f"Step section {index} contains screenshots whose frame timestamps appear outside the step evidence window.",
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


def _insert_key_visual_overview_images(
    manual_text: str,
    frames: List[Frame],
    frame_assets: Dict[int, str],
    frame_analyses: List[Dict[str, Any]],
    ocr_events: List[OCREvent],
) -> str:
    ocr_by_frame = {event.frame_number: event for event in ocr_events}
    selected = select_key_visual_frames(frames, frame_analyses, ocr_by_frame, frame_assets, max_count=4)
    selected_frames = [frame for frame, _score, _reason in selected if frame_assets.get(frame.number) not in manual_text]
    if not selected_frames:
        return manual_text

    strip = _build_step_image_strip_from_frames(selected_frames, frame_assets)
    if not strip:
        return manual_text
    block = "\n".join(["", "**关键画面：**", "", *strip, ""])

    lines = manual_text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^#{2,4}\s+.*(?:视频结构|流程图|整体流程)", line):
            return "\n".join(lines[: index + 1] + block.splitlines() + lines[index + 1 :])
    for index, line in enumerate(lines):
        if re.match(r"^#{2,4}\s+.*概览", line):
            return "\n".join(lines[: index + 1] + block.splitlines() + lines[index + 1 :])
    for index, line in enumerate(lines):
        if re.match(r"^#{2,4}\s+", line):
            return "\n".join(lines[: index + 1] + block.splitlines() + lines[index + 1 :])
    return manual_text.rstrip() + block


def _has_rendered_asset_image(text: str) -> bool:
    return bool(re.search(r"(^|[^`])!\[[^\]]*\]\(manual_assets/[^)]+\)", text))


def _remove_broad_step_heading_image_strips(manual_text: str) -> str:
    lines = manual_text.splitlines()
    output: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        index += 1
        if not re.match(r"^#{2,4}\s+.*图文操作步骤", line):
            continue

        while index < len(lines) and lines[index] == "":
            output.append(lines[index])
            index += 1

        if index < len(lines) and lines[index].startswith("|"):
            table_start = index
            while index < len(lines) and lines[index].startswith("|"):
                index += 1
            table = lines[table_start:index]
            if "manual_assets/" not in "\n".join(table):
                output.extend(table)
    return "\n".join(output)


def _remap_step_section_images(
    section_lines: List[str],
    frames: List[Frame],
    frame_assets: Dict[int, str],
    step_count: int,
) -> List[str]:
    section_text = "\n".join(section_lines)
    selected = _select_section_frames(section_text, frames, step_count)
    selected = [frame for frame in selected if frame.number in frame_assets][:4]
    if not selected:
        return section_lines

    image_numbers = _extract_asset_frame_numbers(section_text)
    selected_numbers = {frame.number for frame in selected}
    should_replace = not image_numbers or any(number not in selected_numbers for number in image_numbers)
    if not should_replace:
        return section_lines

    cleaned = _remove_step_image_blocks(section_lines)
    strip = _build_step_image_strip_from_frames(selected, frame_assets)
    if not strip:
        return cleaned
    return [cleaned[0], "", *strip, "", *cleaned[1:]]


def _extract_asset_frame_numbers(text: str) -> List[int]:
    return [
        int(match.group(1))
        for match in re.finditer(r"manual_assets/frame_(\d+)\.(?:jpg|jpeg|png|webp)", text)
    ]


def _remove_step_image_blocks(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            j = i
            table_lines = []
            while j < len(lines) and lines[j].startswith("|"):
                table_lines.append(lines[j])
                j += 1
            if "manual_assets/" in "\n".join(table_lines):
                cleaned_table = _clean_asset_images_from_table(table_lines)
                if cleaned_table:
                    cleaned.extend(cleaned_table)
                i = j
                continue
            if _is_empty_placeholder_table(table_lines):
                i = j
                continue
        if "manual_assets/" in line:
            i += 1
            continue
        cleaned.append(line)
        i += 1
    return _collapse_blank_lines(cleaned)


def _collapse_blank_lines(lines: List[str]) -> List[str]:
    result: List[str] = []
    for line in lines:
        if line == "" and result and result[-1] == "":
            continue
        result.append(line)
    while len(result) > 1 and result[-1] == "":
        result.pop()
    return result


def _clean_asset_images_from_table(lines: List[str]) -> List[str]:
    cleaned = [_clean_asset_images_from_line(line) for line in lines]
    body = "\n".join(cleaned)
    meaningful = re.sub(r"\|", " ", body)
    meaningful = re.sub(r":?-{3,}:?", " ", meaningful)
    meaningful = re.sub(r"\b\d{1,5}s\b", " ", meaningful)
    meaningful = re.sub(r"\s+", "", meaningful)
    if not meaningful:
        return []
    return cleaned


def _is_empty_placeholder_table(lines: List[str]) -> bool:
    if len(lines) < 2:
        return False
    body = "\n".join(lines[2:] if re.search(r":?-{3,}:?", lines[1]) else lines)
    if "manual_assets/" in body:
        return False
    without_pipes = re.sub(r"[|\s:：\-—>→=]", "", body)
    return without_pipes == ""


def _clean_asset_images_from_line(line: str) -> str:
    line = re.sub(r"!\[[^\]]*\]\(manual_assets/frame_\d+\.(?:jpg|jpeg|png|webp)\)", "", line)
    line = re.sub(r"\s*(?:→|->|=>)\s*(?=\|)", " ", line)
    line = re.sub(r"(?<=\|)\s*(?:→|->|=>)\s*", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def _render_asset_references(manual_text: str) -> str:
    """Convert model-emitted asset paths into real Markdown images."""
    manual_text = re.sub(
        r"！(?=\[[^\]]*\]\(manual_assets/[^)]+\))",
        "!",
        manual_text,
    )
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

    return _build_step_image_strip_from_frames(selected, frame_assets)


def _build_step_image_strip_from_frames(selected: List[Frame], frame_assets: Dict[int, str]) -> List[str]:
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


def _select_section_frames(section_text: str, frames: List[Frame], step_count: int | None = None) -> List[Frame]:
    window = _extract_section_time_window(section_text)
    if window:
        start, end = window
        matches = [frame for frame in frames if start <= frame.timestamp <= end]
        if matches:
            return _spread_frames(matches, max_count=4)

    step_match = re.search(r"步骤\s*(\d+)", section_text)
    if not step_match:
        return []
    step_index = max(int(step_match.group(1)) - 1, 0)
    bucket_count = max(step_count or 4, 1)
    bucket_size = max((len(frames) + bucket_count - 1) // bucket_count, 1)
    start_index = min(step_index * bucket_size, len(frames) - 1)
    end_index = len(frames) if step_index >= bucket_count - 1 else min(start_index + bucket_size, len(frames))
    return _spread_frames(frames[start_index:end_index], max_count=4)


def _extract_section_time_window(section_text: str) -> tuple[float, float] | None:
    section_text = _strip_existing_asset_images(section_text)
    timestamps: List[float] = []

    for start_text, end_text in _TIMESTAMP_RANGE_RE.findall(section_text):
        start = _parse_timestamp_text(start_text)
        end = _parse_timestamp_text(end_text)
        if start is not None and end is not None:
            return (max(min(start, end) - 3, 0), max(start, end) + 3)

    for value in _SECONDS_RE.findall(section_text):
        timestamps.append(float(value))

    for value in _TIMESTAMP_RE.findall(section_text):
        parsed = _parse_timestamp_text(value)
        if parsed is not None:
            timestamps.append(parsed)

    if not timestamps:
        return None
    if len(timestamps) == 1:
        timestamp = timestamps[0]
        return (max(timestamp - 3, 0), timestamp + 3)
    return (max(min(timestamps) - 3, 0), max(timestamps) + 3)


def _strip_existing_asset_images(section_text: str) -> str:
    return re.sub(
        r"!\[[^\]]*\]\(manual_assets/frame_\d+\.(?:jpg|jpeg|png|webp)\)",
        "",
        section_text,
    )


def _parse_timestamp_text(text: str) -> float | None:
    parts = [part for part in text.strip().split(":") if part != ""]
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return None


def _section_image_time_mismatch(section_text: str) -> bool:
    window = _extract_section_time_window(section_text)
    if not window:
        return False
    start, end = window
    for timestamp in _extract_image_alt_timestamps(section_text):
        if timestamp < start or timestamp > end:
            return True
    return False


def _extract_image_alt_timestamps(section_text: str) -> List[float]:
    timestamps = []
    for alt_text in re.findall(r"!\[([^\]]*)\]\(manual_assets/frame_\d+\.(?:jpg|jpeg|png|webp)\)", section_text):
        for value in _SECONDS_RE.findall(alt_text):
            timestamps.append(float(value))
        for value in _TIMESTAMP_RE.findall(alt_text):
            parsed = _parse_timestamp_text(value)
            if parsed is not None:
                timestamps.append(parsed)
    return timestamps


def _spread_frames(frames: List[Frame], max_count: int) -> List[Frame]:
    if len(frames) <= max_count:
        return frames
    if max_count <= 1:
        return [frames[0]]
    step = (len(frames) - 1) / (max_count - 1)
    indexes = sorted({round(i * step) for i in range(max_count)})
    return [frames[index] for index in indexes]


_TIMESTAMP_RANGE_RE = re.compile(
    r"\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(?:-|~|–|—|至|到)\s*\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?"
)
_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)")
_SECONDS_RE = re.compile(r"(?<!\d)(\d{1,5})\s*s\b")
