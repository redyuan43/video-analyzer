from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from .audio_processor import AudioTranscript
from .frame import Frame
from .ocr import OCREvent


def write_visual_review(
    *,
    output_dir: Path,
    video_path: Path | None,
    frames: list[Frame],
    transcript: AudioTranscript | None,
    ocr_events: list[OCREvent],
    frame_analyses: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    review_dir = output_dir / "visual_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    sheets = _write_contact_sheets(review_dir, output_dir, frames)
    html_path = output_dir / "visual_review.html"
    html_path.write_text(
        _render_visual_review_html(
            output_dir=output_dir,
            review_dir=review_dir,
            video_path=video_path,
            frames=frames,
            transcript=transcript,
            ocr_events=ocr_events,
            frame_analyses=frame_analyses,
            metadata=metadata,
            sheets=sheets,
        ),
        encoding="utf-8",
    )
    summary = {
        "path": str(html_path),
        "contact_sheet_dir": str(review_dir),
        "frame_count": len(frames),
        "contact_sheet_count": len(sheets),
        "frames_per_sheet": 9,
        "ab_test": {
            "name": "visual_review_contact_sheets",
            "baseline": "inspect individual frame images and separate transcript/evidence files",
            "treatment": "inspect visual_review.html with 3x3 contact sheets and evidence snippets",
            "primary_metric": "image_items_to_review",
            "observed_delta": {
                "image_items": len(sheets) - len(frames),
                "image_review_reduction_ratio": round(1.0 - (len(sheets) / max(len(frames), 1)), 4) if frames else 0.0,
            },
        },
    }
    return html_path, summary


def write_run_manifest(
    *,
    output_dir: Path,
    results: dict[str, Any],
    visual_review_path: Path | None = None,
    dedup_audit_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    metadata = results.get("metadata") or {}
    transcript = results.get("transcript") or {}
    ocr_events = results.get("ocr_events") or []
    frame_analyses = results.get("frame_analyses") or []
    manual = results.get("operation_manual") or {}
    lines = [
        "# RUN_MANIFEST",
        "",
        "## 先读顺序",
        "",
        "1. `RUN_MANIFEST.md`：确认任务、模型、证据覆盖和风险。",
        "2. `visual_review.html`：快速审阅原视频、关键帧、transcript、OCR/VL 摘要。",
        "3. `manual_evidence.md` 和 `transcript.md`：核对具体断言。",
        "4. `analysis.json` / `orin/*.json`：需要结构化细节时再读。",
        "",
        "## 运行概况",
        "",
        f"- Task: {metadata.get('task') or ''}",
        f"- Text model: {metadata.get('text_model') or metadata.get('model') or ''}",
        f"- ASR provider: {metadata.get('asr_provider') or ''}",
        f"- OCR provider: {metadata.get('ocr_provider') or ''}",
        f"- Frames extracted: {metadata.get('frames_extracted')}",
        f"- VL frames processed: {metadata.get('vl_frames_processed')}",
        f"- Transcript segments: {len(transcript.get('segments') or [])}",
        f"- OCR events: {len(ocr_events)}",
        f"- Frame analyses: {len(frame_analyses)}",
        "",
        "## 关键产物",
        "",
        "- `operation_manual.md` / `operation_manual.quality_failed.md`",
        "- `manual_evidence.md`",
        "- `transcript.md`",
        "- `analysis.json`",
        "- `qa/answer_index.json`",
    ]
    if visual_review_path:
        lines.append(f"- `{_relpath(visual_review_path, output_dir)}`")
    if dedup_audit_path:
        lines.append(f"- `{_relpath(dedup_audit_path, output_dir)}`")
    lines.extend(
        [
            "",
            "## A/B 参考收益",
            "",
        ]
    )
    for ab in _collect_ab_tests(metadata):
        observed = ab.get("observed_delta")
        if observed is None and "coverage_delta" in ab:
            observed = {"coverage_delta": ab.get("coverage_delta")}
        lines.append(f"- {ab.get('name')}: {json.dumps(observed or {}, ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            f"- Quality gate passed: {manual.get('quality_gate_passed')}",
            f"- Evidence path: {manual.get('evidence_path') or 'manual_evidence.md'}",
            "- 评论、网页补证据和模型推断只能作为低置信补充；操作步骤以 OCR/VL/ASR 证据为准。",
            "",
        ]
    )
    path = output_dir / "RUN_MANIFEST.md"
    body = "\n".join(lines).rstrip() + "\n"
    analysis_path = output_dir / "analysis.json"
    analysis_chars = analysis_path.stat().st_size if analysis_path.is_file() else len(json.dumps(results, ensure_ascii=False))
    manifest_delta = {
        "first_read_chars": len(body) - analysis_chars,
        "first_read_reduction_ratio": round(1.0 - (len(body) / max(analysis_chars, 1)), 4),
    }
    body = body.replace(
        "\n## 证据边界\n",
        f"\n- agent_first_run_manifest: {json.dumps(manifest_delta, ensure_ascii=False)}\n\n## 证据边界\n",
        1,
    )
    path.write_text(body, encoding="utf-8")
    summary = {
        "path": str(path),
        "chars": len(body),
        "analysis_json_chars": analysis_chars,
        "ab_test": {
            "name": "agent_first_run_manifest",
            "baseline": "agent reads analysis.json or scattered markdown artifacts first",
            "treatment": "agent reads concise RUN_MANIFEST.md before evidence drilldown",
            "primary_metric": "first_read_chars",
            "observed_delta": {
                "first_read_chars": len(body) - analysis_chars,
                "first_read_reduction_ratio": round(1.0 - (len(body) / max(analysis_chars, 1)), 4),
            },
        },
    }
    return path, summary


def _collect_ab_tests(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for name in ("frame_dedup_audit", "visual_review", "run_manifest"):
        payload = metadata.get(name) or {}
        ab = payload.get("ab_test") if isinstance(payload, dict) else None
        if isinstance(ab, dict):
            tests.append(ab)
    cue_anchors = (
        (((metadata.get("frame_extraction") or {}).get("strategy_observations") or {}).get("paper_algorithm_trace") or {}).get("cue_anchors")
        or {}
    )
    cue_ab = cue_anchors.get("ab_test") if isinstance(cue_anchors, dict) else None
    if isinstance(cue_ab, dict):
        tests.append({**cue_ab, "coverage_delta": cue_anchors.get("coverage_delta")})
    return tests


def _write_contact_sheets(review_dir: Path, output_dir: Path, frames: list[Frame]) -> list[Path]:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return []
    if not frames:
        return []
    sheets: list[Path] = []
    cols = 3
    rows = 3
    cell_width = 420
    label_height = 28
    ordered = sorted(frames, key=lambda item: item.timestamp)
    for sheet_index in range(0, len(ordered), cols * rows):
        batch = ordered[sheet_index : sheet_index + cols * rows]
        images = []
        for frame in batch:
            try:
                images.append((frame, Image.open(frame.path).convert("RGB")))
            except Exception:
                continue
        if not images:
            continue
        first = images[0][1]
        cell_height = int(first.height * cell_width / max(first.width, 1)) + label_height
        sheet = Image.new("RGB", (cols * cell_width, rows * cell_height), "black")
        draw = ImageDraw.Draw(sheet)
        for index, (frame, image) in enumerate(images):
            resized = image.resize((cell_width, cell_height - label_height))
            x = (index % cols) * cell_width
            y = (index // cols) * cell_height
            sheet.paste(resized, (x, y + label_height))
            draw.text((x + 8, y + 6), f"frame_{frame.number}  {format_seconds(frame.timestamp)}", fill="white")
        path = review_dir / f"contact_sheet_{len(sheets) + 1:03d}.jpg"
        sheet.save(path, quality=85)
        sheets.append(path)
    return sheets


def _render_visual_review_html(
    *,
    output_dir: Path,
    review_dir: Path,
    video_path: Path | None,
    frames: list[Frame],
    transcript: AudioTranscript | None,
    ocr_events: list[OCREvent],
    frame_analyses: list[dict[str, Any]],
    metadata: dict[str, Any],
    sheets: list[Path],
) -> str:
    video_tag = ""
    if video_path and video_path.exists():
        video_tag = f'<video src="{html.escape(os.path.relpath(video_path, output_dir))}" controls playsinline></video>'
    sheet_cards = "\n".join(
        f'<a href="{html.escape(_relpath(path, output_dir))}" target="_blank"><img src="{html.escape(_relpath(path, output_dir))}" loading="lazy"><span>{html.escape(path.name)}</span></a>'
        for path in sheets
    )
    frame_cards = "\n".join(
        f'<a href="{html.escape(_relpath(frame.path, output_dir))}" target="_blank"><img src="{html.escape(_relpath(frame.path, output_dir))}" loading="lazy"><span>frame_{frame.number} {format_seconds(frame.timestamp)}</span></a>'
        for frame in sorted(frames, key=lambda item: item.timestamp)[:240]
    )
    transcript_text = transcript.text if transcript else ""
    ocr_text = "\n".join(
        f"[{format_seconds(event.timestamp)}] frame_{event.frame_number}: {(event.text or '').strip()[:500]}"
        for event in ocr_events[:80]
        if (event.text or "").strip()
    )
    vl_text = "\n".join(
        f"[{format_seconds(float(item.get('timestamp') or 0.0))}] frame_{item.get('frame_number')}: {str(item.get('response') or item.get('summary') or '')[:500]}"
        for item in frame_analyses[:80]
    )
    counts = {
        "frames": len(frames),
        "contact_sheets": len(sheets),
        "ocr_events": len(ocr_events),
        "vl_events": len(frame_analyses),
        "pipeline_mode": (metadata.get("frame_selection") or {}).get("pipeline_mode"),
    }
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>visual review</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#101214;color:#e8e5dc;font:14px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{position:sticky;top:0;background:#101214;border-bottom:1px solid #2e3338;padding:14px 22px;z-index:2}}h1{{font-size:18px;margin:0}}main{{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(320px,.9fr);gap:18px;padding:18px;max-width:1600px;margin:0 auto}}video{{width:100%;max-height:58vh;background:#000;border:1px solid #2e3338}}section{{margin-bottom:20px}}h2{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}}.grid a{{display:block;position:relative;border:1px solid #2e3338;background:#181b1f;text-decoration:none;color:#e8e5dc}}.grid img{{width:100%;display:block}}.grid span{{position:absolute;left:0;right:0;bottom:0;background:rgba(16,18,20,.78);font-size:11px;padding:4px 6px}}pre{{white-space:pre-wrap;background:#181b1f;border:1px solid #2e3338;padding:12px;max-height:42vh;overflow:auto}}.counts{{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}}.counts span{{border:1px solid #3a4047;padding:4px 8px;background:#181b1f}}</style>
</head>
<body>
<header><h1>visual review</h1><div class="counts">{''.join(f'<span>{html.escape(str(k))}: {html.escape(str(v))}</span>' for k, v in counts.items())}</div></header>
<main>
<div>
<section>{video_tag}</section>
<section><h2>Contact Sheets</h2><div class="grid">{sheet_cards or '<p>No contact sheets.</p>'}</div></section>
<section><h2>Frames</h2><div class="grid">{frame_cards or '<p>No frames.</p>'}</div></section>
</div>
<div>
<section><h2>Transcript</h2><pre>{html.escape(transcript_text[:30000] or 'No transcript.')}</pre></section>
<section><h2>OCR Evidence</h2><pre>{html.escape(ocr_text or 'No OCR text events.')}</pre></section>
<section><h2>VL Evidence</h2><pre>{html.escape(vl_text or 'No VL events.')}</pre></section>
</div>
</main>
</body>
</html>
"""


def _relpath(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def format_seconds(value: float) -> str:
    seconds = int(max(float(value or 0.0), 0.0))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
