"""Artifact writers for operation-manual runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from .audio_processor import AudioTranscript


def write_transcript_markdown(transcript: Optional[AudioTranscript], path: Path) -> Optional[Path]:
    if not transcript:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transcript",
        "",
        f"- Language: {transcript.language or ''}",
        f"- Segments: {len(transcript.segments or [])}",
        "",
    ]
    segments = transcript.segments or []
    wrote_segment = False
    if segments:
        for segment in segments:
            text = str(
                first_present(
                    segment,
                    ("text", "Text", "content", "Content", "transcript", "Transcript", "raw_output", "raw_text"),
                )
                or ""
            ).strip()
            if not text:
                continue
            speaker = str(first_present(segment, ("speaker", "Speaker")) or "").strip()
            if speaker:
                text = f"{speaker}: {text}"
            start = first_present(segment, ("start_time", "start", "Start"))
            end = first_present(segment, ("end_time", "end", "End"))
            if start is not None or end is not None:
                lines.append(f"- [{format_timestamp(start)} - {format_timestamp(end)}] {text}")
            else:
                lines.append(f"- {text}")
            wrote_segment = True
    if not wrote_segment and transcript.text:
        lines.extend(["## Full Text", "", transcript.text])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def first_present(values: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in values and values[key] not in (None, ""):
            return values[key]
    return None


def format_timestamp(value: object) -> str:
    try:
        seconds = int(float(value or 0))
    except (TypeError, ValueError):
        seconds = 0
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def write_orin_artifacts(output_dir: Path, results: dict, page_context: str) -> Path:
    """Archive raw/intermediate evidence without changing the public outputs."""
    orin_dir = output_dir / "orin"
    orin_dir.mkdir(parents=True, exist_ok=True)

    metadata = results.get("metadata") or {}
    transcript = results.get("transcript")
    asr = results.get("asr")
    ocr_events = results.get("ocr_events") or []
    visual_events = results.get("visual_events") or []
    frame_analyses = results.get("frame_analyses") or []

    write_json(orin_dir / "metadata.json", metadata)
    if page_context:
        (orin_dir / "page_context.md").write_text(page_context, encoding="utf-8")
    if transcript:
        write_json(orin_dir / "transcript.json", transcript)
        write_transcript_markdown(
            AudioTranscript(
                text=transcript.get("text") or "",
                segments=transcript.get("segments") or [],
                language=metadata.get("audio_language") or "",
            ),
            orin_dir / "transcript.md",
        )
    if asr:
        write_json(orin_dir / "asr.json", asr)
    if ocr_events:
        write_json(orin_dir / "ocr_events.json", ocr_events)
        for index, event in enumerate(ocr_events):
            write_json(orin_dir / f"ocr_event_{index:03d}.json", event)
    if visual_events:
        write_json(orin_dir / "visual_events.json", visual_events)
        for index, event in enumerate(visual_events):
            write_json(orin_dir / f"visual_event_{index:03d}.json", event)
    if frame_analyses:
        write_json(orin_dir / "frame_analyses.json", frame_analyses)
        for index, analysis in enumerate(frame_analyses):
            write_json(orin_dir / f"frame_analysis_{index:03d}.json", analysis)

    copy_page_context_sources(metadata.get("page_context") or {}, orin_dir)
    return orin_dir


def copy_page_context_sources(page_context_metadata: dict, orin_dir: Path) -> None:
    source_paths = [
        page_context_metadata.get("description_file"),
        (page_context_metadata.get("comments") or {}).get("json_file"),
        (page_context_metadata.get("comments") or {}).get("selected_json_file"),
        (page_context_metadata.get("comments") or {}).get("markdown_file"),
        (page_context_metadata.get("subtitles") or {}).get("raw_file"),
        (page_context_metadata.get("subtitles") or {}).get("text_file"),
    ]
    for value in source_paths:
        if not value:
            continue
        source = Path(value)
        if not source.exists() or not source.is_file():
            continue
        target = orin_dir / source.name
        if target.resolve() == source.resolve():
            continue
        shutil.copy2(source, target)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
