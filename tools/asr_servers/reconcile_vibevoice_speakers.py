#!/usr/bin/env python3
"""Re-run VibeVoice speaker reconciliation from saved chunk results."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VIBEVOICE_ROOT = Path("/home/ai/github/VibeVoice-bench")
DEFAULT_VIBEVOICE_PYTHON = Path("/home/ai/vllm-p40-nightly-test/bin/python")


def reexec_with_vibevoice_python() -> None:
    if os.environ.get("VIDEO_ANALYZER_RECONCILE_IN_VIBEVOICE_PYTHON") == "1":
        return
    configured = Path(os.environ.get("VIBEVOICE_PYTHON", str(DEFAULT_VIBEVOICE_PYTHON))).expanduser()
    if not configured.exists() or Path(sys.executable) == configured:
        return
    env = dict(os.environ)
    env["VIDEO_ANALYZER_RECONCILE_IN_VIBEVOICE_PYTHON"] = "1"
    os.execve(str(configured), [str(configured), __file__, *sys.argv[1:]], env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="operation-manual run directory")
    parser.add_argument("--vibevoice-root", type=Path, default=DEFAULT_VIBEVOICE_ROOT)
    parser.add_argument("--speaker-upper-bound", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="print result without writing files")
    parser.add_argument("--no-backup", action="store_true", help="skip backup before writing")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def backup_outputs(run_dir: Path) -> Path:
    backup_dir = run_dir / "qa" / f"vibevoice-speaker-reconcile-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        run_dir / "transcript.md",
        run_dir / "analysis.json",
        run_dir / "orin" / "transcript.json",
        run_dir / "orin" / "transcript.md",
        run_dir / "orin" / "asr.json",
    ]
    for path in paths:
        if path.is_file():
            destination = backup_dir / path.relative_to(run_dir).as_posix().replace("/", "__")
            shutil.copy2(path, destination)
    return backup_dir


def public_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        public.append(
            {
                "Start": round(float(segment.get("start_time") or 0.0), 2),
                "End": round(float(segment.get("end_time") or 0.0), 2),
                "Speaker": segment.get("speaker_id", "unknown"),
                "Content": text,
            }
        )
    return public


def transcript_text(segments: list[dict[str, Any]], fallback: str) -> str:
    if not segments:
        return fallback.strip()
    return "\n".join(str(segment.get("Content") or "").strip() for segment in segments if segment.get("Content")).strip()


def summarize(transcript: Any, quality_report: dict[str, Any], mode: str) -> dict[str, Any]:
    text = transcript.text or ""
    return {
        "language": transcript.language,
        "text_preview": text[:500],
        "text_length": len(text),
        "segment_count": len(transcript.segments or []),
        "metadata_keys": sorted(transcript.metadata.keys()),
        "quality_report": quality_report,
        "mode": mode,
        "provider": transcript.metadata.get("provider"),
    }


def find_vibevoice_metadata(metadata: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if metadata.get("chunk_results"):
        return dict(metadata), "metadata"
    deep_metadata = metadata.get("deep_transcript_metadata")
    if isinstance(deep_metadata, dict) and deep_metadata.get("chunk_results"):
        return dict(deep_metadata), "metadata.deep_transcript_metadata"
    raise SystemExit(
        "missing VibeVoice chunk_results in transcript metadata; checked metadata.chunk_results "
        "and metadata.deep_transcript_metadata.chunk_results"
    )


def update_vibevoice_metadata(
    metadata: dict[str, Any],
    source_metadata: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    updated = dict(metadata)
    if source_path == "metadata.deep_transcript_metadata":
        updated["deep_transcript_metadata"] = source_metadata
        updated["offline_reconcile_source"] = source_path
        updated["quality_report"] = source_metadata.get("quality_report")
        updated["mode"] = source_metadata.get("mode")
        return updated
    return source_metadata


def main() -> int:
    reexec_with_vibevoice_python()
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    vibevoice_root = args.vibevoice_root.expanduser().resolve()
    sys.path.insert(0, str(ROOT_DIR))
    sys.path.insert(0, str(vibevoice_root / "src" / "demo"))
    sys.path.insert(0, str(vibevoice_root / "src"))

    from vibevoice_asr_meeting_workflow import load_audio_array
    from vibevoice_asr_reconcile import reconcile_chunks
    from video_analyzer.artifacts import write_transcript_markdown
    from video_analyzer.audio_processor import AudioTranscript

    transcript_path = run_dir / "orin" / "transcript.json"
    if not transcript_path.is_file():
        raise SystemExit(f"missing transcript json: {transcript_path}")
    payload = load_json(transcript_path)
    metadata = dict(payload.get("metadata") or {})
    vibevoice_metadata, metadata_source = find_vibevoice_metadata(metadata)
    chunk_results = vibevoice_metadata["chunk_results"]

    audio_path = run_dir / "audio.wav"
    if not audio_path.is_file():
        raise SystemExit(f"missing audio file: {audio_path}")

    started = time.time()
    audio_array, sample_rate = load_audio_array(str(audio_path))
    reconciled = reconcile_chunks(
        chunk_results=chunk_results,
        full_audio=audio_array,
        sample_rate=int(sample_rate),
        speaker_upper_bound=max(1, args.speaker_upper_bound),
    )
    segments = public_segments(reconciled["segments"])
    text = transcript_text(segments, vibevoice_metadata.get("raw_text") or payload.get("text") or "")
    vibevoice_metadata.update(
        {
            "segments_before_offline_reconcile": len(payload.get("segments") or []),
            "offline_reconcile_elapsed_seconds": round(time.time() - started, 3),
            "audit_chunks": reconciled["audit_chunks"],
            "quality_report": reconciled["quality_report"],
            "speaker_map": reconciled.get("speaker_map"),
            "mode": "offline_ray_chunk_reconcile",
        }
    )
    metadata = update_vibevoice_metadata(metadata, vibevoice_metadata, metadata_source)
    language = payload.get("language") or (load_json(run_dir / "analysis.json").get("metadata", {}) if (run_dir / "analysis.json").is_file() else {}).get("audio_language") or "unknown"
    transcript = AudioTranscript(text=text, segments=segments, language=language, metadata=metadata)
    updated_payload = {"text": text, "segments": segments, "metadata": metadata}
    speakers = sorted({segment.get("Speaker") for segment in segments if segment.get("Speaker")})
    result = {
        "elapsed_seconds": round(time.time() - started, 3),
        "segment_count": len(segments),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "quality_report": reconciled["quality_report"],
    }

    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    backup_dir = None if args.no_backup else backup_outputs(run_dir)
    write_json(transcript_path, updated_payload)
    write_transcript_markdown(transcript, run_dir / "transcript.md")
    write_transcript_markdown(transcript, run_dir / "orin" / "transcript.md")
    write_json(run_dir / "qa" / "vibevoice_speaker_quality_report.json", reconciled["quality_report"])

    asr_path = run_dir / "orin" / "asr.json"
    asr = load_json(asr_path) if asr_path.is_file() else {}
    summary = summarize(transcript, reconciled["quality_report"], str(metadata.get("mode") or "offline_ray_chunk_reconcile"))
    asr.update(
        {
            "strategy": "provider:vibevoice",
            "providers_run": ["vibevoice"],
            "failures": [],
            "merge_notes": ["refreshed transcript with offline automatic speaker reconciliation"],
            "deep_transcript": summary,
            "merged_transcript": summary,
        }
    )
    write_json(asr_path, asr)

    analysis_path = run_dir / "analysis.json"
    if analysis_path.is_file():
        analysis = load_json(analysis_path)
        analysis["transcript"] = updated_payload
        analysis["asr"] = asr
        analysis.setdefault("metadata", {})["asr_quality_report"] = reconciled["quality_report"]
        write_json(analysis_path, analysis)

    if backup_dir:
        result["backup_dir"] = str(backup_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
