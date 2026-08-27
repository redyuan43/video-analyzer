#!/usr/bin/env python3
"""Transcribe uploaded audio with the Video Analyzer long-talk ASR path."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_analyzer.artifacts import write_json, write_transcript_markdown  # noqa: E402
from video_analyzer.asr_providers import extract_audio_to_wav  # noqa: E402
from video_analyzer.config import Config  # noqa: E402
from video_analyzer.local_model_runtime import local_model_runtime_session  # noqa: E402
from video_analyzer.transcription_pipeline import (  # noqa: E402
    ParallelBranchError,
    transcribe_and_diarize_configured_audio,
)


logger = logging.getLogger("audio_transcription")
MANIFEST_SCHEMA_VERSION = 2
ARTIFACT_REVISION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the selected ASR provider and speaker diarization without downstream analysis."
    )
    parser.add_argument("media_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile")
    parser.add_argument("--source-name", default="")
    parser.add_argument(
        "--asr-provider",
        default=os.environ.get(
            "VIDEO_ANALYZER_AUDIO_ASR_PROVIDER", "firered_3dspeaker"
        ),
    )
    return parser.parse_args()


def load_long_talk_config(args: argparse.Namespace) -> Config:
    config = Config(args.config)
    config.update_from_args(
        argparse.Namespace(
            task="operation_manual",
            profile=args.profile,
            client=None,
            asr_provider=getattr(args, "asr_provider", "firered_3dspeaker"),
        )
    )
    return config


def transcript_payload(transcript: Any) -> dict[str, Any]:
    return {
        "text": str(getattr(transcript, "text", "") or ""),
        "segments": list(getattr(transcript, "segments", []) or []),
        "language": str(getattr(transcript, "language", "") or ""),
        "metadata": dict(getattr(transcript, "metadata", {}) or {}),
    }


def speaker_count(report: dict[str, Any], transcript: Any) -> int:
    for key in (
        "final_speaker_count",
        "detected_speaker_count",
        "assigned_speaker_count",
        "speaker_count",
    ):
        try:
            value = int(report.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    speakers = {
        str(segment.get("speaker"))
        for segment in (getattr(transcript, "segments", []) or [])
        if isinstance(segment, dict) and segment.get("speaker") not in (None, "")
    }
    return len(speakers)


def require_valid_diarization_alignment(
    transcript: Any,
    report: dict[str, Any],
) -> None:
    if not isinstance(report, dict):
        raise RuntimeError("speaker diarization produced no valid alignment report")
    error = str(report.get("error") or "").strip()
    if error:
        raise RuntimeError(f"speaker diarization failed: {error}")
    if not str(getattr(transcript, "text", "") or "").strip():
        raise RuntimeError("speaker diarization produced no valid aligned transcript")
    segments = getattr(transcript, "segments", None)
    if not isinstance(segments, list):
        raise RuntimeError("speaker diarization produced no valid aligned segments")
    for segment in segments:
        if not isinstance(segment, dict) or not str(segment.get("text") or "").strip():
            continue
        speaker = next(
            (
                segment.get(key)
                for key in ("speaker", "speaker_id", "speakerId", "Speaker")
                if segment.get(key) not in (None, "")
            ),
            None,
        )
        if speaker is not None and str(speaker).strip():
            return
    raise RuntimeError("speaker diarization produced no valid aligned segments")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def artifact_record(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output_dir)),
        "revision": ARTIFACT_REVISION,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    try:
        return run(args)
    except BaseException as exc:
        manifest_path = output_dir / "transcription_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "artifact_revision": ARTIFACT_REVISION,
                "stages": {},
                "artifacts": {},
            }
        failed_stage = str(manifest.pop("active_stage", None) or "initialization")
        manifest.update({
            "status": "failed",
            "failed_stage": failed_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        manifest.setdefault("stages", {})[failed_stage] = {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        }
        atomic_write_json(manifest_path, manifest)
        raise


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    media_path = Path(args.media_path).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not media_path.is_file():
        raise FileNotFoundError(f"missing media file: {media_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    config = load_long_talk_config(args)
    source_sha256 = sha256_file(media_path)
    manifest_path = output_dir / "transcription_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_revision": ARTIFACT_REVISION,
        "status": "running",
        "provider": args.asr_provider,
        "source_sha256": source_sha256,
        "stages": {},
        "artifacts": {},
    }
    manifest["active_stage"] = "audio_extraction"
    atomic_write_json(manifest_path, manifest)

    stage_started = time.perf_counter()
    audio_path = extract_audio_to_wav(media_path, output_dir)
    if audio_path is None:
        raise RuntimeError(f"audio extraction produced no audio stream: {media_path}")
    manifest["stages"]["audio_extraction"] = {
        "status": "succeeded",
        "elapsed_seconds": round(time.perf_counter() - stage_started, 3),
    }
    manifest["active_stage"] = "asr"
    atomic_write_json(manifest_path, manifest)

    asr_started = time.perf_counter()
    with local_model_runtime_session(config.config, logger, str(output_dir)):
        try:
            transcript, asr_result, diarization_report = (
                transcribe_and_diarize_configured_audio(
                    audio_path,
                    output_dir,
                    config,
                    use_asr_strategy=False,
                    logger=logger,
                    runtime_lock_held=True,
                )
            )
        except ParallelBranchError as exc:
            manifest["active_stage"] = (
                "diarization_alignment"
                if exc.branch == "diarization"
                else "asr"
            )
            atomic_write_json(manifest_path, manifest)
            raise
        if transcript is None or not str(getattr(transcript, "text", "") or "").strip():
            failures = "; ".join(asr_result.failures or asr_result.merge_notes)
            raise RuntimeError(
                f"Required transcript was not produced. {failures}".strip()
            )
        provider_raw = (
            transcript.metadata.get("transcript_raw")
            if isinstance(transcript.metadata, dict)
            else None
        )
        raw_transcript_data = (
            dict(provider_raw)
            if isinstance(provider_raw, dict)
            else transcript_payload(transcript)
        )
        raw_path = output_dir / "transcript_raw.json"
        atomic_write_json(raw_path, raw_transcript_data)
        manifest["stages"]["asr"] = {
            "status": "succeeded",
            "provider": args.asr_provider,
            "elapsed_seconds": round(time.perf_counter() - asr_started, 3),
            "metadata": asr_result.to_metadata(),
        }
        manifest["artifacts"]["transcript_raw"] = artifact_record(raw_path, output_dir)
        manifest["active_stage"] = "diarization_alignment"
        atomic_write_json(manifest_path, manifest)

        require_valid_diarization_alignment(transcript, diarization_report)
        manifest["stages"]["diarization_alignment"] = {
            "status": "succeeded",
            "elapsed_seconds": float(diarization_report.get("elapsed_seconds") or 0),
            "metadata": diarization_report,
        }

    asr_result.transcript = transcript
    transcript_data = transcript_payload(transcript)
    aligned_path = output_dir / "transcript_aligned.json"
    atomic_write_json(aligned_path, transcript_data)
    asr_metadata = asr_result.to_metadata()
    payload = {
        "pipeline": "mobile-audio-transcription",
        "source": {
            "name": args.source_name or media_path.name,
            "media_path": str(media_path),
            "audio_path": str(audio_path),
            "sha256": source_sha256,
        },
        "transcript": transcript_data,
        "asr": asr_metadata,
        "speaker_diarization": diarization_report,
        "speaker_count": speaker_count(diarization_report, transcript),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    transcript_path = write_transcript_markdown(transcript, output_dir / "transcript.md")
    orin_dir = output_dir / "orin"
    orin_dir.mkdir(parents=True, exist_ok=True)
    write_json(orin_dir / "transcript.json", transcript_data)
    write_json(orin_dir / "asr.json", asr_metadata)
    result_path = output_dir / "transcription.json"
    write_json(result_path, payload)
    manifest["status"] = "succeeded"
    manifest.pop("active_stage", None)
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    manifest["artifacts"].update(
        {
            "transcript_aligned": artifact_record(aligned_path, output_dir),
            "transcript_markdown": artifact_record(transcript_path, output_dir),
            "transcription": artifact_record(result_path, output_dir),
            "asr": artifact_record(orin_dir / "asr.json", output_dir),
        }
    )
    diarization_path = output_dir / "qa" / "speaker_diarization_report.json"
    if diarization_path.is_file():
        manifest["artifacts"]["speaker_diarization"] = artifact_record(diarization_path, output_dir)
    atomic_write_json(manifest_path, manifest)

    print(f"[transcription] provider: {args.asr_provider}")
    print(f"[transcription] transcript: {transcript_path}")
    print(f"[transcription] result: {result_path}")
    print(f"[done] run_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
