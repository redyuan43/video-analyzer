import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_audio_transcription
from video_analyzer.asr_providers import ASRStrategyResult
from video_analyzer.audio_processor import AudioTranscript


class AudioTranscriptionTests(unittest.TestCase):
    def test_mobile_audio_config_uses_firered_3dspeaker_provider(self):
        with patch.object(run_audio_transcription, "Config") as config_class:
            config = config_class.return_value
            run_audio_transcription.load_long_talk_config(
                argparse.Namespace(
                    config="config",
                    profile="deepseek_v4_flash",
                    asr_provider="firered_3dspeaker",
                )
            )

        args = config.update_from_args.call_args.args[0]
        self.assertEqual(args.task, "operation_manual")
        self.assertEqual(args.asr_provider, "firered_3dspeaker")

    def test_main_writes_transcription_without_summary_pipeline(self):
        transcript = AudioTranscript(
            text="hello",
            segments=[{"start": 0, "end": 1, "speaker": 0, "text": "hello"}],
            language="en",
        )
        result = ASRStrategyResult(
            strategy="provider:firered_3dspeaker",
            transcript=transcript,
            deep_transcript=transcript,
            providers_run=["firered_3dspeaker"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "demo.mp3"
            media.write_bytes(b"audio")
            output = root / "output"
            args = [
                "run_audio_transcription.py",
                str(media),
                "--output",
                str(output),
                "--profile",
                "deepseek_v4_flash",
            ]
            with (
                patch("sys.argv", args),
                patch.object(run_audio_transcription, "load_long_talk_config") as load_config,
                patch.object(run_audio_transcription, "extract_audio_to_wav", return_value=media),
                patch.object(
                    run_audio_transcription,
                    "transcribe_and_diarize_configured_audio",
                    return_value=(
                        transcript,
                        result,
                        {"final_speaker_count": 1},
                    ),
                ) as transcribe,
                patch.object(run_audio_transcription, "local_model_runtime_session") as runtime,
            ):
                runtime.return_value.__enter__.return_value = None
                runtime.return_value.__exit__.return_value = None
                load_config.return_value.config = {}
                self.assertEqual(run_audio_transcription.main(), 0)

            payload = json.loads((output / "transcription.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["pipeline"], "mobile-audio-transcription")
            self.assertEqual(payload["asr"]["providers_run"], ["firered_3dspeaker"])
            self.assertEqual(payload["speaker_count"], 1)
            self.assertTrue((output / "transcript_raw.json").is_file())
            self.assertTrue((output / "transcript_aligned.json").is_file())
            manifest = json.loads(
                (output / "transcription_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["provider"], "firered_3dspeaker")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["artifact_revision"], 1)
            for artifact in manifest["artifacts"].values():
                self.assertEqual(len(artifact["sha256"]), 64)
                self.assertGreater(artifact["size"], 0)
            self.assertFalse((output / "analysis.json").exists())
            self.assertFalse((output / "operation_manual.md").exists())
            self.assertFalse((output / "study_guide.json").exists())
            self.assertFalse((output / "audio_template_analysis.json").exists())
            self.assertFalse(transcribe.call_args.kwargs["use_asr_strategy"])

    def assert_failed_manifest(self, failure_target, failed_stage):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "demo.mp3"
            media.write_bytes(b"audio")
            output = root / "output"
            args = ["run_audio_transcription.py", str(media), "--output", str(output)]
            transcript = AudioTranscript(text="hello", segments=[], language="en")
            result = ASRStrategyResult(strategy="provider:test", transcript=transcript)
            with (
                patch("sys.argv", args),
                patch.object(
                    run_audio_transcription,
                    "load_long_talk_config",
                ) as load_config,
                patch.object(
                    run_audio_transcription,
                    "extract_audio_to_wav",
                    return_value=media,
                ),
                patch.object(
                    run_audio_transcription,
                    "transcribe_and_diarize_configured_audio",
                    return_value=(transcript, result, {}),
                ),
                patch.object(
                    run_audio_transcription,
                    "local_model_runtime_session",
                ) as runtime,
                patch.object(run_audio_transcription, failure_target, side_effect=RuntimeError("acceptance failure")),
            ):
                runtime.return_value.__enter__.return_value = None
                runtime.return_value.__exit__.return_value = None
                load_config.return_value.config = {}
                with self.assertRaisesRegex(RuntimeError, "acceptance failure"):
                    run_audio_transcription.main()
            manifest = json.loads((output / "transcription_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_stage"], failed_stage)
            self.assertEqual(manifest["error_type"], "RuntimeError")
            self.assertEqual(manifest["error"], "acceptance failure")

    def test_asr_failure_atomically_marks_manifest_failed(self):
        self.assert_failed_manifest(
            "transcribe_and_diarize_configured_audio",
            "asr",
        )

    def test_diarization_failure_atomically_marks_manifest_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "demo.mp3"
            media.write_bytes(b"audio")
            output = root / "output"
            args = ["run_audio_transcription.py", str(media), "--output", str(output)]
            with (
                patch("sys.argv", args),
                patch.object(run_audio_transcription, "load_long_talk_config") as load_config,
                patch.object(run_audio_transcription, "extract_audio_to_wav", return_value=media),
                patch.object(
                    run_audio_transcription,
                    "transcribe_and_diarize_configured_audio",
                    side_effect=run_audio_transcription.ParallelBranchError(
                        "diarization",
                        RuntimeError("3D-Speaker unavailable"),
                    ),
                ),
                patch.object(run_audio_transcription, "local_model_runtime_session") as runtime,
            ):
                runtime.return_value.__enter__.return_value = None
                runtime.return_value.__exit__.return_value = None
                load_config.return_value.config = {}
                with self.assertRaisesRegex(RuntimeError, "diarization branch failed"):
                    run_audio_transcription.main()

            manifest = json.loads(
                (output / "transcription_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["failed_stage"], "diarization_alignment")

    def test_diarization_error_report_atomically_marks_manifest_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "demo.mp3"
            media.write_bytes(b"audio")
            output = root / "output"
            args = ["run_audio_transcription.py", str(media), "--output", str(output)]
            transcript = AudioTranscript(
                text="hello",
                segments=[{"start": 0, "end": 1, "text": "hello"}],
                language="en",
            )
            result = ASRStrategyResult(strategy="provider:test", transcript=transcript)
            with (
                patch("sys.argv", args),
                patch.object(run_audio_transcription, "load_long_talk_config") as load_config,
                patch.object(run_audio_transcription, "extract_audio_to_wav", return_value=media),
                patch.object(
                    run_audio_transcription,
                    "transcribe_and_diarize_configured_audio",
                    return_value=(
                        transcript,
                        result,
                        {"error": "3D-Speaker unavailable"},
                    ),
                ),
                patch.object(run_audio_transcription, "local_model_runtime_session") as runtime,
            ):
                runtime.return_value.__enter__.return_value = None
                runtime.return_value.__exit__.return_value = None
                load_config.return_value.config = {}
                with self.assertRaisesRegex(RuntimeError, "3D-Speaker unavailable"):
                    run_audio_transcription.main()

            manifest_path = output / "transcription_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_stage"], "diarization_alignment")
            self.assertEqual(manifest["error_type"], "RuntimeError")
            self.assertIn("3D-Speaker unavailable", manifest["error"])
            self.assertFalse((output / "transcript_aligned.json").exists())


if __name__ == "__main__":
    unittest.main()
