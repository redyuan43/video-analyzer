import hashlib
import json
import logging
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from video_analyzer import transcription_pipeline


class TranscriptionPipelineTests(unittest.TestCase):
    def test_provided_transcript_preserves_structure_and_skips_providers(self):
        source = {
            "text": "hello world",
            "language": "en",
            "segments": [
                {
                    "start": 0.125,
                    "end": 1.875,
                    "speaker": "speaker-a",
                    "text": "hello world",
                    "confidence": 0.99,
                }
            ],
            "metadata": {"device": "xnote", "nested": {"kept": True}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.json"
            raw = json.dumps(source, ensure_ascii=False).encode("utf-8")
            path.write_bytes(raw)
            with (
                patch.object(
                    transcription_pipeline,
                    "transcribe_with_strategy",
                ) as strategy_mock,
                patch.object(
                    transcription_pipeline,
                    "transcribe_with_provider_result",
                ) as provider_mock,
                patch.object(
                    transcription_pipeline,
                    "process_transcript_speakers",
                ) as diarization_mock,
            ):
                transcript, result = transcription_pipeline.transcribe_configured_audio(
                    None,
                    Path(tmp) / "output",
                    object(),
                    use_asr_strategy=False,
                    logger=logging.getLogger(__name__),
                    provided_transcript_path=path,
                )

        self.assertEqual(transcript.segments, source["segments"])
        self.assertEqual(transcript.metadata["device"], "xnote")
        self.assertEqual(transcript.metadata["nested"], {"kept": True})
        self.assertEqual(
            transcript.metadata["provided_transcript"]["source_sha256"],
            hashlib.sha256(raw).hexdigest(),
        )
        self.assertEqual(result.strategy, "provided_transcript")
        self.assertEqual(result.providers_run, [])
        strategy_mock.assert_not_called()
        provider_mock.assert_not_called()
        diarization_mock.assert_not_called()

    def test_standalone_diarization_runs_in_parallel_with_asr(self):
        barrier = threading.Barrier(2, timeout=1)
        transcript = transcription_pipeline.AudioTranscript(
            text="hello",
            segments=[{"start": 0, "end": 1, "text": "hello"}],
            language="en",
        )
        result = transcription_pipeline.ASRStrategyResult(
            strategy="provider:firered_asr2",
            transcript=transcript,
            providers_run=["firered_asr2"],
        )

        class FakeConfig:
            config = {}

            def get(self, key, default=None):
                return {
                    "asr": {"provider": "firered_asr2"},
                    "speaker_diarization": {
                        "enabled": True,
                        "assignment_enabled": True,
                        "backend": "wespeaker",
                    },
                }.get(key, default)

        def transcribe(*_args, **_kwargs):
            barrier.wait()
            time.sleep(0.05)
            return transcript, result

        def prepare(*_args, **_kwargs):
            barrier.wait()
            time.sleep(0.05)
            return ([{"start": 0, "end": 1, "speaker": "speaker-1"}], {})

        with (
            patch.object(
                transcription_pipeline,
                "transcribe_configured_audio",
                side_effect=transcribe,
            ),
            patch.object(
                transcription_pipeline,
                "prepare_speaker_assignment",
                side_effect=prepare,
            ),
            patch.object(
                transcription_pipeline,
                "apply_speaker_diarization",
                return_value=(transcript, {"final_speaker_count": 1}),
            ) as apply_mock,
        ):
            got_transcript, got_result, report = (
                transcription_pipeline.transcribe_and_diarize_configured_audio(
                    Path("/tmp/audio.wav"),
                    Path("/tmp/output"),
                    FakeConfig(),
                    use_asr_strategy=False,
                    logger=logging.getLogger(__name__),
                )
            )

        self.assertIs(got_transcript, transcript)
        self.assertIs(got_result, result)
        self.assertEqual(report["final_speaker_count"], 1)
        self.assertEqual(
            apply_mock.call_args.kwargs["prepared_assignment"][0][0]["speaker"],
            "speaker-1",
        )

    def test_combined_provider_keeps_its_internal_parallel_pipeline(self):
        transcript = transcription_pipeline.AudioTranscript(
            text="hello",
            segments=[{"start": 0, "end": 1, "speaker": "speaker-1", "text": "hello"}],
            language="en",
        )
        result = transcription_pipeline.ASRStrategyResult(
            strategy="provider:firered_3dspeaker",
            transcript=transcript,
            providers_run=["firered_3dspeaker"],
        )

        class FakeConfig:
            config = {}

            def get(self, key, default=None):
                return {
                    "asr": {"provider": "firered_3dspeaker"},
                    "speaker_diarization": {"enabled": True},
                }.get(key, default)

        with (
            patch.object(
                transcription_pipeline,
                "transcribe_configured_audio",
                return_value=(transcript, result),
            ),
            patch.object(
                transcription_pipeline,
                "prepare_speaker_assignment",
            ) as prepare_mock,
            patch.object(
                transcription_pipeline,
                "apply_speaker_diarization",
                return_value=(transcript, {"skipped": True}),
            ),
        ):
            transcription_pipeline.transcribe_and_diarize_configured_audio(
                Path("/tmp/audio.wav"),
                Path("/tmp/output"),
                FakeConfig(),
                use_asr_strategy=False,
                logger=logging.getLogger(__name__),
            )

        prepare_mock.assert_not_called()

    def test_parallel_marker_is_written_to_diarization_report(self):
        transcript = transcription_pipeline.AudioTranscript(
            text="hello",
            segments=[{"start": 0, "end": 1, "text": "hello"}],
            language="en",
        )

        class FakeConfig:
            config = {}

            def get(self, key, default=None):
                if key == "speaker_diarization":
                    return {"enabled": True, "backend": "wespeaker"}
                return default

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch.object(
                transcription_pipeline,
                "process_transcript_speakers",
                return_value=(transcript, {"enabled": True}),
            ):
                _refined, report = transcription_pipeline.apply_speaker_diarization(
                    Path("/tmp/audio.wav"),
                    transcript,
                    output_dir,
                    FakeConfig(),
                    logger=logging.getLogger(__name__),
                    prepared_assignment=([], {"enabled": True, "notes": []}),
                )

            stored = json.loads(
                (output_dir / "qa" / "speaker_diarization_report.json").read_text()
            )

        self.assertTrue(report["parallel_with_asr"])
        self.assertTrue(stored["parallel_with_asr"])


if __name__ == "__main__":
    unittest.main()
