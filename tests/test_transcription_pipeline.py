import hashlib
import json
import logging
import tempfile
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


if __name__ == "__main__":
    unittest.main()
