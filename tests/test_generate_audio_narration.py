import unittest
from unittest.mock import MagicMock

from tools.generate_audio_narration import fetch_indextts_timeline


class GenerateAudioNarrationTests(unittest.TestCase):
    def test_fetch_indextts_timeline_normalizes_segments(self):
        response = MagicMock()
        response.json.return_value = {
            "status": "succeeded",
            "segment_results": [
                {
                    "index": 1,
                    "text": "第二句。",
                    "start_seconds": 1.2,
                    "end_seconds": 1.7,
                    "duration_seconds": 0.5,
                },
                {
                    "index": 0,
                    "text": "第一句。",
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "duration_seconds": 1.0,
                },
            ],
        }
        session = MagicMock()
        session.get.return_value = response

        timeline = fetch_indextts_timeline(
            session,
            "http://127.0.0.1:8092",
            "speech-1",
            2,
        )

        self.assertEqual(timeline["request_id"], "speech-1")
        self.assertEqual(timeline["segment_count"], 2)
        self.assertEqual(
            [item["text"] for item in timeline["segments"]],
            ["第一句。", "第二句。"],
        )
        session.get.assert_called_once_with(
            "http://127.0.0.1:8092/v1/audio/speech/jobs/speech-1",
            timeout=30,
        )

    def test_fetch_indextts_timeline_rejects_missing_timestamps(self):
        response = MagicMock()
        response.json.return_value = {
            "status": "succeeded",
            "segment_results": [{"index": 0, "text": "第一句。"}],
        }
        session = MagicMock()
        session.get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "timeline is incomplete"):
            fetch_indextts_timeline(
                session,
                "http://127.0.0.1:8092/v1",
                "speech-1",
                1,
            )


if __name__ == "__main__":
    unittest.main()
