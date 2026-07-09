import tempfile
import unittest
from pathlib import Path

from video_analyzer.artifacts import write_transcript_markdown
from video_analyzer.audio_processor import AudioTranscript


class TranscriptArtifactTests(unittest.TestCase):
    def test_writes_vibevoice_content_segments(self):
        transcript = AudioTranscript(
            text="完整正文",
            segments=[
                {
                    "Start": 7.55,
                    "End": 21.34,
                    "Speaker": "Speaker A",
                    "Content": "大家好，我是罗永浩。",
                }
            ],
            language="zh",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transcript.md"
            write_transcript_markdown(transcript, path)

            content = path.read_text(encoding="utf-8")

        self.assertIn("- Segments: 1", content)
        self.assertIn("[00:00:07 - 00:00:21] Speaker A: 大家好，我是罗永浩。", content)

    def test_falls_back_to_full_text_when_segments_have_no_text(self):
        transcript = AudioTranscript(
            text="完整正文",
            segments=[{"Start": 0, "End": 1}],
            language="zh",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transcript.md"
            write_transcript_markdown(transcript, path)

            content = path.read_text(encoding="utf-8")

        self.assertIn("## Full Text", content)
        self.assertIn("完整正文", content)

    def test_writes_numeric_zero_speaker_label(self):
        transcript = AudioTranscript(
            text="完整正文",
            segments=[{"Start": 0, "End": 1, "Speaker": 0, "Content": "你好"}],
            language="zh",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transcript.md"
            write_transcript_markdown(transcript, path)

            content = path.read_text(encoding="utf-8")

        self.assertIn("说话人 1: 你好", content)


if __name__ == "__main__":
    unittest.main()
