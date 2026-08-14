import json
import tempfile
import unittest
from pathlib import Path

from video_analyzer.analysis_progress import write_analysis_progress


class AnalysisProgressTests(unittest.TestCase):
    def test_node_progress_preserves_start_and_explicit_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_analysis_progress(
                output_dir,
                "asr",
                node_updates={
                    "asr": {
                        "status": "running",
                        "message": "transcribing",
                    }
                },
            )
            write_analysis_progress(
                output_dir,
                "asr_done",
                node_updates={
                    "asr": {
                        "status": "succeeded",
                        "message": "done",
                        "duration_seconds": 12.5,
                    }
                },
            )

            payload = json.loads(
                (output_dir / "progress.json").read_text(encoding="utf-8")
            )
            state = payload["node_states"]["asr"]
            self.assertEqual(state["status"], "succeeded")
            self.assertEqual(state["duration_seconds"], 12.5)
            self.assertTrue(state["started_at"])
            self.assertTrue(state["finished_at"])


if __name__ == "__main__":
    unittest.main()
