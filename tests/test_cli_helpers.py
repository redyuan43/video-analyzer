from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from video_analyzer.cli_helpers import analyze_frames_for_vl
from video_analyzer.frame import Frame
from video_analyzer.frame_selection import FrameDecision


class _Analyzer:
    def analyze_frame(self, frame, **_kwargs):
        return {"response": f"frame-{frame.number}"}


class AnalyzeFramesForVlTests(unittest.TestCase):
    def test_concurrent_results_are_collected_in_frame_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            frames = []
            decisions = []
            for number in range(3):
                path = directory / f"frame_{number}.jpg"
                path.write_bytes(f"frame-{number}".encode())
                frames.append(
                    Frame(
                        number=number,
                        path=path,
                        timestamp=float(number),
                        score=0.0,
                    )
                )
                decisions.append(
                    FrameDecision(
                        frame_number=number,
                        timestamp=float(number),
                        selected_for_vl=True,
                        selection_score=1.0,
                        reason="selected",
                        skip_reason="",
                        ocr_status="empty",
                        ocr_chars=0,
                        ocr_summary="",
                        visual_change_score=1.0,
                    )
                )

            results = analyze_frames_for_vl(
                _Analyzer(),
                frames,
                [],
                {0, 1, 2},
                decisions,
                concurrency=3,
                context_before=0,
                context_after=0,
                context_max_gap=0.0,
            )

        self.assertEqual(
            [result["response"] for result in results],
            ["frame-0", "frame-1", "frame-2"],
        )
        self.assertTrue(all(result["status"] == "succeeded" for result in results))


if __name__ == "__main__":
    unittest.main()
