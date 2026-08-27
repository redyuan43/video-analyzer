import hashlib
import tempfile
import unittest
from pathlib import Path

from video_analyzer.cli import load_ocr_checkpoint, write_ocr_checkpoint
from video_analyzer.frame import Frame
from video_analyzer.frame_selection import FrameSelectionOptions, select_vl_frames
from video_analyzer.ocr import OCREvent
from video_analyzer.vl_checkpoint import analysis_signature, load_vl_checkpoint, write_vl_checkpoint


class VLFrameSelectionTests(unittest.TestCase):
    def frames(self, count: int) -> list[Frame]:
        return [
            Frame(number=index, path=Path(f"/tmp/frame-{index:04d}.jpg"), timestamp=float(index * 7.5), score=float(index % 31))
            for index in range(count)
        ]

    def test_auto_policy_respects_time_capacity_and_keeps_timeline_coverage(self):
        selected, _decisions, metadata = select_vl_frames(
            frames=self.frames(355),
            ocr_events=[],
            transcript=None,
            video_duration_seconds=2660.0,
            options=FrameSelectionOptions(
                pipeline_mode="deep",
                vl_frame_policy="auto",
                vl_target_seconds=2700,
                vl_seconds_per_frame=25,
            ),
        )

        self.assertEqual(metadata["vl_time_capacity"], 98)
        self.assertLessEqual(len(selected), 98)
        self.assertIn(0, selected)
        self.assertIn(354, selected)

    def test_explicit_all_bypasses_time_target(self):
        selected, _decisions, metadata = select_vl_frames(
            frames=self.frames(40),
            ocr_events=[],
            transcript=None,
            video_duration_seconds=300.0,
            options=FrameSelectionOptions(vl_frame_policy="all", vl_target_seconds=60, vl_seconds_per_frame=25),
        )

        self.assertEqual(len(selected), 40)
        self.assertTrue(metadata["vl_time_target_bypassed"])
        self.assertEqual(metadata["vl_time_capacity"], 2)


class CheckpointTests(unittest.TestCase):
    def test_vl_checkpoint_reuses_matching_successes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.jpg"
            image.write_bytes(b"frame-content")
            frame = Frame(number=7, path=image, timestamp=12.5, score=1.0)
            signature = analysis_signature({"model": "vl-model"})
            checkpoint = root / "frame_analyses.partial.json"
            write_vl_checkpoint(
                checkpoint,
                [
                    {
                        "frame_number": 7,
                        "frame_sha256": hashlib.sha256(b"frame-content").hexdigest(),
                        "status": "failed",
                        "analysis_signature": signature,
                        "response": "Error analyzing frame 7",
                    },
                    {
                        "frame_number": 7,
                        "frame_sha256": hashlib.sha256(b"frame-content").hexdigest(),
                        "status": "succeeded",
                        "analysis_signature": signature,
                        "response": "Frame 7 result",
                    },
                ],
                signature=signature,
                signature_payload={"model": "vl-model"},
            )
            loaded, metadata = load_vl_checkpoint(checkpoint, [frame], signature)

        self.assertEqual(loaded[7]["response"], "Frame 7 result")
        self.assertEqual(metadata["valid_successes"], 1)

    def test_ocr_checkpoint_reuses_matching_successes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.jpg"
            image.write_bytes(b"ocr-frame")
            frame = Frame(number=3, path=image, timestamp=5.0, score=0.0)
            signature = analysis_signature({"provider": "local"})
            checkpoint = root / "ocr_events.partial.json"
            event = OCREvent(3, 5.0, "test", "ok", "按钮", [])
            write_ocr_checkpoint(checkpoint, {3: event}, {3: frame}, signature, {"provider": "local"})
            loaded = load_ocr_checkpoint(checkpoint, [frame], signature)

        self.assertEqual(loaded[3].text, "按钮")


if __name__ == "__main__":
    unittest.main()
