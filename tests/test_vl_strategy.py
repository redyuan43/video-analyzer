import json
import tempfile
import unittest
from pathlib import Path

from video_analyzer.frame import Frame
from video_analyzer.frame_selection import FrameSelectionOptions, select_vl_frames
from video_analyzer.ocr import OCREvent
from video_analyzer.cli import load_ocr_checkpoint, write_ocr_checkpoint
from video_analyzer.vl_checkpoint import analysis_signature, load_vl_checkpoint, write_vl_checkpoint


class VLFrameSelectionTests(unittest.TestCase):
    def _frames(self, count: int) -> list[Frame]:
        return [
            Frame(
                number=index,
                path=Path(f"/tmp/frame-{index:04d}.jpg"),
                timestamp=float(index * 7.5),
                score=float(index % 31),
            )
            for index in range(count)
        ]

    def test_deep_auto_respects_time_capacity_and_preserves_timeline_coverage(self):
        frames = self._frames(355)

        selected, _decisions, metadata = select_vl_frames(
            frames=frames,
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

        self.assertEqual(metadata["vl_frame_policy_resolved"], "auto")
        self.assertEqual(metadata["vl_time_capacity"], 98)
        self.assertLessEqual(len(selected), 98)
        self.assertIn(0, selected)
        self.assertIn(354, selected)
        self.assertTrue(any(number >= 320 for number in selected))
        self.assertFalse(metadata["vl_time_target_bypassed"])

    def test_explicit_all_bypasses_time_target(self):
        frames = self._frames(40)

        selected, _decisions, metadata = select_vl_frames(
            frames=frames,
            ocr_events=[],
            transcript=None,
            video_duration_seconds=300.0,
            options=FrameSelectionOptions(
                pipeline_mode="deep",
                vl_frame_policy="all",
                vl_target_seconds=60,
                vl_seconds_per_frame=25,
            ),
        )

        self.assertEqual(len(selected), 40)
        self.assertTrue(metadata["vl_time_target_bypassed"])
        self.assertEqual(metadata["vl_projected_seconds"], 1000.0)
        self.assertEqual(metadata["vl_time_capacity"], 2)

    def test_explicit_max_is_a_hard_cap_even_below_auto_minimum(self):
        frames = self._frames(100)

        selected, _decisions, metadata = select_vl_frames(
            frames=frames,
            ocr_events=[],
            transcript=None,
            video_duration_seconds=3600.0,
            options=FrameSelectionOptions(
                pipeline_mode="deep",
                vl_frame_policy="auto",
                explicit_max_frames=5,
                vl_target_seconds=2700,
                vl_seconds_per_frame=25,
            ),
        )

        self.assertEqual(len(selected), 5)
        self.assertEqual(metadata["vl_budget_resolved"], 5)


class VLCheckpointTests(unittest.TestCase):
    def test_v2_reuses_only_successes_with_matching_signature_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.jpg"
            image.write_bytes(b"frame-content")
            frame = Frame(number=7, path=image, timestamp=12.5, score=1.0)
            signature_payload = {"model": "vl-model", "prompt": "describe"}
            signature = analysis_signature(signature_payload)
            checkpoint = root / "frame_analyses.partial.json"
            valid = {
                "frame_number": 7,
                "timestamp": 12.5,
                "frame_sha256": __import__("hashlib").sha256(b"frame-content").hexdigest(),
                "status": "succeeded",
                "analysis_signature": signature,
                "response": "Frame 7 result",
            }
            failed = {
                **valid,
                "status": "failed",
                "response": "Error analyzing frame 7: backend unavailable",
            }
            write_vl_checkpoint(
                checkpoint,
                [failed, valid],
                signature=signature,
                signature_payload=signature_payload,
            )

            loaded, metadata = load_vl_checkpoint(checkpoint, [frame], signature)
            mismatched, mismatch_metadata = load_vl_checkpoint(
                checkpoint,
                [frame],
                analysis_signature({"model": "different"}),
            )

        self.assertEqual(loaded[7]["response"], "Frame 7 result")
        self.assertEqual(metadata["valid_successes"], 1)
        self.assertEqual(mismatched, {})
        self.assertFalse(mismatch_metadata["signature_match"])

    def test_legacy_ordered_checkpoint_maps_frames_and_excludes_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for number in range(3):
                image = root / f"{number}.jpg"
                image.write_bytes(f"frame-{number}".encode())
                frames.append(Frame(number=number, path=image, timestamp=float(number), score=0.0))
            checkpoint = root / "frame_analyses.partial.json"
            checkpoint.write_text(
                json.dumps(
                    [
                        {"response": "Frame 0 result"},
                        {"response": "Error analyzing frame 1: malformed output"},
                        {"response": "Frame 2 result"},
                    ]
                ),
                encoding="utf-8",
            )
            signature = analysis_signature({"model": "vl-model"})

            loaded, metadata = load_vl_checkpoint(
                checkpoint,
                frames,
                signature,
                allow_legacy_ordered=True,
            )

        self.assertEqual(set(loaded), {0, 2})
        self.assertEqual(metadata["failed_entries"], 1)
        self.assertTrue(metadata["legacy_migrated"])


class OCRCheckpointTests(unittest.TestCase):
    def test_ocr_checkpoint_reuses_only_matching_successful_frame_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.jpg"
            image.write_bytes(b"ocr-frame")
            frame = Frame(number=3, path=image, timestamp=5.0, score=0.0)
            event = OCREvent(
                frame_number=3,
                timestamp=5.0,
                provider="easyocr",
                status="ok",
                text="按钮",
                items=[],
            )
            signature_payload = {"provider": "easyocr", "model": "local"}
            signature = analysis_signature(signature_payload)
            checkpoint = root / "ocr_events.partial.json"
            write_ocr_checkpoint(
                checkpoint,
                {3: event},
                {3: frame},
                signature,
                signature_payload,
            )

            loaded = load_ocr_checkpoint(checkpoint, [frame], signature)
            mismatched = load_ocr_checkpoint(
                checkpoint,
                [frame],
                analysis_signature({"provider": "different"}),
            )

        self.assertEqual(loaded[3].text, "按钮")
        self.assertEqual(mismatched, {})


if __name__ == "__main__":
    unittest.main()
