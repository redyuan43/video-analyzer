import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from video_analyzer.frame import Frame
from video_analyzer.ocr import OCREvent
from video_analyzer.ocr_keyframes import (
    build_ocr_text_events,
    image_textness,
    resolve_ocr_keyframe_budget,
    select_ocr_keyframes,
)


class OCRKeyframeStrategyTests(unittest.TestCase):
    def test_auto_budget_scales_for_long_video(self):
        budget = resolve_ocr_keyframe_budget(
            video_duration_seconds=60 * 60,
            pipeline_mode="balanced",
            candidate_count=500,
            value="auto",
        )
        self.assertGreater(budget, 48)

    def test_text_events_dedupe_near_duplicate_ocr(self):
        events = [
            OCREvent(0, 0.0, "dots", "ok", "File menu Settings", []),
            OCREvent(1, 2.0, "dots", "ok", "File  menu   Settings", []),
            OCREvent(2, 10.0, "dots", "ok", "Error: missing token", []),
        ]

        deduped = build_ocr_text_events(events)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["frame_numbers"], [0, 1])
        self.assertEqual(deduped[1]["frame_numbers"], [2])

    def test_textness_prefers_text_image_over_blank(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blank = root / "blank.jpg"
            text = root / "text.jpg"
            Image.new("RGB", (640, 360), "white").save(blank)
            image = Image.new("RGB", (640, 360), "white")
            draw = ImageDraw.Draw(image)
            for row in range(8):
                draw.text((40, 30 + row * 36), f"command --flag value-{row}", fill="black")
            image.save(text)

            blank_score = image_textness(blank).textness_score
            text_score = image_textness(text).textness_score

        self.assertGreater(text_score, blank_score)

    def test_select_ocr_keyframes_records_reasons_and_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = []
            for index in range(12):
                path = root / f"frame_{index}.jpg"
                image = Image.new("RGB", (320, 180), "white")
                draw = ImageDraw.Draw(image)
                if index in {3, 8}:
                    draw.text((20, 60), f"Important UI text {index}", fill="black")
                image.save(path)
                frames.append(Frame(index, path, index * 10.0, 0.0))

            selected, decisions, metadata = select_ocr_keyframes(
                frames=frames,
                transcript=None,
                video_duration_seconds=120.0,
                pipeline_mode="balanced",
                strategy="scan-text",
                budget=4,
                scan_frames_count=120,
            )

        self.assertEqual(len(selected), 4)
        self.assertEqual(metadata["scan_frames_count"], 120)
        self.assertEqual(metadata["ocr_candidate_frames_count"], 12)
        self.assertEqual(metadata["ocr_frames_count"], 4)
        self.assertTrue(all(item.reason or item.skip_reason for item in decisions))


if __name__ == "__main__":
    unittest.main()
