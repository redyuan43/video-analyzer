import unittest

from video_analyzer.manual import embed_step_images, review_operation_manual_markdown


class ManualQualityTests(unittest.TestCase):
    def test_error_placeholder_fails_quality_gate(self):
        issues = review_operation_manual_markdown("Error generating operation manual: timed out")

        self.assertTrue(any(issue["code"] == "manual_generation_error" for issue in issues))

    def test_fullwidth_image_marker_is_normalized(self):
        manual = "### 操作步骤\n\n！[Sequencer 时间轴](manual_assets/frame_034.jpg)\n"

        normalized = embed_step_images(manual, [], {})
        issues = review_operation_manual_markdown(normalized)

        self.assertIn("![Sequencer 时间轴](manual_assets/frame_034.jpg)", normalized)
        self.assertFalse(any(issue["code"] == "raw_asset_path" for issue in issues))


if __name__ == "__main__":
    unittest.main()
