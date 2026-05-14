import unittest

from video_analyzer.manual import review_operation_manual_markdown


class ManualQualityTests(unittest.TestCase):
    def test_error_placeholder_fails_quality_gate(self):
        issues = review_operation_manual_markdown("Error generating operation manual: timed out")

        self.assertTrue(any(issue["code"] == "manual_generation_error" for issue in issues))


if __name__ == "__main__":
    unittest.main()
