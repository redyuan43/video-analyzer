import unittest
from pathlib import Path


class AutoGPUSettingsUITests(unittest.TestCase):
    def test_model_and_profile_fields_accept_auto(self):
        root = Path(__file__).resolve().parents[1]
        template = (
            root
            / "video-analyzer-ui"
            / "video_analyzer_ui"
            / "templates"
            / "index.html"
        ).read_text(encoding="utf-8")
        javascript = (
            root
            / "video-analyzer-ui"
            / "video_analyzer_ui"
            / "static"
            / "js"
            / "main.js"
        ).read_text(encoding="utf-8")

        self.assertIn('id="modelWorkerCount" type="text"', template)
        self.assertIn('id="modelConcurrency" type="text"', template)
        self.assertIn('id="profileVlConcurrency" type="text"', template)
        self.assertIn("function parseAutoPositiveInteger", javascript)
        self.assertIn("settings.vl_concurrency = parseAutoPositiveInteger", javascript)
        self.assertIn("'vision_runtime'", javascript)
        self.assertIn("'ocr_worker_count'", javascript)


if __name__ == "__main__":
    unittest.main()
