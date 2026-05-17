import json
import tempfile
import unittest
from pathlib import Path

from video_analyzer.multidoc import parse_chapters
from tools.augment_video_docs_images import main as augment_main
from tools.prepare_video_doc_export import rewrite_image_paths


class VideoDocImageTests(unittest.TestCase):
    def test_parse_chapters_splits_long_transcript_without_page_chapters(self):
        transcript = {
            "segments": [
                {"start_time": index * 35.0, "end_time": (index + 1) * 35.0, "text": f"segment {index}"}
                for index in range(49)
            ]
        }

        chapters = parse_chapters("", transcript)

        self.assertGreaterEqual(len(chapters), 6)
        self.assertLessEqual(len(chapters), 10)
        self.assertNotEqual(chapters[0]["end"], "00:00:00")
        self.assertTrue(chapters[0]["title"].startswith("自动分段"))

    def test_augment_video_docs_images_inserts_final_and_representative_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            final_dir = run_dir / "baoyu_images" / "final"
            final_dir.mkdir(parents=True)
            (final_dir / "02-infographic-knowledge-notes.png").write_bytes(b"png")
            manual_assets = run_dir / "manual_assets"
            manual_assets.mkdir()
            (manual_assets / "frame_000.jpg").write_bytes(b"jpg")
            (manual_assets / "frame_001.jpg").write_bytes(b"jpg")
            doc_dir = run_dir / "docs_analysis_chapters"
            doc_dir.mkdir()
            doc = doc_dir / "knowledge_notes_v2.md"
            doc.write_text("# 知识笔记\n\n正文\n", encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["augment_video_docs_images.py", str(run_dir)]
                augment_main()
            finally:
                sys.argv = old_argv

            text = doc.read_text(encoding="utf-8")
            self.assertIn("../baoyu_images/final/02-infographic-knowledge-notes.png", text)
            self.assertIn("../manual_assets/frame_000.jpg", text)

            old_argv = sys.argv
            try:
                sys.argv = ["augment_video_docs_images.py", str(run_dir)]
                augment_main()
            finally:
                sys.argv = old_argv
            text_again = doc.read_text(encoding="utf-8")
            self.assertEqual(text_again.count("02-infographic-knowledge-notes.png"), 1)

    def test_export_rewrite_removes_parent_directory_image_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            final_dir = run_dir / "baoyu_images" / "final"
            final_dir.mkdir(parents=True)
            (final_dir / "03-infographic-deep-report.png").write_bytes(b"png")
            doc_dir = run_dir / "docs_analysis_chapters"
            doc_dir.mkdir()
            doc = doc_dir / "deep_report_v2.md"
            text = "![逐章深度报告视觉摘要](../baoyu_images/final/03-infographic-deep-report.png)\n"

            rewritten = rewrite_image_paths(text, doc, run_dir, run_dir)

            self.assertIn("(baoyu_images/final/03-infographic-deep-report.png)", rewritten)
            self.assertNotIn("../baoyu_images", rewritten)


if __name__ == "__main__":
    unittest.main()
