import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_analyzer.multidoc import parse_chapters
from tools.augment_video_docs_images import main as augment_main
from tools.md_to_mobile_pdf import main as mobile_pdf_main
from tools.md_to_mobile_pdf import render_markdown
from tools.md_to_mobile_pdf import render_mermaid_blocks
from tools.md_to_mobile_pdf import wrap_final_images
from tools.pdf_to_long_png import main as long_png_main
from tools.prepare_baoyu_image_prompts import main as prepare_prompts_main
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

    def test_operation_manual_reuses_knowledge_notes_image_and_removes_legacy_01(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            final_dir = run_dir / "baoyu_images" / "final"
            final_dir.mkdir(parents=True)
            (final_dir / "02-infographic-knowledge-notes.png").write_bytes(b"png")
            (final_dir / "01-image-cards-operation-manual.png").write_bytes(b"stale")
            doc = run_dir / "operation_manual.md"
            doc.write_text(
                "# 操作手册\n\n"
                "## 1. 概览\n\n"
                "正文。\n\n"
                "## 4. 图文操作步骤\n\n"
                "![操作手册视觉摘要](baoyu_images/final/01-image-cards-operation-manual.png)\n\n"
                "步骤。\n",
                encoding="utf-8",
            )

            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["augment_video_docs_images.py", str(run_dir)]
                augment_main()
            finally:
                sys.argv = old_argv

            text = doc.read_text(encoding="utf-8")
            self.assertIn("baoyu_images/final/02-infographic-knowledge-notes.png", text)
            self.assertNotIn("01-image-cards-operation-manual.png", text)
            self.assertEqual(text.count("02-infographic-knowledge-notes.png"), 1)
            self.assertLess(text.index("02-infographic-knowledge-notes.png"), text.index("## 4. 图文操作步骤"))

    def test_operation_manual_removes_legacy_01_even_when_replacement_image_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "baoyu_images" / "final").mkdir(parents=True)
            doc = run_dir / "operation_manual.md"
            doc.write_text(
                "# 操作手册\n\n"
                "## 1. 概览\n\n"
                "正文。\n\n"
                "![操作手册视觉摘要](baoyu_images/final/01-image-cards-operation-manual.png)\n",
                encoding="utf-8",
            )

            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["augment_video_docs_images.py", str(run_dir)]
                augment_main()
            finally:
                sys.argv = old_argv

            text = doc.read_text(encoding="utf-8")
            self.assertNotIn("01-image-cards-operation-manual.png", text)
            self.assertNotIn("02-infographic-knowledge-notes.png", text)

    def test_prepare_baoyu_prompts_drops_operation_manual_image_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "docs_analysis").mkdir()
            (run_dir / "docs_analysis" / "knowledge_notes.md").write_text("# 知识笔记\n\n内容\n", encoding="utf-8")
            (run_dir / "docs_analysis" / "deep_report.md").write_text("# 深度报告\n\n内容\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# 证据\n\n内容\n", encoding="utf-8")
            prompt_dir = run_dir / "baoyu_images" / "prompts"
            prompt_dir.mkdir(parents=True)
            (prompt_dir / "01-image-cards-operation-manual.md").write_text("stale", encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["prepare_baoyu_image_prompts.py", str(run_dir)]
                prepare_prompts_main()
            finally:
                sys.argv = old_argv

            self.assertFalse((prompt_dir / "01-image-cards-operation-manual.md").exists())
            self.assertTrue((prompt_dir / "02-infographic-knowledge-notes.md").exists())
            self.assertTrue((prompt_dir / "03-infographic-deep-report.md").exists())
            self.assertTrue((prompt_dir / "04-infographic-manual-evidence.md").exists())

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

    def test_mobile_pdf_export_writes_non_empty_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            source = run_dir / "sample.md"
            target = run_dir / "sample.pdf"
            source.write_text("# 标题\n\n正文段落，适合手机阅读。\n\n| 项 | 值 |\n| --- | --- |\n| A | 很长的一段中文内容 |\n", encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["md_to_mobile_pdf.py", str(source), str(target)]
                mobile_pdf_main()
            finally:
                sys.argv = old_argv

            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.read_bytes()[:4], b"%PDF")

    def test_pdf_to_long_png_writes_non_empty_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            source = run_dir / "sample.md"
            pdf = run_dir / "sample.pdf"
            png = run_dir / "sample.long.png"
            source.write_text("# 标题\n\n第一页。\n\n第二段内容。\n", encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["md_to_mobile_pdf.py", str(source), str(pdf)]
                mobile_pdf_main()
                sys.argv = ["pdf_to_long_png.py", str(pdf), str(png), "--dpi", "96"]
                long_png_main()
            finally:
                sys.argv = old_argv

            self.assertGreater(png.stat().st_size, 1000)
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_mobile_pdf_export_rewrites_linear_flowchart_to_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)

            rendered = render_mermaid_blocks(
                "```mermaid\nflowchart TD\nA[Start] --> B{Check}\nB --> C[End]\n```\n",
                work_dir,
            )

            self.assertIn('class="mobile-flowchart"', rendered)
            self.assertIn("Start", rendered)
            self.assertIn("Check", rendered)
            self.assertIn("End", rendered)
            self.assertNotIn("```mermaid", rendered)
            self.assertFalse((work_dir / "mermaid_001.png").exists())

    def test_mobile_pdf_export_allows_standalone_flowchart_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)

            rendered = render_mermaid_blocks(
                "```mermaid\n"
                "flowchart TD\n"
                "A[绪论: 市场在重估什么?] --> B{第一章: 市场担忧};\n"
                "B --> B1[外部: 港股大科技估值环境不友好];\n"
                "F[结论: 买的是防守和提效, 不是短期爆发];\n"
                "```\n",
                work_dir,
            )

            self.assertIn('class="mobile-flowchart"', rendered)
            self.assertIn("结论: 买的是防守和提效, 不是短期爆发", rendered)
            self.assertNotIn("```mermaid", rendered)
            self.assertFalse((work_dir / "mermaid_001.png").exists())

    def test_mobile_pdf_export_normalizes_inline_table_after_intro(self):
        body = render_markdown(
            "对比竞品数据（2026年3月QuestMobile）： "
            "| 产品 | 月活（MAU） | 日活（DAU） | "
            "| :--- | :--- | :--- | "
            "| 豆包（字节） | 3.45亿 | 1.4亿 | "
            "| 元宝（腾讯） | 5735万 | 1800万 |\n"
        )

        self.assertIn("<table>", body)
        self.assertIn("豆包（字节）", body)
        self.assertIn("元宝（腾讯）", body)
        self.assertIn("2026年3月QuestMobile", body)

    def test_mobile_pdf_export_rewrites_non_linear_mermaid_to_png_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)

            def fake_run(command, check):
                Path(command[command.index("-o") + 1]).write_bytes(b"png")

            with patch("tools.md_to_mobile_pdf.subprocess.run", side_effect=fake_run):
                rendered = render_mermaid_blocks("```mermaid\nsequenceDiagram\nA->>B: Hi\n```\n", work_dir)

            self.assertIn("![Mermaid diagram 1](file://", rendered)
            self.assertNotIn("```mermaid", rendered)
            self.assertTrue((work_dir / "mermaid_001.png").exists())

    def test_mobile_pdf_wraps_final_images_as_full_pages(self):
        body = render_markdown(
            "![视觉摘要](baoyu_images/final/01-image-cards-operation-manual.png)\n\n"
            "![普通帧](manual_assets/frame_000.jpg)\n"
        )

        wrapped = wrap_final_images(body)

        self.assertIn('class="final-image-page"', wrapped)
        self.assertIn("baoyu_images/final/01-image-cards-operation-manual.png", wrapped)
        self.assertIn("<p><img alt=\"普通帧\" src=\"manual_assets/frame_000.jpg\"", wrapped)
        self.assertEqual(wrapped.count('class="final-image-page"'), 1)


if __name__ == "__main__":
    unittest.main()
