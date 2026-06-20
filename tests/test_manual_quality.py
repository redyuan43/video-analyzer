import unittest
from types import SimpleNamespace

from video_analyzer.cli import append_evidence_boundary_section
from video_analyzer.manual import embed_step_images, render_text_evidence_map, review_operation_manual_markdown


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

    def test_text_evidence_map_summarizes_frame_evidence_without_generated_image(self):
        frames = [SimpleNamespace(number=0, timestamp=0.0), SimpleNamespace(number=1, timestamp=2.0)]
        ocr_by_frame = {
            0: SimpleNamespace(status="ok", text="标题文字", provider="dots"),
            1: SimpleNamespace(status="not_run", text="", provider=""),
        }
        frame_analyses = [{"response": "画面显示标题页"}, {"response": ""}]

        lines = render_text_evidence_map(frames, frame_analyses, ocr_by_frame)
        text = "\n".join(lines)

        self.assertIn("## 文字证据地图", text)
        self.assertIn("OCR 成功：1/2", text)
        self.assertIn("| 0.00s | Frame 0 | `ok` | 标题文字 | 画面显示标题页 |", text)
        self.assertNotIn("04-infographic-manual-evidence", text)

    def test_evidence_boundary_section_is_added_for_weak_visual_evidence(self):
        markdown = "# 手册\n\n## 步骤\n\n打开设置。"
        result = append_evidence_boundary_section(
            markdown,
            {"vl_frame_policy_resolved": "none", "vl_frames_processed": 0},
            {"ocr_text_events_count": 0},
            [SimpleNamespace(status="error")],
        )

        self.assertIn("## 证据边界与需复核", result)
        self.assertIn("未运行或未选中 VL", result)
        self.assertIn("OCR 没有形成稳定文本事件", result)

    def test_evidence_boundary_section_is_not_added_without_warnings(self):
        markdown = "# 手册\n\n## 步骤\n\n打开设置。"
        result = append_evidence_boundary_section(
            markdown,
            {"vl_frame_policy_resolved": "auto", "vl_frames_processed": 3},
            {"ocr_text_events_count": 2},
            [SimpleNamespace(status="ok")],
        )

        self.assertEqual(result, markdown)

    def test_evidence_boundary_uses_selected_vl_frame_count(self):
        markdown = "# 手册\n\n## 步骤\n\n打开设置。"
        result = append_evidence_boundary_section(
            markdown,
            {"vl_frame_policy_resolved": "auto", "vl_frames_count": 3, "frames": [1, 2, 3]},
            {"ocr_text_events_count": 2},
            [SimpleNamespace(status="ok")],
        )

        self.assertEqual(result, markdown)


if __name__ == "__main__":
    unittest.main()
