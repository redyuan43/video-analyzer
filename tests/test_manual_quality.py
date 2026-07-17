import unittest
from types import SimpleNamespace

from video_analyzer.cli import append_evidence_boundary_section
from video_analyzer.manual import (
    embed_step_images,
    render_raw_evidence_text,
    render_text_evidence_map,
    review_operation_manual_markdown,
)


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

    def test_text_evidence_map_strips_markdown_images_from_ocr_summary(self):
        frames = [SimpleNamespace(number=4, timestamp=14.0)]
        ocr_by_frame = {
            4: SimpleNamespace(status="ok", text="![](images/0.jpg) 项目结构图 | Goal 验证", provider="dots"),
        }
        frame_analyses = [{"response": "Frame 4 shows ![](images/1.jpg) workflow diagram"}]

        text = "\n".join(render_text_evidence_map(frames, frame_analyses, ocr_by_frame))

        self.assertIn("项目结构图 \\| Goal 验证", text)
        self.assertIn("Frame 4 shows workflow diagram", text)
        self.assertNotIn("![](images/0.jpg)", text)
        self.assertNotIn("![](images/1.jpg)", text)

    def test_raw_evidence_text_is_fenced_to_prevent_broken_image_rendering(self):
        text = render_raw_evidence_text("![](images/0.jpg)\n项目结构图")

        self.assertEqual(text, "```\n项目结构图\n```")
        self.assertNotIn("![](images/0.jpg)", text)

    def test_embed_step_images_promotes_key_visual_diagram_to_structure_section(self):
        frames = [
            SimpleNamespace(number=4, timestamp=14.0),
            SimpleNamespace(number=24, timestamp=100.0),
        ]
        assets = {
            4: "manual_assets/frame_004.jpg",
            24: "manual_assets/frame_024.jpg",
        }
        analyses = [
            {"response": "The scene is a workflow diagram with Brainstorming, Goal: 实现, Goal: 验证, 人工验证, 任务完成."},
            {"response": "The scene shows a terminal."},
        ]
        ocr_events = [
            SimpleNamespace(frame_number=4, text="做这个游戏Demo的整个工作流"),
        ]
        manual = "# 手册\n\n### 2. 视频结构与流程图\n\n整体流程如下。\n"

        result = embed_step_images(manual, frames, assets, frame_analyses=analyses, ocr_events=ocr_events)

        self.assertIn("**关键画面：**", result)
        self.assertIn("![14s / Frame 4](manual_assets/frame_004.jpg)", result)

    def test_embed_step_images_handles_nested_step_headings_and_removes_empty_image_table(self):
        frames = [
            SimpleNamespace(number=5, timestamp=18.0),
            SimpleNamespace(number=6, timestamp=20.0),
            SimpleNamespace(number=8, timestamp=24.0),
            SimpleNamespace(number=24, timestamp=100.0),
        ]
        assets = {
            5: "manual_assets/frame_005.jpg",
            6: "manual_assets/frame_006.jpg",
            8: "manual_assets/frame_008.jpg",
            24: "manual_assets/frame_024.jpg",
        }
        manual = "\n".join(
            [
                "# 手册",
                "",
                "### 4. 图文操作步骤",
                "",
                "| 18s |",
                "| --- |",
                "| ![18s / Frame 5](manual_assets/frame_005.jpg) |",
                "",
                "#### 步骤一：安装必备 Skill",
                "",
                "对应视频时间戳：约 00:18 - 00:24。",
                "",
                "| Skill 列表 | Brainstorming Skill 文件视图 |",
                "| :---: | :---: |",
                "| | |",
                "",
                "#### 步骤二：脑暴",
                "",
                "对应视频时间戳：约 01:40。",
            ]
        )

        result = embed_step_images(manual, frames, assets)

        self.assertIn("![18s / Frame 5](manual_assets/frame_005.jpg)", result)
        self.assertIn("![20s / Frame 6](manual_assets/frame_006.jpg)", result)
        self.assertIn("![24s / Frame 8](manual_assets/frame_008.jpg)", result)
        self.assertEqual(result.count("manual_assets/frame_005.jpg"), 1)
        self.assertNotIn("Skill 列表", result)

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
