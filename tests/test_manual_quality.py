import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from video_analyzer.audio_processor import AudioTranscript
from video_analyzer.cli import append_evidence_boundary_section
from video_analyzer.manual import (
    build_operation_manual_prompt,
    embed_step_images,
    generate_operation_manual,
    render_raw_evidence_text,
    render_text_evidence_map,
    resolve_manual_prompt_char_budget,
    review_operation_manual_markdown,
)
from video_analyzer.ocr import OCREvent


class ManualQualityTests(unittest.TestCase):
    def test_long_manual_prompt_respects_context_derived_character_budget(self):
        budget = resolve_manual_prompt_char_budget(100_000)
        transcript = AudioTranscript(
            text="这是长访谈内容。" * 20_000,
            segments=[
                {"start": index, "end": index + 1, "text": "逐段转写内容" * 30}
                for index in range(400)
            ],
            language="zh-CN",
        )
        frames = [
            SimpleNamespace(number=index, timestamp=float(index), path=Path(f"/tmp/frame_{index}.jpg"))
            for index in range(100)
        ]
        analyses = [{"response": "视觉分析" * 500} for _ in frames]
        ocr_events = [
            OCREvent(index, float(index), "test", "ok", "OCR文本" * 300, [])
            for index in range(100)
        ]

        prompt = build_operation_manual_prompt(
            analyses,
            frames,
            transcript,
            {"strategy": "deep"},
            ocr_events,
            "页面上下文" * 10_000,
            "zh-CN",
            {index: f"manual_assets/frame_{index:03d}.jpg" for index in range(100)},
            max_prompt_chars=budget,
        )

        self.assertLessEqual(len(prompt), budget)
        self.assertIn("Return Markdown with these sections:", prompt)
        self.assertIn("Transcript已压缩", prompt)

    def test_manual_generation_uses_configured_fallback_after_primary_error(self):
        primary = MagicMock()
        primary.generate.side_effect = RuntimeError("context exceeded")
        fallback = MagicMock()
        fallback.generate.return_value = {
            "response": "# 操作手册\n\n## 概览\n\n完成。",
            "context": "not persisted",
        }
        callback = MagicMock()

        result = generate_operation_manual(
            client=primary,
            text_model="local-model",
            frame_analyses=[],
            frames=[],
            transcript=None,
            asr_metadata={},
            ocr_events=[],
            page_context="",
            language="zh-CN",
            temperature=0.2,
            fallback_client=fallback,
            fallback_model="deepseek-v4-pro",
            fallback_temperature=1.0,
            fallback_status_callback=callback,
        )

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_model"], "deepseek-v4-pro")
        self.assertNotIn("context", result)
        fallback.generate.assert_called_once()
        callback.assert_any_call(
            "succeeded",
            "fallback model deepseek-v4-pro completed",
        )

    def test_error_placeholder_fails_quality_gate(self):
        issues = review_operation_manual_markdown("Error generating operation manual: timed out")

        self.assertTrue(any(issue["code"] == "manual_generation_error" for issue in issues))

    def test_fullwidth_image_marker_is_normalized(self):
        manual = "### 操作步骤\n\n！[Sequencer 时间轴](manual_assets/frame_034.jpg)\n"

        normalized = embed_step_images(manual, [], {})
        issues = review_operation_manual_markdown(normalized)

        self.assertIn("![Sequencer 时间轴](manual_assets/frame_034.jpg)", normalized)
        self.assertFalse(any(issue["code"] == "raw_asset_path" for issue in issues))

    def test_markdown_asset_link_is_not_reported_as_raw_path(self):
        manual = (
            "### 主要物料清单 (BOM)\n\n"
            "| 组件 | 截图参考 |\n"
            "| --- | --- |\n"
            "| 主控板 | [查看](manual_assets/frame_011.jpg) |\n"
        )

        issues = review_operation_manual_markdown(manual)

        self.assertFalse(any(issue["code"] == "raw_asset_path" for issue in issues))

    def test_step_detection_does_not_cross_from_an_unrelated_heading(self):
        manual = "\n".join(
            [
                "### 硬件工具",
                "",
                "| 组件 | 截图参考 |",
                "| --- | --- |",
                "| 主控板 | [查看](manual_assets/frame_011.jpg) |",
                "",
                "## 4. 图文操作步骤",
                "",
                "### 步骤 1: 安装",
                "",
                "![安装画面](manual_assets/frame_012.jpg)",
            ]
        )

        issues = review_operation_manual_markdown(manual)

        self.assertFalse(any(issue["code"] == "step_asset_not_rendered" for issue in issues))

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
