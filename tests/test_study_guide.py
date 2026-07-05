import json
import tempfile
import unittest
from pathlib import Path

from video_analyzer.study_guide import build_study_artifacts


class StudyGuideTests(unittest.TestCase):
    def test_detects_hard_evidence_gaps_and_blocks_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "orin").mkdir()
            (run_dir / "frames").mkdir()
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "transcript": {"segments": []},
                        "ocr_events": [],
                        "frame_analyses": [{"response": "Error analyzing frame 0: missing frame"}],
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "frames_manifest.json").write_text(
                json.dumps({"frames": [{"frame_number": 0, "path": "frames/frame_0.jpg", "timestamp": 0.0}]}),
                encoding="utf-8",
            )

            result = build_study_artifacts(run_dir, skip_review=True)

            gaps = result["evidence_gaps"]
            categories = {item["category"] for item in gaps["items"]}
            self.assertIn("frame_missing", categories)
            self.assertIn("asr_empty", categories)
            self.assertIn("ocr_empty", categories)
            self.assertIn("vl_failed", categories)
            self.assertEqual(result["publish_decision"]["status"], "blocked")
            self.assertTrue((run_dir / "study_guide.json").is_file())
            self.assertTrue((run_dir / "study_chapters" / "chapter_01.json").is_file())

    def test_normalizes_vibevoice_segments_and_builds_chapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "orin").mkdir()
            (run_dir / "frames").mkdir()
            (run_dir / "frames" / "frame_0.jpg").write_bytes(b"jpg")
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "transcript": {
                            "segments": [
                                {"Start": 0.0, "End": 10.0, "Content": "开场介绍"},
                                {"Start": 250.0, "End": 260.0, "Content": "硬件选择"},
                                {"Start": 520.0, "End": 530.0, "Content": "固件配置"},
                            ]
                        },
                        "ocr_events": [{"frame_number": 0, "timestamp": 0.0, "status": "ok", "text": "标题"}],
                        "frame_analyses": [{"response": "画面显示项目标题"}],
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "frames_manifest.json").write_text(
                json.dumps({"frames": [{"frame_number": 0, "path": "frames/frame_0.jpg", "timestamp": 0.0}]}),
                encoding="utf-8",
            )

            result = build_study_artifacts(run_dir, skip_review=True)

            self.assertGreaterEqual(result["summary"]["chapters"], 2)
            guide = json.loads((run_dir / "study_guide.json").read_text(encoding="utf-8"))
            self.assertTrue(any(item["source_type"] == "asr" and "硬件选择" in item["text"] for item in guide["evidence"]))
            self.assertTrue(all("自动分段" not in chapter["title"] for chapter in guide["chapters"]))
            self.assertTrue(any("硬件选择" in chapter["title"] or "固件配置" in chapter["title"] for chapter in guide["chapters"]))

    def test_learning_copy_uses_chapter_content_instead_of_repeated_manual(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "orin").mkdir()
            (run_dir / "frames").mkdir()
            (run_dir / "orin" / "page_context.md").write_text(
                "- 00:00:00 - 00:02:00: 背景说明\n"
                "- 00:02:00 - 00:05:00: 参数配置\n",
                encoding="utf-8",
            )
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "transcript": {
                            "segments": [
                                {
                                    "Start": 10.0,
                                    "End": 30.0,
                                    "Content": "先解释系统为什么需要这个流程，并明确最终要验证的目标。",
                                },
                                {
                                    "Start": 140.0,
                                    "End": 180.0,
                                    "Content": "进入参数配置页面，依次填写服务地址、模型名称和并发数量。",
                                },
                            ]
                        },
                        "ocr_events": [],
                        "frame_analyses": [],
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "operation_manual.md").write_text(
                "这是一段全局手册内容，如果每个章节都展示它，学习视图就会重复。",
                encoding="utf-8",
            )
            (run_dir / "frames_manifest.json").write_text(
                json.dumps({"version": 1, "source": "audio_only", "frames": []}),
                encoding="utf-8",
            )

            build_study_artifacts(run_dir, skip_review=True)

            guide = json.loads((run_dir / "study_guide.json").read_text(encoding="utf-8"))
            first, second = guide["chapters"][:2]
            self.assertIn("为什么需要这个流程", first["summary"])
            self.assertIn("参数配置页面", second["summary"])
            self.assertNotEqual(first["summary"], second["summary"])
            self.assertFalse(any(point.endswith("...") for chapter in guide["chapters"] for point in chapter["key_points"]))

    def test_audio_only_run_does_not_require_visual_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "orin").mkdir()
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "transcript": {
                            "text": "播客正文",
                            "segments": [{"Start": 0.0, "End": 10.0, "Content": "播客正文"}],
                        },
                        "ocr_events": [],
                        "frame_analyses": [],
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "frames_manifest.json").write_text(
                json.dumps({"version": 1, "source": "audio_only", "frames": []}),
                encoding="utf-8",
            )

            result = build_study_artifacts(run_dir, skip_review=True)

            categories = {item["category"] for item in result["evidence_gaps"]["items"]}
            self.assertNotIn("frames_manifest_missing", categories)
            self.assertNotIn("ocr_empty", categories)
            self.assertNotIn("vl_empty", categories)
            self.assertEqual(result["publish_decision"]["status"], "publishable")


if __name__ == "__main__":
    unittest.main()
