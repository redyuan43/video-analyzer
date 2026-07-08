import json
import tempfile
import unittest
from pathlib import Path

from video_analyzer.study_guide import build_study_artifacts


class FakeStudyCardClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, model, temperature, num_predict, **kwargs):
        self.prompts.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "num_predict": num_predict,
            }
        )
        if not self.responses:
            raise AssertionError("No fake response configured")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {"response": response}


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

    def test_study_card_synthesis_updates_all_chapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "orin").mkdir()
            (run_dir / "orin" / "page_context.md").write_text(
                "- 00:00:00 - 00:02:00: 他说\n"
                "- 00:02:00 - 00:05:00: 呃\n",
                encoding="utf-8",
            )
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "transcript": {
                            "segments": [
                                {
                                    "Start": 10.0,
                                    "End": 70.0,
                                    "Content": "他说，你的工作是提供你想看到的文化。后来大家因为这种文化聚集起来，也会反馈和共创。",
                                },
                                {
                                    "Start": 150.0,
                                    "End": 230.0,
                                    "Content": "呃，伪纪录片创作需要设计直人和怪人，也要快速建立冲突和人物关系。",
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
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "frames_manifest.json").write_text(
                json.dumps({"version": 1, "source": "audio_only", "frames": []}),
                encoding="utf-8",
            )
            client = FakeStudyCardClient(
                [
                    json.dumps(
                        {
                            "title": "文化氛围的主动塑造",
                            "summary": "本章讨论创作者如何通过持续提供自己想看到的文化，吸引同频的人聚集。文化不是单向输出，而是在反馈和共创中逐渐形成。",
                            "key_points": ["先提供想看到的文化，再等待同频者聚集。", "社群反馈会继续塑造创作者的内容方向。", "文化建设是创作者和受众之间的共同过程。"],
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "title": "伪纪录片的叙事设计",
                            "summary": "本章概括伪纪录片创作为什么困难：它需要在自然感、人物关系和喜剧结构之间取得平衡。创作者必须快速建立冲突，并让观众理解人物功能。",
                            "key_points": ["伪纪录片的真实感来自精心设计。", "直人和怪人的关系是喜剧结构核心。", "开场需要快速建立冲突和人物关系。"],
                        },
                        ensure_ascii=False,
                    ),
                ]
            )

            build_study_artifacts(
                run_dir,
                skip_review=True,
                study_card_llm_base_url="http://agx.taild500c8.ts.net:11434/v1",
                study_card_model="qwen3:4b-instruct",
                study_card_client=client,
            )

            guide = json.loads((run_dir / "study_guide.json").read_text(encoding="utf-8"))
            titles = [chapter["title"] for chapter in guide["chapters"]]
            self.assertEqual(titles, ["文化氛围的主动塑造", "伪纪录片的叙事设计"])
            self.assertEqual(len(client.prompts), 2)
            self.assertTrue(all(chapter["card_synthesis"]["status"] == "generated" for chapter in guide["chapters"]))
            self.assertEqual(guide["chapters"][0]["source_title"], "他说")
            self.assertEqual(guide["chapters"][1]["source_title"], "呃")
            cards = (run_dir / "study_cards.md").read_text(encoding="utf-8")
            self.assertIn("- 摘要：本章讨论创作者如何通过持续提供自己想看到的文化", cards)
            self.assertIn("- 要点：", cards)
            self.assertNotIn("- 主旨：", cards)
            self.assertNotIn("## 01. 他说", cards)
            self.assertNotIn("## 02. 呃", cards)

    def test_study_card_synthesis_falls_back_per_chapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "orin").mkdir()
            (run_dir / "orin" / "page_context.md").write_text(
                "- 00:00:00 - 00:03:00: 呃\n",
                encoding="utf-8",
            )
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "transcript": {
                            "segments": [
                                {
                                    "Start": 10.0,
                                    "End": 120.0,
                                    "Content": "呃，这一段讨论如何把开放麦积累的素材筛选出来，再决定哪些适合发展成专场段子。",
                                }
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
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "frames_manifest.json").write_text(
                json.dumps({"version": 1, "source": "audio_only", "frames": []}),
                encoding="utf-8",
            )
            primary = FakeStudyCardClient(["not json"])
            fallback = FakeStudyCardClient(
                [
                    json.dumps(
                        {
                            "title": "开放麦素材的筛选",
                            "summary": "本章总结创作者如何从开放麦和日常写作中筛选素材。重点是判断素材是否有发展成成熟单口段子的潜力。",
                            "key_points": ["先大量写作，再判断素材潜力。", "开放麦反馈帮助筛选可发展的段子。", "不同素材适合不同表达载体。"],
                        },
                        ensure_ascii=False,
                    )
                ]
            )

            build_study_artifacts(
                run_dir,
                llm_base_url="http://fallback.example/v1",
                text_model="large-fallback",
                skip_review=True,
                study_card_llm_base_url="http://agx.taild500c8.ts.net:11434/v1",
                study_card_model="qwen3:4b-instruct",
                study_card_client=primary,
                study_card_fallback_client=fallback,
            )

            chapter = json.loads((run_dir / "study_chapters" / "chapter_01.json").read_text(encoding="utf-8"))
            self.assertEqual(chapter["title"], "开放麦素材的筛选")
            self.assertEqual(chapter["card_synthesis"]["status"], "fallback_model")
            self.assertEqual(chapter["card_synthesis"]["model"], "large-fallback")
            self.assertIn("primary qwen3:4b-instruct", chapter["card_synthesis"]["reason"])

    def test_study_card_prompt_uses_full_chapter_evidence_before_public_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "orin").mkdir()
            (run_dir / "orin" / "page_context.md").write_text(
                "- 00:00:00 - 00:10:00: 长章节\n",
                encoding="utf-8",
            )
            segments = [
                {
                    "Start": float(index),
                    "End": float(index) + 0.5,
                    "Content": f"片段{index:02d}：这是长章节中的第 {index} 个观点。",
                }
                for index in range(1, 41)
            ]
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "transcript": {"segments": segments},
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
            client = FakeStudyCardClient(
                [
                    json.dumps(
                        {
                            "title": "长章节完整归纳",
                            "summary": "本章根据完整转写归纳长章节的多阶段讨论。摘要覆盖章节后半段观点，而不是只停留在开头片段。",
                            "key_points": ["完整证据用于归纳。", "公开证据列表仍保持精简。", "后半段观点不会被截断。"],
                        },
                        ensure_ascii=False,
                    )
                ]
            )

            build_study_artifacts(
                run_dir,
                skip_review=True,
                study_card_llm_base_url="http://agx.taild500c8.ts.net:11434/v1",
                study_card_model="qwen3:4b-instruct",
                study_card_client=client,
            )

            self.assertIn("片段40", client.prompts[0]["prompt"])
            guide = json.loads((run_dir / "study_guide.json").read_text(encoding="utf-8"))
            chapter = guide["chapters"][0]
            self.assertEqual(chapter["title"], "长章节完整归纳")
            self.assertEqual(len(chapter["evidence"]), 30)
            self.assertNotIn("_synthesis_evidence", chapter)
            chapter_file = json.loads((run_dir / "study_chapters" / "chapter_01.json").read_text(encoding="utf-8"))
            self.assertEqual(len(chapter_file["evidence"]), 30)
            self.assertNotIn("_synthesis_evidence", chapter_file)

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
