import json
import tempfile
import unittest
from pathlib import Path

from video_analyzer import multidoc


class MultidocStudyContextTests(unittest.TestCase):
    def test_load_evidence_uses_study_chapters_and_normalizes_vibevoice_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            orin_dir = run_dir / "orin"
            orin_dir.mkdir()
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "transcript": {
                            "segments": [
                                {"Start": 1.0, "End": 3.0, "Content": "第一段内容"},
                                {"Start": 5.0, "End": 7.0, "Content": "第二段内容"},
                            ]
                        },
                        "ocr_events": [],
                        "frame_analyses": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "study_guide.json").write_text(
                json.dumps(
                    {
                        "overview": {"summary": "系统化学习摘要"},
                        "chapters": [
                            {
                                "chapter_id": "chapter_01",
                                "start": "00:00:00",
                                "end": "00:00:10",
                                "title": "学习章节",
                                "summary": "重点说明",
                                "evidence_ids": ["ev001"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "evidence_gaps.json").write_text(
                json.dumps({"summary": {"total": 1}, "items": [{"id": "gap001", "severity": "warning"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (run_dir / "publish_decision.json").write_text(
                json.dumps({"status": "publish_with_warnings"}, ensure_ascii=False),
                encoding="utf-8",
            )

            evidence = multidoc.load_evidence(run_dir, multidoc.read_json(run_dir / "analysis.json"))
            evidence_map = multidoc.build_evidence_map_json(evidence)

        self.assertEqual(evidence["chapters"][0]["title"], "学习章节")
        self.assertEqual(evidence["transcript"]["segments"][0]["text"], "第一段内容")
        self.assertIn("系统化学习摘要", evidence["study_context_text"])
        self.assertEqual(evidence_map["has_study_guide"], True)
        self.assertEqual(evidence_map["study_chapter_count"], 1)
        self.assertEqual(evidence_map["evidence_gap_count"], 1)
        self.assertEqual(evidence_map["publish_decision"], "publish_with_warnings")

    def test_chapter_generation_is_resumable_and_uses_bounded_evidence_packets(self):
        class FakeClient:
            def __init__(self):
                self.prompts = []

            def generate(self, *, prompt, **_kwargs):
                self.prompts.append(prompt)
                if len(self.prompts) <= 2:
                    return {
                        "response": json.dumps(
                            {
                                "chapter_summary": f"章节 {len(self.prompts)} 摘要",
                                "key_facts": [{"claim": "可验证事实", "evidence_ids": ["asr_001"]}],
                                "analysis": ["分析结论"],
                                "manual_review": ["补充说明"],
                                "cautions": [],
                                "citations": ["asr_001"],
                            },
                            ensure_ascii=False,
                        )
                    }
                return {
                    "response": json.dumps(
                        {
                            "overview": "全片总览",
                            "cross_chapter_conclusions": ["跨章结论"],
                            "limitations": ["存在转写误差"],
                        },
                        ensure_ascii=False,
                    )
                }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "orin").mkdir()
            (run_dir / "analysis.json").write_text("{}", encoding="utf-8")
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            chapter_dir = run_dir / "study_chapters"
            chapter_dir.mkdir()
            for index in (1, 2):
                (chapter_dir / f"chapter_{index:02d}.json").write_text(
                    json.dumps(
                        {
                            "chapter_id": f"chapter_{index:02d}",
                            "index": index,
                            "title": f"章节 {index}",
                            "start": f"00:0{index}:00",
                            "end": f"00:0{index}:30",
                            "summary": f"摘要 {index}",
                            "key_points": ["要点"],
                            "evidence": [
                                {
                                    "id": "asr_001",
                                    "source_type": "asr",
                                    "timestamp_label": "00:00:01",
                                    "confidence": 0.75,
                                    "text": "这是可验证的转写证据。",
                                },
                                {
                                    "id": "manual_001",
                                    "source_type": "manual",
                                    "timestamp_label": "00:00:00",
                                    "confidence": 0.7,
                                    "text": "手册内容" * 5000,
                                },
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            client = FakeClient()
            result = multidoc.run_multidoc_analysis(run_dir, client=client, text_model="test")

            self.assertEqual(result["generation"]["chapter_count"], 2)
            self.assertEqual(len(client.prompts), 3)
            self.assertTrue((run_dir / "docs_analysis" / "knowledge_notes.md").is_file())
            self.assertTrue((run_dir / "docs_analysis_chapters" / "knowledge_notes_v2.md").is_file())
            self.assertTrue((run_dir / "docs_analysis_chapters" / "deep_report_v2.md").is_file())
            self.assertIn("全片总览", (run_dir / "docs_analysis" / "deep_report.md").read_text(encoding="utf-8"))
            self.assertLess(len(client.prompts[0]), 20000)

            cached_client = FakeClient()
            multidoc.run_multidoc_analysis(run_dir, client=cached_client, text_model="test")
            self.assertEqual(cached_client.prompts, [])

    def test_chapter_budget_grows_for_large_evidence(self):
        small = {"evidence": [{"text": "短证据"}], "key_points": []}
        large = {"evidence": [{"text": "长证据" * 5000}], "key_points": ["a"] * 10}

        self.assertGreater(multidoc.chapter_output_budget(large), multidoc.chapter_output_budget(small))
        self.assertLessEqual(multidoc.chapter_output_budget(large), multidoc.MAX_CHAPTER_OUTPUT_TOKENS)

    def test_truncated_json_response_never_leaks_into_markdown(self):
        packet = {
            "chapter_id": "chapter_01",
            "index": 1,
            "title": "章节",
            "start": "00:00:00",
            "end": "00:00:30",
            "summary": "已有章节摘要",
            "key_points": ["已有关键点"],
            "review_flags": [],
            "evidence": [{"id": "asr_001", "text": "证据", "source_type": "asr"}],
        }
        raw = (
            '{"chapter_summary":"模型已经完成的章节摘要",'
            '"key_facts":[{"claim":"未完成'
        )

        result = multidoc.normalize_chapter_result(raw, packet)
        rendered = multidoc.render_knowledge_notes([result], {"overview": "总览"})

        self.assertEqual(result["chapter_summary"], "模型已经完成的章节摘要")
        self.assertEqual(result["key_facts"][0]["claim"], "已有关键点")
        self.assertNotIn('{"chapter_summary"', rendered)

    def test_invalid_overview_response_falls_back_to_chapter_summaries(self):
        chapters = [
            {"chapter_summary": "第一章摘要"},
            {"chapter_summary": "第二章摘要"},
        ]
        raw = '{"chapter_summary":"错误地返回了章节 JSON","key_facts":['

        overview = multidoc.normalize_overview(raw, chapters)

        self.assertEqual(overview["overview"], "第一章摘要\n第二章摘要")
        self.assertNotIn('{"chapter_summary"', overview["overview"])


if __name__ == "__main__":
    unittest.main()
