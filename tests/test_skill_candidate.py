import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_analyzer import skill_distillation as distill


class FakeClient:
    def generate(self, prompt, **kwargs):
        if "证据分析器" in prompt:
            return response(
                {
                    "summary": "演示如何先确认需求，再执行并验证结果。",
                    "structure": [
                        {
                            "title": "执行流程",
                            "summary": "需求、实施、验证",
                            "source_ids": ["transcript:0000", "page:0000"],
                        }
                    ],
                    "methods": [
                        {
                            "title": "先定义完成标准",
                            "summary": "实施前明确结果和验证方式",
                            "source_ids": ["transcript:0000", "visual:0000"],
                        }
                    ],
                    "concepts": [],
                    "cases": [],
                    "failures": [],
                    "limitations": [],
                }
            )
        if "全局理解" in prompt:
            return response(
                {
                    "title": "Demo Workflow",
                    "summary": "从需求澄清到结果验证的完整流程。",
                    "source_kind": "video",
                    "structure": [],
                    "methods": [],
                    "concepts": [],
                    "cases": [],
                    "failures": [],
                    "critique": [],
                    "coverage": {
                        "included_source_types": ["transcript", "visual"],
                        "known_gaps": [],
                    },
                }
            )
        if "提取器" in prompt:
            return response(
                {
                    "candidates": [
                        {
                            "title": "定义完成标准",
                            "summary": "开始执行前先定义可验证的完成条件。",
                            "source_ids": ["transcript:0000", "visual:0000"],
                            "source_quote": "先确认目标，再检查结果。",
                            "tags": ["workflow"],
                            "execution_hint": "写出完成标准",
                            "boundaries": ["纯信息查询"],
                        }
                    ]
                }
            )
        if "三项验证" in prompt:
            return response(
                {
                    "accepted": [
                        {
                            "id": "define-done",
                            "title": "定义完成标准",
                            "type": "framework",
                            "merged_from": ["f001", "p001"],
                            "source_ids": ["transcript:0000", "visual:0000"],
                            "source_quote": "先确认目标，再检查结果。",
                            "summary": "把模糊目标转成可判断的完成条件。",
                            "tags": ["workflow"],
                            "v1": {
                                "passed": True,
                                "reason": "口播与后续画面验证形成两个语境。",
                                "evidence_ids": ["transcript:0000", "page:0000"],
                            },
                            "v2": {
                                "passed": True,
                                "novel_question": "如何判断迁移已完成？",
                                "derived_answer": "先列出服务、数据和回归检查。",
                            },
                            "v3": {
                                "passed": True,
                                "reason": "强调实施前定义判定条件。",
                            },
                        }
                    ],
                    "rejected": [],
                }
            )
        if "构造成简洁" in prompt:
            return response(
                {
                    "name": "define-done-criteria",
                    "title": "定义完成标准",
                    "description": "用户准备执行复杂任务但尚未说明怎样算完成时使用；不适用于纯信息查询。",
                    "reading": {
                        "quote": "先确认目标，再检查结果。",
                        "source_ids": ["transcript:0000", "visual:0000"],
                        "source_note": "00:00 与 01:10",
                    },
                    "interpretation": "先把目标转成可以检查的结果，再开始行动。",
                    "applications": [],
                    "triggers": {
                        "scenarios": ["实施或迁移前"],
                        "language_signals": ["怎样才算完成"],
                        "distinctions": [],
                    },
                    "execution": [
                        {
                            "title": "列出标准",
                            "instruction": "列出可观察结果。",
                            "done_when": "每项都有检查方法。",
                            "stop_condition": "",
                        }
                    ],
                    "boundaries": {
                        "do_not_use": ["纯事实查询"],
                        "failure_modes": [],
                        "limitations": [],
                    },
                    "tags": ["workflow"],
                    "related_skills": [],
                }
            )
        if "分析同一来源" in prompt:
            return response({"links": []})
        if "生成触发压力测试" in prompt:
            return response(
                {
                    "test_cases": [
                        test_case("should-trigger-01", "should_trigger", "怎样才算迁移完成？", "define-done-criteria"),
                        test_case("should-trigger-02", "should_trigger", "先帮我定验收标准", "define-done-criteria"),
                        test_case("should-trigger-03", "should_trigger", "这个任务完成条件是什么", "define-done-criteria"),
                        test_case("should-not-trigger-01", "should_not_trigger", "API 参数是什么", None),
                        test_case("should-not-trigger-02", "should_not_trigger", "现在几点", None),
                        test_case("edge-01", "edge_case", "晚饭吃什么", None),
                    ]
                }
            )
        if "独立触发评测器" in prompt:
            return response(
                {
                    "results": [
                        judge("should-trigger-01", "define-done-criteria"),
                        judge("should-trigger-02", "define-done-criteria"),
                        judge("should-trigger-03", "define-done-criteria"),
                        judge("should-not-trigger-01", None),
                        judge("should-not-trigger-02", None),
                        judge("edge-01", None),
                    ]
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:120]}")


class SkillDistillationTests(unittest.TestCase):
    def test_full_pipeline_pauses_twice_then_delivers_tested_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            write_raw_evidence(run_dir)
            runtime = fake_runtime()

            with patch.object(distill, "resolve_model_runtime", return_value=runtime):
                distill.initialize_distillation(run_dir)
                pipeline = distill.SkillDistillationPipeline(run_dir)
                first = pipeline.run_until_pause()
                self.assertEqual(first["status"], "waiting_overview_review")
                self.assertTrue((distill.pack_dir(run_dir) / "BOOK_OVERVIEW.md").is_file())

                pipeline.review_overview("confirm")
                second = pipeline.run_until_pause()
                self.assertEqual(second["status"], "waiting_candidate_review")
                self.assertEqual(second["candidates"]["accepted_count"], 1)

                selected = second["candidates"]["selected_ids"]
                pipeline.review_candidates(selected)
                final = pipeline.run_until_pause()

            self.assertEqual(final["status"], "succeeded")
            self.assertEqual(final["profile"], "deepseek_v4_pro")
            self.assertEqual(final["skills"]["passed"], 1)
            self.assertEqual(final["skills"]["test_progress"]["phase"], "completed")
            self.assertEqual(final["skills"]["test_progress"]["stage_percent"], 100)
            skill_dir = distill.pack_dir(run_dir) / "distilled_skills" / "define-done-criteria"
            self.assertTrue((skill_dir / "SKILL.md").is_file())
            self.assertTrue((skill_dir / "test-prompts.json").is_file())
            self.assertTrue((distill.pack_dir(run_dir) / "DIGEST.md").is_file())

    def test_build_resume_reuses_complete_candidate_skill_without_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            write_raw_evidence(run_dir)
            runtime = fake_runtime()

            with patch.object(distill, "resolve_model_runtime", return_value=runtime):
                distill.initialize_distillation(run_dir)
                pipeline = distill.SkillDistillationPipeline(run_dir)
                pipeline.run_until_pause()
                pipeline.review_overview("confirm")
                waiting = pipeline.run_until_pause()
                pipeline.review_candidates(waiting["candidates"]["selected_ids"])

                verified = json.loads(
                    (distill.pack_dir(run_dir) / "verified.json").read_text(encoding="utf-8")
                )
                candidate = verified["accepted"][0]
                skill_dir = distill.pack_dir(run_dir) / "distilled_skills" / "existing-skill"
                skill_dir.mkdir(parents=True)
                skill_dir.joinpath("SKILL.md").write_text("# Existing\n", encoding="utf-8")
                skill_dir.joinpath("skill.json").write_text(
                    json.dumps(
                        {
                            "candidate_id": candidate["id"],
                            "name": "existing-skill",
                            "title": "Existing skill",
                        }
                    ),
                    encoding="utf-8",
                )

                with patch.object(
                    distill,
                    "call_json",
                    side_effect=AssertionError("resume must not rebuild a complete skill"),
                ):
                    pipeline._build_skills()

            self.assertEqual(pipeline.state["skills"]["count"], 1)
            self.assertEqual(pipeline.state["skills"]["items"][0]["name"], "existing-skill")
            self.assertEqual(pipeline.state["skills"]["items"][0]["status"], "built")

    def test_manifest_only_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "RUN_MANIFEST.md").write_text("# Manifest\n", encoding="utf-8")
            with patch.object(distill, "resolve_model_runtime", return_value=fake_runtime()):
                distill.initialize_distillation(run_dir)
                with self.assertRaises(FileNotFoundError):
                    distill.SkillDistillationPipeline(run_dir).run_until_pause()
            self.assertEqual(distill.load_state(run_dir)["status"], "failed")

    def test_enable_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            repo_root = Path(tmp) / "repo"
            run_dir.mkdir()
            repo_root.mkdir()
            write_completed_pack(run_dir)

            first = distill.enable_distilled_skills(run_dir, repo_root)
            self.assertTrue(first["enabled"])
            with self.assertRaises(FileExistsError):
                distill.enable_distilled_skills(run_dir, repo_root)
            overwritten = distill.enable_distilled_skills(run_dir, repo_root, overwrite=True)

            self.assertTrue(overwritten["enabled"])
            self.assertTrue(
                (repo_root / ".codex" / "skills" / "define-done-criteria" / "SKILL.md").is_file()
            )

    def test_load_evidence_uses_raw_sources_and_ignores_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            write_raw_evidence(run_dir)
            (run_dir / "RUN_MANIFEST.md").write_text("# Manifest only indexes files\n", encoding="utf-8")

            records, sources = distill.load_evidence_records(run_dir)

            self.assertEqual({item["source_type"] for item in records}, {"transcript", "ocr", "visual", "page"})
            self.assertNotIn("run_manifest", {item["type"] for item in sources})
            self.assertIn("transcript:0000", {item["id"] for item in records})

    def test_overview_normalization_drops_invented_source_ids(self):
        overview = {
            "structure": [
                {
                    "title": "Valid",
                    "source_ids": ["transcript:0000", "chunk_index: 2"],
                },
                {
                    "title": "Unsupported",
                    "source_ids": ["chunk_index: 3"],
                },
            ],
            "methods": [],
            "concepts": [],
            "cases": [],
            "failures": [],
            "critique": [],
            "coverage": {
                "included_source_types": ["transcript", "made-up"],
                "known_gaps": ["missing validation"],
            },
        }

        normalized = distill.normalize_overview(
            overview,
            {
                "transcript:0000": {
                    "id": "transcript:0000",
                    "source_type": "transcript",
                }
            },
        )

        self.assertEqual(len(normalized["structure"]), 1)
        self.assertEqual(normalized["structure"][0]["source_ids"], ["transcript:0000"])
        self.assertEqual(normalized["coverage"]["included_source_types"], ["transcript"])

    def test_overview_normalization_drops_visual_only_core_items_when_narrative_exists(self):
        overview = {
            "structure": [
                {"title": "Core", "source_ids": ["transcript:0000", "visual:0000"]},
                {"title": "Incidental page", "source_ids": ["visual:0001", "ocr:0000"]},
            ],
            "methods": [],
            "concepts": [],
            "cases": [],
            "failures": [],
            "critique": [
                {"issue": "Visual limitation", "source_ids": ["visual:0001"]},
            ],
            "coverage": {},
        }
        records = {
            "transcript:0000": {"source_type": "transcript"},
            "visual:0000": {"source_type": "visual"},
            "visual:0001": {"source_type": "visual"},
            "ocr:0000": {"source_type": "ocr"},
        }

        normalized = distill.normalize_overview(overview, records)

        self.assertEqual([item["title"] for item in normalized["structure"]], ["Core"])
        self.assertEqual(len(normalized["critique"]), 1)

    def test_normalize_tests_adds_missing_sibling_confusion_case(self):
        tests = distill.normalize_tests(
            "current-skill",
            {
                "test_cases": [
                    test_case("trigger-01", "should_trigger", "请处理当前方法", "current-skill"),
                    test_case("trigger-02", "should_trigger", "继续处理当前方法", "current-skill"),
                    test_case("trigger-03", "should_trigger", "再处理当前方法", "current-skill"),
                    test_case("negative-01", "should_not_trigger", "现在几点", None),
                    test_case("negative-02", "should_not_trigger", "API 参数是什么", None),
                    test_case("edge-01", "edge_case", "不确定该不该触发", None),
                ]
            },
            [
                {"name": "current-skill", "title": "当前 Skill"},
                {
                    "name": "sibling-skill",
                    "title": "相邻 Skill",
                    "triggers": {"scenarios": ["用户需要相邻 Skill 的专属流程"]},
                },
            ],
        )

        confusion = next(
            item
            for item in tests["test_cases"]
            if item["id"] == "should-not-trigger-sibling-confusion"
        )
        self.assertEqual(confusion["expected_skill"], "sibling-skill")
        self.assertEqual(confusion["type"], "should_not_trigger")
        self.assertIn("相邻 Skill", confusion["prompt"])

    def test_verification_normalization_rejects_candidates_omitted_by_model(self):
        records = {
            "visual:0000": {
                "id": "visual:0000",
                "source_type": "visual",
                "timestamp": 10,
            }
        }
        candidates = [
            {
                "id": "p001",
                "title": "Single-source principle",
                "type": "principle",
                "source_ids": ["visual:0000"],
            }
        ]

        verified = distill.normalize_verification(
            {"accepted": [], "rejected": []},
            records,
            candidates,
        )

        self.assertEqual(len(verified["rejected"]), 1)
        self.assertEqual(verified["rejected"][0]["id"], "p001")
        self.assertEqual(
            verified["rejected"][0]["failed_checks"],
            ["verification_omitted"],
        )

    def test_evidence_events_merge_modalities_from_same_time_window(self):
        records, events = distill.assign_evidence_events(
            [
                {
                    "id": "transcript:0001",
                    "source_type": "transcript",
                    "path": "orin/transcript.json",
                    "timestamp": 356,
                },
                {
                    "id": "ocr:0001",
                    "source_type": "ocr",
                    "path": "orin/ocr_events.json",
                    "timestamp": 358,
                },
                {
                    "id": "visual:0001",
                    "source_type": "visual",
                    "path": "orin/frame_analyses.json",
                    "timestamp": 359,
                },
                {
                    "id": "visual:0002",
                    "source_type": "visual",
                    "path": "orin/frame_analyses.json",
                    "timestamp": 420,
                },
            ],
            window_seconds=30,
        )

        self.assertEqual(records[0]["event_id"], records[1]["event_id"])
        self.assertEqual(records[1]["event_id"], records[2]["event_id"])
        self.assertNotEqual(records[2]["event_id"], records[3]["event_id"])
        self.assertEqual(len(events), 2)

    def test_single_event_candidate_is_classified_as_single_case(self):
        records = {
            "transcript:0001": {
                "id": "transcript:0001",
                "source_type": "transcript",
                "event_id": "video-event:0011",
            },
            "visual:0001": {
                "id": "visual:0001",
                "source_type": "visual",
                "event_id": "video-event:0011",
            },
        }
        candidate = {
            "id": "p001",
            "title": "Ortholinear layout",
            "type": "principle",
            "source_ids": ["transcript:0001", "visual:0001"],
        }
        result = {
            "evaluations": [
                {
                    **candidate,
                    "v1": {
                        "passed": True,
                        "evidence_ids": ["transcript:0001", "visual:0001"],
                    },
                    "v2": {"passed": True},
                    "v3": {"passed": True},
                }
            ]
        }

        verified = distill.normalize_verification(
            result,
            records,
            [candidate],
            multimodal_audits={
                "p001": {
                    "status": "succeeded",
                    "claim_supported": True,
                    "execution_supported": True,
                    "contradiction": False,
                }
            },
        )

        self.assertEqual(verified["accepted"], [])
        self.assertEqual(len(verified["single_case"]), 1)
        self.assertEqual(
            verified["single_case"][0]["v1"]["independent_context_count"],
            1,
        )

    def test_multimodal_contradiction_rejects_visual_candidate(self):
        records = {
            "visual:0001": {
                "id": "visual:0001",
                "source_type": "visual",
                "event_id": "video-event:0001",
            },
            "visual:0002": {
                "id": "visual:0002",
                "source_type": "visual",
                "event_id": "video-event:0002",
            },
        }
        candidate = {
            "id": "p001",
            "title": "Unsupported visual claim",
            "type": "principle",
            "source_ids": ["visual:0001", "visual:0002"],
        }
        result = {
            "evaluations": [
                {
                    **candidate,
                    "v1": {
                        "passed": True,
                        "evidence_ids": ["visual:0001", "visual:0002"],
                    },
                    "v2": {"passed": True},
                    "v3": {"passed": True},
                }
            ]
        }

        verified = distill.normalize_verification(
            result,
            records,
            [candidate],
            multimodal_audits={
                "p001": {
                    "status": "succeeded",
                    "claim_supported": False,
                    "execution_supported": False,
                    "contradiction": True,
                }
            },
        )

        self.assertEqual(verified["accepted"], [])
        self.assertEqual(verified["single_case"], [])
        self.assertEqual(
            verified["rejected"][0]["failed_checks"],
            ["v1", "multimodal"],
        )

    def test_glossary_items_do_not_count_as_rejected_skills(self):
        glossary = [
            {
                "id": "g001",
                "title": "QMK",
                "type": "term",
                "source_ids": ["transcript:0000"],
            }
        ]

        verified = distill.normalize_verification(
            {"evaluations": []},
            {},
            [],
            glossary=glossary,
        )

        self.assertEqual(verified["rejected"], [])
        self.assertEqual(verified["glossary"][0]["evidence_level"], "glossary")

    def test_stage_progress_includes_running_stage_fraction(self):
        state = {
            "status": "running",
            "current_stage": "test",
            "stages": {
                name: {
                    "status": "succeeded" if name in {
                        "source",
                        "overview",
                        "extract",
                        "verify",
                        "build",
                        "link",
                    } else "pending"
                }
                for name in distill.PIPELINE_STAGES
            },
        }
        state["stages"]["test"] = {
            "status": "running",
            "progress_percent": 50,
        }

        progress = distill.stage_progress(state)

        self.assertEqual(progress["percent"], 81)


def fake_runtime():
    client = FakeClient()
    return distill.ModelRuntime(
        profile_name="deepseek_v4_pro",
        base_url="https://api.deepseek.com",
        generation_model="deepseek-v4-pro",
        review_model="deepseek-v4-pro",
        generation_temperature=1.0,
        review_temperature=1.0,
        generation_client=client,
        review_client=client,
        concurrency=5,
    )


def response(payload):
    return {"response": json.dumps(payload, ensure_ascii=False)}


def test_case(case_id, case_type, prompt, expected_skill):
    return {
        "id": case_id,
        "type": case_type,
        "prompt": prompt,
        "expected_behavior": "按预期选择 skill",
        "expected_skill": expected_skill,
        "notes": "fixture",
    }


def judge(case_id, selected):
    return {
        "id": case_id,
        "selected_skill": selected,
        "would_trigger": selected is not None,
        "reason": "fixture",
        "action": "fixture",
    }


def write_raw_evidence(run_dir):
    orin = run_dir / "orin"
    orin.mkdir()
    (orin / "transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "Start": 0,
                        "End": 8,
                        "Speaker": "A",
                        "Content": "先确认目标和完成标准，再开始执行。",
                    }
                ],
                "text": "先确认目标和完成标准，再开始执行。",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (orin / "ocr_events.json").write_text(
        json.dumps(
            [{"frame_number": 1, "timestamp": 10, "status": "ok", "text": "验收标准"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (orin / "frame_analyses.json").write_text(
        json.dumps(
            [
                {
                    "frame_number": 9,
                    "timestamp": 70,
                    "status": "succeeded",
                    "response": "画面展示执行完成后逐项检查验收标准。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (orin / "page_context.md").write_text(
        "# Page Context Evidence: Demo\n\n## Page Description\n\n展示一个可复核工作流。\n",
        encoding="utf-8",
    )


def write_completed_pack(run_dir):
    root = distill.pack_dir(run_dir)
    skill = root / "distilled_skills" / "define-done-criteria"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: define-done-criteria\ndescription: test\n---\n\n# Test\n",
        encoding="utf-8",
    )
    state = {
        "version": 1,
        "method": distill.METHOD_NAME,
        "run_dir": str(run_dir),
        "status": "succeeded",
        "current_stage": None,
        "profile": "deepseek_v4_pro",
        "generation_model": "deepseek-v4-pro",
        "review_model": "deepseek-v4-pro",
        "created_at": distill.utc_now(),
        "updated_at": distill.utc_now(),
        "stages": {name: {"status": "succeeded"} for name in distill.PIPELINE_STAGES},
        "warnings": [],
        "overview": {"reviewed": True},
        "candidates": {"reviewed": True, "selected_ids": ["define-done"]},
        "skills": {
            "count": 1,
            "passed": 1,
            "failed": 0,
            "items": [
                {
                    "name": "define-done-criteria",
                    "title": "定义完成标准",
                    "status": "passed",
                }
            ],
        },
        "installed": {"paths": []},
        "artifacts": {},
    }
    distill.save_state(run_dir, state)


if __name__ == "__main__":
    unittest.main()
