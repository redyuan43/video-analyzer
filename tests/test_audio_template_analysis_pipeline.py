import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_analyzer.audio_processor import AudioTranscript


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "pipelines" / "run_audio_template_analysis.py"
CATALOG_PATH = (
    ROOT
    / "video-analyzer-ui"
    / "video_analyzer_ui"
    / "static"
    / "data"
    / "audio_prompt_templates.json"
)
SPEC = importlib.util.spec_from_file_location("run_audio_template_analysis_pipeline", MODULE_PATH)
run_audio_template_analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_audio_template_analysis)


class FakeClient:
    def __init__(self, responder):
        self.responder = responder
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        response = self.responder(prompt, len(self.prompts))
        return {"response": response}


class AudioTemplateCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = run_audio_template_analysis.load_templates(CATALOG_PATH)

    def test_catalog_is_exact_doway_chinese_server_export(self):
        self.assertEqual(len(self.templates), 382)
        ids = [item["id"] for item in self.templates]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(template_id.isdigit() for template_id in ids))
        for item in self.templates:
            prompt = item["prompt_original"]
            server = item["server"]
            self.assertTrue(prompt.strip())
            self.assertEqual(item["source_repo"], "Doway AI server")
            self.assertEqual(item["source_path"], "analysis/doway_prompts/server_prompts_zh.json")
            self.assertEqual(server["source_language"], "zh")
            self.assertEqual(server["requested_language"], "zh")
            self.assertEqual(server["response_language"], "zh")
            self.assertEqual(str(server["template_id"]), item["id"])
            self.assertEqual(
                server["prompt_sha256"],
                hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            )

    def test_catalog_accepts_valid_subset_without_hard_coded_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(json.dumps(self.templates[:1], ensure_ascii=False), encoding="utf-8")
            loaded = run_audio_template_analysis.load_templates(path)
        self.assertEqual([item["id"] for item in loaded], [self.templates[0]["id"]])

    def test_catalog_rejects_empty_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                run_audio_template_analysis.load_templates(path)

    def test_unknown_explicit_template_id_is_visible_error(self):
        client = FakeClient(lambda _prompt, _call: self.fail("selector must not be called"))
        with self.assertRaisesRegex(ValueError, "unknown explicit Doway template_id"):
            run_audio_template_analysis.choose_template(
                client=client,
                model="fake-selector",
                templates=self.templates,
                transcript_text="项目会议讨论了上线安排",
                focus_prompt="",
                explicit_template_id="999999999999",
            )
        self.assertEqual(client.prompts, [])

    def test_inherited_text_selector_keeps_lmstudio_reasoning_options(self):
        class FakeConfig:
            def get_runtime_profile(self, _profile_name):
                return {
                    "template_selector_inherit": "text",
                    "text_base_url": "http://100.90.114.26:18081/v1",
                    "text_model": "prism-ml/bonsai-27b",
                    "reasoning_effort": "none",
                }

            def get(self, key, default=None):
                return {} if key == "study_cards" else default

        with patch.object(
            run_audio_template_analysis,
            "GenericOpenAIAPIClient",
        ) as client_class:
            _client, model, base_url, _temperature = (
                run_audio_template_analysis.build_template_selector_client(FakeConfig())
            )

        self.assertEqual(model, "prism-ml/bonsai-27b")
        self.assertEqual(base_url, "http://100.90.114.26:18081/v1")
        self.assertEqual(
            client_class.call_args.kwargs["extra_body"],
            {"reasoning_effort": "none"},
        )

    def test_cloud_route_switches_summary_and_selector_fallback_models(self):
        class FakeConfig:
            def __init__(self):
                self.config = {
                    "active_runtime_profile": "audio_nx1",
                    "runtime_profiles": {
                        "audio_nx1": {
                            "workflow_id": "audio_nx1",
                            "text_base_url": "http://127.0.0.1:18103/v1",
                            "text_model": "local-summary",
                            "text_fallback_enabled": True,
                            "text_fallback_base_url": "https://api.deepseek.com",
                            "text_fallback_model": "deepseek-v4-flash",
                            "text_fallback_api_key_env": "DEEPSEEK_API_KEY",
                            "template_selector_inherit": "text",
                            "template_selector_fallback_enabled": True,
                            "template_selector_fallback_base_url": "https://api.deepseek.com",
                            "template_selector_fallback_model": "deepseek-v4-flash",
                            "template_selector_fallback_api_key_env": "DEEPSEEK_API_KEY",
                            "template_selector_fallback_options": {
                                "temperature": 0.1
                            },
                            "audio_cloud_fallback": {"enabled": False},
                        }
                    },
                }

            def get_runtime_profile(self, profile_name):
                return self.config["runtime_profiles"][profile_name]

        config = FakeConfig()

        run_audio_template_analysis.apply_cloud_fallback(
            config,
            "audio_nx1",
        )

        profile = config.config["runtime_profiles"]["audio_nx1"]
        self.assertEqual(profile["text_model"], "deepseek-v4-flash")
        self.assertEqual(profile["text_base_url"], "https://api.deepseek.com")
        self.assertEqual(
            profile["template_selector_model"],
            "deepseek-v4-flash",
        )
        self.assertEqual(
            profile["template_selector_base_url"],
            "https://api.deepseek.com",
        )
        self.assertEqual(profile["template_selector_temperature"], 0.1)

    def test_invalid_selector_result_uses_content_form_fallback(self):
        client = FakeClient(
            lambda _prompt, _call: json.dumps(
                {"template_id": "outside-candidates", "scene": "会议", "confidence": 0.9}
            )
        )
        selected, classification = run_audio_template_analysis.choose_template(
            client=client,
            model="fake-selector",
            templates=self.templates,
            transcript_text="会议讨论产品上线、负责人、截止时间和待办事项。",
            focus_prompt="关注行动项",
        )
        self.assertIn(selected["id"], {item["id"] for item in self.templates})
        self.assertTrue(selected["id"].isdigit())
        self.assertEqual(selected["id"], "100005")
        self.assertEqual(classification["method"], "content-form-fallback")
        self.assertEqual(classification["content_form"], "meeting")
        self.assertEqual(classification["template_id"], selected["id"])
        self.assertNotIn("outside-candidates", classification["reason"])
        self.assertNotIn("outside-candidates", classification["warnings"][0])

    def test_template_shards_cover_catalog_once_and_stay_balanced(self):
        first = run_audio_template_analysis.build_template_shards(self.templates)
        second = run_audio_template_analysis.build_template_shards(self.templates)

        self.assertEqual(
            [[item["id"] for item in shard] for shard in first],
            [[item["id"] for item in shard] for shard in second],
        )
        self.assertEqual(sorted(len(shard) for shard in first), [76, 76, 76, 77, 77])
        flattened = [item["id"] for shard in first for item in shard]
        self.assertEqual(len(flattened), 382)
        self.assertEqual(len(set(flattened)), 382)

    def test_template_shard_prompts_fit_small_model_context_budget(self):
        prompts = [
            run_audio_template_analysis.render_template_shard_prompt(
                shard,
                "主持人与嘉宾持续讨论自主智能体。" * 2000,
                "",
                index,
                5,
            )
            for index, shard in enumerate(
                run_audio_template_analysis.build_template_shards(self.templates),
                1,
            )
        ]

        self.assertTrue(
            all(
                len(prompt)
                <= run_audio_template_analysis.TEMPLATE_SELECTOR_PROMPT_CHAR_LIMIT
                for prompt in prompts
            )
        )
        self.assertTrue(all("完整要求：" not in prompt for prompt in prompts))
        self.assertTrue(all(len(re.findall(r"^id=", prompt, re.MULTILINE)) >= 76 for prompt in prompts))

    def test_parallel_tournament_prefers_content_form_and_writes_audit(self):
        selected_id = "400000"

        def respond(prompt, call):
            ids = re.findall(r"^id=(\d+)$", prompt, flags=re.MULTILINE)
            if call <= 5:
                ranked_ids = ids[:3]
                if selected_id in ids and selected_id not in ranked_ids:
                    ranked_ids[-1] = selected_id
                ranked = [
                    {
                        "template_id": template_id,
                        "form_fit": 90 - index,
                        "domain_fit": 70,
                        "instruction_fit": 80,
                        "reason": "访谈形态匹配",
                    }
                    for index, template_id in enumerate(ranked_ids)
                ]
                return json.dumps(
                    {
                        "content_form": "interview",
                        "domain": "人工智能",
                        "ranked": ranked,
                    }
                )
            finalists = re.findall(r"^id=(\d+)$", prompt, flags=re.MULTILINE)
            winner = selected_id if selected_id in finalists else finalists[0]
            runner_up = next(item for item in finalists if item != winner)
            return json.dumps(
                {
                    "template_id": winner,
                    "runner_up_id": runner_up,
                    "content_form": "interview",
                    "domain": "人工智能",
                    "scene": "播客访谈",
                    "confidence": 0.92,
                    "margin": 18,
                    "reason": "内容是主持人与嘉宾的连续问答。",
                }
            )

        client = FakeClient(respond)
        with tempfile.TemporaryDirectory() as tmp:
            selected, classification = run_audio_template_analysis.choose_template(
                client=client,
                model="prism-ml/bonsai-27b",
                templates=self.templates,
                transcript_text="主持人：欢迎嘉宾。嘉宾：今天讨论模型部署。",
                focus_prompt="保留关键观点",
                output_dir=Path(tmp),
            )
            audit = json.loads(
                (Path(tmp) / "template_selection.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(client.prompts), 6)
        self.assertEqual(selected["id"], selected_id)
        self.assertEqual(classification["method"], "bonsai_parallel_tournament")
        self.assertEqual(classification["content_form"], "interview")
        self.assertEqual(classification["audit_path"], "template_selection.json")
        self.assertEqual(len(audit["shards"]), 5)
        self.assertLessEqual(len(audit["finalists"]), 15)
        self.assertTrue(
            all("template" not in finalist for finalist in audit["finalists"])
        )

    def test_shard_parser_completes_missing_third_candidate(self):
        shard = [
            template
            for template in self.templates
            if template.get("first_category") == "interview"
        ][:5]
        response = json.dumps(
            {
                "content_form": "interview",
                "domain": "科技",
                "ranked": [
                    {
                        "template_id": shard[0]["id"],
                        "form_fit": 95,
                        "domain_fit": 80,
                        "instruction_fit": 90,
                        "reason": "问答结构匹配",
                    },
                    {
                        "template_id": shard[1]["id"],
                        "form_fit": 90,
                        "domain_fit": 75,
                        "instruction_fit": 85,
                        "reason": "播客访谈匹配",
                    },
                ],
            }
        )

        parsed = run_audio_template_analysis.parse_template_shard_result(
            response,
            shard,
            1,
        )

        self.assertEqual(len(parsed["ranked"]), 3)
        self.assertEqual(parsed["ranked"][2]["form_fit"], 0.0)
        self.assertTrue(parsed["warnings"])

    def test_default_summary_intent_excludes_podcast_creation_template(self):
        summary = next(
            item for item in self.templates if item["id"] == "2000088"
        )
        marketing = next(
            item for item in self.templates if item["id"] == "2000093"
        )
        repurposing = next(
            item for item in self.templates if item["id"] == "2000252"
        )
        finalists = [
            {"template": summary},
            {"template": marketing},
            {"template": repurposing},
        ]

        accepted, excluded = (
            run_audio_template_analysis.filter_finalists_for_output_intent(
                finalists,
                "",
                "主持人欢迎嘉宾参加本期播客。",
            )
        )
        transformed, transformed_excluded = (
            run_audio_template_analysis.filter_finalists_for_output_intent(
                finalists,
                "请改写成营销播客脚本",
                "主持人欢迎嘉宾参加本期播客。",
            )
        )

        self.assertEqual(
            [item["template"]["id"] for item in accepted],
            ["2000088"],
        )
        self.assertEqual(
            [item["template"]["id"] for item in excluded],
            ["2000093", "2000252"],
        )
        self.assertEqual(transformed, finalists)
        self.assertEqual(transformed_excluded, [])

    def test_podcast_interview_excludes_specialized_interview_scenarios(self):
        template_ids = ("2000088", "2000092", "2000097", "2000098")
        finalists = [
            {
                "template": next(
                    item for item in self.templates if item["id"] == template_id
                )
            }
            for template_id in template_ids
        ]

        accepted, excluded = (
            run_audio_template_analysis.filter_finalists_for_output_intent(
                finalists,
                "",
                "主持人向嘉宾提问，欢迎收听本期播客。",
            )
        )

        self.assertEqual(
            [item["template"]["id"] for item in accepted],
            ["2000088"],
        )
        self.assertEqual(
            {item["template"]["id"] for item in excluded},
            {"2000092", "2000097", "2000098"},
        )

    def test_exact_podcast_scenario_outranks_general_interview_template(self):
        template_ids = ("2000007", "400002", "2000088")
        finalists = [
            {
                "template": next(
                    item for item in self.templates if item["id"] == template_id
                )
            }
            for template_id in template_ids
        ]

        accepted, excluded = (
            run_audio_template_analysis.filter_finalists_for_output_intent(
                finalists,
                "",
                "主持人与嘉宾在播客中持续问答。",
            )
        )

        self.assertEqual(
            [item["template"]["id"] for item in accepted],
            ["2000088"],
        )
        self.assertEqual(
            {item["template"]["id"] for item in excluded},
            {"2000007", "400002"},
        )

    def test_single_finalist_discards_fabricated_runner_up(self):
        template = next(
            item for item in self.templates if item["id"] == "2000088"
        )
        client = FakeClient(
            lambda _prompt, _call: json.dumps(
                {
                    "template_id": "2000088",
                    "runner_up_id": "2000089",
                    "content_form": "interview",
                    "domain": "科技",
                    "scene": "播客访谈",
                    "confidence": 0.95,
                    "margin": 100,
                    "reason": "主持人与嘉宾持续问答。",
                }
            )
        )

        result = run_audio_template_analysis.run_template_final_adjudication(
            client=client,
            model="fake",
            finalists=[
                {
                    "shard": 1,
                    "shard_rank": 1,
                    "shard_content_form": "interview",
                    "shard_domain": "科技",
                    "scores": {},
                    "template": template,
                }
            ],
            transcript_text="主持人与嘉宾在播客中持续问答。",
            focus_prompt="",
            majority_form="interview",
        )

        self.assertEqual(result["template_id"], "2000088")
        self.assertEqual(result["runner_up_id"], "")


class AudioTemplateLongTextTests(unittest.TestCase):
    def test_format_transcript_accepts_capitalized_provider_fields(self):
        transcript = AudioTranscript(
            text="fallback text",
            segments=[
                {
                    "Start": 1.5,
                    "End": 3.0,
                    "Speaker": "host",
                    "Content": "请嘉宾介绍一下。",
                },
                {
                    "Start": 3.0,
                    "End": 5.0,
                    "Speaker": "guest",
                    "Content": "好的。",
                },
            ],
            language="zh",
            metadata={},
        )

        rendered = run_audio_template_analysis.format_transcript_for_analysis(
            transcript
        )

        self.assertIn("[00:01-00:03] host: 请嘉宾介绍一下。", rendered)
        self.assertIn("[00:03-00:05] guest: 好的。", rendered)
        self.assertNotEqual(rendered, transcript.text)

    def test_asr_preflight_recognizes_only_speech_text(self):
        self.assertFalse(run_audio_template_analysis.has_meaningful_speech(""))
        self.assertFalse(run_audio_template_analysis.has_meaningful_speech("[Music]"))
        self.assertFalse(run_audio_template_analysis.has_meaningful_speech("[Environmental Sounds]"))
        self.assertTrue(run_audio_template_analysis.has_meaningful_speech("今天我们讨论项目计划"))

    def test_asr_preflight_samples_full_long_audio_span(self):
        offsets = run_audio_template_analysis.preflight_offsets(10_800)

        self.assertEqual(len(offsets), 5)
        self.assertEqual(offsets[0], 0.0)
        self.assertEqual(offsets[-1], 10_770)

    def test_short_transcript_keeps_single_content_call(self):
        client = FakeClient(lambda _prompt, _call: "短文本总结")
        summary = run_audio_template_analysis.summarize_with_template(
            client=client,
            model="fake-content",
            template={"id": "2", "title_zh": "模板", "prompt_original": "完整模板要求\n{transcript}"},
            transcript_text="[00:00-00:01] A: 短文本",
            focus_prompt="补充关注点",
            language="zh-CN",
            temperature=0.0,
        )
        self.assertEqual(summary, "短文本总结")
        self.assertEqual(len(client.prompts), 1)

    def test_long_transcript_maps_head_middle_tail_and_reduces_with_full_template(self):
        def respond(prompt, _call):
            if "连续分块" in prompt:
                markers = [marker for marker in ("MARK_HEAD", "MARK_MIDDLE", "MARK_TAIL") if marker in prompt]
                return "MAP_" + "_".join(markers)
            return "最终总结"

        transcript = "\n".join(
            (
                "[00:00-00:10] A: MARK_HEAD " + "甲" * 5000,
                "[00:10-00:20] B: MARK_MIDDLE " + "乙" * 5000,
                "[00:20-00:30] A: MARK_TAIL " + "丙" * 5000,
            )
        )
        template_prompt = "PROMPT_HEAD\n{{date}}\n按时间线完整总结\nPROMPT_TAIL"
        client = FakeClient(respond)
        with (
            patch.object(run_audio_template_analysis, "SUMMARY_SINGLE_PASS_CHARS", 9000),
            patch.object(run_audio_template_analysis, "SUMMARY_MAP_CHUNK_CHARS", 6000),
        ):
            summary = run_audio_template_analysis.summarize_with_template(
                client=client,
                model="fake-content",
                template={"id": "2", "title_zh": "模板", "prompt_original": template_prompt},
                transcript_text=transcript,
                focus_prompt="只作为补充",
                language="zh-CN",
                temperature=0.0,
                source_name="20260709113245.mp3",
            )

        self.assertEqual(summary, "最终总结")
        final_prompt = client.prompts[-1]
        self.assertIn("MAP_MARK_HEAD", final_prompt)
        self.assertIn("MAP_MARK_MIDDLE", final_prompt)
        self.assertIn("MAP_MARK_TAIL", final_prompt)
        self.assertIn("PROMPT_HEAD", final_prompt)
        self.assertIn("PROMPT_TAIL", final_prompt)
        self.assertNotIn("[中间内容已截断]", "\n".join(client.prompts))

    def test_profile_summary_settings_force_balanced_map_chunks(self):
        def respond(prompt, _call):
            if "连续分块" in prompt:
                return "- 独立事实\n- 独立结论"
            return "- 结论一\n- 结论二\n- 结论三"

        transcript = "\n".join(
            f"[00:{index:02d}-00:{index + 1:02d}] A: " + chr(0x4E00 + index) * 3500
            for index in range(6)
        )
        client = FakeClient(respond)
        settings = run_audio_template_analysis.summary_generation_settings(
            {
                "summary_single_pass_chars": 12000,
                "summary_map_chunk_chars": 8000,
                "summary_reduce_batch_chars": 24000,
                "summary_map_max_tokens": 1200,
            }
        )

        run_audio_template_analysis.summarize_with_template(
            client=client,
            model="fake-content",
            template={
                "id": "400000",
                "title_zh": "采访笔记",
                "prompt_original": "总结 {transcript}",
            },
            transcript_text=transcript,
            focus_prompt="",
            language="zh-CN",
            temperature=0.2,
            settings=settings,
        )

        map_prompts = [prompt for prompt in client.prompts if "连续分块" in prompt]
        self.assertEqual(len(map_prompts), 3)
        self.assertIn("第 1/3 个连续分块", map_prompts[0])
        self.assertIn("第 3/3 个连续分块", map_prompts[-1])

    def test_summary_quality_gate_retries_once_and_rejects_repetition(self):
        repeated = "\n".join(
            f"- [00:{index:02d}-00:{index + 1:02d}] 相同观点"
            for index in range(6)
        )
        corrected = "\n".join(
            f"- [00:{index:02d}-00:{index + 1:02d}] 独立观点{index}"
            for index in range(6)
        )
        client = FakeClient(
            lambda prompt, call: corrected if call == 2 else repeated
        )

        summary = run_audio_template_analysis.summarize_with_template(
            client=client,
            model="fake-content",
            template={
                "id": "400000",
                "title_zh": "采访笔记",
                "prompt_original": "总结 {transcript}",
            },
            transcript_text="短转写",
            focus_prompt="",
            language="zh-CN",
            temperature=0.2,
        )

        self.assertEqual(summary, corrected)
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("前一次输出未通过质量检查", client.prompts[-1])
        self.assertTrue(
            run_audio_template_analysis.assess_summary_quality(summary)["passed"]
        )

    def test_summary_quality_gate_reports_duplicate_time_ranges(self):
        summary = "\n".join(
            (
                "- [30:20-30:58] 行为规范的来源",
                "- [30:20-30:58] 行为规范的作用",
            )
        )

        quality = run_audio_template_analysis.assess_summary_quality(summary)

        self.assertFalse(quality["passed"])
        self.assertEqual(quality["duplicate_time_range_count"], 1)

    def test_deterministic_summary_deduplication_removes_empty_questions(self):
        summary = """问题4：行为规范
- [30:20-30:58] 独立观点

问题5：行为评估
- [37:32-37:53] 重复观点

问题6：本体
- [41:30-41:32] 重复观点

问题7：自我改进
- [59:58-01:01:09] 另一个独立观点"""

        deduplicated, removed = (
            run_audio_template_analysis.deduplicate_summary_lines(summary)
        )

        self.assertEqual(removed, 2)
        self.assertEqual(deduplicated.count("重复观点"), 1)
        self.assertNotIn("问题6：本体", deduplicated)
        self.assertIn("问题3：自我改进", deduplicated)
        self.assertTrue(
            run_audio_template_analysis.assess_summary_quality(
                deduplicated
            )["passed"]
        )

    def test_doway_placeholders_use_known_values_and_mark_unknowns(self):
        rendered = run_audio_template_analysis.render_template_prompt(
            "{{date}}|{{recordStartTime}}|{{recordEndTime}}|{{duration}}|{{location}}|{unknown}|{transcript}",
            transcript="正文",
            focus_prompt="补充",
            recording_time="2026年7月9日 11:32:45",
        )
        self.assertEqual(
            rendered,
            "2026年7月9日|2026年7月9日 11:32:45|未提供|未提供|未提供|未提供|正文",
        )

    def test_client_template_block_is_removed_from_user_supplement(self):
        focus = """【模板指令开始】
模板：旧客户端副本
这里是可能被截断的模板正文
【模板指令结束】

【用户补充】
请特别关注风险项"""
        self.assertEqual(
            run_audio_template_analysis.client_focus_supplement(focus),
            "请特别关注风险项",
        )

    def test_empty_core_summary_response_is_not_hidden(self):
        client = FakeClient(lambda _prompt, _call: "   ")
        with self.assertRaisesRegex(RuntimeError, "empty response"):
            run_audio_template_analysis.summarize_with_template(
                client=client,
                model="fake-content",
                template={"id": "2", "title_zh": "模板", "prompt_original": "总结 {transcript}"},
                transcript_text="正文",
                focus_prompt="",
                language="zh-CN",
                temperature=0.0,
            )

    def test_bonsai_content_disables_thinking_and_rejects_reasoning_fallback(self):
        class RecordingClient(FakeClient):
            def __init__(self, payload):
                super().__init__(lambda _prompt, _call: "")
                self.payload = payload
                self.kwargs = []

            def generate(self, prompt, **kwargs):
                self.prompts.append(prompt)
                self.kwargs.append(kwargs)
                return self.payload

        success = RecordingClient({"response": "最终摘要", "response_source": "content"})
        result = run_audio_template_analysis.generate_required_text(
            success,
            "prompt",
            model="prism-ml/bonsai-27b",
            temperature=0.0,
            num_predict=100,
            stage="test",
        )
        self.assertEqual(result, "最终摘要")
        self.assertEqual(
            success.kwargs[0]["extra_body"],
            run_audio_template_analysis.selector_request_extra_body(),
        )

        reasoning = RecordingClient(
            {
                "response": "Here's a thinking process:",
                "response_source": "reasoning_content",
            }
        )
        with self.assertRaisesRegex(RuntimeError, "reasoning_content"):
            run_audio_template_analysis.generate_required_text(
                reasoning,
                "prompt",
                model="prism-ml/bonsai-27b",
                temperature=0.0,
                num_predict=100,
                stage="test",
            )


class AudioTemplateStructuredOutputTests(unittest.TestCase):
    def test_structured_output_keeps_existing_fields_and_adds_mobile_fields(self):
        guide_response = json.dumps(
            {
                "title": "项目复盘",
                "summary": "结构化摘要",
                "keywords": ["上线", "复盘"],
                "action_items": [{"task": "修复问题", "owner": "张三", "deadline": "未提供"}],
                "chapters": [
                    {
                        "index": 1,
                        "title": "问题定位",
                        "start": "00:00",
                        "end": "00:30",
                        "summary": "定位问题",
                        "key_points": ["发现故障", "确认原因"],
                    }
                ],
            },
            ensure_ascii=False,
        )
        client = FakeClient(lambda _prompt, _call: guide_response)
        transcript = AudioTranscript(
            text="发现故障并确认原因",
            segments=[{"start": 0, "end": 30, "speaker": "A", "text": "发现故障并确认原因"}],
            language="zh",
            metadata={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            guide_path = run_audio_template_analysis.build_light_study_guide(
                client, "fake-content", output, transcript, "最终摘要", 0.0
            )
            guide = json.loads(guide_path.read_text(encoding="utf-8"))
            self.assertEqual(guide["title"], "项目复盘")
            self.assertEqual(guide["keywords"], ["上线", "复盘"])
            self.assertEqual(guide["action_items"][0]["task"], "修复问题")
            self.assertIn("chapters", guide)
            self.assertIn("mindmap", guide)

            analysis_path = run_audio_template_analysis.write_analysis_json(
                output_dir=output,
                media_path=output / "demo.mp3",
                transcript=transcript,
                asr_result=None,
                speaker_report={"enabled": False},
                selected_template={"id": "2", "title_zh": "模板", "prompt_original": "原文"},
                classification={"method": "explicit"},
                summary="最终摘要",
                selector_base_url="fake://selector",
                selector_model="fake-selector",
                content_base_url="fake://content",
                content_model="fake-content",
                manual_path=output / "operation_manual.md",
                evidence_path=output / "manual_evidence.md",
                study_guide_path=guide_path,
                elapsed_seconds=1.0,
                timings={
                    "asr_seconds": 0.5,
                    "template_selector_seconds": 0.2,
                    "manual_generation_seconds": 0.3,
                },
            )
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            audio_analysis = analysis["audio_template_analysis"]
            self.assertEqual(analysis["pipeline_profile"], "audio_nx1")
            self.assertEqual(analysis["metadata"]["pipeline_profile"], "audio_nx1")
            self.assertEqual(audio_analysis["pipeline_profile"], "audio_nx1")
            self.assertEqual(audio_analysis["summary"], "最终摘要")
            self.assertEqual(audio_analysis["title"], "项目复盘")
            self.assertEqual(audio_analysis["keywords"], ["上线", "复盘"])
            self.assertEqual(audio_analysis["action_items"][0]["owner"], "张三")
            self.assertIn("study_guide", audio_analysis)
            self.assertIn("mindmap", audio_analysis)
            self.assertEqual(analysis["structured_content"]["summary"], "最终摘要")
            self.assertEqual(analysis["metadata"]["timings"]["asr_seconds"], 0.5)
            self.assertEqual(
                analysis["metadata"]["timings"]["template_selector_seconds"],
                0.2,
            )
            self.assertEqual(
                analysis["metadata"]["timings"]["manual_generation_seconds"],
                0.3,
            )
            self.assertEqual(analysis["metadata"]["timings"]["total_seconds"], 1.0)
            self.assertTrue(
                analysis["operation_manual"]["quality_gate_passed"]
            )
            self.assertEqual(
                analysis["audio_template_analysis"]["quality"]["duplicate_ratio"],
                0.0,
            )

            overview = run_audio_template_analysis.render_study_overview(guide)
            self.assertIn("## 内容脑图", overview)
            self.assertIn("```mermaid\nmindmap", overview)
            self.assertIn("问题定位", overview)

    def test_operation_manual_hides_internal_selector_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = run_audio_template_analysis.write_operation_manual(
                Path(tmp),
                {"title_zh": "采访笔记"},
                {
                    "scene": "访谈",
                    "reason": (
                        "\x1b[36mray::TemplateSelectorShardActor.select()\x1b[0m\n"
                        'Traceback File "/home/ai/private.py" OpenAIAPIError'
                    ),
                },
                "干净总结",
                "",
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn("自动模板选择暂不可用", text)
        self.assertNotIn("ray::", text)
        self.assertNotIn("Traceback", text)
        self.assertNotIn("/home/ai", text)


if __name__ == "__main__":
    unittest.main()
