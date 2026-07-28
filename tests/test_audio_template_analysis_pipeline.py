import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_analyzer.audio_processor import AudioTranscript


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "run_audio_template_analysis.py"
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

    def test_catalog_rejects_wrong_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(json.dumps(self.templates[:1], ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 382"):
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

    def test_invalid_small_model_id_uses_legal_keyword_fallback(self):
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
        self.assertEqual(classification["method"], "keyword-fallback")
        self.assertEqual(classification["template_id"], selected["id"])


class AudioTemplateLongTextTests(unittest.TestCase):
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
            )
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            audio_analysis = analysis["audio_template_analysis"]
            self.assertEqual(audio_analysis["summary"], "最终摘要")
            self.assertEqual(audio_analysis["title"], "项目复盘")
            self.assertEqual(audio_analysis["keywords"], ["上线", "复盘"])
            self.assertEqual(audio_analysis["action_items"][0]["owner"], "张三")
            self.assertIn("study_guide", audio_analysis)
            self.assertIn("mindmap", audio_analysis)
            self.assertEqual(analysis["structured_content"]["summary"], "最终摘要")


if __name__ == "__main__":
    unittest.main()
