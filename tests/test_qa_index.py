import json
import tempfile
import unittest
from pathlib import Path

from video_analyzer.doc_chat import ask_video_docs_result
from video_analyzer.qa_index import build_qa_index, retrieve_context


class FakeClient:
    def __init__(self):
        self.prompt = ""

    def generate(self, *, prompt, model, temperature, num_predict):
        self.prompt = prompt
        return {"response": "根据 manual_evidence.md，API Token 在 Settings 页面填写。"}


class QaIndexTests(unittest.TestCase):
    def test_build_index_and_retrieve_relevant_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "operation_manual.md").write_text(
                "# 操作手册\n\n## 部署步骤\n\n点击 Settings，然后填写 API Token 并保存。",
                encoding="utf-8",
            )
            (run_dir / "manual_evidence.md").write_text(
                "# 证据\n\n[00:01:02] frame_012 显示 Settings 页面和 API Token 输入框。",
                encoding="utf-8",
            )
            (run_dir / "RUN_MANIFEST.md").write_text(
                "# RUN_MANIFEST\n\n先读 visual_review.html，再核对 manual_evidence.md。",
                encoding="utf-8",
            )
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "vl_frames_processed": 0,
                            "frame_selection": {"vl_frame_policy_resolved": "none"},
                            "ocr_keyframes": {"ocr_text_events_count": 0},
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            index = build_qa_index(run_dir)
            context = retrieve_context(run_dir, "API Token 在哪里填写？", max_context_chars=300)

            self.assertEqual(index["source_count"], 4)
            self.assertGreaterEqual(index["chunk_count"], 4)
            self.assertTrue(any(item["name"] == "run_manifest" for item in index["sources"]))
            self.assertTrue((run_dir / "qa" / "answer_index.json").is_file())
            self.assertTrue((run_dir / "qa" / "source_chunks.jsonl").is_file())
            joined = "\n".join(chunk["text"] for chunk in context["chunks"])
            self.assertIn("API Token", joined)
            self.assertTrue(any(item["code"] == "vl_skipped" for item in context["warnings"]))

    def test_doc_chat_uses_retrieved_context_and_returns_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "operation_manual.md").write_text(
                "# 操作手册\n\nAPI Token 在 Settings 页面填写。",
                encoding="utf-8",
            )
            (run_dir / "manual_evidence.md").write_text(
                "# 证据\n\nFrame_003 显示 Settings 页面里的 API Token 输入框。",
                encoding="utf-8",
            )
            (run_dir / "analysis.json").write_text(
                json.dumps({"metadata": {"vl_frames_processed": 0}}, ensure_ascii=False),
                encoding="utf-8",
            )
            build_qa_index(run_dir)
            client = FakeClient()

            result = ask_video_docs_result(run_dir, "API Token 在哪里填写？", client, "fake-model")

        self.assertIn("manual_evidence", client.prompt)
        self.assertIn("API Token", client.prompt)
        self.assertIn("Settings", result["answer"])
        self.assertTrue(result["citations"])
        self.assertTrue(any(item["confidence"] == "high" for item in result["citations"]))
        self.assertTrue(any(item["code"] == "vl_skipped" for item in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
