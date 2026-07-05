import json
import tempfile
import unittest
from pathlib import Path

from video_analyzer.web_evidence import SearchResult, build_web_evidence, parse_duckduckgo_html


class FakeClient:
    def generate(self, **kwargs):
        return {
            "response": json.dumps(
                {
                    "status": "partial_external_support",
                    "used_for": "补充官方背景说明",
                    "uncertainty_note": "外部资料只能说明项目背景，视频操作步骤仍需复核。",
                    "sources": [
                        {
                            "url": "https://docs.example.com/tool",
                            "source_confidence": "high",
                            "used_for": "官方文档",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        }


class WebEvidenceTests(unittest.TestCase):
    def test_parses_duckduckgo_html_results(self):
        html = """
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.example.com%2Ftool">Official Docs</a>
        <div class="result__snippet">Setup guide</div>
        """

        results = parse_duckduckgo_html(html, max_results=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://docs.example.com/tool")
        self.assertEqual(results[0].title, "Official Docs")

    def test_marks_video_only_gap_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "evidence_gaps.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "gap_0001",
                                "category": "vl_empty",
                                "severity": "warning",
                                "message": "VL frame analyses 为空",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = build_web_evidence(run_dir, no_network=False, search_fn=lambda **_: [], fetch_fn=lambda **_: "")

            item = result["web_evidence"]["items"][0]
            self.assertEqual(item["status"], "video_only_gap")
            self.assertEqual(result["summary"]["video_only_gap"], 1)
            self.assertTrue((run_dir / "web_evidence.json").is_file())
            self.assertTrue((run_dir / "web_evidence.md").is_file())

    def test_collects_and_reviews_external_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "evidence_gaps.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "gap_0001",
                                "category": "prior_review_blocks_publish",
                                "severity": "error",
                                "message": "既有复核文档建议阻止发布",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "operation_manual.md").write_text("# Demo Tool\n\n配置 API Token。", encoding="utf-8")

            def search_fn(query, max_results, timeout_sec):
                return [SearchResult("Official Docs", "https://docs.example.com/tool", "Setup guide")]

            def fetch_fn(url, timeout_sec):
                return "Official setup guide for Demo Tool."

            result = build_web_evidence(
                run_dir,
                client=FakeClient(),
                search_fn=search_fn,
                fetch_fn=fetch_fn,
            )

            item = result["web_evidence"]["items"][0]
            self.assertEqual(item["status"], "partial_external_support")
            self.assertEqual(item["sources"][0]["source_confidence"], "high")
            self.assertEqual(result["summary"]["partial_external_support"], 1)


if __name__ == "__main__":
    unittest.main()
