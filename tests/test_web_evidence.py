import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from video_analyzer.web_evidence import (
    SearchResult,
    build_web_evidence,
    extract_review_claims,
    parse_duckduckgo_html,
    run_amd_brave_bridge,
)


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


class FactAuditClient:
    def generate(self, **kwargs):
        prompt = kwargs["prompt"]
        if "视频原始断言" in prompt:
            return {
                "response": json.dumps(
                    {
                        "verdict": "supported",
                        "confidence": "high",
                        "conclusion": "官方资料直接确认该产品能力。",
                        "recommended_wording": "官方资料显示该产品具备此能力。",
                        "source_ids": ["source_01"],
                    },
                    ensure_ascii=False,
                )
            }
        return FakeClient().generate(**kwargs)


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

    def test_triage_non_web_route_skips_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "evidence_gaps.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "gap_0001",
                                "category": "prior_review_conditional_publish",
                                "severity": "warning",
                                "message": "既有复核文档建议附条件发布",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "evidence_triage.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "gap_id": "gap_0001",
                                "evidence_class": "review_decision",
                                "resolution_route": "review_parse",
                                "publish_impact": "warning",
                                "recommendation": "解析最终发布建议，不需要联网。",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def search_fn(**kwargs):
                raise AssertionError("search should not be called")

            result = build_web_evidence(run_dir, no_network=False, search_fn=search_fn, fetch_fn=lambda **_: "")

            item = result["web_evidence"]["items"][0]
            self.assertEqual(item["status"], "not_applicable")
            self.assertEqual(result["summary"]["not_applicable"], 1)

    def test_extracts_and_audits_review_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "operation_manual.md").write_text(
                """# Demo

## 需复核项

| 序号 | 内容 | 证据来源 | 复核建议 |
| --- | --- | --- | --- |
| 1 | 示例产品于 2026 年发布，续航达到 500 公里 | 播客 ASR | 查询官方公告 |
""",
                encoding="utf-8",
            )

            def search_fn(query, max_results, timeout_sec):
                self.assertIn("示例产品", query)
                return [SearchResult("Official release", "https://docs.example.com/release", "Official release")]

            result = build_web_evidence(
                run_dir,
                client=FactAuditClient(),
                search_fn=search_fn,
                fetch_fn=lambda url, timeout_sec: "Official release confirms the claim.",
            )

            claim = result["web_evidence"]["claims"][0]
            self.assertEqual(claim["claim"], "示例产品于 2026 年发布，续航达到 500 公里")
            self.assertEqual(claim["verdict"], "supported")
            self.assertEqual(claim["source_ids"], ["source_01"])
            self.assertEqual(result["summary"]["supported"], 1)
            markdown = (run_dir / "web_evidence.md").read_text(encoding="utf-8")
            self.assertIn("视频断言事实审计", markdown)
            self.assertIn("有直接支持：1", markdown)

    def test_claim_without_cited_source_is_downgraded(self):
        class InvalidFactAuditClient:
            def generate(self, **kwargs):
                return {
                    "response": json.dumps(
                        {
                            "verdict": "contradicted",
                            "confidence": "high",
                            "conclusion": "没有来源也判为冲突。",
                            "recommended_wording": "不要这样写。",
                            "source_ids": [],
                        },
                        ensure_ascii=False,
                    )
                }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "operation_manual.md").write_text("## 需复核项\n\n- 某项具体事实\n", encoding="utf-8")
            result = build_web_evidence(
                run_dir,
                client=InvalidFactAuditClient(),
                search_fn=lambda **_: [SearchResult("Primary", "https://docs.example.com/fact", "")],
                fetch_fn=lambda **_: "Primary fact.",
            )

            claim = result["web_evidence"]["claims"][0]
            self.assertEqual(claim["verdict"], "not_enough_evidence")
            self.assertEqual(claim["source_ids"], [])

    def test_amd_brave_bridge_never_places_key_in_command(self):
        completed = SimpleNamespace(returncode=0, stdout='{"web":{"results":[]}}', stderr="")
        with patch("video_analyzer.web_evidence.subprocess.run", return_value=completed) as run:
            payload = run_amd_brave_bridge(
                "search",
                "Anthropic Claude Code",
                timeout_sec=5,
                ssh_target="AMD",
                remote_env_file="~/.lmstudio/credentials/brave-search.env",
            )

        self.assertEqual(payload["web"]["results"], [])
        command = run.call_args.kwargs.get("args") or run.call_args.args[0]
        self.assertNotIn("BRAVE_API_KEY", " ".join(command))
        self.assertNotIn("secret", " ".join(command).lower())
        self.assertIn("BRAVE_API_KEY", run.call_args.kwargs["input"])
        self.assertIn('ENV_FILE="$HOME/${ENV_FILE:2}"', run.call_args.kwargs["input"])


if __name__ == "__main__":
    unittest.main()
