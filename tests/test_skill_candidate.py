import json
import tempfile
import unittest
from pathlib import Path

from video_analyzer.skill_candidate import build_tool_skill_candidate, enable_tool_skill_candidate


class SkillCandidateTests(unittest.TestCase):
    def test_builds_reviewable_skill_candidate_from_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            write_artifacts(run_dir)

            summary = build_tool_skill_candidate(run_dir)

            skill_path = run_dir / "skills" / "tool_skill_candidate" / "SKILL.md"
            review_path = run_dir / "skills" / "tool_skill_candidate" / "skill_review.json"
            reference_path = run_dir / "skills" / "tool_skill_candidate" / "references" / "evidence_summary.md"
            self.assertTrue(summary["available"])
            self.assertTrue(skill_path.is_file())
            self.assertTrue(review_path.is_file())
            self.assertTrue(reference_path.is_file())
            skill_text = skill_path.read_text(encoding="utf-8")
            self.assertIn("name:", skill_text)
            self.assertIn("description:", skill_text)
            self.assertIn("Demo Tool Setup", skill_text)
            review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(review["status"], "needs_review")
            self.assertTrue(review["review_required"])

    def test_enable_copies_candidate_to_project_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            run_dir = Path(tmp) / "run"
            root.mkdir()
            run_dir.mkdir()
            write_artifacts(run_dir)
            build_tool_skill_candidate(run_dir)

            summary = enable_tool_skill_candidate(run_dir, root)

            target = root / ".codex" / "skills" / summary["skill_name"] / "SKILL.md"
            self.assertTrue(summary["enabled"])
            self.assertTrue(target.is_file())
            self.assertIn("Demo Tool Setup", target.read_text(encoding="utf-8"))

    def test_chinese_title_uses_run_directory_slug_for_skill_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "BV123456" / "operation-manual-001"
            run_dir.mkdir(parents=True)
            write_artifacts(run_dir, title="中文工具配置")

            summary = build_tool_skill_candidate(run_dir)

        self.assertNotEqual(summary["skill_name"], "tool_skill_candidate")
        self.assertIn("bv123456", summary["skill_name"])

    def test_rewrites_auto_segment_titles_in_candidate_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            write_artifacts(run_dir, title="Page Context Evidence: Demo Tool")
            (run_dir / "study_guide.json").write_text(
                json.dumps(
                    {
                        "title": "Page Context Evidence: Demo Tool",
                        "overview": {"summary": "全片分为 2 个学习章节，核心路径包括：自动分段 01、自动分段 02。"},
                        "chapters": [
                            {
                                "index": 1,
                                "title": "自动分段 01",
                                "summary": "Today we're going to configure the Demo Tool API Token and verify the connection.",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = build_tool_skill_candidate(run_dir)
            text = (run_dir / summary["skill_path"]).read_text(encoding="utf-8")

        self.assertIn("# Demo Tool", text)
        self.assertIn("Configure The Demo Tool API Token", text)
        self.assertNotIn("自动分段 01", text)

    def test_skill_description_escapes_yaml_colon_titles(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            write_artifacts(run_dir, title="Demo Tool: Setup")

            summary = build_tool_skill_candidate(run_dir)
            text = (run_dir / summary["skill_path"]).read_text(encoding="utf-8")

        self.assertIn('description: "Use when the user asks', text)
        self.assertIn("Demo Tool: Setup", text)


def write_artifacts(run_dir: Path, title: str = "Demo Tool Setup") -> None:
    (run_dir / "operation_manual.md").write_text(
        f"# {title}\n\n1. 打开设置页面。\n2. 填写 API Token。\n",
        encoding="utf-8",
    )
    (run_dir / "manual_evidence.md").write_text(
        "# Evidence\n\nframe_001 显示 API Token 输入框。\n",
        encoding="utf-8",
    )
    (run_dir / "transcript.md").write_text(
        "- [00:00:01 - 00:00:05] 打开设置并填写 API Token。\n",
        encoding="utf-8",
    )
    (run_dir / "study_guide.json").write_text(
        json.dumps(
            {
                "title": title,
                "overview": {"summary": "演示如何配置 Demo Tool。"},
                "chapters": [
                    {
                        "index": 1,
                        "title": "打开设置",
                        "summary": "进入设置页面并定位 API Token 输入框。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
