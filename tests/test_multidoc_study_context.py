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


if __name__ == "__main__":
    unittest.main()
