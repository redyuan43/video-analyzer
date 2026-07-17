import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from video_analyzer.audio_processor import AudioTranscript
from video_analyzer.frame import Frame
from video_analyzer.frame_dedup_audit import audit_frame_deduplication
from video_analyzer.ocr import OCREvent
from video_analyzer.review_artifacts import write_run_manifest, write_visual_review


class ReviewArtifactTests(unittest.TestCase):
    def test_sliding_window_rgb_dedup_audit_reports_duplicate_treatment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for index, color in enumerate(["white", "white", "black"]):
                path = root / f"frame_{index}.jpg"
                Image.new("RGB", (80, 60), color).save(path)
                frames.append(Frame(index, path, index * 1.0, 0.0))

            audit = audit_frame_deduplication(frames)

        self.assertTrue(audit["enabled"])
        self.assertEqual(audit["summary"]["baseline_frame_count"], 3)
        self.assertEqual(audit["summary"]["treatment_drop_count"], 1)
        self.assertEqual(audit["records"][1]["treatment_action"], "drop")
        self.assertEqual(audit["ab_test"]["name"], "frame_dedup_sliding_window_rgb_diff")

    def test_visual_review_and_run_manifest_are_written_with_ab_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frames = []
            for index in range(10):
                path = run_dir / "frames" / f"frame_{index}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (160, 90), "white")
                draw = ImageDraw.Draw(image)
                draw.text((10, 30), f"frame {index}", fill="black")
                image.save(path)
                frames.append(Frame(index, path, index * 3.0, float(index)))
            video_path = run_dir / "video.mp4"
            video_path.write_bytes(b"fake")
            transcript = AudioTranscript(
                text="hello world",
                segments=[{"start": 0, "end": 2, "text": "hello world"}],
                language="en",
            )
            ocr_events = [OCREvent(0, 0.0, "test", "ok", "API Token", [])]
            metadata = {"task": "operation_manual", "frames_extracted": len(frames)}

            review_path, review_summary = write_visual_review(
                output_dir=run_dir,
                video_path=video_path,
                frames=frames,
                transcript=transcript,
                ocr_events=ocr_events,
                frame_analyses=[{"frame_number": 0, "timestamp": 0.0, "response": "Settings screen"}],
                metadata=metadata,
            )
            results = {
                "metadata": {
                    "task": "operation_manual",
                    "frames_extracted": len(frames),
                    "visual_review": review_summary,
                    "frame_dedup_audit": {
                        "ab_test": {
                            "name": "frame_dedup_sliding_window_rgb_diff",
                            "observed_delta": {"image_review_reduction_ratio": 0.2},
                        }
                    },
                },
                "transcript": {"segments": transcript.segments},
                "ocr_events": [ocr_events[0].to_dict()],
                "frame_analyses": [{"frame_number": 0}],
                "operation_manual": {"quality_gate_passed": True, "evidence_path": "manual_evidence.md"},
            }
            (run_dir / "analysis.json").write_text(json.dumps(results), encoding="utf-8")

            manifest_path, manifest_summary = write_run_manifest(
                output_dir=run_dir,
                results=results,
                visual_review_path=review_path,
                dedup_audit_path=run_dir / "frame_dedup_audit.json",
            )

            self.assertTrue(review_path.is_file())
            self.assertTrue((run_dir / "visual_review" / "contact_sheet_001.jpg").is_file())
            self.assertIn("visual_review_contact_sheets", review_summary["ab_test"]["name"])
            self.assertTrue(manifest_path.is_file())
            manifest_text = manifest_path.read_text(encoding="utf-8")
            self.assertIn("RUN_MANIFEST", manifest_text)
            self.assertIn("agent_first_run_manifest", manifest_text)
            self.assertEqual(manifest_summary["ab_test"]["name"], "agent_first_run_manifest")


if __name__ == "__main__":
    unittest.main()
