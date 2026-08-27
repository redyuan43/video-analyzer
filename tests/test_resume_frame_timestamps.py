import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "tools" / "pipelines" / "resume_operation_manual_from_frames.py"
SPEC = importlib.util.spec_from_file_location("resume_operation_manual_from_frames", SCRIPT_PATH)
resume_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = resume_module
SPEC.loader.exec_module(resume_module)


class ResumeFrameTimestampTests(unittest.TestCase):
    def test_load_frames_uses_source_analysis_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True)
            for number in range(3):
                (frames_dir / f"frame_{number}.jpg").write_bytes(b"")
            source_analysis = Path(tmp) / "source-analysis.json"
            source_analysis.write_text(
                json.dumps(
                    {
                        "ocr_events": [
                            {"frame_number": 0, "timestamp": 0.0},
                            {"frame_number": 1, "timestamp": 17.5},
                            {"frame_number": 2, "timestamp": 91.25},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = resume_module.load_frames(
                run_dir,
                video_duration_seconds=120.0,
                source_analysis=source_analysis,
                allow_estimated_timestamps=False,
            )

            self.assertEqual([frame.timestamp for frame in result.frames], [0.0, 17.5, 91.25])
            self.assertFalse(result.metadata["estimated_timestamps"])
            self.assertEqual(result.metadata["timestamp_source"]["source"], "ocr_events")

    def test_load_frames_requires_timestamp_source_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True)
            for number in range(3):
                (frames_dir / f"frame_{number}.jpg").write_bytes(b"")

            with self.assertRaises(RuntimeError):
                resume_module.load_frames(
                    run_dir,
                    video_duration_seconds=120.0,
                    source_analysis=None,
                    allow_estimated_timestamps=False,
                )

            result = resume_module.load_frames(
                run_dir,
                video_duration_seconds=120.0,
                source_analysis=None,
                allow_estimated_timestamps=True,
            )

            self.assertEqual([frame.timestamp for frame in result.frames], [0.0, 60.0, 120.0])
            self.assertTrue(result.metadata["estimated_timestamps"])

    def test_load_frames_uses_manifest_before_estimation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True)
            for number in range(3):
                (frames_dir / f"frame_{number}.jpg").write_bytes(b"")
            (run_dir / "frames_manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source": "jetson",
                        "frames": [
                            {"frame_number": 0, "timestamp": 0.0, "score": 0.1},
                            {"frame_number": 1, "timestamp": 11.0, "score": 0.2},
                            {"frame_number": 2, "timestamp": 73.0, "score": 0.3},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = resume_module.load_frames(
                run_dir,
                video_duration_seconds=120.0,
                source_analysis=None,
                allow_estimated_timestamps=False,
            )

            self.assertEqual([frame.timestamp for frame in result.frames], [0.0, 11.0, 73.0])
            self.assertEqual([frame.score for frame in result.frames], [0.1, 0.2, 0.3])
            self.assertEqual(result.metadata["timestamp_source"]["source"], "frames_manifest")

    def test_load_source_ocr_keyframes_preserves_selection_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_analysis = Path(tmp) / "analysis.json"
            source_analysis.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "ocr_keyframes": {
                                "strategy": "scan-text",
                                "frames": [
                                    {"frame_number": 1, "selected_for_ocr": True},
                                    {"frame_number": 2, "selected_for_ocr": False},
                                    {"frame_number": 7, "selected_for_ocr": True},
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            selected, metadata = resume_module.load_source_ocr_keyframes(source_analysis)

            self.assertEqual(selected, {1, 7})
            self.assertEqual(metadata["strategy"], "scan-text")
            metadata["frames"][0]["selected_for_ocr"] = False
            reloaded = json.loads(source_analysis.read_text(encoding="utf-8"))
            self.assertTrue(reloaded["metadata"]["ocr_keyframes"]["frames"][0]["selected_for_ocr"])


if __name__ == "__main__":
    unittest.main()
