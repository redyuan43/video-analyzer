import unittest
from pathlib import Path

from video_analyzer.jetson_frames import REMOTE_WORKER_SCRIPT


class JetsonRemoteWorkerScriptTests(unittest.TestCase):
    def test_static_video_candidates_are_filled_with_uniform_coverage(self):
        namespace = {}
        exec(REMOTE_WORKER_SCRIPT, namespace)
        paths = [Path(f"preview_{index:06d}.jpg") for index in range(10)]
        sparse_candidates = [{"path": str(paths[0]), "timestamp": 0.0, "score": 255.0}]

        filled = namespace["add_uniform_coverage_candidates"](
            sparse_candidates,
            paths,
            segment_start=0.0,
            segment_duration=90.0,
            sample_fps=0.1,
            max_frames=5,
        )

        self.assertEqual(len(filled), 5)
        self.assertEqual(filled[0]["timestamp"], 0.0)
        self.assertEqual(filled[-1]["timestamp"], 90.0)


if __name__ == "__main__":
    unittest.main()
