import os
import tempfile
from pathlib import Path
from unittest import TestCase
from PIL import Image

from video_analyzer.jetson_frames import (
    JetsonFrameWorker,
    REMOTE_WORKER_SCRIPT,
    _candidate_observation_metrics_from_items,
    _rsync_host,
    _rsync_ssh_args,
    _select_jetson_candidate_metadata,
    _ssh_host_args,
)


class JetsonFrameSelectionTests(TestCase):
    def test_agx_default_uses_control_host_instead_of_stale_lan_ip(self):
        old_value = os.environ.pop("JETSON_AGX_LAN_HOST", None)
        try:
            self.assertEqual(_ssh_host_args("agx")[-1], "agx")
            self.assertEqual(_rsync_host("agx"), "agx")
            self.assertEqual(_rsync_ssh_args("agx"), [])
        finally:
            if old_value is not None:
                os.environ["JETSON_AGX_LAN_HOST"] = old_value

    def test_agx_lan_host_override_is_explicit(self):
        old_value = os.environ.get("JETSON_AGX_LAN_HOST")
        try:
            os.environ["JETSON_AGX_LAN_HOST"] = "agx-lan"

            ssh_args = _ssh_host_args("agx")

            self.assertEqual(ssh_args[-1], "agx@agx-lan")
            self.assertIn("HostKeyAlias=agx-lan", ssh_args)
            self.assertEqual(_rsync_host("agx"), "agx@agx-lan")
            self.assertTrue(_rsync_ssh_args("agx"))
        finally:
            if old_value is None:
                os.environ.pop("JETSON_AGX_LAN_HOST", None)
            else:
                os.environ["JETSON_AGX_LAN_HOST"] = old_value

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

    def test_static_shortcut_limits_candidates_before_full_preview_scan(self):
        namespace = {}
        exec(REMOTE_WORKER_SCRIPT, namespace)
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for index in range(12):
                image_path = Path(temp_dir) / f"probe_{index:06d}.jpg"
                Image.new("L", (320, 180), color=128).save(image_path)
                paths.append((image_path, index * 60.0))

            candidates = namespace["build_static_coverage_candidates"](
                paths,
                segment_start=0.0,
                segment_duration=720.0,
                max_frames=4,
            )

        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(item["static_shortcut"] for item in candidates))
        self.assertEqual(candidates[0]["timestamp"], 0.0)
        self.assertEqual(candidates[-1]["timestamp"], 660.0)

    def test_materialize_skips_missing_highres_outputs(self):
        namespace = {}
        exec(REMOTE_WORKER_SCRIPT, namespace)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "out"

            def fake_extract(_video, _timestamp, output_path):
                if _timestamp < 10.0:
                    output_path.write_bytes(b"jpg")
                return "ffmpeg-nvdec"

            namespace["extract_highres_candidate"] = fake_extract
            materialized, backends, missing = namespace["materialize_highres_candidates"](
                Path(temp_dir) / "video.mp4",
                output_dir,
                [
                    {"path": "preview_000000.jpg", "timestamp": 0.0, "score": 1.0},
                    {"path": "preview_000001.jpg", "timestamp": 10.0, "score": 2.0},
                ],
            )

        self.assertEqual(len(materialized), 1)
        self.assertEqual(backends, ["ffmpeg-nvdec"])
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["index"], 1)

    def make_worker(self, name: str) -> JetsonFrameWorker:
        return JetsonFrameWorker(
            host="agx",
            start_seconds=0.0,
            duration_seconds=120.0,
            output_dir=Path(name),
        )

    def test_metadata_selection_preserves_budget_without_highres_files(self):
        worker = self.make_worker("jetson_00_agx")
        manifest = {
            "candidates": [
                {"path": "preview/preview_000000.jpg", "timestamp": 0.0, "score": 10.0},
                {"path": "preview/preview_000001.jpg", "timestamp": 10.0, "score": 90.0},
                {"path": "preview/preview_000002.jpg", "timestamp": 20.0, "score": 20.0},
                {"path": "preview/preview_000003.jpg", "timestamp": 40.0, "score": 30.0},
                {"path": "preview/preview_000004.jpg", "timestamp": 60.0, "score": 40.0},
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            selected_by_worker, observation = _select_jetson_candidate_metadata(
                [(worker, manifest, Path(temp_dir))],
                candidate_budget=3,
            )

        selected = selected_by_worker["jetson_00_agx"]["candidates"]
        self.assertEqual(len(selected), 3)
        self.assertEqual(observation["raw_candidate_points"], 5)
        self.assertEqual(observation["final_candidate_points"], 3)
        self.assertEqual(observation["candidate_budget"], 3)

    def test_shadow_metrics_capture_coverage_redundancy_and_stable_text(self):
        metrics = _candidate_observation_metrics_from_items(
            [
                {"timestamp": 0.0, "textness_score": 0.35, "visual_score": 4.0},
                {"timestamp": 1.0, "textness_score": 0.2, "visual_score": 30.0},
                {"timestamp": 70.0, "textness_score": 0.5, "visual_score": 5.0},
            ]
        )

        self.assertEqual(metrics["candidate_count"], 3)
        self.assertEqual(metrics["near_duplicate_gap_count"], 1)
        self.assertEqual(metrics["stable_text_candidate_count"], 2)
        self.assertGreater(metrics["coverage_60s_bucket_ratio"], 0)

    def test_remote_worker_paper_features_are_reported(self):
        namespace = {}
        exec(REMOTE_WORKER_SCRIPT, namespace)
        original_has_module = namespace["has_module"]
        namespace["has_module"] = (
            lambda name: False
            if name in {"open_clip", "transnetv2_pytorch", "torch"}
            else original_has_module(name)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "preview_000001.jpg"
            Image.new("L", (320, 180), color=255).save(image_path)
            candidates = [{"path": str(image_path), "timestamp": 0.0, "score": 1.0}]

            gffv = namespace["gffv_feature"](image_path)
            projection = namespace["sspa_projection"](image_path)
            clip_status = namespace["attach_clip_embeddings"](candidates)
            transnet_status = namespace["attach_transnet_shot_boundaries"](Path(temp_dir) / "video.mp4", candidates)

        self.assertEqual(len(gffv), 8)
        self.assertIsInstance(projection, float)
        self.assertIn(clip_status["status"], {"ok", "fail"})
        self.assertIn(transnet_status["status"], {"ok", "fail", "unavailable"})

    def test_remote_worker_fails_cpu_only_heavy_paper_backends(self):
        namespace = {}
        exec(REMOTE_WORKER_SCRIPT, namespace)

        class TorchStub:
            class cuda:
                @staticmethod
                def is_available():
                    return False

        import sys
        import types

        open_clip_stub = types.SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not load clip on CPU"))
        )
        transnet_stub = types.SimpleNamespace(TransNetV2=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not load transnet on CPU")))
        old_torch = sys.modules.get("torch")
        old_open_clip = sys.modules.get("open_clip")
        old_transnet = sys.modules.get("transnetv2_pytorch")
        old_enable_transnet = os.environ.get("VIDEO_ANALYZER_ENABLE_TRANSNET")
        try:
            os.environ["VIDEO_ANALYZER_ENABLE_TRANSNET"] = "1"
            sys.modules["torch"] = TorchStub
            sys.modules["open_clip"] = open_clip_stub
            sys.modules["transnetv2_pytorch"] = transnet_stub
            namespace["has_module"] = lambda name: name in {"torch", "open_clip", "transnetv2_pytorch"}
            clip_status = namespace["attach_clip_embeddings"]([{"path": "missing.jpg"}])
            transnet_status = namespace["attach_transnet_shot_boundaries"](Path("missing.mp4"), [{"timestamp": 0.0}])
        finally:
            if old_enable_transnet is None:
                os.environ.pop("VIDEO_ANALYZER_ENABLE_TRANSNET", None)
            else:
                os.environ["VIDEO_ANALYZER_ENABLE_TRANSNET"] = old_enable_transnet
            for name, old in [("torch", old_torch), ("open_clip", old_open_clip), ("transnetv2_pytorch", old_transnet)]:
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

        self.assertEqual(clip_status["status"], "fail")
        self.assertEqual(clip_status["reason"], "cuda_unavailable")
        self.assertEqual(transnet_status["status"], "fail")
        self.assertEqual(transnet_status["reason"], "cuda_unavailable")
