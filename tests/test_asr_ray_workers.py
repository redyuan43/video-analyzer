import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.asr_servers import asr_ray_workers
from tools.asr_servers.asr_ray_workers import AsrChunk, ChunkResult


class AsrRayWorkerTests(unittest.TestCase):
    def test_materialize_segment_chunks_enforces_hard_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(asr_ray_workers, "_extract_audio_segment") as extract:
                chunks = asr_ray_workers.materialize_segment_chunks(
                    directory / "source.wav",
                    directory,
                    [(1.0, 120.0)],
                    hard_limit_seconds=50.0,
                )

        self.assertEqual(
            [(chunk.start, chunk.length) for chunk in chunks],
            [(1.0, 50.0), (51.0, 50.0), (101.0, 19.0)],
        )
        self.assertEqual(extract.call_count, 3)

    def test_single_chunk_dispatch_does_not_require_ray(self):
        chunk = AsrChunk(0, Path("/tmp/chunk.wav"), 0.0, 5.0)
        with (
            patch.object(asr_ray_workers, "ray", None),
            patch.object(
                asr_ray_workers,
                "post_audio",
                return_value={"success": True, "text": "短音频"},
            ) as post_audio,
        ):
            results = asr_ray_workers.dispatch_asr_chunks(
                ["http://127.0.0.1:18400/api/asr/transcribe"],
                [chunk],
                {},
                request_timeout=30,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].endpoint, "http://127.0.0.1:18400/api/asr/transcribe")
        post_audio.assert_called_once()

    def test_single_chunk_retries_on_another_endpoint(self):
        chunk = AsrChunk(0, Path("/tmp/chunk.wav"), 0.0, 5.0)
        with (
            patch.object(asr_ray_workers, "ray", None),
            patch.object(
                asr_ray_workers,
                "post_audio",
                side_effect=[
                    TimeoutError("worker-1 timeout"),
                    {"success": True, "text": "retry succeeded"},
                ],
            ) as post_audio,
        ):
            results = asr_ray_workers.dispatch_asr_chunks(
                ["worker-1", "worker-2"],
                [chunk],
                {},
                request_timeout=30,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].endpoint, "worker-2")
        self.assertEqual(results[0].attempt, 2)
        self.assertEqual(results[0].first_error, "worker-1 timeout")
        self.assertEqual(post_audio.call_count, 2)

    def test_multi_chunk_dispatch_shuts_down_owned_ray_runtime(self):
        class FakeRef:
            def __init__(self, value):
                self.value = value

        class FakeMethod:
            def __init__(self, callback):
                self.callback = callback

            def remote(self, *args):
                return FakeRef(self.callback(*args))

        class FakeActor:
            def __init__(self, endpoint):
                self.endpoint = endpoint
                self.ready = FakeMethod(lambda: endpoint)
                self.transcribe = FakeMethod(self._transcribe)

            def _transcribe(self, item, _form):
                index, _path, start, length = item
                return (
                    index,
                    start,
                    length,
                    {"success": True, "text": f"chunk-{index}"},
                    self.endpoint,
                    0.5,
                )

        class FakeActorFactory:
            @staticmethod
            def remote(endpoint, _request_timeout):
                return FakeActor(endpoint)

        class FakeRay:
            def __init__(self):
                self.initialized = False
                self.init_calls = 0
                self.shutdown_calls = 0
                self.kill_calls = 0

            def is_initialized(self):
                return self.initialized

            def init(self, **_kwargs):
                self.initialized = True
                self.init_calls += 1

            def get(self, ref, timeout=None):
                del timeout
                if isinstance(ref, list):
                    return [item.value for item in ref]
                return ref.value

            def wait(self, refs, **_kwargs):
                return [refs[0]], refs[1:]

            def kill(self, _actor, no_restart=True):
                self.kill_calls += 1
                self.assert_no_restart = no_restart

            def shutdown(self):
                self.initialized = False
                self.shutdown_calls += 1

        runtime = FakeRay()
        chunks = [
            AsrChunk(0, Path("/tmp/a.wav"), 0.0, 5.0),
            AsrChunk(1, Path("/tmp/b.wav"), 5.0, 5.0),
        ]
        with (
            patch.object(asr_ray_workers, "ray", runtime),
            patch.object(
                asr_ray_workers,
                "HttpAsrWorker",
                FakeActorFactory,
                create=True,
            ),
        ):
            results = asr_ray_workers.dispatch_asr_chunks(
                ["worker-1", "worker-2"],
                chunks,
                {},
                request_timeout=30,
            )

        self.assertEqual([item.endpoint for item in results], ["worker-1", "worker-2"])
        self.assertEqual(runtime.init_calls, 1)
        self.assertEqual(runtime.kill_calls, 2)
        self.assertEqual(runtime.shutdown_calls, 1)
        self.assertFalse(runtime.initialized)

    def test_merge_preserves_retry_and_bounds_timestamps(self):
        results = [
            ChunkResult(
                chunk=AsrChunk(0, Path("/tmp/a.wav"), 0.0, 10.0),
                payload={
                    "success": True,
                    "text": "第一段共同内容",
                    "segments": [{"start": -1, "end": 11, "text": "第一段共同内容"}],
                },
                endpoint="worker-1",
                elapsed_seconds=1.25,
                attempt=1,
            ),
            ChunkResult(
                chunk=AsrChunk(1, Path("/tmp/b.wav"), 8.0, 10.0),
                payload={
                    "success": True,
                    "text": "共同内容第二段",
                    "words": [{"start": 1, "end": 30, "word": "第二段"}],
                },
                endpoint="worker-2",
                elapsed_seconds=1.5,
                attempt=2,
                first_error="worker-1 timeout",
            ),
        ]

        payload = asr_ray_workers.merge_asr_results(
            "test",
            results,
            segmentation_mode="fixed",
            audio_duration_seconds=15.0,
            worker_count=5,
        )

        self.assertTrue(payload["success"])
        self.assertEqual(payload["dispatch_mode"], "ray")
        self.assertEqual(payload["worker_count"], 5)
        self.assertEqual(payload["chunk_results"][1]["attempt"], 2)
        self.assertEqual(
            payload["chunk_results"][1]["retry_error"],
            "worker-1 timeout",
        )
        self.assertEqual(payload["segments"][0]["start"], 0.0)
        self.assertEqual(payload["segments"][0]["end"], 11.0)
        self.assertEqual(payload["words"][0]["end"], 15.0)

    def test_fixed_chunks_with_synthetic_timestamps_need_review(self):
        results = [
            ChunkResult(
                chunk=AsrChunk(0, Path("/tmp/a.wav"), 0.0, 10.0),
                payload={"success": True, "text": "第一段"},
                endpoint="worker-1",
                elapsed_seconds=1.0,
                attempt=1,
            ),
            ChunkResult(
                chunk=AsrChunk(1, Path("/tmp/b.wav"), 8.0, 10.0),
                payload={"success": True, "text": "第二段"},
                endpoint="worker-2",
                elapsed_seconds=1.0,
                attempt=1,
            ),
        ]

        payload = asr_ray_workers.merge_asr_results(
            "test",
            results,
            segmentation_mode="fixed",
            audio_duration_seconds=15.0,
            worker_count=2,
        )

        acceptance = payload["acceptance"]
        self.assertTrue(payload["success"])
        self.assertEqual(acceptance["status"], "needs_review")
        self.assertEqual(acceptance["timestamp_source"], "chunk_bounds")
        self.assertEqual(acceptance["chunk_coverage"]["coverage_ratio"], 1.0)

    def test_fixed_chunks_with_empty_output_fail_acceptance(self):
        results = [
            ChunkResult(
                chunk=AsrChunk(0, Path("/tmp/a.wav"), 0.0, 10.0),
                payload={"success": True, "text": ""},
                endpoint="worker-1",
                elapsed_seconds=1.0,
                attempt=1,
            )
        ]

        payload = asr_ray_workers.merge_asr_results(
            "test",
            results,
            segmentation_mode="fixed",
            audio_duration_seconds=10.0,
            worker_count=1,
        )

        self.assertFalse(payload["success"])
        self.assertEqual(payload["acceptance"]["status"], "failed")
        self.assertIn("empty_chunk_output", payload["acceptance"]["failure_reasons"])

    def test_word_timestamps_are_preferred_over_chunk_bounds(self):
        result = ChunkResult(
            chunk=AsrChunk(0, Path("/tmp/a.wav"), 0.0, 10.0),
            payload={
                "success": True,
                "text": "完整",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 10.0,
                        "text": "完整",
                        "timestamp_source": "chunk_bounds",
                    }
                ],
                "words": [{"start": 0.1, "end": 9.9, "text": "完整"}],
            },
            endpoint="worker-1",
            elapsed_seconds=1.0,
            attempt=1,
        )

        payload = asr_ray_workers.merge_asr_results(
            "test",
            [result],
            segmentation_mode="fixed",
            audio_duration_seconds=10.0,
            worker_count=1,
        )

        self.assertEqual(payload["acceptance"]["timestamp_source"], "words")
        self.assertEqual(payload["acceptance"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
