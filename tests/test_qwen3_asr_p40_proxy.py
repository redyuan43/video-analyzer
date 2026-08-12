import io
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qwen3_asr_p40_proxy as proxy
from tools.asr_ray_workers import AsrChunk, ChunkResult


class Qwen3AsrP40ProxyTests(unittest.TestCase):
    def setUp(self):
        self.client = proxy.app.test_client()

    def test_fixed_chunks_use_shared_ray_dispatch(self):
        chunks = [
            AsrChunk(0, Path("/tmp/a.wav"), 0.0, 100.0),
            AsrChunk(1, Path("/tmp/b.wav"), 90.0, 100.0),
        ]
        results = [
            ChunkResult(
                chunk=chunks[0],
                payload={"success": True, "text": "第一段共同内容"},
                endpoint="http://127.0.0.1:18300/api/asr/transcribe",
                elapsed_seconds=1.0,
                attempt=1,
            ),
            ChunkResult(
                chunk=chunks[1],
                payload={"success": True, "text": "共同内容第二段"},
                endpoint="http://127.0.0.1:18301/api/asr/transcribe",
                elapsed_seconds=1.0,
                attempt=1,
            ),
        ]
        with (
            patch.object(proxy, "ensure_workers"),
            patch.object(proxy, "audio_duration", return_value=200.0),
            patch.object(proxy, "parsed_workers", return_value=[(0, 18300), (1, 18301)]),
            patch.object(proxy, "materialize_fixed_chunks", return_value=chunks) as materialize,
            patch.object(proxy, "dispatch_asr_chunks", return_value=results) as dispatch,
        ):
            response = self.client.post(
                "/api/asr/transcribe",
                data={
                    "audio": (io.BytesIO(b"fake audio"), "sample.wav"),
                    "single_pass_max_duration_sec": "150",
                    "chunk_duration_sec": "100",
                    "chunk_overlap_sec": "10",
                },
                content_type="multipart/form-data",
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["dispatch_mode"], "ray")
        self.assertEqual(payload["chunk_count"], 2)
        self.assertEqual(payload["worker_count"], 2)
        self.assertEqual(materialize.call_args.kwargs["chunk_seconds"], 100.0)
        self.assertEqual(materialize.call_args.kwargs["overlap_seconds"], 10.0)
        self.assertEqual(
            dispatch.call_args.args[0],
            [
                "http://127.0.0.1:18300/api/asr/transcribe",
                "http://127.0.0.1:18301/api/asr/transcribe",
            ],
        )

    def test_gpu_three_is_rejected(self):
        with patch.object(proxy, "WORKER_SPECS", ["3:18300"]):
            with self.assertRaisesRegex(ValueError, "GPU 3 is reserved"):
                proxy.parsed_workers()


if __name__ == "__main__":
    unittest.main()
