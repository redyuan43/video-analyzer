import io
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import firered_asr2_p40_proxy as proxy
from tools.asr_ray_workers import AsrChunk, ChunkResult


class FireRedAsr2ProxyTests(unittest.TestCase):
    def setUp(self):
        self.client = proxy.app.test_client()

    def test_vad_no_speech_is_an_explicit_failure(self):
        with (
            patch.object(proxy, "audio_duration", return_value=12.0),
            patch.object(proxy, "normalize_audio", side_effect=lambda _source, output: output),
            patch.object(proxy, "detect_speech_segments", return_value=[]),
        ):
            response = self.client.post(
                "/api/asr/transcribe",
                data={
                    "audio": (io.BytesIO(b"fake audio"), "sample.wav"),
                    "segmentation_mode": "vad",
                },
                content_type="multipart/form-data",
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 422)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"], "no_speech_detected")
        self.assertEqual(payload["segmentation_mode"], "vad")
        self.assertEqual(payload["chunk_count"], 0)

    def test_vad_runtime_failure_falls_back_to_fixed_chunks(self):
        chunk = AsrChunk(0, Path("/tmp/chunk.wav"), 0.0, 12.0)
        result = ChunkResult(
            chunk=chunk,
            payload={"success": True, "text": "回退成功", "language": "zh"},
            endpoint="http://127.0.0.1:18400/api/asr/transcribe",
            elapsed_seconds=1.0,
            attempt=1,
        )
        with (
            patch.object(proxy, "audio_duration", return_value=12.0),
            patch.object(proxy, "normalize_audio", side_effect=lambda _source, output: output),
            patch.object(
                proxy,
                "detect_speech_segments",
                side_effect=RuntimeError("VAD unavailable"),
            ),
            patch.object(proxy, "materialize_fixed_chunks", return_value=[chunk]),
            patch.object(proxy, "dispatch_asr_chunks", return_value=[result]),
        ):
            response = self.client.post(
                "/api/asr/transcribe",
                data={
                    "audio": (io.BytesIO(b"fake audio"), "sample.wav"),
                    "segmentation_mode": "vad",
                },
                content_type="multipart/form-data",
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["text"], "回退成功")
        self.assertEqual(payload["segmentation_mode"], "fixed")
        self.assertTrue(payload["fallback_used"])
        self.assertEqual(payload["fallback_reason"], "VAD unavailable")


if __name__ == "__main__":
    unittest.main()
