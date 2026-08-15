import os
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from video_analyzer.tencent_hy_asr import (
    build_signed_url,
    missing_tencent_credentials,
    read_pcm_chunks,
    scale_pcm_s16le,
    transcribe_pcm_chunk,
    transcribe_pcm_chunk_once,
    transcribe_with_tencent_hy_asr,
)


class TencentHyASRTests(unittest.TestCase):
    def write_wav(self, path: Path, seconds: float) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * int(16000 * seconds))

    def test_signed_url_uses_wss_and_does_not_expose_secret_key(self):
        credentials = {
            "app_id": "123456",
            "secret_id": "secret-id",
            "secret_key": "never-in-url",
        }
        with patch("video_analyzer.tencent_hy_asr.time.time", return_value=1000):
            url = build_signed_url(
                "wss://asr.cloud.tencent.com/asr/v2",
                credentials,
                {"engine_model_type": "Hy-ASR-3.0-preview"},
            )

        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.scheme, "wss")
        self.assertEqual(parsed.path, "/asr/v2/123456")
        self.assertEqual(query["engine_model_type"], ["Hy-ASR-3.0-preview"])
        self.assertEqual(query["secretid"], ["secret-id"])
        self.assertIn("signature", query)
        self.assertNotIn(credentials["secret_key"], url)

    def test_long_wav_is_split_below_preview_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            self.write_wav(audio_path, 120)
            chunks = read_pcm_chunks(audio_path, 55)

        self.assertEqual(len(chunks), 3)
        self.assertEqual([item[1] for item in chunks], [0, 55, 110])
        self.assertLessEqual(len(chunks[0][2]), 55 * 16000 * 2)

    def test_parallel_chunk_results_restore_global_timestamps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            self.write_wav(audio_path, 70)
            env = {
                "TENCENTCLOUD_APP_ID": "1",
                "TENCENTCLOUD_SECRET_ID": "id",
                "TENCENTCLOUD_SECRET_KEY": "key",
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch(
                    "video_analyzer.tencent_hy_asr.transcribe_pcm_chunk",
                    return_value=(
                        [{"start": 1.0, "end": 2.0, "text": "测试"}],
                        [{"code": 0, "final": 1}],
                    ),
                ),
            ):
                transcript = transcribe_with_tencent_hy_asr(
                    audio_path,
                    options={"chunk_duration_sec": 55, "parallel_chunks": 2},
                )

        self.assertEqual(transcript.text, "测试\n测试")
        self.assertEqual(
            [segment["start"] for segment in transcript.segments],
            [1.0, 56.0],
        )
        self.assertEqual(transcript.metadata["chunk_count"], 2)

    def test_preview_parallelism_is_capped_to_account_safe_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            self.write_wav(audio_path, 280)
            env = {
                "TENCENTCLOUD_APP_ID": "1",
                "TENCENTCLOUD_SECRET_ID": "id",
                "TENCENTCLOUD_SECRET_KEY": "key",
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch(
                    "video_analyzer.tencent_hy_asr.transcribe_pcm_chunk",
                    return_value=([], [{"code": 0, "final": 1}]),
                ),
            ):
                transcript = transcribe_with_tencent_hy_asr(
                    audio_path,
                    options={"chunk_duration_sec": 55, "parallel_chunks": 10},
                )

        self.assertEqual(transcript.metadata["parallel_chunks"], 6)

    def test_audio_packets_wait_for_successful_handshake(self):
        actions = []
        end_sent = __import__("threading").Event()

        class FakeWebSocket:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def recv(self, timeout):
                if "handshake_received" not in actions:
                    actions.append("handshake_received")
                    return '{"code":0,"message":"success"}'
                if not end_sent.wait(timeout):
                    raise TimeoutError("end message was not sent")
                return '{"code":0,"message":"success","final":1}'

            def send(self, payload):
                if isinstance(payload, bytes):
                    actions.append("audio_sent")
                else:
                    actions.append("end_sent")
                    end_sent.set()

        credentials = {
            "app_id": "123456",
            "secret_id": "secret-id",
            "secret_key": "secret-key",
        }
        with patch(
            "video_analyzer.tencent_hy_asr.connect",
            return_value=FakeWebSocket(),
        ):
            transcribe_pcm_chunk_once(
                b"\0" * 6400,
                "wss://asr.cloud.tencent.com/asr/v2",
                credentials,
                {"send_realtime_factor": 100},
            )

        self.assertLess(
            actions.index("handshake_received"),
            actions.index("audio_sent"),
        )

    def test_failed_chunk_cancels_queued_work_without_draining_all_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            self.write_wav(audio_path, 300)
            env = {
                "TENCENTCLOUD_APP_ID": "1",
                "TENCENTCLOUD_SECRET_ID": "id",
                "TENCENTCLOUD_SECRET_KEY": "key",
            }
            calls = []

            def transcribe_chunk(*_args):
                calls.append(len(calls))
                if len(calls) == 1:
                    raise RuntimeError("decode failed")
                time.sleep(0.2)
                return [], [{"code": 0, "final": 1}]

            with (
                patch.dict(os.environ, env, clear=False),
                patch(
                    "video_analyzer.tencent_hy_asr.transcribe_pcm_chunk",
                    side_effect=transcribe_chunk,
                ),
            ):
                started = time.monotonic()
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"chunk 0 at 0\.000s failed",
                ):
                    transcribe_with_tencent_hy_asr(
                        audio_path,
                        options={
                            "chunk_duration_sec": 30,
                            "parallel_chunks": 2,
                        },
                    )
                elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.15)
        self.assertLess(len(calls), 10)

    def test_decode_error_retries_once_with_reduced_pcm_gain(self):
        pcm_data = b"\xe8\x03\x18\xfc"
        expected_scaled = b"\xbc\x02\x44\xfd"
        with patch(
            "video_analyzer.tencent_hy_asr.transcribe_pcm_chunk_once",
            side_effect=[
                RuntimeError("Tencent Hy-ASR error 4007: decode failed"),
                ([{"text": "恢复成功"}], [{"code": 0, "final": 1}]),
            ],
        ) as transcribe_once:
            sentences, _responses = transcribe_pcm_chunk(
                pcm_data,
                "wss://asr.cloud.tencent.com/asr/v2",
                {"app_id": "1", "secret_id": "id", "secret_key": "key"},
                {
                    "max_attempts": 1,
                    "decode_error_gain_fallback": 0.7,
                },
            )

        self.assertEqual(sentences[0]["text"], "恢复成功")
        self.assertEqual(transcribe_once.call_count, 2)
        self.assertEqual(transcribe_once.call_args_list[1].args[0], expected_scaled)

    def test_scale_pcm_s16le_preserves_sample_count(self):
        pcm_data = b"\xff\x7f\x00\x80\xe8\x03\x18\xfc"
        scaled = scale_pcm_s16le(pcm_data, 0.7)
        self.assertEqual(len(scaled), len(pcm_data))
        self.assertEqual(scaled[-4:], b"\xbc\x02\x44\xfd")

    def test_missing_credentials_reports_all_required_environment_names(self):
        env_names = (
            "TENCENTCLOUD_APP_ID",
            "TENCENTCLOUD_SECRET_ID",
            "TENCENTCLOUD_SECRET_KEY",
        )
        with patch.dict(os.environ, {}, clear=True):
            missing = missing_tencent_credentials(
                {"env_file": "/tmp/video-analyzer-missing-tencentcloud.env"}
            )
        self.assertEqual(missing, list(env_names))


if __name__ == "__main__":
    unittest.main()
