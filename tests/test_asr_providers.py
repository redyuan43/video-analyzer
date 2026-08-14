import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from video_analyzer.asr_providers import (
    merge_asr_transcripts,
    transcribe_with_firered_asr2,
    transcribe_with_http_asr,
    transcribe_with_provider_result,
)
from video_analyzer.audio_processor import AudioTranscript


class ASRProviderTests(unittest.TestCase):
    def test_http_asr_preserves_vibevoice_metadata(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "success": True,
            "text": "你好",
            "segments": [{"Start": 0, "End": 1, "Speaker": "Speaker A", "Content": "你好"}],
            "language": "zh",
            "provider": "vibevoice_vllm_p40",
            "mode": "ray_chunk_reconcile",
            "quality_report": {"needs_review": True, "global_speaker_count": 18},
            "audit_chunks": {"speaker_clusters": []},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"fake wav")
            with patch("video_analyzer.asr_providers.requests.post", return_value=response):
                transcript = transcribe_with_http_asr(audio_path, "http://127.0.0.1:18012/api/asr/transcribe")

        self.assertIsNotNone(transcript)
        assert transcript is not None
        self.assertEqual(transcript.metadata["provider"], "vibevoice_vllm_p40")
        self.assertEqual(transcript.metadata["mode"], "ray_chunk_reconcile")
        self.assertEqual(transcript.metadata["quality_report"]["global_speaker_count"], 18)
        self.assertIn("audit_chunks", transcript.metadata)

    def test_http_asr_rejects_failed_coverage_acceptance(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "success": True,
            "text": "不完整",
            "segments": [],
            "acceptance": {
                "status": "failed",
                "failure_reasons": ["empty_chunk_output"],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"fake wav")
            with patch("video_analyzer.asr_providers.requests.post", return_value=response):
                transcript = transcribe_with_http_asr(audio_path, "http://127.0.0.1:18014/api/asr/transcribe")

        self.assertIsNone(transcript)

    def test_merge_asr_transcripts_preserves_source_metadata(self):
        fast = AudioTranscript(text="fast", segments=[{"text": "fast"}], language="zh", metadata={"provider": "fast"})
        deep = AudioTranscript(
            text="deep",
            segments=[{"text": "deep"}],
            language="zh",
            metadata={"quality_report": {"global_speaker_count": 2}},
        )

        merged = merge_asr_transcripts(fast, deep)

        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(merged.metadata["source"], "merged_remote_http_vibevoice")
        self.assertEqual(merged.metadata["fast_transcript_metadata"]["provider"], "fast")
        self.assertEqual(merged.metadata["deep_transcript_metadata"]["quality_report"]["global_speaker_count"], 2)

    def test_firered_asr_forwards_profile_chunk_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"fake wav")
            with (
                patch(
                    "video_analyzer.asr_providers._wav_duration",
                    return_value=600,
                ),
                patch(
                    "video_analyzer.asr_providers.transcribe_with_http_asr",
                ) as transcribe,
            ):
                transcribe_with_firered_asr2(
                    audio_path,
                    "http://127.0.0.1:18014/api/asr/transcribe",
                    options={
                        "segmentation_mode": "vad",
                        "vad_max_segment_sec": 50,
                        "single_pass_max_duration_sec": 50,
                        "chunk_duration_sec": 30,
                        "chunk_overlap_sec": 3,
                    },
                )

        self.assertEqual(
            transcribe.call_args.kwargs["extra_data"],
            {
                "segmentation_mode": "vad",
                "vad_max_segment_sec": 50,
                "single_pass_max_duration_sec": 50,
                "chunk_duration_sec": 30,
                "chunk_overlap_sec": 3,
            },
        )

    def test_tencent_provider_forwards_endpoint_and_options(self):
        transcript = AudioTranscript(text="腾讯转写", segments=[], language="zh")
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"fake wav")
            with patch(
                "video_analyzer.asr_providers.transcribe_with_tencent_hy_asr",
                return_value=transcript,
            ) as transcribe:
                result = transcribe_with_provider_result(
                    provider="tencent_hy_asr",
                    audio_path=audio_path,
                    language="zh",
                    whisper_model="medium",
                    device="cpu",
                    vibevoice_config={
                        "tencent_hy_asr_endpoint": "wss://asr.cloud.tencent.com/asr/v2",
                        "tencent_hy_asr_options": {"parallel_chunks": 4},
                    },
                )

        self.assertIs(result.transcript, transcript)
        self.assertEqual(result.providers_run, ["tencent_hy_asr"])
        self.assertEqual(
            transcribe.call_args.args[1],
            "wss://asr.cloud.tencent.com/asr/v2",
        )
        self.assertEqual(transcribe.call_args.args[2]["parallel_chunks"], 4)


if __name__ == "__main__":
    unittest.main()
