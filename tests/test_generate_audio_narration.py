import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.generate_audio_narration import (
    fetch_indextts_timeline,
    markdown_to_spoken_text,
    render_tts,
    validate_spoken_text,
)


class GenerateAudioNarrationTests(unittest.TestCase):
    def test_markdown_to_spoken_text_excludes_tts_reference_appendices(self):
        markdown = """
# 音频讲解稿：测试

## 开场

这是需要朗读的正文。

## 术语与读法

- API：诶辟艾

## TTS 分段建议

- 第一段单独合成

## 时长估算

- 约两分钟
"""

        spoken = markdown_to_spoken_text(markdown)

        self.assertIn("这是需要朗读的正文。", spoken)
        self.assertNotIn("术语与读法", spoken)
        self.assertNotIn("诶辟艾", spoken)
        self.assertNotIn("TTS 分段建议", spoken)
        self.assertNotIn("时长估算", spoken)

    def test_markdown_to_spoken_text_keeps_body_without_appendices(self):
        spoken = markdown_to_spoken_text("# 标题\n\n## 第一部分\n\n正文内容。")

        self.assertEqual(spoken, "标题\n第一部分\n正文内容。\n")

    def test_validate_spoken_text_rejects_reference_appendix(self):
        with self.assertRaisesRegex(ValueError, "non-spoken appendix"):
            validate_spoken_text("正文内容。\n术语与读法\nAPI：诶辟艾")

    def test_fetch_indextts_timeline_normalizes_segments(self):
        response = MagicMock()
        response.json.return_value = {
            "status": "succeeded",
            "segment_results": [
                {
                    "index": 1,
                    "text": "第二句。",
                    "start_seconds": 1.2,
                    "end_seconds": 1.7,
                    "duration_seconds": 0.5,
                },
                {
                    "index": 0,
                    "text": "第一句。",
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "duration_seconds": 1.0,
                },
            ],
        }
        session = MagicMock()
        session.get.return_value = response

        timeline = fetch_indextts_timeline(
            session,
            "http://127.0.0.1:8092",
            "speech-1",
            2,
        )

        self.assertEqual(timeline["request_id"], "speech-1")
        self.assertEqual(timeline["segment_count"], 2)
        self.assertEqual(
            [item["text"] for item in timeline["segments"]],
            ["第一句。", "第二句。"],
        )
        session.get.assert_called_once_with(
            "http://127.0.0.1:8092/v1/audio/speech/jobs/speech-1",
            timeout=30,
        )

    def test_fetch_indextts_timeline_rejects_missing_timestamps(self):
        response = MagicMock()
        response.json.return_value = {
            "status": "succeeded",
            "segment_results": [{"index": 0, "text": "第一句。"}],
        }
        session = MagicMock()
        session.get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "timeline is incomplete"):
            fetch_indextts_timeline(
                session,
                "http://127.0.0.1:8092/v1",
                "speech-1",
                1,
            )

    def test_render_mimo_tts_decodes_chat_completion_wav(self):
        response = MagicMock()
        response.status_code = 200
        response.ok = True
        response.content = b""
        response.headers = {}
        response.json.return_value = {
            "choices": [{"message": {"audio": {"data": "UklGRg=="}}}]
        }
        session = MagicMock()
        session.post.return_value = response
        config = {
            "provider": "xiaomi_mimo_tts",
            "base_url": "https://api.xiaomimimo.com/v1",
            "model": "mimo-v2.5-tts",
            "voice": "冰糖",
            "speed": 0.9,
            "timeout": 180,
            "api_key_env": "XIAOMI_MIMO_API_KEY",
            "extra_params": {},
            "style_prompt": "清晰地播报",
        }

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"XIAOMI_MIMO_API_KEY": "test-key"}, clear=False
        ), patch(
            "tools.generate_audio_narration.requests.Session",
            return_value=session,
        ):
            output = Path(tmp) / "mimo.wav"
            timeline = render_tts("这是测试。", output, config)
            rendered = output.read_bytes()

        self.assertIsNone(timeline)
        self.assertEqual(rendered, b"RIFF")
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(
            session.post.call_args.args[0],
            "https://api.xiaomimimo.com/v1/chat/completions",
        )
        self.assertEqual(
            session.post.call_args.kwargs["headers"]["api-key"], "test-key"
        )
        self.assertEqual(
            payload,
            {
                "model": "mimo-v2.5-tts",
                "messages": [
                    {"role": "user", "content": "清晰地播报"},
                    {"role": "assistant", "content": "这是测试。"},
                ],
                "audio": {"format": "wav", "voice": "冰糖"},
                "stream": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
