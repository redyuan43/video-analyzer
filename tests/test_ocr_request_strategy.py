import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from video_analyzer.frame import Frame
from video_analyzer.ocr import (
    DotsMOCRVLLMProvider,
    OCREvent,
    OpenAICompatibleVisionOCRProvider,
    _encode_image,
    _ocr_cache_key,
    normalize_unlimited_ocr_output,
)


class OCRRequestStrategyTests(unittest.TestCase):
    def test_encode_image_resizes_large_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frame.jpg"
            Image.new("RGB", (1920, 1080), "white").save(path, format="JPEG", quality=95)

            original = base64.b64decode(_encode_image(path, 0))
            resized = base64.b64decode(_encode_image(path, 1280))

            self.assertLess(len(resized), len(original))
            with Image.open(path) as source:
                self.assertEqual(source.size, (1920, 1080))

    def test_cache_key_includes_prompt_and_request_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frame.jpg"
            Image.new("RGB", (100, 80), "white").save(path, format="JPEG")
            frame = Frame(number=1, timestamp=1.0, path=path, score=0.0)

            keys = {
                _ocr_cache_key(frame, "dots_mocr_vllm", "model", "prompt_scene_spotting", "a,b", 4096, 1280),
                _ocr_cache_key(frame, "dots_mocr_vllm", "model", "prompt_layout_json", "a,b", 4096, 1280),
                _ocr_cache_key(frame, "dots_mocr_vllm", "model", "prompt_scene_spotting", "a,b", 10000, 1280),
                _ocr_cache_key(frame, "dots_mocr_vllm", "model", "prompt_scene_spotting", "a,b", 4096, 0),
                _ocr_cache_key(frame, "unlimited_ocr", "model", "prompt_scene_spotting", "a,b", 4096, 1920, "gundam"),
                _ocr_cache_key(frame, "unlimited_ocr", "model", "prompt_scene_spotting", "a,b", 4096, 1920, "base"),
            }

            self.assertEqual(len(keys), 6)

    def test_tailnet_ocr_clients_bypass_proxy_environment(self):
        dots = DotsMOCRVLLMProvider(base_url="http://spark-31d6.taild500c8.ts.net:8000/v1")
        fallback = OpenAICompatibleVisionOCRProvider(
            base_url="http://100.90.114.26:18081/v1",
            model="model",
        )

        self.assertFalse(dots.session.trust_env)
        self.assertFalse(fallback.session.trust_env)

    def test_unlimited_output_parses_text_and_drops_image_placeholders(self):
        raw = (
            "![](images/0.jpg)\n"
            "<|det|>image [0, 0, 100, 100]<|/det|>![](images/1.jpg)\n"
            "<|det|>text [100, 20, 400, 80]<|/det|>MSFT 微软 389.100 +1.94%\n"
            "<|det|>text [100, 90, 300, 130]<|/det|>最高价 394.200"
        )

        text, items, quality_status, failures = normalize_unlimited_ocr_output(raw)

        self.assertEqual(text, "MSFT 微软 389.100 +1.94%\n最高价 394.200")
        self.assertEqual(items[1]["bbox"], [100, 20, 400, 80])
        self.assertEqual(quality_status, "passed")
        self.assertEqual(failures, [])

    def test_unlimited_output_rejects_repeated_formula_hallucination(self):
        repeated = "\n".join(
            "<|det|>formula [0, 0, 10, 10]<|/det|>\\( \\therefore m = \\frac{3}{11} \\)"
            for _ in range(8)
        )

        text, _items, quality_status, failures = normalize_unlimited_ocr_output(repeated)

        self.assertEqual(
            text,
            "\n".join("\\( \\therefore m = \\frac{3}{11} \\)" for _ in range(8)),
        )
        self.assertEqual(quality_status, "quality_failed")
        self.assertIn("repetitive_output", failures)
        self.assertIn("repetitive_formula_output", failures)

    def test_unlimited_output_rejects_abnormally_long_video_frame_result(self):
        raw = "MSFT 389.100 " * 700

        _text, _items, quality_status, failures = normalize_unlimited_ocr_output(raw)

        self.assertEqual(quality_status, "quality_failed")
        self.assertIn("abnormally_long_output", failures)

    def test_unlimited_output_rejects_low_character_diversity(self):
        raw = "ㅂㅂㅂㅂㅂㅂㅂㅅㅅㅅㅅㅁㅁㅁㅁ" * 30

        _text, _items, quality_status, failures = normalize_unlimited_ocr_output(raw)

        self.assertEqual(quality_status, "quality_failed")
        self.assertIn("low_character_diversity", failures)

    def test_unlimited_output_rejects_placeholder_text(self):
        _text, _items, quality_status, failures = normalize_unlimited_ocr_output(
            'Text: "____"'
        )

        self.assertEqual(quality_status, "quality_failed")
        self.assertIn("placeholder_output", failures)

    def test_unlimited_output_retries_image_only_result(self):
        _text, _items, quality_status, failures = normalize_unlimited_ocr_output(
            "![](images/0.jpg)"
        )

        self.assertEqual(quality_status, "quality_failed")
        self.assertIn("image_only_output", failures)

    def test_unlimited_output_drops_image_marker_flood_when_text_is_useful(self):
        raw = "PLTR 176.090\n" + "\n".join(
            f"![](images/{index}.jpg)" for index in range(8)
        )

        text, _items, quality_status, failures = normalize_unlimited_ocr_output(raw)

        self.assertEqual(text, "PLTR 176.090")
        self.assertEqual(quality_status, "passed")
        self.assertEqual(failures, [])

    def test_unlimited_output_trims_incomplete_trailing_detection(self):
        raw = (
            "PLTR Palantir\n176.090 +2.37%\n"
            "![](images/0.jpg)\n"
            "<|det|>text [699, 284, 835,"
        )

        text, items, quality_status, failures = normalize_unlimited_ocr_output(raw)

        self.assertEqual(text, "PLTR Palantir\n176.090 +2.37%")
        self.assertEqual(items, [])
        self.assertEqual(quality_status, "passed")
        self.assertEqual(failures, [])

    def test_unlimited_output_salvages_text_from_malformed_table_detection(self):
        raw = (
            "<|det|>table [16, 91, 160, 110]<|/det|>"
            "<table><tr><td>名称</td><td>最新价</td><td>涨跌幅</td></tr></table>\n"
            "<|det|>table_[<img></td><td><img></td></tr>"
            "<tr><td>Palantir PLTR</td><td>176.090</td><td>+2.37%</td></tr>\n"
            "<|det|>chart [9, 116, 162, 755]<|/det|>"
        )

        text, items, quality_status, failures = normalize_unlimited_ocr_output(raw)

        self.assertIn("Palantir PLTR", text)
        self.assertIn("176.090", text)
        self.assertIn("+2.37%", text)
        self.assertEqual(items[-1]["category"], "chart")
        self.assertEqual(quality_status, "quality_failed")
        self.assertIn("unbalanced_detection_markers", failures)

    def test_unlimited_output_rejects_repeated_non_text_noise(self):
        raw = "![](images/0.jpg)\n" + "\n".join("[Non-Text]" for _ in range(40))

        text, _items, quality_status, failures = normalize_unlimited_ocr_output(raw)

        self.assertEqual(text, "")
        self.assertEqual(quality_status, "quality_failed")
        self.assertIn("image_only_output", failures)
        self.assertIn("low_information_output", failures)

    def test_ocr_event_roundtrip_preserves_quality_metadata(self):
        event = OCREvent(
            1,
            2.0,
            "unlimited_ocr:test",
            "ok",
            "MSFT",
            [{"category": "text", "text": "MSFT"}],
            raw_text="<|det|>text [0, 0, 1, 1]<|/det|>MSFT",
            quality_status="passed",
            generation_metadata={"image_mode": "gundam"},
        )

        restored = OCREvent.from_dict(event.to_dict())

        self.assertEqual(restored.raw_text, event.raw_text)
        self.assertEqual(restored.quality_status, "passed")
        self.assertEqual(restored.generation_metadata, {"image_mode": "gundam"})

    def test_unlimited_provider_sends_runtime_controls(self):
        good = "<|det|>text [0, 0, 100, 30]<|/det|>PLTR 176.090 +2.37%"
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": good}}],
            "ocr_metadata": {"effective_max_length": 8192},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frame.jpg"
            Image.new("RGB", (1920, 1080), "black").save(path)
            provider = DotsMOCRVLLMProvider(
                base_url="http://127.0.0.1:18088/v1",
                provider_name="unlimited_ocr",
                image_mode="gundam",
                max_tokens=8192,
                max_image_long_side=0,
            )
            provider.selected_base_url = "http://127.0.0.1:18088/v1"
            provider.session = Mock()
            provider.session.post.return_value = response

            event = provider.analyze_frame(Frame(1, path, 2.0, 0.0))

        self.assertEqual(event.status, "ok")
        self.assertEqual(event.text, "PLTR 176.090 +2.37%")
        self.assertEqual(event.quality_status, "passed")
        payload = provider.session.post.call_args.kwargs["json"]
        self.assertEqual(payload["images_config"], {"image_mode": "gundam", "max_crop_blocks": 8})
        self.assertEqual(payload["custom_params"]["ngram_sizes"], [3, 35])
        self.assertEqual(payload["custom_params"]["window_size"], 128)
        self.assertEqual(payload["custom_params"]["repetition_penalty"], 1.1)
        self.assertEqual(payload["max_tokens"], 8192)

    def test_unlimited_provider_observes_quality_warning_without_dropping_text(self):
        bad = "\n".join(
            "<|det|>formula [0, 0, 10, 10]<|/det|>\\( \\therefore m = \\frac{3}{11} \\)"
            for _ in range(8)
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": bad}}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frame.jpg"
            Image.new("RGB", (1920, 1080), "black").save(path)
            provider = DotsMOCRVLLMProvider(
                base_url="http://127.0.0.1:18088/v1",
                provider_name="unlimited_ocr",
                image_mode="gundam",
            )
            provider.selected_base_url = "http://127.0.0.1:18088/v1"
            provider.session = Mock()
            provider.session.post.return_value = response

            event = provider.analyze_frame(Frame(1, path, 2.0, 0.0))

        self.assertEqual(event.status, "ok")
        self.assertEqual(event.quality_status, "quality_failed")
        self.assertTrue(event.text)
        self.assertIn("repetitive_formula_output", event.generation_metadata["quality_warnings"])
        self.assertEqual(event.generation_metadata["quality_gate_mode"], "observe")
        self.assertEqual(provider.session.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
