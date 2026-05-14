import base64
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from video_analyzer.frame import Frame
from video_analyzer.ocr import DotsMOCRVLLMProvider, OpenAICompatibleVisionOCRProvider, _encode_image, _ocr_cache_key


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
            }

            self.assertEqual(len(keys), 4)

    def test_tailnet_ocr_clients_bypass_proxy_environment(self):
        dots = DotsMOCRVLLMProvider(base_url="http://spark-31d6.taild500c8.ts.net:8000/v1")
        fallback = OpenAICompatibleVisionOCRProvider(
            base_url="http://100.90.114.26:18081/v1",
            model="model",
        )

        self.assertFalse(dots.session.trust_env)
        self.assertFalse(fallback.session.trust_env)


if __name__ == "__main__":
    unittest.main()
