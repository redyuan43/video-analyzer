#!/usr/bin/env python3
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from video_analyzer.config import Config
from video_analyzer.analyzer import VideoAnalyzer
from video_analyzer.manual import build_operation_manual_prompt, embed_step_images, prepare_frame_assets, write_frame_evidence_index
from video_analyzer.asr_providers import transcribe_with_provider, transcribe_with_remote_http
from video_analyzer.ocr import DotsMOCRVLLMProvider, run_ocr
from video_analyzer.audio_processor import AudioTranscript

try:
    import cv2
    import numpy as np
    from video_analyzer.frame import VideoProcessor
except ModuleNotFoundError:
    cv2 = None
    np = None
    VideoProcessor = None


class OperationManualTests(unittest.TestCase):
    def test_operation_manual_config_uses_lmstudio_defaults(self):
        args = argparse.Namespace(
            video_path="video.mp4",
            config="config",
            output=None,
            client=None,
            ollama_url=None,
            api_key=None,
            api_url=None,
            model=None,
            duration=None,
            keep_frames=False,
            whisper_model=None,
            start_stage=1,
            max_frames=10,
            log_level="INFO",
            prompt="",
            language=None,
            device="cpu",
            temperature=None,
            task="operation_manual",
            manual_language=None,
            llm_base_url=None,
            vision_model=None,
            text_model=None,
            ocr_provider=None,
            ocr_base_url=None,
            asr_provider=None,
            context_file=None,
        )
        config = Config("config")
        config.update_from_args(args)

        self.assertEqual(config.get("clients")["default"], "openai_api")
        self.assertEqual(config.get("clients")["openai_api"]["api_key"], "0")
        self.assertEqual(config.get("clients")["openai_api"]["api_url"], "http://127.0.0.1:1234/v1")
        self.assertEqual(config.get("clients")["openai_api"]["model"], "sayanything-hauhaucs-aggressive@?")
        self.assertEqual(config.get("asr")["provider"], "auto")
        self.assertEqual(config.get("ocr")["fallback_model"], "sayanything-hauhaucs-aggressive@?")

    def test_ocr_provider_parses_dots_mocr_json(self):
        if cv2 is None:
            self.skipTest("opencv-python is not installed")
        provider = DotsMOCRVLLMProvider(base_url="http://ocr.test/v1")
        frame_path = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
        cv2.imwrite(str(frame_path), np.zeros((20, 20, 3), dtype=np.uint8))
        frame = Mock(number=3, timestamp=1.25, path=frame_path)

        get_response = Mock()
        get_response.raise_for_status.return_value = None
        post_response = Mock()
        post_response.raise_for_status.return_value = None
        post_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '[{"bbox":[1,2,3,4],"category":"Text","text":"mystyle.json"}]'
                    }
                }
            ]
        }

        with patch("video_analyzer.ocr.requests.get", return_value=get_response), patch(
            "video_analyzer.ocr.requests.post", return_value=post_response
        ):
            event = provider.analyze_frame(frame)

        self.assertEqual(event.status, "ok")
        self.assertEqual(event.text, "mystyle.json")
        self.assertEqual(event.items[0]["bbox"], [1, 2, 3, 4])
        frame_path.unlink()

    def test_run_ocr_reports_unavailable_once(self):
        frames = [Mock(number=0, timestamp=0.0), Mock(number=1, timestamp=1.0)]
        with patch("video_analyzer.ocr.requests.get", side_effect=RuntimeError("offline")) as get:
            events = run_ocr(frames, "auto", "http://ocr.test/v1", "model", "prompt_scene_spotting")

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].status, "unavailable")
        self.assertEqual(get.call_count, 1)

    def test_run_ocr_falls_back_to_openai_vision(self):
        if cv2 is None:
            self.skipTest("opencv-python is not installed")
        frame_path = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
        cv2.imwrite(str(frame_path), np.zeros((20, 20, 3), dtype=np.uint8))
        frames = [Mock(number=0, timestamp=0.0, path=frame_path)]

        fallback_models = Mock()
        fallback_models.raise_for_status.return_value = None
        fallback_response = Mock()
        fallback_response.status_code = 200
        fallback_response.raise_for_status.return_value = None
        fallback_response.json.return_value = {
            "choices": [{"message": {"content": '[{"category":"Text","text":"风格预设"}]'}}]
        }

        def fake_get(url, **kwargs):
            if "ocr.test" in url:
                raise RuntimeError("dots offline")
            return fallback_models

        with patch("video_analyzer.ocr.requests.get", side_effect=fake_get), patch(
            "video_analyzer.ocr.requests.post", return_value=fallback_response
        ):
            events = run_ocr(
                frames,
                "auto",
                "http://ocr.test/v1",
                "model",
                "prompt_scene_spotting",
                fallback_base_url="http://127.0.0.1:1234/v1",
                fallback_model="qwen/qwen3-vl-30b",
            )

        self.assertEqual(events[0].status, "ok")
        self.assertIn("风格预设", events[0].text)
        self.assertTrue(events[0].provider.startswith("openai_vision"))
        frame_path.unlink()

    def test_manual_prompt_separates_context_and_evidence(self):
        transcript = AudioTranscript(text="说，用我的 mystyle 风格", segments=[], language="zh")
        frame = Mock(number=0, timestamp=2.0)
        prompt = build_operation_manual_prompt(
            frame_analyses=[{"response": "A flow shows AI extracting style"}],
            frames=[frame],
            transcript=transcript,
            ocr_events=[],
            page_context="GitHub: https://github.com/Agents365-ai/drawio-skill",
            language="zh-CN",
            frame_assets={0: "manual_assets/frame_000.jpg"},
        )

        self.assertIn("Page description/context", prompt)
        self.assertIn("Frame evidence", prompt)
        self.assertIn("manual_assets/frame_000.jpg", prompt)
        self.assertIn("Do not append a large screenshot gallery", prompt)
        self.assertIn("Mermaid flowchart", prompt)
        self.assertIn("需复核", prompt)

    def test_evidence_index_is_separate_from_user_manual_assets(self):
        if cv2 is None:
            self.skipTest("opencv-python is not installed")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            frame_path = output_dir / "source.jpg"
            cv2.imwrite(str(frame_path), np.zeros((20, 20, 3), dtype=np.uint8))
            frame = Mock(number=0, timestamp=1.5, path=frame_path)
            ocr = Mock(frame_number=0, timestamp=1.5, provider="ocr", status="ok", text="按钮", items=[], error=None)
            assets = prepare_frame_assets([frame], output_dir)
            evidence_path = write_frame_evidence_index(
                frames=[frame],
                output_dir=output_dir,
                ocr_events=[ocr],
                frame_analyses=[{"response": "shows a button"}],
                frame_assets=assets,
            )

            self.assertTrue((output_dir / assets[0]).exists())
            self.assertEqual(evidence_path.name, "manual_evidence.md")
            evidence = evidence_path.read_text(encoding="utf-8")
            self.assertIn("帧证据索引", evidence)
            self.assertIn("按钮", evidence)

    def test_step_images_are_embedded_near_matching_step(self):
        frames = [
            Mock(number=0, timestamp=3.0),
            Mock(number=1, timestamp=20.0),
            Mock(number=2, timestamp=23.0),
            Mock(number=3, timestamp=64.0),
        ]
        assets = {
            0: "manual_assets/frame_000.jpg",
            1: "manual_assets/frame_001.jpg",
            2: "manual_assets/frame_002.jpg",
            3: "manual_assets/frame_003.jpg",
        }
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：基础输入（视频证据 `[00:03]~[00:23]`）",
                "操作说明。",
                "### 步骤 2：风格预设（视频证据 `[01:04]`）",
                "操作说明。",
            ]
        )

        updated = embed_step_images(manual, frames, assets)

        first_step = updated.split("### 步骤 2")[0]
        self.assertIn("manual_assets/frame_000.jpg", first_step)
        self.assertIn("manual_assets/frame_002.jpg", first_step)
        self.assertIn("| 3s |", first_step)

    def test_frame_context_is_bounded_for_many_frames(self):
        analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        analyzer.previous_analyses = [{"response": f"frame {i} " + ("x" * 1000)} for i in range(8)]

        formatted = VideoAnalyzer._format_previous_analyses(analyzer)

        self.assertIn("Frame 5", formatted)
        self.assertIn("Frame 7", formatted)
        self.assertNotIn("Frame 4", formatted)
        self.assertLess(len(formatted), 2600)

    def test_vibevoice_empty_python_config_uses_known_environment(self):
        audio_path = Mock()
        audio_path.__fspath__ = Mock(return_value="/tmp/audio.wav")
        with patch("video_analyzer.asr_providers.VIBEVOICE_PYTHONS", [Path("/env/bin/python")]), patch(
            "video_analyzer.asr_providers.VIBEVOICE_SCRIPT", Path("/script.py")
        ), patch("video_analyzer.asr_providers.Path.exists", return_value=True), patch(
            "video_analyzer.asr_providers.subprocess.run"
        ) as run:
            run.return_value.stdout = "--- Raw Output ---\nhello\n--- Structured Output (0 segments) ---"
            transcript = transcribe_with_provider(
                provider="vibevoice",
                audio_path=Path("/tmp/audio.wav"),
                language="",
                whisper_model="medium",
                device="cpu",
                vibevoice_config={"python": ""},
            )

        self.assertEqual(transcript.text, "hello")
        self.assertEqual(run.call_args[0][0][0], "/env/bin/python")

    def test_remote_http_asr_uses_first_successful_endpoint(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        audio_path.write_bytes(b"RIFF")
        failed = Mock()
        failed.raise_for_status.side_effect = RuntimeError("offline")
        ok = Mock()
        ok.raise_for_status.return_value = None
        ok.json.return_value = {"success": True, "text": "远程识别文本", "language": "zh"}

        with patch("video_analyzer.asr_providers.requests.post", side_effect=[failed, ok]) as post:
            transcript = transcribe_with_remote_http(
                audio_path,
                ["http://spark/asr", "http://edge/asr"],
            )

        self.assertEqual(transcript.text, "远程识别文本")
        self.assertEqual(post.call_args_list[0].args[0], "http://spark/asr")
        self.assertEqual(post.call_args_list[1].args[0], "http://edge/asr")
        audio_path.unlink()

    def test_screen_keyframes_keep_static_and_changed_ui(self):
        if cv2 is None:
            self.skipTest("opencv-python is not installed")
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "screen.mp4"
            output_dir = Path(temp_dir) / "frames"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (160, 90),
            )
            for i in range(20):
                frame = np.full((90, 160, 3), 255, dtype=np.uint8)
                cv2.putText(frame, "step1" if i < 10 else "step2", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                writer.write(frame)
            writer.release()

            frames = VideoProcessor(video_path, output_dir, "model").extract_screen_keyframes(
                frames_per_minute=120,
                max_frames=10,
                change_threshold=1.0,
            )

            self.assertGreaterEqual(len(frames), 2)
            self.assertTrue(all(frame.path.exists() for frame in frames))

    def test_density_budget_keeps_coverage_and_high_change_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VideoProcessor(Path(temp_dir) / "video.mp4", Path(temp_dir) / "frames", "model")
            candidates = [
                (0, Path(temp_dir) / "0.jpg", 0.0, 1.0),
                (1, Path(temp_dir) / "1.jpg", 5.0, 2.0),
                (2, Path(temp_dir) / "2.jpg", 10.0, 100.0),
                (3, Path(temp_dir) / "3.jpg", 25.0, 3.0),
                (4, Path(temp_dir) / "4.jpg", 35.0, 90.0),
                (5, Path(temp_dir) / "5.jpg", 50.0, 4.0),
                (6, Path(temp_dir) / "6.jpg", 65.0, 80.0),
            ]

            selected = processor._select_density_budget(candidates, max_frames=5, coverage_interval_seconds=20)
            selected_timestamps = [item[2] for item in selected]

            self.assertIn(0.0, selected_timestamps)
            self.assertIn(25.0, selected_timestamps)
            self.assertIn(50.0, selected_timestamps)
            self.assertIn(65.0, selected_timestamps)
            self.assertIn(10.0, selected_timestamps)


if __name__ == "__main__":
    unittest.main()
