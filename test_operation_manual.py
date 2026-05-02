#!/usr/bin/env python3
import argparse
import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from video_analyzer.config import Config
from video_analyzer.analyzer import VideoAnalyzer
from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.manual import (
    build_operation_manual_prompt,
    embed_step_images,
    prepare_frame_assets,
    review_operation_manual_markdown,
    write_frame_evidence_index,
)
from video_analyzer.asr_providers import (
    REMOTE_ASR_URLS,
    REMOTE_VIBEVOICE_URLS,
    merge_asr_transcripts,
    transcribe_with_provider,
    transcribe_with_provider_result,
    transcribe_with_remote_http,
    transcribe_with_vibevoice_remote,
    transcribe_with_strategy,
    transcribe_with_vibevoice,
)
from video_analyzer.ocr import DOTS_MOCR_ENDPOINTS, DotsMOCRVLLMProvider, run_ocr
from video_analyzer.audio_processor import AudioTranscript

RUNNER_PATH = Path(__file__).resolve().parent / "tools" / "run_operation_manual_from_url.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_operation_manual_from_url", RUNNER_PATH)
run_operation_manual_from_url = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER_SPEC.loader.exec_module(run_operation_manual_from_url)

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
            asr_strategy=None,
            remote_asr_url=None,
            vibevoice_url=None,
            context_file=None,
        )
        config = Config("config")
        config.update_from_args(args)

        self.assertEqual(config.get("clients")["default"], "openai_api")
        self.assertEqual(config.get("clients")["openai_api"]["api_key"], "0")
        self.assertEqual(config.get("clients")["openai_api"]["api_url"], "http://127.0.0.1:1234/v1")
        self.assertEqual(
            config.get("clients")["openai_api"]["model"],
            "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive@?",
        )
        self.assertEqual(config.get("asr")["provider"], "auto")
        self.assertEqual(config.get("asr")["strategy"], "balanced")
        self.assertEqual(config.get("asr")["vibevoice"]["remote_urls"], [])
        self.assertEqual(config.get("asr")["vibevoice"]["deep_remote_urls"][1], "http://192.168.100.236:8003/api/asr/transcribe")
        self.assertTrue(config.get("asr")["vibevoice"]["use_native_chunking"])
        self.assertEqual(
            config.get("ocr")["fallback_model"],
            "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive@?",
        )

    def test_url_runner_builds_page_context_markdown(self):
        info = {
            "title": "Hermes Bridge",
            "id": "BV123",
            "uploader": "tester",
            "upload_date": "20260502",
            "duration": 90,
            "description": "安装命令和视频简介",
            "chapters": [{"start_time": 0, "end_time": 30, "title": "开始"}],
            "tags": ["Hermes", "Android"],
        }

        text = run_operation_manual_from_url.build_context_markdown(info, "https://example.test/video")

        self.assertIn("# Hermes Bridge", text)
        self.assertIn("安装命令和视频简介", text)
        self.assertIn("00:00:00 - 00:00:30: 开始", text)
        self.assertIn("Hermes, Android", text)

    def test_url_runner_builds_page_context_bundle_with_subtitles_and_comments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_dir = Path(temp_dir)
            (video_dir / "download.zh-CN.vtt").write_text(
                "\n".join(
                    [
                        "WEBVTT",
                        "",
                        "00:00:01.000 --> 00:00:03.000",
                        "<c>打开终端输入 install.sh</c>",
                    ]
                ),
                encoding="utf-8",
            )
            info = {
                "title": "Hermes Bridge",
                "id": "BV123",
                "uploader": "tester",
                "description": "安装命令和视频简介",
                "subtitles": {"zh-CN": [{"ext": "vtt"}]},
                "comments": [
                    {"author": "tester", "text": "置顶补充：新版入口在 Settings", "author_is_uploader": True},
                    {"author": "viewer", "text": "普通评论里说旧版本按钮不同", "like_count": 9},
                ],
            }
            args = argparse.Namespace(
                include_subtitles=True,
                include_comments=True,
                subtitle_langs="zh-CN,zh,en",
                max_comments=1,
            )

            bundle = run_operation_manual_from_url.build_page_context_bundle(
                info,
                "https://example.test/video",
                run_operation_manual_from_url.build_context_markdown(info, "https://example.test/video"),
                video_dir,
                args,
            )

            self.assertIn("Evidence weights for manual generation", bundle["markdown"])
            self.assertIn("author subtitles", bundle["markdown"])
            self.assertIn("打开终端输入 install.sh", bundle["markdown"])
            self.assertIn("置顶补充", bundle["markdown"])
            self.assertNotIn("普通评论里说旧版本按钮不同", bundle["markdown"])
            self.assertTrue((video_dir / "subtitles" / "download.zh-CN.vtt").exists())
            self.assertTrue((video_dir / "comments.json").exists())
            self.assertEqual(bundle["metadata"]["subtitles"]["language"], "zh-CN")
            self.assertEqual(bundle["metadata"]["comments"]["selected_count"], 1)

    def test_url_runner_cleans_json3_subtitles_with_timestamps(self):
        payload = {
            "events": [
                {
                    "tStartMs": 1200,
                    "dDurationMs": 1800,
                    "segs": [{"utf8": "点击"}, {"utf8": "安装按钮"}],
                }
            ]
        }

        text = run_operation_manual_from_url.clean_json3_subtitles(json.dumps(payload, ensure_ascii=False))

        self.assertIn("[00:00:01 - 00:00:03] 点击安装按钮", text)

    def test_url_runner_defaults_to_edge_vibevoice_and_spark_ocr(self):
        args = argparse.Namespace(
            python=".venv/bin/python",
            vibevoice_url="http://192.168.100.236:8003/api/asr/transcribe",
            ocr_base_url="http://192.168.100.169:8000/v1",
            llm_base_url="http://127.0.0.1:1234/v1",
            vision_model="sayanything-hauhaucs-aggressive@?",
            text_model="redhatai_qwen3.6-35b-a3b-nvfp4",
            manual_language="zh-CN",
            max_frames=24,
            log_level="INFO",
            duration=None,
            no_keep_frames=False,
        )

        command = run_operation_manual_from_url.build_analyzer_command(
            args,
            Path("video.mp4"),
            Path("description.md"),
            Path("run"),
        )

        self.assertIn("--asr-provider", command)
        self.assertIn("vibevoice", command)
        self.assertIn("http://192.168.100.236:8003/api/asr/transcribe", command)
        self.assertIn("http://192.168.100.169:8000/v1", command)

    def test_url_runner_rejects_unsafe_run_name_before_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "video"
            parent.mkdir()

            with self.assertRaises(ValueError):
                run_operation_manual_from_url.safe_child_dir(parent, "/tmp/delete-me")

            safe = run_operation_manual_from_url.safe_child_dir(parent, "../delete-me")
            self.assertEqual(safe, (parent / "delete-me").resolve())
            self.assertIn(parent.resolve(), safe.parents)

    def test_url_runner_reports_quality_failed_manual_path_from_analysis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis_path = Path(temp_dir) / "analysis.json"
            manual_path = Path(temp_dir) / "operation_manual.quality_failed.md"
            analysis_path.write_text(
                json.dumps({"operation_manual": {"manual_path": str(manual_path)}}),
                encoding="utf-8",
            )

            self.assertEqual(run_operation_manual_from_url.read_manual_path(analysis_path), manual_path)

    def test_operation_manual_preserves_user_configured_asr_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "config.json").write_text(
                """
{
  "task": "describe",
  "clients": {
    "default": "openai_api",
    "openai_api": {"api_key": "0", "api_url": "http://127.0.0.1:1234/v1", "model": "model"},
    "ollama": {"url": "http://localhost:11434", "model": "model"},
    "temperature": 0.0
  },
  "operation_manual": {"llm_base_url": "http://127.0.0.1:1234/v1", "vision_model": "vision", "text_model": "text"},
  "ocr": {"provider": "none"},
  "asr": {"provider": "vibevoice", "strategy": "balanced", "vibevoice": {}},
  "audio": {"language": "zh", "whisper_model": "medium", "device": "cpu"},
  "prompts": [],
  "prompt_dir": "prompts",
  "output_dir": "output"
}
""".strip(),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                video_path="video.mp4",
                config=str(config_dir),
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
                asr_strategy=None,
                remote_asr_url=None,
                vibevoice_url=None,
                context_file=None,
            )

            config = Config(str(config_dir))
            config.update_from_args(args)

            self.assertEqual(config.get("asr")["provider"], "vibevoice")

    def test_vibevoice_cli_options_configure_fast_and_deep_remote_urls(self):
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
            asr_strategy="deep",
            remote_asr_url=["http://agx/asr"],
            vibevoice_url=["http://spark/vibevoice"],
            context_file=None,
        )
        config = Config("config")
        config.update_from_args(args)

        vibevoice = config.get("asr")["vibevoice"]
        self.assertEqual(vibevoice["remote_urls"], ["http://agx/asr"])
        self.assertEqual(vibevoice["deep_remote_urls"], ["http://spark/vibevoice"])

    def test_openai_client_uses_reasoning_content_when_content_is_empty(self):
        client = GenericOpenAIAPIClient("0", "http://127.0.0.1:1234/v1", max_retries=1)
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "截图里显示终端安装命令。",
                    }
                }
            ]
        }

        with patch("video_analyzer.clients.generic_openai_api.requests.post", return_value=response):
            result = client.generate(
                prompt="describe",
                image_path=None,
                model="vision",
                temperature=0.0,
                num_predict=32,
            )

        self.assertEqual(result["response"], "截图里显示终端安装命令。")
        self.assertEqual(result["response_source"], "reasoning_content")

    def test_openai_client_rejects_reasoning_content_fallback_for_public_endpoint(self):
        client = GenericOpenAIAPIClient("key", "https://api.example.com/v1", max_retries=1)
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "private reasoning",
                    }
                }
            ]
        }

        with patch("video_analyzer.clients.generic_openai_api.requests.post", return_value=response):
            with self.assertRaises(Exception) as raised:
                client.generate(prompt="describe", model="vision")

        self.assertIn("reasoning_content fallback is only allowed", str(raised.exception))

    def test_dots_mocr_default_endpoints_prefer_lan_before_localhost(self):
        self.assertEqual(DOTS_MOCR_ENDPOINTS[0], "http://192.168.100.169:8000/v1")
        self.assertEqual(DOTS_MOCR_ENDPOINTS[1], "http://192.168.100.131:8000/v1")

    def test_asr_default_endpoints_use_lan_not_tailscale(self):
        self.assertEqual(REMOTE_ASR_URLS[0], "http://192.168.100.117:8001/api/asr/transcribe")
        self.assertEqual(REMOTE_VIBEVOICE_URLS[0], "http://192.168.100.169:8002/api/asr/transcribe")
        self.assertEqual(REMOTE_VIBEVOICE_URLS[1], "http://192.168.100.236:8003/api/asr/transcribe")
        self.assertFalse(any("taild500c8" in url for url in [*REMOTE_ASR_URLS, *REMOTE_VIBEVOICE_URLS]))

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
            asr_metadata={
                "strategy": "deep",
                "providers_run": ["remote_http", "vibevoice"],
                "fast_transcript": {"text_length": 4},
                "deep_transcript": {"text_length": 12},
            },
            ocr_events=[],
            page_context="GitHub: https://github.com/Agents365-ai/drawio-skill",
            language="zh-CN",
            frame_assets={0: "manual_assets/frame_000.jpg"},
        )

        self.assertIn("Page context evidence package", prompt)
        self.assertIn("Frame evidence", prompt)
        self.assertIn("manual_assets/frame_000.jpg", prompt)
        self.assertIn("ASR strategy evidence", prompt)
        self.assertIn("VibeVoice", prompt)
        self.assertIn("author subtitles", prompt)
        self.assertIn("社区补充/常见问题", prompt)
        self.assertIn("Do not append a large screenshot gallery", prompt)
        self.assertIn("Screenshots must be real Markdown images", prompt)
        self.assertIn("Never write screenshot paths as plain text", prompt)
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

    def test_asset_paths_are_rendered_as_markdown_images(self):
        frames = [Mock(number=0, timestamp=3.0)]
        assets = {0: "manual_assets/frame_000.jpg"}
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：截图",
                "| 帧截图 | 说明 |",
                "|--------|------|",
                "| `manual_assets/frame_000.jpg` | 预览 |",
            ]
        )

        updated = embed_step_images(manual, frames, assets)

        self.assertIn("![frame_000](manual_assets/frame_000.jpg)", updated)
        self.assertNotIn("`manual_assets/frame_000.jpg`", updated)

    def test_code_span_images_are_unwrapped_to_rendered_images(self):
        frames = [Mock(number=0, timestamp=3.0)]
        assets = {0: "manual_assets/frame_000.jpg"}
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：截图",
                "`![入口截图](manual_assets/frame_000.jpg)`",
            ]
        )

        updated = embed_step_images(manual, frames, assets)

        self.assertIn("![入口截图](manual_assets/frame_000.jpg)", updated)
        self.assertNotIn("`![入口截图](manual_assets/frame_000.jpg)`", updated)

    def test_operation_manual_review_flags_raw_asset_paths(self):
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：打开页面",
                "| 截图 | 说明 |",
                "| --- | --- |",
                "| `manual_assets/frame_000.jpg` | 页面入口 |",
            ]
        )

        issues = review_operation_manual_markdown(manual)

        self.assertIn("raw_asset_path", {issue["code"] for issue in issues})

    def test_operation_manual_review_flags_code_span_images(self):
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：打开页面",
                "`![入口截图](manual_assets/frame_000.jpg)`",
            ]
        )

        issues = review_operation_manual_markdown(manual)

        self.assertIn("image_in_code_span", {issue["code"] for issue in issues})

    def test_operation_manual_review_accepts_rendered_step_images(self):
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：打开页面",
                "![入口截图](manual_assets/frame_000.jpg)",
                "按页面提示继续。",
            ]
        )

        issues = review_operation_manual_markdown(manual)

        self.assertNotIn("raw_asset_path", {issue["code"] for issue in issues})
        self.assertNotIn("step_asset_not_rendered", {issue["code"] for issue in issues})

    def test_operation_manual_review_warns_when_step_has_no_screenshot(self):
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：打开页面",
                "按页面提示继续。",
            ]
        )

        issues = review_operation_manual_markdown(manual)

        self.assertIn("step_missing_screenshot", {issue["code"] for issue in issues})

    def test_frame_context_is_bounded_for_many_frames(self):
        analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        analyzer.previous_analyses = [{"response": f"frame {i} " + ("x" * 1000)} for i in range(8)]

        formatted = VideoAnalyzer._format_previous_analyses(analyzer)

        self.assertIn("Frame 5", formatted)
        self.assertIn("Frame 7", formatted)
        self.assertNotIn("Frame 4", formatted)
        self.assertLess(len(formatted), 2600)

    def test_vibevoice_defaults_to_remote_only(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        audio_path.write_bytes(b"RIFF")

        with patch("video_analyzer.asr_providers.transcribe_with_vibevoice_remote", return_value=None) as remote, patch(
            "video_analyzer.asr_providers.subprocess.run"
        ) as run:
            transcript = transcribe_with_vibevoice(audio_path, {"deep_remote_urls": ["http://spark/vibevoice"]})

        self.assertIsNone(transcript)
        remote.assert_called_once()
        run.assert_not_called()
        audio_path.unlink()

    def test_balanced_uses_successful_vibevoice_without_qwen_asr_fallback(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000 * 240)
        deep = AudioTranscript(text="vibevoice reliable text", segments=[{"start_time": 0.0, "end_time": 1.0}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=None), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice", return_value=deep
        ), patch("video_analyzer.asr_providers.transcribe_with_capswriter") as capswriter, patch(
            "video_analyzer.asr_providers.AudioProcessor"
        ) as processor:
            result = transcribe_with_strategy("balanced", audio_path, "", "medium", "cpu", {"remote_urls": []})

        self.assertEqual(result.transcript.text, "vibevoice reliable text")
        capswriter.assert_not_called()
        processor.assert_not_called()
        audio_path.unlink()

    def test_vibevoice_remote_success_avoids_local_subprocess(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        audio_path.write_bytes(b"RIFF")
        remote_transcript = AudioTranscript(text="remote vibe", segments=[{"provider": "vibevoice_remote"}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_vibevoice_remote", return_value=remote_transcript), patch(
            "video_analyzer.asr_providers.subprocess.run"
        ) as run:
            transcript = transcribe_with_vibevoice(audio_path, {"deep_remote_urls": ["http://spark/vibevoice"]})

        self.assertEqual(transcript.text, "remote vibe")
        run.assert_not_called()
        audio_path.unlink()

    def test_vibevoice_passes_configured_remote_urls(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        audio_path.write_bytes(b"RIFF")

        with patch("video_analyzer.asr_providers.transcribe_with_vibevoice_remote", return_value=None) as remote:
            transcript = transcribe_with_vibevoice(audio_path, {"deep_remote_urls": ["http://spark/vibevoice"]})

        self.assertIsNone(transcript)
        remote.assert_called_once()
        self.assertEqual(remote.call_args.args[0], audio_path)
        self.assertEqual(remote.call_args.args[1], ["http://spark/vibevoice"])
        self.assertTrue(remote.call_args.kwargs["options"]["use_native_chunking"])
        audio_path.unlink()

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
        self.assertEqual(post.call_args_list[0].kwargs["timeout"], (30, 900))
        audio_path.unlink()

    def test_vibevoice_enables_native_chunking_without_owning_policy(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        audio_path.write_bytes(b"RIFF")
        ok = Mock()
        ok.raise_for_status.return_value = None
        ok.json.return_value = {
            "success": True,
            "text": "native chunk text",
            "language": "zh",
            "segments": [{"mode": "chunk_reconcile"}],
        }

        with patch("video_analyzer.asr_providers.requests.post", return_value=ok) as post:
            transcript = transcribe_with_vibevoice(
                audio_path,
                {
                    "deep_remote_urls": ["http://spark/vibevoice"],
                },
            )

        self.assertEqual(transcript.text, "native chunk text")
        self.assertEqual(post.call_args.kwargs["data"]["use_native_chunking"], "True")
        self.assertNotIn("chunk_parallel_workers", post.call_args.kwargs["data"])
        self.assertNotIn("chunk_duration_sec", post.call_args.kwargs["data"])
        audio_path.unlink()

    def test_vibevoice_distributes_long_audio_across_remote_workers(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000 * 500)
        spark = AudioTranscript(text="spark half", segments=[{"start_time": 1.0, "end_time": 2.0, "text": "spark half"}], language="zh")
        edge = AudioTranscript(text="edge half", segments=[{"start_time": 1.0, "end_time": 2.0, "text": "edge half"}], language="zh")

        with patch("video_analyzer.asr_providers._split_audio_evenly") as split_audio, patch(
            "video_analyzer.asr_providers.transcribe_with_http_asr", side_effect=[spark, edge]
        ) as post:
            split_audio.return_value = [(Path("spark.wav"), 0.0), (Path("edge.wav"), 250.0)]
            transcript = transcribe_with_vibevoice_remote(
                audio_path,
                ["http://spark/vibevoice", "http://edge/vibevoice"],
                {"use_native_chunking": True, "distributed_min_seconds": 420, "distributed_workers": 2},
            )

        self.assertEqual(transcript.text, "spark half\nedge half")
        self.assertEqual({call.args[1] for call in post.call_args_list}, {"http://spark/vibevoice", "http://edge/vibevoice"})
        self.assertTrue(all(call.kwargs["extra_data"]["use_native_chunking"] is False for call in post.call_args_list))
        self.assertIn(251.0, [segment.get("start_time") for segment in transcript.segments])
        audio_path.unlink()

    def test_remote_http_empty_url_list_disables_builtin_endpoints(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        audio_path.write_bytes(b"RIFF")

        with patch("video_analyzer.asr_providers.requests.post") as post:
            transcript = transcribe_with_remote_http(audio_path, [])

        self.assertIsNone(transcript)
        post.assert_not_called()
        audio_path.unlink()

    def test_explicit_auto_provider_records_actual_attempts(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        audio_path.write_bytes(b"RIFF")
        transcript = AudioTranscript(text="vibe text", segments=[{"raw_output": "vibe text"}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=None), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice", return_value=transcript
        ), patch("video_analyzer.asr_providers.transcribe_with_capswriter") as capswriter:
            result = transcribe_with_provider_result("auto", audio_path, "", "medium", "cpu", {})

        self.assertEqual(result.transcript.text, "vibe text")
        self.assertEqual(result.providers_run, ["vibevoice"])
        capswriter.assert_not_called()
        audio_path.unlink()

    def test_asr_strategy_fast_uses_remote_http_only(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        audio_path.write_bytes(b"RIFF")
        transcript = AudioTranscript(text="fast text", segments=[{"start": 0.0, "end": 1.0, "text": "fast text"}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=transcript) as remote, patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice"
        ) as vibe:
            result = transcribe_with_strategy("fast", audio_path, "", "medium", "cpu", {})

        self.assertEqual(result.transcript.text, "fast text")
        self.assertEqual(result.providers_run, ["remote_http"])
        remote.assert_called_once()
        vibe.assert_not_called()
        audio_path.unlink()

    def test_asr_strategy_deep_merges_vibevoice_text_with_fast_timestamps(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        audio_path.write_bytes(b"RIFF")
        fast = AudioTranscript(
            text="用风格 预设",
            segments=[
                {"start": 0.0, "end": 3.0, "text": "用风格"},
                {"start": 3.0, "end": 6.0, "text": "预设"},
            ],
            language="zh",
        )
        deep = AudioTranscript(text="使用 mystyle 风格。生成可复用预设。", segments=[{"raw_output": "deep"}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=fast), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice", return_value=deep
        ):
            result = transcribe_with_strategy("deep", audio_path, "", "medium", "cpu", {})

        self.assertEqual(result.transcript.text, deep.text)
        self.assertEqual(result.transcript.segments[0]["start"], 0.0)
        self.assertEqual(result.transcript.segments[0]["source"], "merged_remote_http_vibevoice")
        self.assertEqual(result.transcript.segments[0]["text"], "使用 mystyle 风格。")
        self.assertEqual(result.transcript.segments[1]["text"], "生成可复用预设。")
        self.assertEqual(result.providers_run, ["remote_http", "vibevoice"])
        audio_path.unlink()

    def test_asr_strategy_balanced_skips_vibevoice_for_short_good_fast_transcript(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        fast = AudioTranscript(text="short good transcript", segments=[{"start": 0.0, "end": 1.0, "text": "short"}], language="en")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=fast), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice"
        ) as vibe:
            result = transcribe_with_strategy("balanced", audio_path, "", "medium", "cpu", {})

        self.assertEqual(result.transcript.text, "short good transcript")
        self.assertIn("skipped VibeVoice", " ".join(result.merge_notes))
        vibe.assert_not_called()
        audio_path.unlink()

    def test_merge_asr_transcripts_keeps_fast_timestamps_and_deep_terms(self):
        fast = AudioTranscript(
            text="old term",
            segments=[{"start": 1.0, "end": 2.0, "text": "old term"}],
            language="zh",
        )
        deep = AudioTranscript(text="mystyle 风格。", segments=[], language="zh")

        merged = merge_asr_transcripts(fast, deep)

        self.assertEqual(merged.text, "mystyle 风格。")
        self.assertEqual(merged.segments[0]["start"], 1.0)
        self.assertEqual(merged.segments[0]["text"], "mystyle 风格。")

    def test_balanced_empty_fast_transcript_falls_back_after_vibevoice_failure(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        empty_fast = AudioTranscript(text="", segments=[], language="unknown")
        caps = AudioTranscript(text="caps text", segments=[{"start": 0.0, "end": 1.0, "text": "caps text"}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=empty_fast), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice", return_value=None
        ), patch("video_analyzer.asr_providers.transcribe_with_capswriter", return_value=caps), patch(
            "video_analyzer.asr_providers.AudioProcessor"
        ) as processor:
            result = transcribe_with_strategy("balanced", audio_path, "", "medium", "cpu", {})

        self.assertEqual(result.transcript.text, "caps text")
        self.assertEqual(result.providers_run, ["remote_http", "vibevoice", "capswriter_http"])
        processor.assert_not_called()
        audio_path.unlink()

    def test_balanced_weak_fast_transcript_falls_back_when_vibevoice_fails(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        weak = AudioTranscript(text="嗯嗯", segments=[{"start": 0.0, "end": 1.0, "text": "嗯嗯"}], language="zh")
        caps = AudioTranscript(text="caps repaired text", segments=[{"start": 0.0, "end": 1.0, "text": "caps repaired text"}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=weak), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice", return_value=None
        ), patch("video_analyzer.asr_providers.transcribe_with_capswriter", return_value=caps):
            result = transcribe_with_strategy("balanced", audio_path, "", "medium", "cpu", {})

        self.assertEqual(result.transcript.text, "caps repaired text")
        self.assertEqual(result.providers_run, ["remote_http", "vibevoice", "capswriter_http"])
        audio_path.unlink()

    def test_balanced_long_audio_runs_vibevoice(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000 * 181)
        fast = AudioTranscript(text="fast text", segments=[{"start": 0.0, "end": 181.0, "text": "fast text"}], language="zh")
        deep = AudioTranscript(text="deep text。", segments=[{"raw_output": "deep text"}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=fast), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice", return_value=deep
        ):
            result = transcribe_with_strategy("balanced", audio_path, "", "medium", "cpu", {})

        self.assertEqual(result.providers_run, ["remote_http", "vibevoice"])
        self.assertEqual(result.transcript.text, "deep text。")
        audio_path.unlink()

    def test_balanced_short_weak_fast_transcript_runs_vibevoice(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        weak = AudioTranscript(text="嗯嗯", segments=[{"start": 0.0, "end": 1.0, "text": "嗯嗯"}], language="zh")
        deep = AudioTranscript(text="真实讲解内容。", segments=[{"raw_output": "真实讲解内容"}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=weak), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice", return_value=deep
        ):
            result = transcribe_with_strategy("balanced", audio_path, "", "medium", "cpu", {})

        self.assertEqual(result.providers_run, ["remote_http", "vibevoice"])
        self.assertEqual(result.transcript.text, "真实讲解内容。")
        audio_path.unlink()

    def test_screen_keyframes_keep_static_and_changed_ui(self):
        if cv2 is None or VideoProcessor is None:
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
        if VideoProcessor is None:
            self.skipTest("opencv-python is not installed")
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
