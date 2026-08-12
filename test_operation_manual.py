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
from video_analyzer.cli import create_operation_manual_text_client
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
    default_capswriter_url,
    default_vibevoice_urls,
    merge_asr_transcripts,
    transcribe_with_provider,
    transcribe_with_provider_result,
    transcribe_with_remote_http,
    transcribe_with_vibevoice_remote,
    transcribe_with_strategy,
    transcribe_with_vibevoice,
)
from video_analyzer.ocr import DOTS_MOCR_ENDPOINTS, DotsMOCRVLLMProvider, default_ocr_endpoints, run_ocr
from video_analyzer.audio_processor import AudioTranscript
from video_analyzer.doc_chat import ask_video_docs, build_doc_chat_prompt, load_video_docs
from video_analyzer.frame import Frame
from video_analyzer.frame_selection import (
    FrameSelectionOptions,
    build_frame_context_window,
    resolve_candidate_frame_budget,
    resolve_vl_context_gap_seconds,
    resolve_vl_frame_budget,
    select_vl_frames,
)
from video_analyzer.jetson_frames import resolve_jetson_sample_fps, split_jetson_workers
from video_analyzer.ocr import OCREvent

RUNNER_PATH = Path(__file__).resolve().parent / "tools" / "run_operation_manual_from_url.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_operation_manual_from_url", RUNNER_PATH)
run_operation_manual_from_url = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER_SPEC.loader.exec_module(run_operation_manual_from_url)

MULTIDOC_PATH = Path(__file__).resolve().parent / "tools" / "run_multidoc_analysis.py"
MULTIDOC_SPEC = importlib.util.spec_from_file_location("run_multidoc_analysis", MULTIDOC_PATH)
run_multidoc_analysis = importlib.util.module_from_spec(MULTIDOC_SPEC)
assert MULTIDOC_SPEC and MULTIDOC_SPEC.loader
MULTIDOC_SPEC.loader.exec_module(run_multidoc_analysis)

NARRATION_PATH = Path(__file__).resolve().parent / "tools" / "generate_audio_narration.py"
NARRATION_SPEC = importlib.util.spec_from_file_location("generate_audio_narration", NARRATION_PATH)
generate_audio_narration = importlib.util.module_from_spec(NARRATION_SPEC)
assert NARRATION_SPEC and NARRATION_SPEC.loader
NARRATION_SPEC.loader.exec_module(generate_audio_narration)

try:
    import cv2
    import numpy as np
    from video_analyzer.frame import VideoProcessor
except ModuleNotFoundError:
    cv2 = None
    np = None
    VideoProcessor = None


class OperationManualTests(unittest.TestCase):
    def test_candidate_frames_auto_scales_with_duration(self):
        short_budget = resolve_candidate_frame_budget(
            video_duration_seconds=4 * 60,
            pipeline_mode="balanced",
            candidate_frames="auto",
        )
        long_budget = resolve_candidate_frame_budget(
            video_duration_seconds=60 * 60,
            pipeline_mode="balanced",
            candidate_frames="auto",
        )

        self.assertGreater(long_budget, short_budget * 10)
        self.assertGreaterEqual(long_budget, 300)

    def test_balanced_vl_budget_scales_with_duration_and_density(self):
        short_frames = [Frame(i, Path(f"frame_{i}.jpg"), i * 10.0, 5.0) for i in range(24)]
        long_frames = [Frame(i, Path(f"frame_{i}.jpg"), i * 12.0, 5.0) for i in range(300)]
        low_ocr = [
            OCREvent(i, frame.timestamp, "test", "ok", "ok", [])
            for i, frame in enumerate(short_frames)
        ]
        high_ocr = [
            OCREvent(i, frame.timestamp, "test", "ok", "按钮 设置 命令 参数 文件名 " * 20, [])
            for i, frame in enumerate(long_frames)
        ]
        transcript = AudioTranscript(
            text="dense transcript",
            segments=[{"start": i * 15, "end": i * 15 + 5, "text": "step"} for i in range(120)],
            language="zh",
        )
        options = FrameSelectionOptions(pipeline_mode="balanced")

        short_budget = resolve_vl_frame_budget(short_frames, low_ocr, None, 4 * 60, options)
        long_budget = resolve_vl_frame_budget(long_frames, high_ocr, transcript, 60 * 60, options)

        self.assertNotEqual(short_budget, 12)
        self.assertGreater(long_budget, short_budget)
        self.assertGreater(long_budget, 100)

    def test_pipeline_modes_control_vl_selection(self):
        frames = [Frame(i, Path(f"frame_{i}.jpg"), i * 10.0, 10.0) for i in range(20)]
        ocr_events = [OCREvent(i, frame.timestamp, "test", "ok", "按钮 " * 30, []) for i, frame in enumerate(frames)]

        fast_selected, fast_decisions, fast_meta = select_vl_frames(
            frames,
            ocr_events,
            None,
            4 * 60,
            FrameSelectionOptions(pipeline_mode="fast"),
        )
        deep_selected, deep_decisions, deep_meta = select_vl_frames(
            frames,
            ocr_events,
            None,
            4 * 60,
            FrameSelectionOptions(pipeline_mode="deep"),
        )

        self.assertEqual(fast_selected, set())
        self.assertTrue(all(not decision.selected_for_vl for decision in fast_decisions))
        self.assertEqual(fast_meta["vl_frames_count"], 0)
        self.assertEqual(deep_selected, {frame.number for frame in frames})
        self.assertTrue(all(decision.selected_for_vl for decision in deep_decisions))
        self.assertEqual(deep_meta["vl_frames_count"], len(frames))

    def test_jetson_frame_workers_split_with_overlap(self):
        workers = split_jetson_workers(
            hosts=["nx2", "nx3"],
            video_duration_seconds=100.0,
            output_dir=Path("/tmp/frames"),
            overlap_seconds=2.0,
        )

        self.assertEqual([worker.host for worker in workers], ["nx2", "nx3"])
        self.assertEqual(workers[0].start_seconds, 0.0)
        self.assertAlmostEqual(workers[0].duration_seconds, 52.0)
        self.assertAlmostEqual(workers[1].start_seconds, 48.0)
        self.assertAlmostEqual(workers[1].duration_seconds, 52.0)

    def test_jetson_sample_fps_defaults_by_pipeline_mode(self):
        self.assertEqual(resolve_jetson_sample_fps("auto", "fast"), 1.0)
        self.assertEqual(resolve_jetson_sample_fps("auto", "balanced"), 2.0)
        self.assertEqual(resolve_jetson_sample_fps("auto", "deep"), 3.0)
        self.assertEqual(resolve_jetson_sample_fps("0.5", "fast"), 0.5)

    def test_operation_manual_config_uses_active_local_vision_defaults(self):
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
            vision_base_url=None,
            text_base_url=None,
            vision_model=None,
            text_model=None,
            ocr_provider=None,
            ocr_base_url=None,
            ocr_concurrency=None,
            ocr_cache=None,
            ocr_cache_dir=None,
            asr_provider=None,
            asr_strategy=None,
            remote_asr_url=None,
            vibevoice_url=None,
            context_file=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(temp_dir)
            config.update_from_args(args)

            self.assertEqual(config.get("clients")["default"], "openai_api")
            self.assertEqual(config.get("clients")["openai_api"]["api_key"], "0")
            self.assertEqual(
                config.get("clients")["openai_api"]["api_url"],
                "http://100.96.79.21:18082/v1",
            )
            self.assertEqual(
                config.get("clients")["openai_api"]["model"],
                "minicpm-v-4.5-v100",
            )
            self.assertEqual(
                config.get("operation_manual")["text_base_url"],
                "https://api.deepseek.com",
            )
            self.assertEqual(
                config.get("operation_manual")["text_model"],
                "deepseek-v4-flash",
            )
            self.assertEqual(config.get("asr")["provider"], "auto")
            self.assertEqual(config.get("asr")["strategy"], "balanced")
            self.assertEqual(config.get("asr")["vibevoice"]["remote_urls"], [])
            self.assertEqual(
                config.get("asr")["vibevoice"]["deep_remote_urls"],
                [
                    "http://edge.taild500c8.ts.net:8012/api/asr/transcribe",
                    "http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe",
                ],
            )
            self.assertTrue(config.get("asr")["vibevoice"]["use_native_chunking"])
            self.assertEqual(config.get("asr")["vibevoice"]["chunk_parallel_workers"], 2)
            self.assertEqual(
                config.get("asr")["vibevoice"]["capswriter_url"],
                "http://spark-31d6.taild500c8.ts.net:8001/api/asr/transcribe",
            )
            self.assertEqual(
                config.get("ocr")["fallback_model"],
                "hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive",
            )
            self.assertEqual(
                config.get("ocr")["base_urls"],
                [
                    "http://spark-31d6.taild500c8.ts.net:8000/v1",
                    "http://edge.taild500c8.ts.net:8000/v1",
                ],
            )
            self.assertEqual(config.get("ocr")["concurrency"], "auto")
            self.assertEqual(config.get("ocr")["cache"], "on")

    def test_runtime_profile_merges_user_config_and_allows_cli_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "config.json").write_text(
                json.dumps(
                    {
                        "active_runtime_profile": "lab",
                        "runtime_profiles": {
                            "lab": {
                                "llm_base_url": "http://lab.local/v1",
                                "text_model": "lab-text",
                                "vibevoice_url": "http://lab.local/asr",
                                "ocr_base_url": "http://lab.local/ocr",
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config = Config(str(config_dir))
            profile = config.get_runtime_profile()

            self.assertEqual(profile["llm_base_url"], "http://lab.local/v1")
            self.assertEqual(profile["text_model"], "lab-text")
            self.assertEqual(profile["vibevoice_url"], "http://lab.local/asr")
            self.assertEqual(profile["ocr_base_url"], "http://lab.local/ocr")
            self.assertIn("spark", config.get("runtime_profiles"))
            self.assertIn("deepseek_v4_flash", config.get("runtime_profiles"))
            self.assertNotIn("local_lan", config.get("runtime_profiles"))

            args = argparse.Namespace(
                config=str(config_dir),
                profile=None,
                output_root=None,
                run_name=None,
                max_frames=None,
                pipeline_mode=None,
                candidate_frames=None,
                min_vl_frames=None,
                max_vl_frames=None,
                vl_frame_policy=None,
                vl_concurrency=None,
                vl_context_before=None,
                vl_context_after=None,
                vl_context_max_gap=None,
                manual_language=None,
                vibevoice_url=None,
                ocr_base_url=None,
                ocr_concurrency=None,
                ocr_cache=None,
                ocr_cache_dir=None,
                llm_base_url=None,
                vision_base_url=None,
                text_base_url=None,
                vision_model=None,
                text_model=None,
                include_subtitles=None,
                include_comments=None,
                max_comments=None,
                subtitle_langs=None,
                ytdlp_js_runtimes=None,
            )

            run_operation_manual_from_url.apply_runtime_profile(args)

            self.assertEqual(args.llm_base_url, "http://lab.local/v1")
            self.assertEqual(args.text_model, "lab-text")
            self.assertEqual(args.vibevoice_url, "http://lab.local/asr")
            self.assertEqual(args.ocr_base_url, "http://lab.local/ocr")
            self.assertEqual(args.output_root, "downloads/url-videos")
            self.assertEqual(args.ytdlp_js_runtimes, "auto")

    def test_operation_profile_updates_asr_deep_remote_urls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "config.json").write_text(
                json.dumps(
                    {
                        "active_runtime_profile": "lab",
                        "runtime_profiles": {
                            "lab": {
                                "llm_base_url": "http://lab.local/v1",
                                "text_model": "lab-text",
                                "vibevoice_urls": ["http://127.0.0.1:18012/api/asr/transcribe"],
                                "ocr_base_url": "http://lab.local/ocr",
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(task="operation_manual", profile=None, client=None, asr_provider=None)

            config = Config(str(config_dir))
            config.update_from_args(args)

            self.assertEqual(
                config.get("asr")["vibevoice"]["deep_remote_urls"],
                ["http://127.0.0.1:18012/api/asr/transcribe"],
            )

    def test_endpoint_host_override_updates_runtime_services(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "config.json").write_text(
                json.dumps(
                    {
                        "endpoints": {
                            "hosts": {
                                "edge": "edge-new.taild500c8.ts.net",
                                "spark": "spark-new.taild500c8.ts.net",
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config = Config(str(config_dir))
            profile = config.get_runtime_profile("deepseek_v4_pro")

            self.assertEqual(
                profile["vibevoice_urls"],
                [
                    "http://edge-new.taild500c8.ts.net:8012/api/asr/transcribe",
                    "http://spark-new.taild500c8.ts.net:8012/api/asr/transcribe",
                ],
            )
            self.assertEqual(
                profile["ocr_base_urls"],
                [
                    "http://spark-new.taild500c8.ts.net:8000/v1",
                    "http://edge-new.taild500c8.ts.net:8000/v1",
                ],
            )
            self.assertEqual(
                config.get("asr")["vibevoice"]["deep_remote_urls"],
                profile["vibevoice_urls"],
            )
            self.assertEqual(
                default_vibevoice_urls(config.config),
                [
                    "http://edge-new.taild500c8.ts.net:8012/api/asr/transcribe",
                    "http://spark-new.taild500c8.ts.net:8012/api/asr/transcribe",
                ],
            )
            self.assertEqual(
                default_ocr_endpoints(config.config),
                [
                    "http://spark-new.taild500c8.ts.net:8000/v1",
                    "http://edge-new.taild500c8.ts.net:8000/v1",
                ],
            )
            self.assertEqual(
                default_capswriter_url(config.config),
                "http://spark-new.taild500c8.ts.net:8001/api/asr/transcribe",
            )

    def test_unknown_endpoint_placeholder_fails_fast(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "config.json").write_text(
                json.dumps(
                    {
                        "runtime_profiles": {
                            "broken": {
                                "llm_base_url": "http://{missing_host}:1234/v1",
                                "vibevoice_url": "http://lab.local/asr",
                                "ocr_base_url": "http://lab.local/ocr",
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing_host"):
                Config(str(config_dir))

    def test_url_runner_adds_node_js_runtime_for_youtube_challenges(self):
        args = argparse.Namespace(ytdlp_js_runtimes="auto")
        command = ["yt-dlp", "--dump-single-json", "https://example.test/video"]

        with patch.object(run_operation_manual_from_url.shutil, "which", return_value="/usr/bin/node"):
            run_operation_manual_from_url.add_ytdlp_runtime_args(command, args)

        self.assertIn("--js-runtimes", command)
        self.assertIn("node", command)

    def test_url_runner_can_disable_ytdlp_js_runtime(self):
        args = argparse.Namespace(ytdlp_js_runtimes="none")
        command = ["yt-dlp", "--dump-single-json", "https://example.test/video"]

        run_operation_manual_from_url.add_ytdlp_runtime_args(command, args)

        self.assertNotIn("--js-runtimes", command)

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
            self.assertTrue((video_dir / "selected_comments.json").exists())
            self.assertEqual(bundle["metadata"]["subtitles"]["language"], "zh-CN")
            self.assertEqual(bundle["metadata"]["comments"]["selected_count"], 1)
            self.assertEqual(
                len(json.loads((video_dir / "comments.json").read_text(encoding="utf-8"))),
                2,
            )
            self.assertEqual(
                len(json.loads((video_dir / "selected_comments.json").read_text(encoding="utf-8"))),
                1,
            )

    def test_cli_writes_timestamped_transcript_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript = AudioTranscript(
                text="hello world",
                segments=[{"start_time": 1.2, "end_time": 3.8, "text": "hello world"}],
                language="en",
            )

            path = Path(temp_dir) / "transcript.md"
            from video_analyzer.cli import write_transcript_markdown

            write_transcript_markdown(transcript, path)

            text = path.read_text(encoding="utf-8")
            self.assertIn("# Transcript", text)
            self.assertIn("[00:00:01 - 00:00:03] hello world", text)

    def test_cli_archives_raw_artifacts_under_orin(self):
        from video_analyzer.cli import write_orin_artifacts

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            results = {
                "metadata": {"audio_language": "zh", "page_context": {}},
                "transcript": {
                    "text": "原文",
                    "segments": [{"start_time": 0.0, "end_time": 2.0, "text": "原文"}],
                },
                "asr": {"provider": "vibevoice"},
                "ocr_events": [{"frame_number": 0, "text": "按钮"}],
                "visual_events": [{"response": "shows button"}],
                "frame_analyses": [{"response": "shows button"}],
            }

            orin_dir = write_orin_artifacts(output_dir, results, "page context")

            self.assertEqual(orin_dir, output_dir / "orin")
            self.assertTrue((orin_dir / "metadata.json").exists())
            self.assertTrue((orin_dir / "transcript.md").exists())
            self.assertTrue((orin_dir / "asr.json").exists())
            self.assertTrue((orin_dir / "ocr_event_000.json").exists())
            self.assertTrue((orin_dir / "frame_analysis_000.json").exists())
            self.assertIn("page context", (orin_dir / "page_context.md").read_text(encoding="utf-8"))

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

    def test_multidoc_analysis_requires_existing_analysis_and_orin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                run_multidoc_analysis.run_multidoc_analysis(Path(temp_dir))

            run_dir = Path(temp_dir)
            (run_dir / "analysis.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                run_multidoc_analysis.run_multidoc_analysis(run_dir)

    def test_video_docs_chat_loads_sources_and_builds_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "orin").mkdir()
            (run_dir / "operation_manual.md").write_text("# 手册\n\n点击按钮。", encoding="utf-8")
            (run_dir / "transcript.md").write_text("- [00:00:01 - 00:00:03] 作者说明按钮用途", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("frame_012 OCR: 设置", encoding="utf-8")
            (run_dir / "orin" / "comments.md").write_text("普通评论：可能旧版本不同", encoding="utf-8")

            bundle = load_video_docs(run_dir)
            prompt = build_doc_chat_prompt(bundle, "按钮在哪里？", [])

            self.assertEqual(len(bundle["sources"]), 4)
            self.assertIn("manual_evidence", prompt)
            self.assertIn("comments/comment-only", prompt)
            self.assertIn("按钮在哪里？", prompt)

    def test_video_docs_chat_loads_analysis_json_without_markdown_sidecars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "metadata": {"title": "Demo"},
                        "transcript": {"text": "按钮在右上角"},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            bundle = load_video_docs(run_dir)
            prompt = build_doc_chat_prompt(bundle, "按钮在哪里？", [])

            self.assertEqual([source["name"] for source in bundle["sources"]], ["analysis_json"])
            self.assertIn("## Source: analysis_json", prompt)
            self.assertIn("analysis.json 是结构化聚合产物", prompt)
            self.assertIn("Path: ", prompt)

    def test_video_docs_chat_asks_with_mock_client(self):
        class FakeClient:
            def generate(self, **kwargs):
                self.prompt = kwargs["prompt"]
                return {"response": "根据 transcript.md，答案需要复核。"}

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "operation_manual.md").write_text("主步骤", encoding="utf-8")
            client = FakeClient()

            answer = ask_video_docs(run_dir, "怎么操作？", client, "model")

            self.assertIn("需要复核", answer)
            self.assertIn("怎么操作？", client.prompt)

    def test_start_example_runs_followup_steps(self):
        text = Path("start_example.sh").read_text(encoding="utf-8")

        self.assertIn("tools/run_multidoc_analysis.sh \"$RUN_DIR\" --profile \"$PROFILE\"", text)
        self.assertIn("tools/generate_audio_narration.sh \"$RUN_DIR\" --profile \"$PROFILE\"", text)
        self.assertIn("[done] run_dir: ", text)
        self.assertIn("Using canonical Bilibili URL", text)
        self.assertIn("quote full share URLs that contain &", text)
        self.assertNotIn("Could not infer Bilibili BV id", text)
        self.assertNotIn("VIDEO_ID=", text)
        self.assertNotIn("vd_source=", text)

    def test_audio_narration_resolves_default_and_pdf_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "operation_manual.md").write_text("默认手册", encoding="utf-8")
            chapters = run_dir / "docs_analysis_chapters"
            chapters.mkdir()
            notes = chapters / "knowledge_notes_v2.md"
            notes.write_text("知识笔记", encoding="utf-8")

            self.assertEqual(generate_audio_narration.resolve_source(run_dir, None), run_dir / "operation_manual.md")
            self.assertEqual(generate_audio_narration.resolve_source(run_dir, "knowledge_notes_v2.pdf"), notes)

    def test_start_example_parses_run_dir_marker_from_url_runner_output(self):
        log_text = "\n".join(
            [
                "[download] ready",
                "[done] manual: /tmp/video/operation_manual.md",
                "[done] run_dir: /tmp/video/operation-manual",
            ]
        )

        run_dir = next(
            line.split(": ", 1)[1].strip()
            for line in reversed(log_text.splitlines())
            if line.startswith("[done] run_dir: ")
        )

        self.assertEqual(run_dir, "/tmp/video/operation-manual")

    def test_multidoc_analysis_runs_four_rounds_with_mock_llm(self):
        class FakeClient:
            def __init__(self):
                self.prompts = []

            def generate(self, **kwargs):
                self.prompts.append(kwargs["prompt"])
                return {"response": f"# response {len(self.prompts)}"}

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            orin = run_dir / "orin"
            orin.mkdir()
            (run_dir / "operation_manual.md").write_text("original manual", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("manual evidence", encoding="utf-8")
            (orin / "page_context.md").write_text(
                "- 00:00:00 - 00:01:00: 第一章\n评论只能补充",
                encoding="utf-8",
            )
            (orin / "transcript.md").write_text("- [00:00:00 - 00:00:05] 开场", encoding="utf-8")
            (orin / "transcript.json").write_text(
                json.dumps({"text": "开场", "segments": [{"start_time": 0, "end_time": 5, "text": "开场"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (orin / "ocr_events.json").write_text(json.dumps([{"text": "按钮"}], ensure_ascii=False), encoding="utf-8")
            (orin / "frame_analyses.json").write_text(
                json.dumps([{"response": "shows button"}], ensure_ascii=False),
                encoding="utf-8",
            )
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "metadata": {"text_model": "text-model", "audio_language": "zh"},
                        "transcript": {"text": "开场", "segments": [{"start_time": 0, "end_time": 5, "text": "开场"}]},
                        "ocr_events": [{"text": "按钮"}],
                        "frame_analyses": [{"response": "shows button"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            client = FakeClient()

            summary = run_multidoc_analysis.run_multidoc_analysis(run_dir, client=client)

            output_dir = run_dir / "docs_analysis"
            self.assertEqual(len(client.prompts), 6)
            self.assertTrue((output_dir / "knowledge_notes.md").exists())
            self.assertTrue((output_dir / "deep_report.md").exists())
            self.assertTrue((output_dir / "operation_manual_review.md").exists())
            self.assertTrue((output_dir / "analysis.json").exists())
            self.assertTrue((output_dir / "orin" / "round_01_evidence_map.md").exists())
            self.assertTrue((output_dir / "orin" / "round_02_chapter_analysis.md").exists())
            self.assertTrue((output_dir / "orin" / "round_03_knowledge_notes_draft.md").exists())
            self.assertTrue((output_dir / "orin" / "round_04_review.md").exists())
            self.assertEqual((run_dir / "operation_manual.md").read_text(encoding="utf-8"), "original manual")
            self.assertIn("docs_analysis/orin", summary["orin_dir"])

    def test_multidoc_prompt_keeps_comments_low_trust(self):
        evidence = {
            "page_context": "context",
            "transcript_md": "transcript",
            "ocr_events": [],
            "frame_analyses": [],
        }

        prompt = run_multidoc_analysis.build_evidence_map_prompt(evidence, "zh-CN")

        self.assertIn("评论只能作为社区补充", prompt)
        self.assertIn("OCR/VL", prompt)

    def test_url_runner_defaults_to_spark_services(self):
        args = argparse.Namespace(
            python=".venv/bin/python",
            vibevoice_url="http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe",
            ocr_base_url=[
                "http://spark-31d6.taild500c8.ts.net:8000/v1",
                "http://edge.taild500c8.ts.net:8000/v1",
            ],
            ocr_concurrency="auto",
            ocr_cache="on",
            ocr_cache_dir=".cache/video-analyzer/ocr",
            llm_base_url="http://spark-31d6.taild500c8.ts.net:1234/v1",
            vision_base_url="http://100.96.79.21:18082/v1",
            text_base_url="http://100.90.114.26:18081/v1",
            vision_model="qwen/qwen3-vl-30b",
            text_model="redhatai_qwen3.6-35b-a3b-nvfp4",
            manual_language="zh-CN",
            max_frames=24,
            pipeline_mode="balanced",
            candidate_frames="auto",
            min_vl_frames="auto",
            max_vl_frames="auto",
            vl_frame_policy="auto",
            vl_concurrency=2,
            vl_context_before=3,
            vl_context_after=2,
            vl_context_max_gap="auto",
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
        self.assertIn("http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe", command)
        self.assertIn("http://spark-31d6.taild500c8.ts.net:8000/v1", command)
        self.assertIn("http://edge.taild500c8.ts.net:8000/v1", command)
        self.assertIn("--vision-base-url", command)
        self.assertIn("http://100.96.79.21:18082/v1", command)
        self.assertIn("--text-base-url", command)
        self.assertIn("http://100.90.114.26:18081/v1", command)
        self.assertEqual(command.count("--ocr-base-url"), 2)
        self.assertIn("--ocr-cache", command)
        self.assertNotIn("--ocr-timeout-seconds", command)

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
    "openai_api": {"api_key": "0", "api_url": "http://spark-31d6.taild500c8.ts.net:1234/v1", "model": "model"},
    "ollama": {"url": "http://localhost:11434", "model": "model"},
    "temperature": 0.0
  },
  "operation_manual": {"llm_base_url": "http://spark-31d6.taild500c8.ts.net:1234/v1", "vision_model": "vision", "text_model": "text"},
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
                ocr_concurrency=None,
                ocr_cache=None,
                ocr_cache_dir=None,
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
            ocr_concurrency=None,
            ocr_cache=None,
            ocr_cache_dir=None,
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
        client = GenericOpenAIAPIClient("0", "http://spark-31d6.taild500c8.ts.net:1234/v1", max_retries=1)
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

        with patch.object(client.session, "post", return_value=response):
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

        with patch.object(client.session, "post", return_value=response):
            with self.assertRaises(Exception) as raised:
                client.generate(prompt="describe", model="vision")

        self.assertIn("reasoning_content fallback is only allowed", str(raised.exception))

    def test_openai_client_sends_multiple_images_in_order(self):
        client = GenericOpenAIAPIClient("0", "http://spark-31d6.taild500c8.ts.net:1234/v1", max_retries=1)
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        with tempfile.TemporaryDirectory() as temp_dir:
            image_a = Path(temp_dir) / "a.jpg"
            image_b = Path(temp_dir) / "b.jpg"
            image_a.write_bytes(b"a")
            image_b.write_bytes(b"b")
            with patch.object(client.session, "post", return_value=response) as post:
                client.generate(prompt="describe", image_paths=[str(image_a), str(image_b)], model="vision")

        content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["text", "image_url", "image_url"])
        self.assertIn("YQ==", content[1]["image_url"]["url"])
        self.assertIn("Yg==", content[2]["image_url"]["url"])

    def test_operation_manual_text_client_can_use_separate_base_url(self):
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            "clients": {
                "default": "openai_api",
                "openai_api": {
                    "api_key": "0",
                    "api_url": "http://vision.test/v1",
                    "timeout_seconds": 123,
                },
            },
            "operation_manual": {
                "llm_base_url": "http://legacy.test/v1",
                "vision_base_url": "http://vision.test/v1",
                "text_base_url": "http://text.test/v1",
            },
        }.get(key, default)
        fallback_client = Mock()

        text_client = create_operation_manual_text_client(config, fallback_client)

        self.assertIsInstance(text_client, GenericOpenAIAPIClient)
        self.assertEqual(text_client.base_url, "http://text.test/v1")
        self.assertEqual(text_client.timeout_seconds, 123)

    def test_frame_analysis_can_force_no_think_and_larger_token_budget(self):
        client = Mock()
        client.generate.return_value = {"response": "frame ok"}
        prompt_loader = Mock()
        prompt_loader.get_by_index.side_effect = [
            "Frame prompt. {PREVIOUS_FRAMES} {prompt}",
            "Video prompt.",
        ]
        analyzer = VideoAnalyzer(
            client=client,
            model="vision",
            prompt_loader=prompt_loader,
            temperature=0.0,
            frame_num_predict=1200,
            frame_no_think=True,
        )
        frame = Mock(number=1, timestamp=2.5, path=Path("/tmp/frame.jpg"))

        analyzer.analyze_frame(frame, ocr_text="Button: Start")

        kwargs = client.generate.call_args.kwargs
        self.assertTrue(kwargs["prompt"].startswith("/no_think\n"))
        self.assertIn("OCR evidence", kwargs["prompt"])
        self.assertEqual(kwargs["num_predict"], 1200)

    def test_dots_mocr_default_endpoints_use_spark_and_edge_tailscale(self):
        self.assertEqual(
            DOTS_MOCR_ENDPOINTS,
            [
                "http://spark-31d6.taild500c8.ts.net:8000/v1",
                "http://edge.taild500c8.ts.net:8000/v1",
            ],
        )

    def test_asr_default_endpoints_use_spark_edge_tailscale(self):
        self.assertEqual(REMOTE_ASR_URLS, ["http://spark-31d6.taild500c8.ts.net:8001/api/asr/transcribe"])
        self.assertEqual(
            REMOTE_VIBEVOICE_URLS,
            [
                "http://edge.taild500c8.ts.net:8012/api/asr/transcribe",
                "http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe",
            ],
        )
        self.assertTrue(all("spark-31d6.taild500c8.ts.net" in url for url in REMOTE_ASR_URLS))

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
            events = run_ocr(
                frames,
                "auto",
                "http://ocr.test/v1",
                "model",
                "prompt_scene_spotting",
                warmup_timeout_seconds=0,
            )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].status, "unavailable")
        self.assertEqual(get.call_count, 1)

    def test_run_ocr_uses_cache_without_probe_when_all_frames_hit(self):
        frame_path = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
        frame_path.write_bytes(b"cached-image")
        frames = [Mock(number=0, timestamp=0.0, path=frame_path)]
        ready = Mock()
        ready.raise_for_status.return_value = None
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": '[{"text":"cached text"}]'}}]}

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("video_analyzer.ocr.requests.get", return_value=ready), patch(
                "video_analyzer.ocr.requests.post", return_value=response
            ) as post:
                first = run_ocr(
                    frames,
                    "auto",
                    "http://ocr.test/v1",
                    "model",
                    "prompt_scene_spotting",
                    cache_dir=temp_dir,
                )
            with patch("video_analyzer.ocr.requests.get", side_effect=RuntimeError("offline")) as get, patch(
                "video_analyzer.ocr.requests.post"
            ) as post_again:
                second = run_ocr(
                    frames,
                    "auto",
                    "http://ocr.test/v1",
                    "model",
                    "prompt_scene_spotting",
                    cache_dir=temp_dir,
                )

        self.assertEqual(first[0].cache_status, "miss")
        self.assertEqual(second[0].cache_status, "hit")
        self.assertEqual(second[0].text, "cached text")
        self.assertEqual(post.call_count, 1)
        get.assert_not_called()
        post_again.assert_not_called()
        frame_path.unlink()

    def test_run_ocr_refresh_ignores_cache_and_rewrites(self):
        frame_path = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
        frame_path.write_bytes(b"refresh-image")
        frames = [Mock(number=0, timestamp=0.0, path=frame_path)]
        ready = Mock()
        ready.raise_for_status.return_value = None
        first_response = Mock()
        first_response.raise_for_status.return_value = None
        first_response.json.return_value = {"choices": [{"message": {"content": '[{"text":"old"}]'}}]}
        second_response = Mock()
        second_response.raise_for_status.return_value = None
        second_response.json.return_value = {"choices": [{"message": {"content": '[{"text":"new"}]'}}]}

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("video_analyzer.ocr.requests.get", return_value=ready), patch(
                "video_analyzer.ocr.requests.post", side_effect=[first_response, second_response]
            ) as post:
                run_ocr(
                    frames,
                    "auto",
                    "http://ocr.test/v1",
                    "model",
                    "prompt_scene_spotting",
                    cache_dir=temp_dir,
                )
                refreshed = run_ocr(
                    frames,
                    "auto",
                    "http://ocr.test/v1",
                    "model",
                    "prompt_scene_spotting",
                    cache_mode="refresh",
                    cache_dir=temp_dir,
                )

        self.assertEqual(refreshed[0].cache_status, "refresh")
        self.assertEqual(refreshed[0].text, "new")
        self.assertEqual(post.call_count, 2)
        frame_path.unlink()

    def test_run_ocr_distributes_frames_across_multiple_dots_endpoints(self):
        frame_paths = []
        frames = []
        for index in range(4):
            frame_path = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
            frame_path.write_bytes(f"image-{index}".encode())
            frame_paths.append(frame_path)
            frames.append(Mock(number=index, timestamp=float(index), path=frame_path))

        ready = Mock()
        ready.raise_for_status.return_value = None

        def fake_post(url, **kwargs):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "choices": [{"message": {"content": f'[{{"text":"{url}"}}]'}}]
            }
            return response

        with patch("video_analyzer.ocr.requests.get", return_value=ready), patch(
            "video_analyzer.ocr.requests.post", side_effect=fake_post
        ) as post:
            events = run_ocr(
                frames,
                "auto",
                "auto",
                "model",
                "prompt_scene_spotting",
                base_urls=["http://spark-ocr/v1", "http://edge-ocr/v1"],
                cache_mode="off",
            )

        providers = [event.provider for event in events]
        self.assertEqual([event.frame_number for event in events], [0, 1, 2, 3])
        self.assertEqual(sum("spark-ocr" in provider for provider in providers), 2)
        self.assertEqual(sum("edge-ocr" in provider for provider in providers), 2)
        self.assertEqual(post.call_count, 4)
        for frame_path in frame_paths:
            frame_path.unlink()

    def test_run_ocr_uses_only_healthy_dots_endpoint(self):
        frame_paths = []
        frames = []
        for index in range(2):
            frame_path = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
            frame_path.write_bytes(f"image-{index}".encode())
            frame_paths.append(frame_path)
            frames.append(Mock(number=index, timestamp=float(index), path=frame_path))

        ready = Mock()
        ready.raise_for_status.return_value = None

        def fake_get(url, **kwargs):
            if "edge-ocr" in url:
                raise RuntimeError("offline")
            return ready

        def fake_post(url, **kwargs):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"choices": [{"message": {"content": '[{"text":"ok"}]'}}]}
            return response

        with patch("video_analyzer.ocr.requests.get", side_effect=fake_get), patch(
            "video_analyzer.ocr.requests.post", side_effect=fake_post
        ):
            events = run_ocr(
                frames,
                "auto",
                "auto",
                "model",
                "prompt_scene_spotting",
                base_urls=["http://spark-ocr/v1", "http://edge-ocr/v1"],
                warmup_timeout_seconds=0,
                cache_mode="off",
            )

        self.assertTrue(all("spark-ocr" in event.provider for event in events))
        self.assertTrue(all(event.status == "ok" for event in events))
        for frame_path in frame_paths:
            frame_path.unlink()

    def test_dots_mocr_probe_waits_for_cold_start(self):
        first = RuntimeError("cold start")
        second = RuntimeError("still loading")
        ready = Mock()
        ready.raise_for_status.return_value = None
        provider = DotsMOCRVLLMProvider(
            base_url="http://ocr.test/v1",
            probe_timeout_seconds=5,
            warmup_timeout_seconds=180,
            warmup_retry_interval_seconds=5,
        )

        with patch("video_analyzer.ocr.requests.get", side_effect=[first, second, ready]) as get, patch(
            "video_analyzer.ocr.time.monotonic", side_effect=[0, 5, 10, 15, 110]
        ), patch("video_analyzer.ocr.time.sleep") as sleep:
            selected = provider.probe()

        self.assertEqual(selected, "http://ocr.test/v1")
        self.assertEqual(get.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

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
                fallback_base_url="http://spark-31d6.taild500c8.ts.net:1234/v1",
                fallback_model="qwen/qwen3-vl-30b",
                warmup_timeout_seconds=0,
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

    def test_step_images_replace_model_images_outside_step_bucket(self):
        frames = [Mock(number=i, timestamp=float(i * 10)) for i in range(10)]
        assets = {i: f"manual_assets/frame_{i:03d}.jpg" for i in range(10)}
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：前半段",
                "| 说明 | 图 |",
                "| --- | --- |",
                "| 错图 | ![90s / Frame 9](manual_assets/frame_009.jpg) |",
                "操作说明。",
                "### 步骤 2：后半段",
                "操作说明。",
            ]
        )

        updated = embed_step_images(manual, frames, assets)

        first_step = updated.split("### 步骤 2")[0]
        self.assertNotIn("manual_assets/frame_009.jpg", first_step)
        self.assertIn("| 说明 | 图 |", first_step)
        self.assertIn("| 错图 |", first_step)
        self.assertIn("manual_assets/frame_000.jpg", first_step)
        self.assertIn("manual_assets/frame_004.jpg", first_step)

    def test_step_images_use_plain_time_range_to_select_frames(self):
        frames = [
            Mock(number=0, timestamp=90.0),
            Mock(number=1, timestamp=100.0),
            Mock(number=2, timestamp=160.0),
            Mock(number=3, timestamp=266.0),
            Mock(number=4, timestamp=300.0),
        ]
        assets = {i: f"manual_assets/frame_{i:03d}.jpg" for i in range(5)}
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：Ralph 方案（01:40-04:26）",
                "![旧图](manual_assets/frame_000.jpg)",
                "操作说明。",
            ]
        )

        updated = embed_step_images(manual, frames, assets)

        self.assertNotIn("manual_assets/frame_000.jpg", updated)
        self.assertIn("manual_assets/frame_001.jpg", updated)
        self.assertIn("manual_assets/frame_003.jpg", updated)

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

    def test_operation_manual_review_warns_for_image_time_mismatch(self):
        manual = "\n".join(
            [
                "# 手册",
                "### 步骤 1：操作（00:10-00:20）",
                "![90s / Frame 9](manual_assets/frame_009.jpg)",
            ]
        )

        issues = review_operation_manual_markdown(manual)

        self.assertIn("step_image_time_mismatch", {issue["code"] for issue in issues})

    def test_frame_context_is_bounded_for_many_frames(self):
        analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        analyzer.previous_analyses = [{"response": f"frame {i} " + ("x" * 1000)} for i in range(8)]

        formatted = VideoAnalyzer._format_previous_analyses(analyzer)

        self.assertIn("Frame 5", formatted)
        self.assertIn("Frame 7", formatted)
        self.assertNotIn("Frame 4", formatted)
        self.assertLess(len(formatted), 2600)

    def test_vl_context_window_uses_before_current_after(self):
        frames = [Frame(i, Path(f"frame_{i}.jpg"), i * 5.0, 0.0) for i in range(8)]

        context = build_frame_context_window(
            frames=frames,
            current_frame=frames[4],
            before=3,
            after=2,
            max_gap_seconds="auto",
        )

        self.assertEqual([item.frame.number for item in context], [1, 2, 3, 4, 5, 6])
        self.assertEqual([item.role for item in context], ["previous", "previous", "previous", "current", "next", "next"])

    def test_vl_context_window_does_not_cross_time_breaks(self):
        frames = [
            Frame(0, Path("frame_0.jpg"), 0.0, 0.0),
            Frame(1, Path("frame_1.jpg"), 5.0, 0.0),
            Frame(2, Path("frame_2.jpg"), 10.0, 0.0),
            Frame(3, Path("frame_3.jpg"), 80.0, 0.0),
            Frame(4, Path("frame_4.jpg"), 85.0, 0.0),
        ]

        context = build_frame_context_window(
            frames=frames,
            current_frame=frames[3],
            before=3,
            after=2,
            max_gap_seconds="auto",
        )

        self.assertEqual([item.frame.number for item in context], [3, 4])
        self.assertEqual(resolve_vl_context_gap_seconds(frames, "auto"), 15.0)

    def test_frame_analysis_uses_multiframe_context_images(self):
        client = Mock()
        client.generate.return_value = {"response": "frame ok"}
        prompt_loader = Mock()
        prompt_loader.get_by_index.side_effect = [
            "Frame prompt. {PREVIOUS_FRAMES} {prompt}",
            "Video prompt.",
        ]
        analyzer = VideoAnalyzer(
            client=client,
            model="vision",
            prompt_loader=prompt_loader,
            temperature=0.0,
        )
        frames = [Frame(i, Path(f"/tmp/frame_{i}.jpg"), i * 5.0, 0.0) for i in range(3)]
        context = build_frame_context_window(frames, frames[1], before=1, after=1, max_gap_seconds=10)

        analyzer.analyze_frame(frames[1], ocr_text="Current OCR", context_window=context, context_ocr_texts={0: "before", 2: "after"})

        kwargs = client.generate.call_args.kwargs
        self.assertEqual(kwargs["image_paths"], ["/tmp/frame_0.jpg", "/tmp/frame_1.jpg", "/tmp/frame_2.jpg"])
        self.assertIn("CURRENT frame 1", kwargs["prompt"])
        self.assertIn("OCR context", kwargs["prompt"])

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
        self.assertEqual(post.call_args.kwargs["data"]["single_pass_max_duration_sec"], "420.0")
        self.assertEqual(post.call_args.kwargs["data"]["chunk_parallel_workers"], "2")
        self.assertEqual(post.call_args.kwargs["data"]["chunk_duration_sec"], "120.0")
        audio_path.unlink()

    def test_vibevoice_native_chunking_uses_endpoint_failover_instead_of_client_split(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000 * 500)
        failed = Mock()
        failed.raise_for_status.side_effect = RuntimeError("offline")
        ok = Mock()
        ok.raise_for_status.return_value = None
        ok.json.return_value = {
            "success": True,
            "text": "edge native chunk text",
            "language": "zh",
            "segments": [{"mode": "ray_chunk_reconcile"}],
        }

        with patch("video_analyzer.asr_providers._split_audio_evenly") as split_audio, patch(
            "video_analyzer.asr_providers.requests.post", side_effect=[failed, ok]
        ) as post, patch("video_analyzer.asr_providers.time.sleep"):
            transcript = transcribe_with_vibevoice_remote(
                audio_path,
                ["http://spark/vibevoice", "http://edge/vibevoice"],
                {"use_native_chunking": True, "distributed_min_seconds": 420, "distributed_workers": 2},
            )

        self.assertEqual(transcript.text, "edge native chunk text")
        split_audio.assert_not_called()
        self.assertEqual([call.args[0] for call in post.call_args_list], ["http://spark/vibevoice", "http://edge/vibevoice"])
        self.assertEqual(post.call_args.kwargs["data"]["chunk_parallel_workers"], "2")
        self.assertEqual(post.call_args.kwargs["timeout"], (30.0, 680.0))
        audio_path.unlink()

    def test_vibevoice_distributes_long_audio_across_remote_workers_when_native_chunking_disabled(self):
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
                {"use_native_chunking": False, "distributed_min_seconds": 420, "distributed_workers": 2},
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

    def test_balanced_empty_fast_transcript_does_not_fallback_outside_spark(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        empty_fast = AudioTranscript(text="", segments=[], language="unknown")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=empty_fast), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice", return_value=None
        ), patch("video_analyzer.asr_providers.transcribe_with_capswriter") as capswriter, patch(
            "video_analyzer.asr_providers.AudioProcessor"
        ) as processor:
            result = transcribe_with_strategy("balanced", audio_path, "", "medium", "cpu", {})

        self.assertIsNone(result.transcript)
        self.assertEqual(result.providers_run, ["remote_http", "vibevoice"])
        self.assertIn("remote_http produced no transcript", result.failures)
        self.assertIn("vibevoice produced no transcript", result.failures)
        capswriter.assert_not_called()
        processor.assert_not_called()
        audio_path.unlink()

    def test_balanced_weak_fast_transcript_does_not_fallback_outside_spark(self):
        audio_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        weak = AudioTranscript(text="嗯嗯", segments=[{"start": 0.0, "end": 1.0, "text": "嗯嗯"}], language="zh")

        with patch("video_analyzer.asr_providers.transcribe_with_remote_http", return_value=weak), patch(
            "video_analyzer.asr_providers.transcribe_with_vibevoice", return_value=None
        ), patch("video_analyzer.asr_providers.transcribe_with_capswriter") as capswriter:
            result = transcribe_with_strategy("balanced", audio_path, "", "medium", "cpu", {})

        self.assertEqual(result.transcript.text, "嗯嗯")
        self.assertEqual(result.providers_run, ["remote_http", "vibevoice"])
        self.assertTrue(any("balanced did not fallback outside Spark ASR" in note for note in result.merge_notes))
        capswriter.assert_not_called()
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
