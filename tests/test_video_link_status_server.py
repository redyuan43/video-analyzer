import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from video_analyzer import cli as cli_mod
from video_analyzer import douyin_browser as douyin_browser_mod
from video_analyzer.frame import Frame
from video_analyzer.frame_manifest import write_frame_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO_ROOT / "tools" / "video_link_status_server.py"
URL_CONTEXT_PATH = REPO_ROOT / "video_analyzer" / "url_context.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


server_mod = load_module(SERVER_PATH, "video_link_status_server")
url_context_mod = load_module(URL_CONTEXT_PATH, "video_analyzer_url_context")


class VideoLinkStatusServerTests(unittest.TestCase):
    def setUp(self):
        default_config = json.loads(
            (REPO_ROOT / "video_analyzer" / "config" / "default_config.json").read_text(encoding="utf-8")
        )
        patcher = patch.object(
            server_mod,
            "runtime_config",
            return_value={
                "active_runtime_profile": "deepseek_v4_pro",
                "runtime_profiles": default_config.get("runtime_profiles") or {},
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_focus_prompt_materializes_analysis_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp) / "page_context.md"
            run_dir = Path(tmp) / "run"
            context.write_text("# Context\n", encoding="utf-8")

            focused = url_context_mod.materialize_analysis_context(context, run_dir, "重点关注 CLI 参数")
            self.assertEqual(focused.name, "input_page_context.md")
            text = focused.read_text(encoding="utf-8")
            self.assertIn("## 用户关注重点", text)
            self.assertIn("重点关注 CLI 参数", text)
            self.assertIn("不能覆盖视频", text)
            self.assertEqual((focused.parent / "user_focus_prompt.md").read_text(encoding="utf-8"), "重点关注 CLI 参数\n")

    def test_ytdlp_runtime_args_forward_extractor_args(self):
        args = type(
            "Args",
            (),
            {
                "ytdlp_js_runtimes": "node",
                "ytdlp_remote_components": "ejs:github",
                "ytdlp_extractor_args": "youtube:player_client=mweb,web",
            },
        )()
        command = ["yt-dlp", "--skip-download"]

        url_context_mod.add_ytdlp_runtime_args(command, args)

        self.assertIn("--js-runtimes", command)
        self.assertIn("node", command)
        self.assertIn("--remote-components", command)
        self.assertIn("ejs:github", command)
        self.assertIn("--extractor-args", command)
        self.assertIn("youtube:player_client=mweb,web", command)

    def test_url_runner_uses_automatic_youtube_client_unless_explicitly_overridden(self):
        script = REPO_ROOT / "tools" / "run_operation_manual_from_url.sh"
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "PYTHON": "/bin/echo",
                "VIDEO_ANALYZER_YTDLP_RUNTIME_LOCK": str(Path(tmp) / "yt-dlp.lock"),
            }
            automatic = subprocess.run(
                [str(script), "https://www.youtube.com/watch?v=b7IMBHMjNv8"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            explicit = subprocess.run(
                [
                    str(script),
                    "https://www.youtube.com/watch?v=b7IMBHMjNv8",
                    "--ytdlp-extractor-args",
                    "youtube:player_client=mweb,web",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )

        self.assertNotIn("--ytdlp-extractor-args", automatic.stdout)
        self.assertIn("--ytdlp-extractor-args youtube:player_client=mweb,web", explicit.stdout)

    def test_youtube_format_error_retries_once_without_core_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            error = subprocess.CalledProcessError(
                1,
                ["yt-dlp"],
                output="ERROR: [youtube] b7IMBHMjNv8: Requested format is not available",
            )
            job = {"video_url": "https://www.youtube.com/watch?v=b7IMBHMjNv8", "run_dir": None}

            first_reason = server.retryable_stage_failure_reason(job, "analyze-core", error, str(Path(tmp) / "missing.log"), {})
            second_reason = server.retryable_stage_failure_reason(
                job,
                "analyze-core",
                error,
                str(Path(tmp) / "missing.log"),
                {"auto_retry_attempts": 1},
            )

        self.assertEqual(first_reason, server_mod.YOUTUBE_FORMAT_REQUEUE_MESSAGE)
        self.assertIsNone(second_reason)

    def test_youtube_format_error_does_not_retry_after_core_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "analysis.json").write_text("{}", encoding="utf-8")
            error = subprocess.CalledProcessError(
                1,
                ["yt-dlp"],
                output="ERROR: [youtube] b7IMBHMjNv8: Requested format is not available",
            )
            job = {"video_url": "https://www.youtube.com/watch?v=b7IMBHMjNv8", "run_dir": str(run_dir)}

            reason = server.retryable_stage_failure_reason(job, "analyze-core", error, str(Path(tmp) / "missing.log"), {})

        self.assertIsNone(reason)

    def test_stage_retry_archives_the_first_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            job_id = "a" * 32
            log_path = server.stage_log_path(job_id, "analyze-core")
            log_path.parent.mkdir(parents=True)
            log_path.write_text("first attempt\n", encoding="utf-8")

            current_path, attempt_paths = server.prepare_stage_log_attempt(
                job_id,
                "analyze-core",
                {"attempt": 1, "log_path": str(log_path), "attempt_log_paths": [str(log_path)]},
                2,
            )

            archived_path = server.stage_attempt_log_path(job_id, "analyze-core", 1)
            self.assertEqual(current_path, log_path)
            self.assertEqual(archived_path.read_text(encoding="utf-8"), "first attempt\n")
            self.assertFalse(log_path.exists())
            self.assertEqual(attempt_paths, [str(archived_path), str(log_path)])

    def test_ytdlp_maintenance_script_has_valid_shell_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(REPO_ROOT / "tools" / "ytdlp_runtime_maintenance.sh")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_package_metadata_requires_python_311(self):
        setup_text = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
        ui_project_text = (REPO_ROOT / "video-analyzer-ui" / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('python_requires=">=3.11"', setup_text)
        self.assertIn('requires-python = ">=3.11"', ui_project_text)

    def test_remote_download_command_uses_mi_safe_ytdlp_shape(self):
        args = type(
            "Args",
            (),
            {
                "url": "https://example.com/video",
                "include_subtitles": True,
                "include_comments": True,
                "subtitle_langs": "zh-CN,zh,en",
                "ytdlp_js_runtimes": "auto",
                "ytdlp_remote_components": "ejs:github",
                "ytdlp_extractor_args": "",
                "ytdlp_proxy": None,
                "cookies": None,
                "cookies_from_browser": "",
            },
        )()

        command = url_context_mod.remote_download_command(args, "/home/ivan/Documents/video-analyzer-url-downloads/test")

        self.assertTrue(command.startswith("bash -lc "))
        self.assertIn("yt-dlp", command)
        self.assertIn("--write-subs", command)
        self.assertIn("--write-comments", command)
        self.assertIn("--js-runtimes node", command)
        self.assertIn("--remote-components ejs:github", command)
        self.assertIn("/home/ivan/Documents/video-analyzer-url-downloads/test/download.", command)

    def test_bilibili_ytdlp_commands_include_site_headers(self):
        command = ["yt-dlp", "--skip-download"]

        url_context_mod.add_ytdlp_site_args(command, "https://www.bilibili.com/video/BV1YtVz6eEAz/")

        self.assertIn("--add-header", command)
        self.assertIn("Referer: https://www.bilibili.com/video/BV1YtVz6eEAz/", command)
        self.assertIn("Origin: https://www.bilibili.com", command)
        self.assertTrue(any(header.startswith("User-Agent: Mozilla/5.0") for header in command))

        remote_args = type(
            "Args",
            (),
            {
                "url": "https://www.bilibili.com/video/BV1YtVz6eEAz/",
                "include_subtitles": False,
                "include_comments": False,
                "subtitle_langs": "zh-CN,zh,en",
                "ytdlp_js_runtimes": "none",
                "ytdlp_remote_components": "",
                "ytdlp_extractor_args": "",
                "ytdlp_proxy": None,
                "cookies": None,
                "cookies_from_browser": "",
            },
        )()
        remote_command = url_context_mod.remote_download_command(
            remote_args,
            "/home/ivan/Documents/video-analyzer-url-downloads/test",
        )

        self.assertIn("--add-header", remote_command)
        self.assertIn("Referer: https://www.bilibili.com/video/BV1YtVz6eEAz/", remote_command)
        self.assertIn("Origin: https://www.bilibili.com", remote_command)
        self.assertIn("User-Agent: Mozilla/5.0", remote_command)

    def test_infer_video_id_from_apple_podcast_episode_url(self):
        url = "https://podcasts.apple.com/au/podcast/huberman-lab/id1545953110?i=1000775612409"

        self.assertEqual(url_context_mod.infer_video_id_from_url(url), "1000775612409")
        self.assertEqual(
            url_context_mod.apple_podcasts_episode_parts(url),
            ("1545953110", "1000775612409", "au"),
        )

    def test_fetch_metadata_falls_back_to_apple_lookup(self):
        url = "https://podcasts.apple.com/au/podcast/huberman-lab/id1545953110?i=1000775612409"
        args = type(
            "Args",
            (),
            {
                "ytdlp_js_runtimes": "node",
                "ytdlp_remote_components": "ejs:github",
                "ytdlp_extractor_args": "",
                "ytdlp_proxy": None,
                "cookies": None,
                "cookies_from_browser": "",
            },
        )()
        payload = {
            "resultCount": 2,
            "results": [
                {"kind": "podcast", "trackId": 1545953110, "collectionName": "Huberman Lab"},
                {
                    "kind": "podcast-episode",
                    "trackId": 1000775612409,
                    "trackName": "Raising a Dog & Mastering Calm Assertive Energy | Cesar Millan",
                    "description": "Episode description",
                    "episodeUrl": "https://traffic.megaphone.fm/SCIM6380580289.mp3",
                    "trackTimeMillis": 9503000,
                    "releaseDate": "2026-07-06T07:00:00Z",
                    "collectionName": "Huberman Lab",
                    "artistName": "Scicomm Media",
                    "artworkUrl600": "https://example.com/art.jpg",
                },
            ],
        }

        def fake_urlopen(request, timeout=0):
            self.assertIn("id=1545953110", request.full_url)
            self.assertIn("entity=podcastEpisode", request.full_url)
            self.assertEqual(timeout, 20)
            return BytesIO(json.dumps(payload).encode("utf-8"))

        with patch.object(
            url_context_mod.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["yt-dlp"]),
        ), patch.object(url_context_mod, "urlopen", side_effect=fake_urlopen):
            info = url_context_mod.fetch_metadata(url, args)

        self.assertEqual(info["extractor"], "ApplePodcastsLookup")
        self.assertEqual(info["id"], "1000775612409")
        self.assertEqual(info["title"], "Raising a Dog & Mastering Calm Assertive Energy | Cesar Millan")
        self.assertEqual(info["duration"], 9503)
        self.assertEqual(info["upload_date"], "20260706")
        self.assertEqual(info["_video_analyzer_download_url"], "https://traffic.megaphone.fm/SCIM6380580289.mp3")

    def test_fetch_metadata_non_apple_error_stays_ytdlp_error(self):
        args = type(
            "Args",
            (),
            {
                "ytdlp_js_runtimes": "node",
                "ytdlp_remote_components": "ejs:github",
                "ytdlp_extractor_args": "",
                "ytdlp_proxy": None,
                "cookies": None,
                "cookies_from_browser": "",
            },
        )()

        with patch.object(
            url_context_mod.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["yt-dlp"]),
        ), self.assertRaises(subprocess.CalledProcessError):
            url_context_mod.fetch_metadata("https://example.com/video", args)

    def test_download_video_uses_apple_lookup_direct_media_url(self):
        args = type(
            "Args",
            (),
            {
                "include_subtitles": False,
                "include_comments": False,
                "subtitle_langs": "zh-CN,zh,en",
                "ytdlp_js_runtimes": "none",
                "ytdlp_remote_components": "",
                "ytdlp_extractor_args": "",
                "ytdlp_proxy": None,
                "cookies": None,
                "cookies_from_browser": "",
            },
        )()
        info = {
            "extractor": "ApplePodcastsLookup",
            "_video_analyzer_download_url": "https://traffic.megaphone.fm/SCIM6380580289.mp3",
        }

        with tempfile.TemporaryDirectory() as tmp, patch.object(url_context_mod.subprocess, "run") as run:
            url_context_mod.download_video(
                "https://podcasts.apple.com/au/podcast/huberman-lab/id1545953110?i=1000775612409",
                Path(tmp),
                args,
                info,
            )

        command = run.call_args.args[0]
        self.assertEqual(command[-1], "https://traffic.megaphone.fm/SCIM6380580289.mp3")
        self.assertIn("-f", command)

    def test_douyin_browser_helpers_recognize_url_and_profile(self):
        self.assertTrue(douyin_browser_mod.is_douyin_url("https://v.douyin.com/sdYOxlCtlrY/"))
        self.assertTrue(douyin_browser_mod.is_douyin_url("https://www.douyin.com/video/7656377961499250097"))
        self.assertFalse(douyin_browser_mod.is_douyin_url("https://www.bilibili.com/video/BV1YtVz6eEAz/"))
        self.assertEqual(douyin_browser_mod.infer_douyin_id("https://www.douyin.com/video/7656377961499250097"), "7656377961499250097")
        self.assertEqual(douyin_browser_mod.parse_browser_profile("chrome:Profile 1"), ("chrome", "Profile 1"))
        self.assertEqual(douyin_browser_mod.parse_browser_profile("chrome+gnomekeyring:Default"), ("chrome", "Default"))

    def test_douyin_aweme_detail_maps_to_info(self):
        detail = {
            "aweme_id": "7656377961499250097",
            "desc": "安装这些skill，让Codex接管电路设计和三维建模 #Codex",
            "video": {"duration": 158802},
            "author": {"nickname": "作者", "uid": "123", "sec_uid": "sec"},
            "text_extra": [{"hashtag_name": "Codex"}],
        }

        info = douyin_browser_mod.aweme_detail_to_info(
            detail,
            "https://v.douyin.com/sdYOxlCtlrY/",
            {"final_url": "https://www.douyin.com/video/7656377961499250097"},
        )

        self.assertEqual(info["id"], "7656377961499250097")
        self.assertEqual(info["extractor"], "douyin_browser")
        self.assertEqual(info["duration"], 158.802)
        self.assertEqual(info["uploader"], "作者")
        self.assertIn("Codex", info["tags"])

    def test_infer_video_id_from_bilibili_url(self):
        self.assertEqual(
            url_context_mod.infer_video_id_from_url("https://www.bilibili.com/video/BV1YtVz6eEAz/?spm_id_from=333"),
            "BV1YtVz6eEAz",
        )

    def test_existing_video_dir_for_url_uses_cached_youtube_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp) / "B8OxtGSEfoo"
            video_dir.mkdir()
            (video_dir / "video.mp4").write_bytes(b"video")
            (video_dir / "page_context.md").write_text("# context\n", encoding="utf-8")

            resolved = url_context_mod.existing_video_dir_for_url(
                Path(tmp),
                "https://www.youtube.com/watch?v=B8OxtGSEfoo",
            )

        self.assertEqual(resolved, video_dir)

    def test_existing_video_dir_for_url_accepts_cached_audio_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp) / "B8OxtGSEfoo"
            video_dir.mkdir()
            (video_dir / "audio.m4a").write_bytes(b"audio")
            (video_dir / "page_context.md").write_text("# context\n", encoding="utf-8")

            resolved = url_context_mod.existing_video_dir_for_url(
                Path(tmp),
                "https://www.youtube.com/watch?v=B8OxtGSEfoo",
            )

        self.assertEqual(resolved, video_dir)

    def test_materialize_download_accepts_audio_only_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp)
            (video_dir / "download.m4a").write_bytes(b"audio")

            media_path = url_context_mod.materialize_download(video_dir, video_dir / "video.mp4")

            self.assertEqual(media_path, video_dir / "audio.m4a")
            self.assertEqual(media_path.read_bytes(), b"audio")
            self.assertFalse((video_dir / "download.m4a").exists())

    def test_media_has_video_stream_detects_audio_only_input(self):
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps({"streams": []}),
            stderr="",
        )
        with patch.object(cli_mod.subprocess, "run", return_value=completed):
            self.assertFalse(cli_mod.media_has_video_stream(Path("/tmp/audio.m4a")))

    def test_media_has_video_stream_ignores_attached_picture(self):
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_type": "video", "disposition": {"attached_pic": 1}},
                    ]
                }
            ),
            stderr="",
        )
        with patch.object(cli_mod.subprocess, "run", return_value=completed):
            self.assertFalse(cli_mod.media_has_video_stream(Path("/tmp/audio-with-cover.mp3")))

    def test_media_has_video_stream_detects_video_input(self):
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_type": "video", "disposition": {"attached_pic": 0}},
                    ]
                }
            ),
            stderr="",
        )
        with patch.object(cli_mod.subprocess, "run", return_value=completed):
            self.assertTrue(cli_mod.media_has_video_stream(Path("/tmp/video.mp4")))

    def test_options_include_defaults_and_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            options = server.options()

        self.assertEqual(options["defaults"]["analysis_mode"], "auto")
        self.assertEqual(options["defaults"]["profile"], "deepseek_v4_pro")
        self.assertEqual(options["defaults"]["cookies_from_browser"], "none")
        self.assertEqual(options["defaults"]["download_device"], "local")
        self.assertTrue(options["defaults"]["keep_existing"])
        self.assertTrue(options["defaults"]["include_subtitles"])
        self.assertTrue(options["defaults"]["prefer_subtitle_transcript"])
        self.assertTrue(options["defaults"]["include_comments"])
        self.assertTrue(options["defaults"]["refresh_context"])
        self.assertTrue(options["defaults"]["skip_images"])
        self.assertEqual(options["defaults"]["max_comments"], 3000)
        self.assertIn("balanced", options["choices"]["analysis_modes"])
        self.assertIn("operation-fast", options["choices"]["analysis_modes"])
        self.assertIn("deepseek_v4_pro", options["choices"]["profiles"])
        self.assertIn("mi", options["choices"]["download_devices"])

    def test_options_use_machine_active_runtime_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            config = {
                "active_runtime_profile": "nx2_fallback",
                "runtime_profiles": {
                    "deepseek_v4_pro": {},
                    "nx2_fallback": {},
                },
            }
            with patch.object(server_mod, "runtime_config", return_value=config):
                options = server.options()

        self.assertEqual(options["defaults"]["profile"], "nx2_fallback")

    def test_collect_core_artifacts_includes_review_and_ab_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "frame_dedup_audit.json").write_text("{}", encoding="utf-8")
            (run_dir / "visual_review.html").write_text("<html></html>", encoding="utf-8")
            (run_dir / "RUN_MANIFEST.md").write_text("# RUN_MANIFEST\n", encoding="utf-8")
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "frames_extracted": 3,
                            "frame_dedup_audit": {"summary": {"treatment_drop_count": 1}},
                            "visual_review": {"contact_sheet_count": 1},
                            "run_manifest": {"chars": 120},
                            "ocr_keyframes": {"ocr_text_events_count": 2},
                        },
                        "ocr_events": [{}, {}],
                        "frame_analyses": [{}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)

            artifacts = server.collect_core_artifacts(run_dir)

        self.assertEqual(artifacts["frame_dedup_audit"], str(run_dir / "frame_dedup_audit.json"))
        self.assertEqual(artifacts["visual_review"], str(run_dir / "visual_review.html"))
        self.assertEqual(artifacts["run_manifest"], str(run_dir / "RUN_MANIFEST.md"))
        self.assertEqual(artifacts["core_counts"]["frame_dedup_audit"]["treatment_drop_count"], 1)
        self.assertEqual(artifacts["core_counts"]["visual_review"]["contact_sheet_count"], 1)
        self.assertEqual(artifacts["core_counts"]["run_manifest"]["chars"], 120)

    def test_create_job_saves_common_and_collection_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job(
                {
                    "videoUrl": "https://example.com/video",
                    "analysisMode": "deep",
                    "profile": "deepseek_v4_pro",
                    "runName": "../operation manual!",
                    "cookiesFromBrowser": "none",
                    "downloadDevice": "mi",
                    "skipImages": True,
                    "keepExisting": False,
                    "includeSubtitles": False,
                    "preferSubtitleTranscript": True,
                    "includeComments": False,
                    "maxComments": "12",
                    "subtitleLangs": "en,zh",
                    "refreshContext": True,
                    "focusPrompt": "重点关注部署参数和失败恢复",
                }
            )

        self.assertEqual(job["options"]["analysis_mode"], "deep")
        self.assertEqual(job["options"]["run_name"], "operation-manual")
        self.assertEqual(job["options"]["cookies_from_browser"], "")
        self.assertEqual(job["options"]["download_device"], "mi")
        self.assertTrue(job["options"]["skip_images"])
        self.assertFalse(job["options"]["keep_existing"])
        self.assertFalse(job["options"]["include_subtitles"])
        self.assertTrue(job["options"]["prefer_subtitle_transcript"])
        self.assertFalse(job["options"]["include_comments"])
        self.assertEqual(job["options"]["max_comments"], 12)
        self.assertEqual(job["options"]["subtitle_langs"], "en,zh")
        self.assertTrue(job["options"]["refresh_context"])
        self.assertEqual(job["options"]["focus_prompt"], "重点关注部署参数和失败恢复")

    def test_create_uploaded_media_job_materializes_file_context_and_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            source = Path(tmp) / "demo.mp3"
            source.write_bytes(b"fake audio")
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", repo_root)

            job = server.create_uploaded_media_job(
                {
                    "analysis_mode": "auto",
                    "run_name": "audio-run",
                    "include_subtitles": True,
                    "include_comments": True,
                    "download_device": "mi",
                    "template_id": "tmpl-001",
                    "template_title": "Default Summary",
                    "template_title_zh": "默认总结",
                    "template_category": "通用 / 摘要",
                },
                source,
                "../demo.mp3",
            )
            loaded = server.load_job(job["job_id"])
            self.assertEqual(loaded["source_type"], "upload")
            self.assertEqual(loaded["source_name"], "demo.mp3")
            self.assertEqual(loaded["options"]["download_device"], "local")
            self.assertFalse(loaded["options"]["include_subtitles"])
            self.assertFalse(loaded["options"]["include_comments"])
            self.assertEqual(loaded["options"]["template_id"], "tmpl-001")
            self.assertEqual(loaded["options"]["template_title_zh"], "默认总结")
            self.assertTrue(Path(loaded["media_path"]).is_file())
            self.assertIn("本地上传媒体文件", Path(loaded["page_context_path"]).read_text(encoding="utf-8"))

    def test_create_uploaded_media_job_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            source = Path(tmp) / "empty.mp3"
            source.touch()
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", repo_root)

            with self.assertRaisesRegex(server_mod.BridgeError, "uploaded media file is empty"):
                server.create_uploaded_media_job({"analysis_mode": "auto"}, source, "empty.mp3")

    def test_uploaded_media_probe_uses_ffprobe_and_avoids_ytdlp(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            source = Path(tmp) / "demo.mp3"
            source.write_bytes(b"fake audio")
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", repo_root)
            job = server.create_uploaded_media_job({"analysis_mode": "auto"}, source, "demo.mp3")
            loaded = server.load_job(job["job_id"])
            completed = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout="3600.5\n", stderr="")

            with patch.object(server_mod.subprocess, "run", return_value=completed) as run:
                result = server.stage_probe(loaded)

        self.assertEqual(result["artifacts"]["duration_seconds"], 3600)
        self.assertEqual(loaded["resolved_mode"], "fast")
        self.assertEqual(run.call_args.args[0][0], "ffprobe")

    def test_uploaded_media_operation_uses_audio_template_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            source = Path(tmp) / "demo.mp3"
            source.write_bytes(b"fake audio")
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", repo_root)
            job = server.create_uploaded_media_job(
                {
                    "analysis_mode": "auto",
                    "profile": "deepseek_v4_pro",
                    "focus_prompt": "会议纪要",
                    "template_id": "tmpl-meeting",
                },
                source,
                "demo.mp3",
            )
            loaded = server.load_job(job["job_id"])

            command = server.operation_command(loaded)

        self.assertIn("tools/run_audio_template_analysis.py", command)
        self.assertIn("--template-id", command)
        self.assertEqual(command[command.index("--template-id") + 1], "tmpl-meeting")
        self.assertIn("--focus-prompt", command)
        self.assertIn("会议纪要", command)
        self.assertIn("--output", command)
        self.assertEqual(command[command.index("--profile") + 1], "deepseek_v4_pro")
        self.assertNotIn("video_analyzer.cli", " ".join(command))

    def test_command_mapping_for_collection_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job(
                {
                    "video_url": "https://example.com/video",
                    "analysis_mode": "fast",
                    "profile": "deepseek_v4_pro",
                    "cookies_from_browser": "none",
                    "download_device": "mi",
                    "keep_existing": False,
                    "include_subtitles": False,
                    "prefer_subtitle_transcript": True,
                    "include_comments": False,
                    "max_comments": 8,
                    "subtitle_langs": "en-US,en",
                    "refresh_context": True,
                    "focus_prompt": "只关注配置变更",
                }
            )
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "fast"

        command = server.operation_command(loaded)

        self.assertIn("--no-include-subtitles", command)
        self.assertIn("--prefer-subtitle-transcript", command)
        self.assertIn("--no-include-comments", command)
        self.assertIn("--refresh-context", command)
        self.assertIn("--max-comments", command)
        self.assertIn("8", command)
        self.assertIn("--subtitle-langs", command)
        self.assertIn("en-US,en", command)
        self.assertIn("--focus-prompt", command)
        self.assertIn("只关注配置变更", command)
        self.assertNotIn("--keep-existing", command)
        self.assertNotIn("--cookies-from-browser", command)
        self.assertIn("--download-device", command)
        self.assertIn("mi", command)
        self.assertIn("--frame-extractor", command)
        self.assertEqual(command[command.index("--frame-extractor") + 1], "jetson")
        self.assertIn("--jetson-frame-hosts", command)
        self.assertEqual(command[command.index("--jetson-frame-hosts") + 1], "agx,agx")
        self.assertIn("--jetson-frame-backend", command)
        self.assertEqual(command[command.index("--jetson-frame-backend") + 1], "ray")
        self.assertIn("--jetson-sample-fps", command)
        self.assertEqual(command[command.index("--jetson-sample-fps") + 1], "0.5")
        self.assertIn("--jetson-require-hwdec", command)
        self.assertIn("--resume-existing-core", command)

    def test_deep_operation_uses_agx_frame_extractor(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "deep"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "deep"

        command = server.operation_command(loaded)

        self.assertEqual(command[0], "tools/run_operation_manual_from_url.sh")
        self.assertEqual(command[command.index("--pipeline-mode") + 1], "deep")
        self.assertEqual(command[command.index("--frame-extractor") + 1], "jetson")
        self.assertEqual(command[command.index("--jetson-frame-hosts") + 1], "agx,agx")
        self.assertEqual(command[command.index("--jetson-frame-backend") + 1], "ray")
        self.assertIn("--jetson-require-hwdec", command)
        self.assertIn("--resume-existing-core", command)

    def test_operation_fast_keeps_vl_with_bounded_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "operation-fast"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "operation-fast"

        command = server.operation_command(loaded)

        self.assertEqual(command[command.index("--pipeline-mode") + 1], "fast")
        self.assertEqual(command[command.index("--vl-frame-policy") + 1], "auto")
        self.assertEqual(command[command.index("--min-vl-frames") + 1], "8")
        self.assertEqual(command[command.index("--max-vl-frames") + 1], "16")
        self.assertNotIn("--asr-provider", command)

    def test_url_context_resume_command_passes_core_resume_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            context = run_dir / "input_page_context.md"
            video = Path(tmp) / "video.mp4"
            run_dir.mkdir()
            context.write_text("# Context\n", encoding="utf-8")
            video.write_bytes(b"video")
            args = type(
                "Args",
                (),
                {
                    "python": sys.executable,
                    "llm_base_url": "http://127.0.0.1:8000/v1",
                    "vision_base_url": "http://127.0.0.1:18082/v1",
                    "text_base_url": "http://127.0.0.1:8000/v1",
                    "vision_model": "minicpm-v-4.5-v100",
                    "text_model": "deepseek-v4-pro",
                    "manual_language": "zh-CN",
                    "log_level": "INFO",
                    "pipeline_mode": "deep",
                    "candidate_frames": "auto",
                    "min_vl_frames": "auto",
                    "max_vl_frames": "auto",
                    "vl_frame_policy": "auto",
                    "vl_concurrency": 6,
                    "vl_context_before": 0,
                    "vl_context_after": 0,
                    "vl_context_max_gap": "auto",
                    "transcript_file": str(run_dir / "transcript.md"),
                    "asr_provider": "none",
                    "vibevoice_url": [],
                    "frame_extractor": "jetson",
                    "jetson_frame_hosts": "agx,agx",
                    "jetson_frame_backend": "ray",
                    "jetson_sample_fps": "0.5",
                    "jetson_chunk_overlap_seconds": 2.0,
                    "jetson_frame_weights": "",
                    "jetson_require_hwdec": True,
                    "resume_existing_core": True,
                    "ocr_base_url": ["http://127.0.0.1:18088/v1"],
                    "ocr_concurrency": "5",
                    "ocr_cache": "on",
                    "ocr_cache_dir": ".cache/video-analyzer/ocr",
                    "ocr_keyframe_strategy": "scan-text",
                    "ocr_keyframe_budget": "auto",
                    "ocr_scan_sample_fps": "auto",
                    "no_keep_frames": False,
                    "max_frames": None,
                    "duration": None,
                },
            )()

            command = url_context_mod.build_analyzer_command(args, video, context, run_dir)

        self.assertIn("--resume-existing", command)
        self.assertIn("--transcript-file", command)
        self.assertIn("--asr-provider", command)
        self.assertEqual(command[command.index("--asr-provider") + 1], "none")

    def test_reusable_frames_rejects_local_manifest_for_jetson_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            frames_dir = output_dir / "frames"
            frames_dir.mkdir()
            frame_path = frames_dir / "frame_0.jpg"
            frame_path.write_bytes(b"jpg")
            write_frame_manifest([Frame(0, frame_path, 1.0, 0.5)], output_dir, source="local")

            frames, metadata = cli_mod.reusable_frames_from_manifest(output_dir, "jetson")

        self.assertEqual(frames, [])
        self.assertIn("reuse_rejected_reason", metadata)

    def test_reusable_frames_accepts_jetson_manifest_for_jetson_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            frames_dir = output_dir / "frames"
            frames_dir.mkdir()
            frame_path = frames_dir / "frame_0.jpg"
            frame_path.write_bytes(b"jpg")
            write_frame_manifest([Frame(0, frame_path, 1.0, 0.5)], output_dir, source="jetson")

            frames, metadata = cli_mod.reusable_frames_from_manifest(output_dir, "jetson")

        self.assertEqual(len(frames), 1)
        self.assertEqual(metadata["backend"], "reused_jetson")

    def test_create_jobs_batch_assigns_focus_prompt_per_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            result = server.create_jobs(
                {
                    "video_urls_text": "https://example.com/one\nhttps://example.com/two",
                    "run_name": "batch",
                    "focus_prompts": {
                        "https://example.com/one": "关注安装",
                        "https://example.com/two": "关注排错",
                    },
                    "auto_start": False,
                }
            )

        self.assertEqual([job["options"]["focus_prompt"] for job in result["jobs"]], ["关注安装", "关注排错"])

    def test_create_jobs_batch_uses_global_focus_prompt_as_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            result = server.create_jobs(
                {
                    "video_urls_text": "https://example.com/one\nhttps://example.com/two",
                    "run_name": "batch",
                    "focus_prompt": "关注部署风险",
                    "focus_prompts": {"https://example.com/two": "关注排错"},
                    "auto_start": False,
                }
            )

        self.assertEqual([job["options"]["focus_prompt"] for job in result["jobs"]], ["关注部署风险", "关注排错"])

    def test_command_mapping_defaults_keep_downloads_without_browser_cookies(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "fast"

        command = server.prepare_command(loaded)

        self.assertIn("--download-only", command)
        self.assertIn("--keep-existing", command)
        self.assertNotIn("--cookies-from-browser", command)
        self.assertIn("--include-subtitles", command)
        self.assertIn("--include-comments", command)

    def test_command_mapping_keeps_explicit_browser_cookies(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job(
                {"video_url": "https://example.com/video", "analysis_mode": "fast", "cookies_from_browser": "chrome"}
            )
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "fast"

        command = server.prepare_command(loaded)

        self.assertIn("--cookies-from-browser", command)
        self.assertIn("chrome", command)

    def test_long_talk_prepare_uses_supported_pipeline_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "long-talk-fast"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "long-talk-fast"

        command = server.prepare_command(loaded)

        self.assertIn("--pipeline-mode", command)
        self.assertEqual(command[command.index("--pipeline-mode") + 1], "fast")
        self.assertNotIn("long-talk-fast", command)

    def test_long_talk_operation_uses_wrapper_and_selected_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "long-talk-fast"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "long-talk-fast"

        command = server.operation_command(loaded)

        self.assertEqual(command[0], "tools/run_long_talk_fast_from_url.sh")
        self.assertIn("--profile", command)
        self.assertEqual(command[command.index("--profile") + 1], "deepseek_v4_pro")
        self.assertNotIn("--pipeline-mode", command)

    def test_local_profile_routes_all_frame_work_to_nx2(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            config = {
                "active_runtime_profile": "nx2_fallback",
                "runtime_profiles": {
                    "nx2_fallback": {
                        "frame_extractor": "jetson",
                        "jetson_frame_hosts": "nx2,nx2",
                        "jetson_frame_backend": "ssh",
                        "jetson_sample_fps": 0.5,
                    }
                }
            }
            with patch.object(server_mod, "runtime_config", return_value=config):
                job = server.create_job(
                    {
                        "video_url": "https://example.com/video",
                        "analysis_mode": "long-talk-fast",
                        "profile": "nx2_fallback",
                    }
                )
                loaded = server.load_job(job["job_id"])
                loaded["resolved_mode"] = "long-talk-fast"
                command = server.operation_command(loaded)

        self.assertEqual(command[command.index("--jetson-frame-hosts") + 1], "nx2,nx2")
        self.assertEqual(command[command.index("--jetson-frame-backend") + 1], "ssh")

    def test_long_talk_wrapper_defaults_to_agx_dual_worker(self):
        text = (REPO_ROOT / "tools" / "run_long_talk_fast_from_url.sh").read_text(encoding="utf-8")

        self.assertIn('JETSON_FRAME_HOSTS="${JETSON_FRAME_HOSTS:-agx,agx}"', text)
        self.assertIn('JETSON_FRAME_BACKEND="${JETSON_FRAME_BACKEND:-ray}"', text)
        self.assertIn('if [[ "$JETSON_FRAME_BACKEND" == "ray" ]]', text)
        self.assertIn('--jetson-frame-backend "$JETSON_FRAME_BACKEND"', text)
        self.assertNotIn("nx1,nx2,nx3,nx4", text)

    def test_final_publish_stage_uses_finalize_only_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            loaded["run_dir"] = str(run_dir)

            command = server.final_publish_command(loaded)

        self.assertEqual(command[0], "tools/run_video_doc_final_publish.sh")
        self.assertIn("--finalize-only", command)
        self.assertIn("--skip-send", command)
        self.assertIn("--skip-pdf", command)
        self.assertIn("--jobs", command)
        self.assertEqual(command[command.index("--jobs") + 1], "3")
        self.assertIn("--profile", command)
        self.assertEqual(command[command.index("--profile") + 1], "deepseek_v4_pro")
        self.assertIn("--skip-images", command)

    def test_final_publish_stage_respects_skip_images_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "skip_images": True})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            loaded["run_dir"] = str(run_dir)

            command = server.final_publish_command(loaded)

        self.assertIn("--skip-images", command)

    def test_final_publish_stage_keeps_images_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "skip_images": False})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            loaded["run_dir"] = str(run_dir)

            command = server.final_publish_command(loaded)

        self.assertIn("--skip-images", command)

    def test_final_publish_stage_can_enable_images_with_runtime_flag(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server_mod, "BAOYU_IMAGE_GENERATION_ENABLED", True):
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "skip_images": False})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            loaded["run_dir"] = str(run_dir)

            command = server.final_publish_command(loaded)

        self.assertNotIn("--skip-images", command)

    def test_image_prompts_stage_uses_repo_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            loaded["run_dir"] = str(run_dir)

            command = server.image_prompts_command(loaded)

        self.assertEqual(command[0], server_mod.sys.executable)
        self.assertEqual(command[1], str(REPO_ROOT / "tools" / "prepare_baoyu_image_prompts.py"))
        self.assertEqual(command[2], str(run_dir))

    def test_failed_stage_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "fast"
            loaded["stages"]["probe"] = {"status": "succeeded"}
            loaded["stages"]["prepare"] = {"status": "failed", "error": "old error"}
            server.save_job(loaded)

            def fake_prepare(current_job, log_path, stage_info=None):
                return {"artifacts": {"video_path": "/tmp/video.mp4"}, "stdout_tail": ["ok"]}

            with patch.object(server, "stage_prepare", side_effect=fake_prepare):
                result = server.run_stage(job["job_id"], "prepare")

        self.assertEqual(result["stages"]["prepare"]["status"], "succeeded")
        self.assertNotIn("old error", json.dumps(result["stages"]["prepare"], ensure_ascii=False))

    def test_queued_core_reconciles_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual-reset-003"})
            loaded = server.load_job(job["job_id"])
            video_dir = Path(tmp) / "video"
            run_dir = video_dir / "operation-manual-reset-003"
            (run_dir / "orin").mkdir(parents=True)
            (run_dir / "analysis.json").write_text(json.dumps({"metadata": {}, "ocr_events": [], "frame_analyses": []}), encoding="utf-8")
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
            (run_dir / "study_guide.json").write_text("{}", encoding="utf-8")
            (run_dir / "audio_template_analysis.json").write_text(
                json.dumps({"selected_template": {"id": "actual", "title_zh": "实际模板"}, "classification": {"method": "explicit"}}),
                encoding="utf-8",
            )
            loaded["video_dir"] = str(video_dir)
            loaded["stages"]["probe"] = {"status": "succeeded"}
            loaded["stages"]["prepare"] = {"status": "succeeded"}
            loaded["stages"]["analyze-core"] = {
                "status": "queued",
                "queued_for": "core",
                "retry_reason": "server stopped while this stage was running",
            }
            loaded["runner"] = {
                "status": "queued",
                "current_stage": "analyze-core",
                "queued_for": "core",
                "server_pid": os.getpid(),
            }
            loaded["status"] = "queued"
            server.save_job(loaded)

            recovered = server.load_job(job["job_id"])

        self.assertEqual(recovered["stages"]["analyze-core"]["status"], "succeeded")
        self.assertEqual(recovered["run_dir"], str(run_dir.resolve()))
        self.assertEqual(recovered["runner"]["current_stage"], "verify-core")
        self.assertEqual(recovered["runner"]["queued_for"], "verify")
        self.assertEqual(recovered["stages"]["verify-core"]["status"], "queued")
        self.assertEqual(recovered["stages"]["verify-core"]["queued_for"], "verify")
        self.assertEqual(recovered["artifacts"]["operation_manual"]["value"], str(run_dir.resolve() / "operation_manual.md"))
        public = server.public_job(recovered)
        self.assertEqual(public["result_resources"]["summary_markdown"], "operation_manual.md")
        self.assertEqual(public["result_resources"]["transcript_markdown"], "transcript.md")
        self.assertEqual(public["prompt_template"]["actual"]["id"], "actual")

    def test_document_preview_groups_primary_evidence_process_and_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "run"
            (run_dir / "docs_analysis_chapters").mkdir(parents=True)
            (run_dir / "manual_assets").mkdir()
            (run_dir / "orin").mkdir()
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "docs_analysis_chapters" / "knowledge_notes_v2.md").write_text("# Notes\n", encoding="utf-8")
            (run_dir / "docs_analysis_chapters" / "deep_report_v2.md").write_text("# Report\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "evidence_index.md").write_text("# Index\n", encoding="utf-8")
            (run_dir / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
            (run_dir / "analysis.json").write_text("{}", encoding="utf-8")
            (run_dir / "study_guide.json").write_text("{}", encoding="utf-8")
            (run_dir / "manual_assets" / "frame_000.jpg").write_bytes(b"jpg")
            loaded["run_dir"] = str(run_dir)
            server.save_job(loaded)

            public = server.public_job(server.load_job(job["job_id"]))
            preview = public["document_preview"]

        self.assertEqual([item["path"] for item in preview["primary"]], [
            "operation_manual.md",
            "docs_analysis_chapters/knowledge_notes_v2.md",
            "docs_analysis_chapters/deep_report_v2.md",
        ])
        self.assertIn("manual_evidence.md", [item["path"] for item in preview["evidence"]])
        self.assertNotIn("manual_evidence.md", [item["path"] for item in preview["primary"]])
        self.assertIn("transcript.md", [item["path"] for item in preview["process"]])
        self.assertIn("manual_assets", [item["path"] for item in preview["assets"]])
        self.assertIn("flowchart LR", preview["derivation"]["mermaid"])
        self.assertIn("操作手册", [node["title"] for node in preview["derivation"]["nodes"]])
        self.assertTrue(preview["primary"][0]["url"].endswith("/resources/operation_manual.md"))

    def test_resource_file_serves_only_run_dir_relative_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            loaded["run_dir"] = str(run_dir)
            server.save_job(loaded)

            path, mime_type = server.resource_file(job["job_id"], "operation_manual.md")

            with self.assertRaises(server_mod.BridgeError) as escaped:
                server.resource_file(job["job_id"], "../operation_manual.md")

        self.assertEqual(path, run_dir / "operation_manual.md")
        self.assertTrue(mime_type is None or "markdown" in mime_type or mime_type.startswith("text/"))
        self.assertEqual(escaped.exception.status, server_mod.HTTPStatus.FORBIDDEN)

    def test_verify_core_accepts_quality_failed_manual_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "video" / "operation-manual"
            run_dir.mkdir(parents=True)
            (run_dir / "analysis.json").write_text("{}", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "operation_manual.quality_failed.md").write_text("# Review artifact\n", encoding="utf-8")
            loaded["run_dir"] = str(run_dir)

            result = server.stage_verify_core(loaded)

        self.assertEqual(result["artifacts"]["missing"], [])
        self.assertIn("failed quality gate", result["artifacts"]["warnings"][0]["message"])
        self.assertIn("operation_manual.quality_failed.md", result["artifacts"]["warnings"][0]["message"])

    def test_verify_core_rejects_frame_analysis_resource_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "video" / "operation-manual"
            run_dir.mkdir(parents=True)
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "frame_analyses": [
                            {"response": "Error analyzing frame 12: model-resource-busy"},
                            {"status": "skipped", "response": "VL analysis skipped."},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            loaded["run_dir"] = str(run_dir)

            with self.assertRaises(server_mod.BridgeError) as caught:
                server.stage_verify_core(loaded)

        self.assertIn("core analysis errors", caught.exception.message)

    def test_analyze_core_accepts_quality_failed_manual_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "balanced"
            run_dir = Path(tmp) / "video" / "operation-manual"
            run_dir.mkdir(parents=True)
            (run_dir / "analysis.json").write_text("{}", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "operation_manual.quality_failed.md").write_text("# Review artifact\n", encoding="utf-8")
            log_path = server.stage_log_path(job["job_id"], "analyze-core")

            def fake_run_command(*_args, **_kwargs):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(f"[done] run_dir: {run_dir}\n", encoding="utf-8")
                return {"stdout_tail": ["[done] run_dir: " + str(run_dir)]}

            with patch.object(server, "run_command", fake_run_command):
                result = server.stage_analyze_core(loaded, str(log_path))

        self.assertIn("operation_manual.quality_failed.md", result["artifacts"]["operation_manual"])
        self.assertEqual(loaded["warnings"][0]["stage"], "analyze-core")
        self.assertIn("failed quality gate", loaded["warnings"][0]["message"])

    def test_optional_stage_failure_becomes_warning_when_core_markdown_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "video" / "operation-manual"
            run_dir.mkdir(parents=True)
            (run_dir / "analysis.json").write_text("{}", encoding="utf-8")
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            loaded["run_dir"] = str(run_dir)
            loaded["stages"] = {
                "probe": {"status": "succeeded"},
                "prepare": {"status": "succeeded"},
                "analyze-core": {"status": "succeeded"},
                "verify-core": {"status": "succeeded"},
                "study-guide": {"status": "skipped"},
                "multidoc": {"status": "skipped"},
                "deep-v2": {"status": "skipped"},
                "evidence-review": {"status": "skipped"},
                "image-prompts": {"status": "skipped"},
            }
            server.save_job(loaded)

            with patch.object(server, "run_command", side_effect=RuntimeError("publisher unavailable")):
                result = server.run_stage(job["job_id"], "final-publish")

        self.assertEqual(result["stages"]["final-publish"]["status"], "skipped")
        self.assertTrue(result["stages"]["final-publish"]["soft_failed"])
        self.assertEqual(result["warnings"][0]["stage"], "final-publish")
        self.assertIn("publisher unavailable", result["warnings"][0]["message"])

    def test_skipped_multidoc_with_missing_outputs_can_be_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "video" / "operation-manual"
            run_dir.mkdir(parents=True)
            (run_dir / "analysis.json").write_text("{}", encoding="utf-8")
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            loaded["run_dir"] = str(run_dir)
            loaded["stages"].update(
                {
                    "probe": {"status": "succeeded"},
                    "prepare": {"status": "succeeded"},
                    "analyze-core": {"status": "succeeded"},
                    "verify-core": {"status": "succeeded"},
                    "study-guide": {"status": "succeeded"},
                    "multidoc": {"status": "skipped", "error": "old timeout"},
                }
            )
            server.save_job(loaded)

            with patch.object(server, "run_command", return_value={"stdout_tail": []}) as run_command:
                server.run_stage(job["job_id"], "multidoc")

        run_command.assert_called_once()

    def test_next_stage_recovers_skipped_multidoc_when_outputs_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "video" / "operation-manual"
            run_dir.mkdir(parents=True)
            (run_dir / "analysis.json").write_text("{}", encoding="utf-8")
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            loaded["run_dir"] = str(run_dir)
            loaded["stages"].update(
                {
                    "probe": {"status": "succeeded"},
                    "prepare": {"status": "succeeded"},
                    "analyze-core": {"status": "succeeded"},
                    "verify-core": {"status": "succeeded"},
                    "study-guide": {"status": "succeeded"},
                    "multidoc": {"status": "skipped"},
                }
            )

            self.assertEqual(server.next_stage(loaded), "multidoc")

    def test_rerun_from_stage_resets_downstream_outputs_and_starts_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            loaded["stages"].update(
                {
                    "probe": {"status": "succeeded"},
                    "prepare": {"status": "succeeded"},
                    "analyze-core": {"status": "succeeded"},
                    "verify-core": {"status": "succeeded"},
                    "study-guide": {"status": "succeeded"},
                    "multidoc": {"status": "succeeded"},
                    "deep-v2": {"status": "succeeded"},
                    "evidence-review": {"status": "succeeded"},
                    "final-publish": {"status": "succeeded"},
                }
            )
            loaded["artifacts"] = {
                "docs_analysis": {"value": "old"},
                "chapter_deep_report": {"value": "old"},
                "evidence_review": {"value": "old"},
                "exports": {"value": "old"},
            }
            loaded["warnings"] = [
                {"stage": "multidoc", "message": "old warning"},
                {"stage": "verify-core", "message": "keep"},
            ]
            server.save_job(loaded)

            with patch.object(server, "_run_remaining_stages") as run_remaining:
                result = server.rerun_from_stage(job["job_id"], "multidoc")

            self.assertEqual(result["runner"]["current_stage"], "multidoc")
            refreshed = server.load_job(job["job_id"])
            self.assertEqual(refreshed["stages"]["study-guide"]["status"], "succeeded")
            self.assertNotIn("multidoc", refreshed["stages"])
            self.assertNotIn("final-publish", refreshed["stages"])
            self.assertNotIn("docs_analysis", refreshed["artifacts"])
            self.assertNotIn("exports", refreshed["artifacts"])
            self.assertEqual(refreshed["warnings"], [{"stage": "verify-core", "message": "keep"}])
            run_remaining.assert_called_once()

    def test_final_publish_command_skips_pdfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            loaded["run_dir"] = str(run_dir)

            command = server.final_publish_command(loaded)

        self.assertIn("--skip-pdf", command)
        self.assertIn("--finalize-only", command)

    def test_core_stage_starts_jetson_ray_before_ray_frame_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            command = [
                "tools/run_operation_manual_from_url.sh",
                "https://example.com/video",
                "--jetson-frame-backend",
                "ray",
            ]
            log_path = Path(tmp) / "analyze-core.log"

            completed = subprocess.CompletedProcess(["tools/start_jetson_frame_ray.sh"], 0, stdout="cluster ready", stderr="")
            with patch("tools.video_link_status_server.subprocess.run", return_value=completed) as run:
                result = server.ensure_jetson_ray_ready(command, str(log_path))
            log_text = log_path.read_text(encoding="utf-8")

        self.assertEqual(result["command"], [str(REPO_ROOT / "tools" / "start_jetson_frame_ray.sh")])
        self.assertIn("cluster ready", log_text)
        self.assertEqual(run.call_args.args[0], [str(REPO_ROOT / "tools" / "start_jetson_frame_ray.sh")])

    def test_core_stage_skips_ray_preflight_for_ssh_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            command = ["tools/run_operation_manual_from_url.sh", "https://example.com/video", "--jetson-frame-backend", "ssh"]

            with patch("tools.video_link_status_server.subprocess.run") as run:
                result = server.ensure_jetson_ray_ready(command, str(Path(tmp) / "analyze-core.log"))

        self.assertIsNone(result)
        run.assert_not_called()

    def test_public_job_hides_resolved_and_audio_only_visual_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/audio", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            run_dir = Path(tmp) / "audio" / "operation-manual"
            run_dir.mkdir(parents=True)
            (run_dir / "frames_manifest.json").write_text(
                json.dumps({"version": 1, "source": "audio_only", "frames": []}),
                encoding="utf-8",
            )
            loaded["run_dir"] = str(run_dir)
            loaded["stages"] = {
                "deep-v2": {"status": "skipped", "soft_failed": True},
                "final-publish": {"status": "succeeded"},
            }
            loaded["warnings"] = [
                {"stage": "deep-v2", "message": "no frames"},
                {"stage": "final-publish", "message": "old failure"},
            ]

            public = server.public_job(loaded)

        self.assertEqual(public["warnings"], [])

    def test_study_and_evidence_review_are_separate_stages(self):
        self.assertLess(server_mod.STAGE_ORDER.index("study-guide"), server_mod.STAGE_ORDER.index("multidoc"))
        self.assertLess(server_mod.STAGE_ORDER.index("deep-v2"), server_mod.STAGE_ORDER.index("evidence-review"))
        self.assertLess(server_mod.STAGE_ORDER.index("evidence-review"), server_mod.STAGE_ORDER.index("web-evidence"))
        self.assertLess(server_mod.STAGE_ORDER.index("web-evidence"), server_mod.STAGE_ORDER.index("qa-index"))
        self.assertLess(server_mod.STAGE_ORDER.index("qa-index"), server_mod.STAGE_ORDER.index("image-prompts"))
        self.assertLess(server_mod.STAGE_ORDER.index("evidence-review"), server_mod.STAGE_ORDER.index("image-prompts"))

    def test_study_guide_command_skips_model_review_until_deep_report_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)

            command = server.study_guide_command(loaded)

        self.assertIn("--skip-review", command)

    def test_evidence_review_command_runs_model_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)

            command = server.evidence_review_command(loaded)

        self.assertNotIn("--skip-review", command)
        self.assertEqual(command[1], "tools/run_study_guide.py")

    def test_qa_index_command_builds_existing_run_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            job = server.create_job({"video_url": "https://example.com/video", "profile": "deepseek_v4_pro"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)

            command = server.qa_index_command(loaded)

        self.assertEqual(command[:3], [sys.executable, "-m", "video_analyzer.doc_chat"])
        self.assertIn(str(run_dir), command)
        self.assertIn("--build-index", command)
        self.assertIn("--profile", command)
        self.assertIn("deepseek_v4_pro", command)

    def test_web_evidence_command_uses_existing_run_and_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            job = server.create_job({"video_url": "https://example.com/video", "profile": "deepseek_v4_pro"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)

            command = server.web_evidence_command(loaded)

        self.assertEqual(command[:3], [sys.executable, "-m", "video_analyzer.web_evidence"])
        self.assertIn(str(run_dir), command)
        self.assertIn("--profile", command)
        self.assertIn("deepseek_v4_pro", command)

    def test_qa_summary_reports_index_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            qa_dir = run_dir / "qa"
            qa_dir.mkdir(parents=True)
            (qa_dir / "answer_index.json").write_text(
                json.dumps({"source_count": 2, "chunk_count": 3, "warnings": [{"code": "vl_skipped"}]}),
                encoding="utf-8",
            )
            (qa_dir / "source_chunks.jsonl").write_text("{}\n", encoding="utf-8")

            summary = server.qa_summary(run_dir)

        self.assertTrue(summary["available"])
        self.assertEqual(summary["answer_index"], "qa/answer_index.json")
        self.assertEqual(summary["source_chunks"], "qa/source_chunks.jsonl")
        self.assertEqual(summary["chunk_count"], 3)
        self.assertEqual(summary["warnings"][0]["code"], "vl_skipped")

    def test_qa_answer_is_saved_to_chat_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            job = server.create_job({"video_url": "https://example.com/video", "profile": "deepseek_v4_pro"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)
            server.save_job(loaded)
            answer = {"answer": "这是已保存的回答。", "citations": [], "warnings": [], "context_chars": 12}

            with patch.object(server_mod, "resolve_api_key", return_value="0"), patch.object(
                server_mod, "ask_video_docs_result", return_value=answer
            ):
                result = server.ask_qa(job["job_id"], {"question": "怎么配置？"})
                history = server.qa_history(job["job_id"])

        self.assertEqual(result["answer"], answer["answer"])
        self.assertIn("qa/chat_history/", result["history_path"])
        self.assertEqual(history["count"], 1)
        self.assertEqual(history["messages"][0]["question"], "怎么配置？")
        self.assertEqual(history["messages"][0]["answer"], answer["answer"])
        self.assertIn("created_at", history["messages"][0])

    def test_skill_candidate_generate_and_enable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            jobs_dir = Path(tmp) / "jobs"
            run_dir = Path(tmp) / "run"
            repo_root.mkdir()
            run_dir.mkdir()
            (run_dir / "operation_manual.md").write_text("# Demo Tool Setup\n\n1. 填写 API Token。\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n\nframe_001 显示 API Token 输入框。\n", encoding="utf-8")
            (run_dir / "transcript.md").write_text("- [00:00:01 - 00:00:05] 填写 API Token。\n", encoding="utf-8")
            (run_dir / "study_guide.json").write_text(
                json.dumps(
                    {
                        "title": "Demo Tool Setup",
                        "overview": {"summary": "配置 Demo Tool 的 API Token。"},
                        "chapters": [{"index": 1, "title": "填写 Token", "summary": "在设置页填写 API Token。"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = server_mod.VideoLinkStatusServer(jobs_dir, repo_root)
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)
            server.save_job(loaded)

            generated = server.generate_skill_candidate(job["job_id"])
            enabled = server.enable_skill_candidate(job["job_id"])

            self.assertTrue(generated["available"])
            self.assertEqual(generated["status"], "needs_review")
            self.assertTrue(enabled["enabled"])
            self.assertTrue((repo_root / ".codex" / "skills" / enabled["skill_name"] / "SKILL.md").is_file())

    def test_backfilled_qa_index_on_completed_job_keeps_job_succeeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "operation_manual.md").write_text("# Manual\n\nAPI Token 设置步骤。", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n\nFrame_001 显示设置页。", encoding="utf-8")
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)
            loaded["status"] = "succeeded"
            loaded["stages"] = {
                stage: {"status": "succeeded"}
                for stage in server_mod.STAGE_ORDER
                if stage != "qa-index"
            }
            server.save_job(loaded)

            result = server.run_stage(job["job_id"], "qa-index")

        self.assertEqual(result["stages"]["qa-index"]["status"], "succeeded")
        self.assertEqual(result["status"], "succeeded")
        self.assertIsNone(result["next_stage"])
        self.assertTrue(result["summary"]["qa"]["available"])

    def test_list_jobs_returns_recent_public_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            first = server.create_job({"video_url": "https://example.com/one"})
            loaded_first = server.load_job(first["job_id"])
            loaded_first["created_at"] = "2026-01-01T00:00:00+0800"
            server.save_job(loaded_first)
            second = server.create_job({"video_url": "https://example.com/two"})

            result = server.list_jobs()

        self.assertEqual(result["total"], 2)
        self.assertEqual(result["jobs"][0]["job_id"], second["job_id"])
        self.assertEqual(result["jobs"][1]["job_id"], first["job_id"])
        self.assertEqual(result["summary"]["total"], 2)
        self.assertIn("core", result["resources"])

    def test_list_jobs_uses_lightweight_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            server.create_job({"video_url": "https://example.com/one"})

            with patch.object(server, "core_diagnostics", side_effect=AssertionError("too expensive")):
                result = server.list_jobs()

        self.assertEqual(result["total"], 1)
        self.assertNotIn("core_diagnostics", result["jobs"][0])

    def test_public_job_derives_title_from_info_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            video_dir = Path(tmp) / "video"
            video_dir.mkdir()
            (video_dir / "info.json").write_text(json.dumps({"title": "真实视频标题"}, ensure_ascii=False), encoding="utf-8")
            loaded = server.load_job(job["job_id"])
            loaded["artifacts"] = {"video_path": {"value": str(video_dir / "video.mp4")}}
            server.save_job(loaded)

            public = server.public_job(server.load_job(job["job_id"]))

        self.assertEqual(public["title"], "真实视频标题")
        self.assertEqual(public["display_title"], "真实视频标题")

    def test_delete_job_removes_status_record_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)
            server.save_job(loaded)

            result = server.delete_job(job["job_id"])

            self.assertTrue(result["deleted"])
            self.assertTrue(run_dir.exists())
            self.assertFalse((Path(tmp) / "jobs" / job["job_id"]).exists())

    def test_create_jobs_batch_partially_accepts_valid_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            result = server.create_jobs(
                {
                    "video_urls_text": "https://example.com/one\nnot-a-url\nhttps://example.com/two",
                    "run_name": "batch",
                    "auto_start": False,
                }
            )

        self.assertEqual(result["created"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["jobs"][0]["options"]["run_name"], "batch-001")
        self.assertEqual(result["jobs"][1]["options"]["run_name"], "batch-003")
        self.assertEqual(result["errors"][0]["index"], 2)

    def test_create_jobs_batch_suffixes_duplicate_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            result = server.create_jobs(
                {
                    "videoUrls": ["https://example.com/same", "https://example.com/same"],
                    "runName": "same-run",
                    "autoStart": False,
                }
            )

        self.assertEqual(result["created"], 2)
        self.assertEqual(result["duplicates"], {"https://example.com/same": 2})
        self.assertEqual([job["options"]["run_name"] for job in result["jobs"]], ["same-run-001", "same-run-002"])

    def test_resource_summary_reports_running_and_queued_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            running = server.create_job({"video_url": "https://example.com/running"})
            queued = server.create_job({"video_url": "https://example.com/queued"})
            loaded_running = server.load_job(running["job_id"])
            loaded_running["status"] = "running"
            loaded_running["runner"] = {"status": "running", "current_stage": "analyze-core", "server_pid": os.getpid()}
            loaded_running["stages"]["analyze-core"] = {"status": "running"}
            server.save_job(loaded_running)
            loaded_queued = server.load_job(queued["job_id"])
            loaded_queued["status"] = "queued"
            loaded_queued["runner"] = {
                "status": "queued",
                "current_stage": "analyze-core",
                "queued_for": "core",
                "server_pid": os.getpid(),
            }
            loaded_queued["stages"]["analyze-core"] = {"status": "queued", "queued_for": "core"}
            server.save_job(loaded_queued)

            resources = server.list_jobs()["resources"]

        self.assertEqual(resources["core"]["running_count"], 1)
        self.assertEqual(resources["core"]["queued_count"], 1)
        self.assertEqual(resources["core"]["queued"][0]["job_id"], queued["job_id"])

    def test_stage_waits_in_queue_when_resource_is_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "fast"
            loaded["stages"]["probe"] = {"status": "succeeded"}
            loaded["stages"]["prepare"] = {"status": "succeeded"}
            server.save_job(loaded)
            server.resource_locks["core"].acquire()

            def fake_locked(job_id, stage, continue_runner=False):
                current = server.load_job(job_id)
                current["stages"][stage] = {"status": "succeeded"}
                server.save_job(current)
                return server.public_job(current)

            with patch.object(server, "_run_stage_locked", side_effect=fake_locked):
                import threading
                import time

                thread = threading.Thread(target=server.run_stage, args=(job["job_id"], "analyze-core"))
                thread.start()
                for _ in range(50):
                    queued = server.load_job(job["job_id"])
                    if queued["stages"].get("analyze-core", {}).get("status") == "queued":
                        break
                    time.sleep(0.02)
                queued = server.public_job(server.load_job(job["job_id"]))
                server.resource_locks["core"].release()
                thread.join(timeout=2)

        self.assertEqual(queued["status"], "queued")
        self.assertEqual(queued["queue"]["resource"], "core")
        self.assertEqual(queued["stages"]["analyze-core"]["queue_position"], 1)
        self.assertFalse(thread.is_alive())

    def test_core_resource_is_exclusive_for_p40_subsystems(self):
        self.assertEqual(server_mod.RESOURCE_LIMITS["core"], 1)

    def test_stage_waits_for_live_persisted_resource_user_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            blockers = [
                server.create_job({"video_url": f"https://example.com/blocker-{index}", "analysis_mode": "fast"})
                for index in range(server_mod.RESOURCE_LIMITS["core"])
            ]
            waiting = server.create_job({"video_url": "https://example.com/waiting", "analysis_mode": "fast"})
            for blocker in blockers:
                blocker_loaded = server.load_job(blocker["job_id"])
                blocker_loaded["status"] = "running"
                blocker_loaded["runner"] = {"status": "running", "current_stage": "analyze-core"}
                blocker_loaded["stages"]["analyze-core"] = {
                    "status": "running",
                    "process": {"pid": os.getpid()},
                }
                server.save_job(blocker_loaded)
            waiting_loaded = server.load_job(waiting["job_id"])
            waiting_loaded["resolved_mode"] = "fast"
            waiting_loaded["stages"]["probe"] = {"status": "succeeded"}
            waiting_loaded["stages"]["prepare"] = {"status": "succeeded"}
            server.save_job(waiting_loaded)
            sleep_calls = []

            def release_blocker(_seconds):
                sleep_calls.append(_seconds)
                for blocker in blockers:
                    released = server.load_job(blocker["job_id"])
                    released["status"] = "succeeded"
                    released["runner"] = {"status": "succeeded", "current_stage": None}
                    released["stages"]["analyze-core"] = {"status": "succeeded"}
                    server.save_job(released)

            def fake_locked(job_id, stage, continue_runner=False):
                current = server.load_job(job_id)
                current["stages"][stage] = {"status": "succeeded"}
                server.save_job(current)
                return server.public_job(current)

            with patch.object(server_mod.time, "sleep", side_effect=release_blocker), patch.object(
                server, "_run_stage_locked", side_effect=fake_locked
            ):
                result = server.run_stage(waiting["job_id"], "analyze-core")

        self.assertEqual(sleep_calls, [server_mod.RESOURCE_WAIT_SECONDS])
        self.assertEqual(result["stages"]["analyze-core"]["status"], "succeeded")

    def test_live_resource_users_counts_process_even_when_stage_status_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "running"
            loaded["runner"] = {"status": "running", "current_stage": "analyze-core"}
            loaded["stages"]["analyze-core"] = {"status": "queued", "process": {"pid": os.getpid()}}
            server.save_job(loaded)

            users = server.live_resource_users("core")

        self.assertEqual([user["job_id"] for user in users], [job["job_id"]])

    def test_load_job_requeues_orphaned_running_stage_when_process_is_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "running"
            loaded["runner"] = {"status": "running", "current_stage": "final-publish"}
            loaded["stages"]["final-publish"] = {"status": "running"}
            server.save_job(loaded)

            recovered = server.public_job(server.load_job(job["job_id"]))

        self.assertEqual(recovered["status"], "queued")
        self.assertEqual(recovered["runner"]["status"], "queued")
        self.assertEqual(recovered["runner"]["queued_for"], "final-publish")
        self.assertEqual(recovered["stages"]["final-publish"]["status"], "queued")
        self.assertIn("queued for retry", recovered["runner"]["error"])
        self.assertEqual(recovered["queue"]["resource"], "final-publish")

    def test_load_job_requeues_legacy_process_gone_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "failed"
            loaded["runner"] = {
                "status": "failed",
                "current_stage": "analyze-core",
                "error": server_mod.ORPHANED_PROCESS_GONE_MESSAGE,
            }
            loaded["stages"]["analyze-core"] = {"status": "failed", "error": server_mod.ORPHANED_PROCESS_GONE_MESSAGE}
            server.save_job(loaded)

            recovered = server.load_job(job["job_id"])

        self.assertEqual(recovered["status"], "queued")
        self.assertEqual(recovered["runner"]["status"], "queued")
        self.assertEqual(recovered["runner"]["queued_for"], "core")
        self.assertEqual(recovered["stages"]["analyze-core"]["status"], "queued")
        self.assertNotIn("error", recovered["stages"]["analyze-core"])

    def test_auto_resume_delays_requeued_interrupted_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "running"
            loaded["runner"] = {
                "status": "running",
                "current_stage": "analyze-core",
                "server_pid": 99999999,
            }
            loaded["stages"]["analyze-core"] = {"status": "running"}
            server.save_job(loaded)

            with patch.object(server_mod.VideoLinkStatusServer, "start_run", autospec=True) as start_run:
                resumed = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT, auto_resume=True)
                recovered = resumed.load_job(job["job_id"])

        start_run.assert_not_called()
        self.assertEqual(recovered["status"], "queued")
        self.assertTrue(resumed.auto_retry_info(recovered).get("auto_retry"))

    def test_auto_retry_starts_ready_interrupted_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "queued"
            loaded["runner"] = {
                "status": "queued",
                "current_stage": "final-publish",
                "queued_for": "final-publish",
                "error": server_mod.ORPHANED_PROCESS_REQUEUE_MESSAGE,
                "server_pid": os.getpid(),
            }
            loaded["stages"]["final-publish"] = {
                "status": "queued",
                "queued_at": "2000-01-01T00:00:00+0800",
                "queued_for": "final-publish",
                "retry_reason": server_mod.ORPHANED_PROCESS_REQUEUE_MESSAGE,
            }
            server.save_job(loaded)

            with patch.object(server, "resource_has_running_work", return_value=False), patch.object(server, "start_run") as start_run:
                started = server.auto_retry_queued_jobs_once(now=time.time())

        self.assertEqual(started, [job["job_id"]])
        start_run.assert_called_once_with(job["job_id"])

    def test_auto_retry_waits_until_resource_is_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "queued"
            loaded["runner"] = {
                "status": "queued",
                "current_stage": "final-publish",
                "queued_for": "final-publish",
                "error": server_mod.ORPHANED_PROCESS_REQUEUE_MESSAGE,
                "server_pid": os.getpid(),
            }
            loaded["stages"]["final-publish"] = {
                "status": "queued",
                "queued_at": "2000-01-01T00:00:00+0800",
                "queued_for": "final-publish",
                "retry_reason": server_mod.ORPHANED_PROCESS_REQUEUE_MESSAGE,
            }
            server.save_job(loaded)

            with patch.object(server, "resource_has_running_work", return_value=True), patch.object(server, "start_run") as start_run:
                started = server.auto_retry_queued_jobs_once(now=time.time())

        self.assertEqual(started, [])
        start_run.assert_not_called()

    def test_ray_frame_oom_requeues_core_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "fast"
            loaded["stages"]["probe"] = {"status": "succeeded"}
            loaded["stages"]["prepare"] = {"status": "succeeded"}
            server.save_job(loaded)
            error = subprocess.CalledProcessError(1, ["frame-worker"])
            error.output = (
                "RuntimeError: Ray frame driver failed on agx with code 1\n"
                "ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.\n"
                "Memory on the node was 60.81GB / 61.36GB, which exceeds the memory usage threshold of 0.99.\n"
            )

            with patch.object(server, "stage_analyze_core", side_effect=error):
                result = server.run_stage(job["job_id"], "analyze-core")

            reloaded = server.load_job(job["job_id"])

        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["runner"]["status"], "queued")
        self.assertEqual(result["runner"]["queued_for"], "core")
        self.assertEqual(result["stages"]["analyze-core"]["status"], "queued")
        self.assertEqual(result["stages"]["analyze-core"]["retry_reason"], server_mod.TRANSIENT_RESOURCE_REQUEUE_MESSAGE)
        self.assertIn("Ray frame driver failed", result["stages"]["analyze-core"]["last_error"])
        self.assertTrue(server.auto_retry_info(reloaded).get("auto_retry"))

    def test_load_job_requeues_legacy_ray_frame_oom_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "failed"
            loaded["resolved_mode"] = "fast"
            loaded["runner"] = {
                "status": "failed",
                "current_stage": "analyze-core",
                "error": "analyze-core failed: Ray frame driver failed on agx with code 1",
            }
            loaded["stages"]["probe"] = {"status": "succeeded"}
            loaded["stages"]["prepare"] = {"status": "succeeded"}
            loaded["stages"]["analyze-core"] = {
                "status": "failed",
                "error": (
                    "Ray frame driver failed on agx with code 1\n"
                    "ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.\n"
                    "which exceeds the memory usage threshold of 0.99.\n"
                ),
            }
            server.save_job(loaded)

            recovered = server.load_job(job["job_id"])

        self.assertEqual(recovered["status"], "queued")
        self.assertEqual(recovered["runner"]["status"], "queued")
        self.assertEqual(recovered["runner"]["queued_for"], "core")
        self.assertEqual(recovered["runner"]["error"], server_mod.TRANSIENT_RESOURCE_REQUEUE_MESSAGE)
        self.assertEqual(recovered["stages"]["analyze-core"]["status"], "queued")
        self.assertEqual(recovered["stages"]["analyze-core"]["retry_reason"], server_mod.TRANSIENT_RESOURCE_REQUEUE_MESSAGE)
        self.assertIn("Ray frame driver failed", recovered["stages"]["analyze-core"]["last_error"])

    def test_load_job_keeps_orphaned_stage_running_when_process_is_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "running"
            loaded["runner"] = {"status": "running", "current_stage": "final-publish"}
            loaded["stages"]["final-publish"] = {
                "status": "running",
                "process": {"pid": os.getpid(), "command": ["sleep", "10"]},
            }
            server.save_job(loaded)

            recovered = server.load_job(job["job_id"])

        self.assertEqual(recovered["status"], "running")
        self.assertEqual(recovered["runner"]["status"], "running")
        self.assertTrue(recovered["stages"]["final-publish"]["process"]["alive"])
        self.assertTrue(recovered["stages"]["final-publish"]["process"]["orphaned"])

    def test_stop_job_terminates_running_process_tree_and_marks_failed(self):
        process = subprocess.Popen(["bash", "-c", "sleep 60 & wait"])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
                job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
                loaded = server.load_job(job["job_id"])
                loaded["status"] = "running"
                loaded["runner"] = {"status": "running", "current_stage": "analyze-core"}
                loaded["stages"]["analyze-core"] = {"status": "running", "process": {"pid": process.pid}}
                server.save_job(loaded)

                result = server.stop_job(job["job_id"])
                stopped = server.load_job(job["job_id"])

            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

        self.assertTrue(result["stopped"])
        self.assertIn(process.pid, result["stopped_pids"])
        self.assertEqual(stopped["status"], "failed")
        self.assertEqual(stopped["runner"]["error"], "stopped by user")
        self.assertEqual(stopped["stages"]["analyze-core"]["status"], "failed")

    def test_stop_job_marks_queued_job_failed_without_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "queued"
            loaded["runner"] = {"status": "queued", "current_stage": "analyze-core", "queued_for": "core"}
            loaded["stages"]["analyze-core"] = {"status": "queued", "queued_for": "core"}
            server.save_job(loaded)

            result = server.stop_job(job["job_id"])
            stopped = server.load_job(job["job_id"])

        self.assertTrue(result["stopped"])
        self.assertEqual(result["stopped_pids"], [])
        self.assertEqual(stopped["status"], "failed")
        self.assertEqual(stopped["stages"]["analyze-core"]["error"], "stopped by user")

    def test_load_job_marks_final_publish_succeeded_when_exports_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "final_publish_summary.json").write_text("{}", encoding="utf-8")
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)
            loaded["status"] = "running"
            loaded["runner"] = {"status": "running", "current_stage": "final-publish"}
            for stage in server_mod.STAGE_ORDER:
                if stage != "final-publish":
                    loaded["stages"][stage] = {"status": "succeeded"}
            loaded["stages"]["final-publish"] = {"status": "running", "process": {"pid": 99999999}}
            server.save_job(loaded)

            recovered = server.load_job(job["job_id"])

        self.assertEqual(recovered["status"], "succeeded")
        self.assertEqual(recovered["runner"]["status"], "succeeded")
        self.assertEqual(recovered["stages"]["final-publish"]["status"], "succeeded")

    def test_open_run_dir_launches_code_for_succeeded_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "succeeded"
            loaded["run_dir"] = str(run_dir)
            server.save_job(loaded)

            with patch.object(server_mod.shutil, "which", return_value="/usr/bin/code"), patch.object(
                server_mod.subprocess, "Popen"
            ) as popen:
                result = server.open_run_dir(job["job_id"])

        self.assertTrue(result["opened"])
        self.assertEqual(result["run_dir"], str(run_dir.resolve()))
        self.assertEqual(result["command"], ["code", str(run_dir.resolve())])
        popen.assert_called_once_with(
            ["/usr/bin/code", str(run_dir.resolve())],
            stdin=server_mod.subprocess.DEVNULL,
            stdout=server_mod.subprocess.DEVNULL,
            stderr=server_mod.subprocess.DEVNULL,
            start_new_session=True,
        )

    def test_open_run_dir_rejects_incomplete_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "running"
            loaded["run_dir"] = str(run_dir)
            server.save_job(loaded)

            with self.assertRaises(server_mod.BridgeError) as raised:
                server.open_run_dir(job["job_id"])

        self.assertEqual(raised.exception.status, server_mod.HTTPStatus.CONFLICT)

    def test_probe_auto_routes_long_video_to_long_talk_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})

            with patch.object(server_mod, "probe_duration_seconds", return_value=3600):
                result = server.run_stage(job["job_id"], "probe")

        self.assertEqual(result["resolved_mode"], "long-talk-fast")
        self.assertEqual(result["stages"]["probe"]["status"], "succeeded")

    def test_manual_stage_completion_queues_next_stage_without_fake_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})

            with patch.object(server_mod, "probe_duration_seconds", return_value=600):
                result = server.run_stage(job["job_id"], "probe")

        self.assertEqual(result["stages"]["probe"]["status"], "succeeded")
        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["current_stage"], "prepare")
        self.assertEqual(result["runner"]["status"], "queued")
        self.assertEqual(result["runner"]["current_stage"], "prepare")
        self.assertEqual(result["stages"]["prepare"]["status"], "queued")
        self.assertNotEqual(result["runner"]["status"], "running")

    def test_probe_auto_uses_focus_prompt_for_fast_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "focus_prompt": "快速给我一个摘要，先大概看看"})

            with patch.object(server_mod, "probe_duration_seconds", return_value=600):
                result = server.run_stage(job["job_id"], "probe")

        self.assertEqual(result["resolved_mode"], "fast")
        self.assertIn("quick", result["resolved_mode_reason"])

    def test_probe_auto_uses_focus_prompt_for_deep_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "focus_prompt": "生成最终发布版，步骤和参数不要漏"})

            with patch.object(server_mod, "probe_duration_seconds", return_value=600):
                result = server.run_stage(job["job_id"], "probe")

        self.assertEqual(result["resolved_mode"], "deep")
        self.assertIn("deep", result["resolved_mode_reason"])

    def test_probe_auto_uses_long_talk_for_long_subtitle_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "focus_prompt": "这是长视频播客，按字幕和章节梳理"})

            with patch.object(server_mod, "probe_duration_seconds", return_value=3600):
                result = server.run_stage(job["job_id"], "probe")

        self.assertEqual(result["resolved_mode"], "long-talk-fast")
        self.assertIn("long video", result["resolved_mode_reason"])

    def test_probe_explicit_mode_ignores_focus_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "balanced", "focus_prompt": "快速摘要"})

            with patch.object(server_mod, "probe_duration_seconds", return_value=3600):
                result = server.run_stage(job["job_id"], "probe")

        self.assertEqual(result["resolved_mode"], "balanced")
        self.assertEqual(result["resolved_mode_reason"], "explicit mode selected")

    def test_analyze_core_stage_parses_run_dir_from_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp) / "jobs"
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            server = server_mod.VideoLinkStatusServer(jobs_dir, REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            server.run_stage(job["job_id"], "probe")

            def fake_run(command, log_path, on_start=None, append_log=False):
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                Path(log_path).write_text(f"[done] run_dir: {run_dir}\n", encoding="utf-8")
                return {"stdout_tail": ["ok"]}

            with patch.object(server, "run_command", side_effect=fake_run):
                loaded = server.load_job(job["job_id"])
                loaded["stages"]["prepare"] = {"status": "succeeded"}
                server.save_job(loaded)
                result = server.run_stage(job["job_id"], "operation")

        self.assertEqual(result["run_dir"], str(run_dir.resolve()))
        self.assertEqual(result["stages"]["analyze-core"]["status"], "succeeded")

    def test_start_run_executes_remaining_stages_in_background_runner(self):
        class ImmediateThread:
            def __init__(self, target, args=(), daemon=None):
                self.target = target
                self.args = args
                self.daemon = daemon
                self._alive = False

            def start(self):
                self._alive = True
                self.target(*self.args)
                self._alive = False

            def is_alive(self):
                return self._alive

        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            calls = []

            def fake_run_stage(job_id, stage, continue_runner=False):
                calls.append(stage)
                loaded = server.load_job(job_id)
                loaded["stages"][stage] = {"status": "succeeded", "duration_seconds": 0.001}
                if stage == "probe":
                    loaded["resolved_mode"] = "fast"
                if stage == "analyze-core":
                    loaded["run_dir"] = str(Path(tmp))
                loaded["updated_at"] = server_mod.iso_now()
                server.save_job(loaded)
                return server.public_job(loaded)

            with patch.object(server_mod.threading, "Thread", ImmediateThread), patch.object(
                server, "run_stage", side_effect=fake_run_stage
            ):
                result = server.start_run(job["job_id"])

        self.assertEqual(calls, server_mod.STAGE_ORDER)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["runner"]["status"], "succeeded")
        self.assertEqual(result["progress"]["percent"], 100)
        self.assertIn("/?job=", result["dashboard_url"])

    def test_public_job_includes_progress_and_error_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "failed"
            loaded["stages"]["probe"] = {"status": "succeeded"}
            loaded["stages"]["analyze-core"] = {
                "status": "failed",
                "error": "ASR endpoint timed out",
                "log_path": str(Path(tmp) / "analyze-core.log"),
            }

        result = server.public_job(loaded)

        self.assertEqual(result["progress"]["completed"], 1)
        self.assertEqual(result["progress"]["failed"], 1)
        self.assertEqual(result["error_summary"]["stage"], "analyze-core")
        self.assertIn("ASR endpoint timed out", result["error_summary"]["message"])
        self.assertEqual(result["core_diagnostics"]["status"], "error")
        self.assertEqual(result["core_diagnostics"]["issues"][0]["code"], "core-stage-failed")

    def test_core_diagnostics_reports_efficiency_bottleneck_from_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "operation_manual.md").write_text("# Manual\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            (run_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "frames_extracted": 24,
                            "timings": {
                                "asr_seconds": 40,
                                "ocr_seconds": 20,
                                "vl_seconds": 75,
                                "manual_generation_seconds": 10,
                                "total_seconds": 120,
                            },
                            "ocr_keyframes": {
                                "scan_frames_count": 100,
                                "ocr_candidate_frames_count": 24,
                                "ocr_frames_count": 18,
                                "ocr_text_events_count": 12,
                            },
                            "frame_selection": {
                                "video_duration_seconds": 240,
                                "vl_frames_count": 24,
                            },
                        },
                        "frame_analyses": [],
                    }
                ),
                encoding="utf-8",
            )
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)
            loaded["stages"]["analyze-core"] = {"status": "succeeded"}

            diagnostics = server.public_job(loaded)["core_diagnostics"]

        self.assertEqual(diagnostics["status"], "watch")
        self.assertEqual(diagnostics["efficiency"]["runtime_ratio"], 0.5)
        self.assertEqual(diagnostics["efficiency"]["bottleneck"]["key"], "vl_seconds")
        self.assertEqual(diagnostics["efficiency"]["counts"]["vl_frames"], 24)
        self.assertIn("dominant-core-bottleneck", [item["code"] for item in diagnostics["issues"]])

    def test_core_diagnostics_warns_when_minicpm_vl_concurrency_is_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["stages"]["analyze-core"] = {
                "status": "running",
                "artifacts": {
                    "command": [
                        "python",
                        "-m",
                        "video_analyzer.cli",
                        "--vision-model",
                        "minicpm-v-4.5-v100",
                        "--vl-concurrency",
                        "3",
                    ]
                },
            }

            with patch.object(server, "gpu_snapshot", return_value={"status": "ok", "devices": []}):
                diagnostics = server.public_job(loaded)["core_diagnostics"]

        self.assertEqual(diagnostics["status"], "warning")
        self.assertIn("low-minicpm-vl-concurrency", [item["code"] for item in diagnostics["issues"]])
        self.assertEqual(diagnostics["gpu"]["status"], "ok")

    def test_core_diagnostics_warns_when_minicpm_gpu_workers_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["stages"]["analyze-core"] = {
                "status": "running",
                "artifacts": {
                    "command": [
                        "python",
                        "-m",
                        "video_analyzer.cli",
                        "--vision-model",
                        "minicpm-v-4.5-v100",
                        "--vl-concurrency",
                        "6",
                    ]
                },
            }
            gpu = {
                "status": "ok",
                "devices": [
                    {
                        "index": 0,
                        "name": "Tesla P40",
                        "memory_total_mib": 24576,
                        "memory_used_mib": 7400,
                        "utilization_gpu_percent": 50,
                        "power_draw_w": 68.5,
                        "power_limit_w": 250.0,
                        "processes": [
                            {"pid": 123, "process_name": "llama-server", "used_memory_mib": 7300},
                        ],
                    }
                ],
            }

            with patch.object(server, "gpu_snapshot", return_value=gpu):
                diagnostics = server.public_job(loaded)["core_diagnostics"]

        self.assertEqual(diagnostics["gpu"], gpu)
        self.assertIn("minicpm-gpu-worker-count-low", [item["code"] for item in diagnostics["issues"]])

    def test_public_job_includes_video_preview_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"fake video")
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["video_path"] = str(video)
            loaded["stages"]["probe"] = {"status": "succeeded", "artifacts": {"duration_seconds": 125}}
            result = server.public_job(loaded)

        self.assertTrue(result["preview"]["video_ready"])
        self.assertEqual(result["preview"]["video_url"], f"/api/video-link/jobs/{job['job_id']}/video")
        self.assertEqual(result["preview"]["duration_seconds"], 125)

    def test_public_job_includes_youtube_source_player(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            job = server.create_job({"video_url": "https://www.youtube.com/watch?v=abc123XYZ_0"})

            result = server.public_job(server.load_job(job["job_id"]))

        self.assertEqual(result["source_player"]["provider"], "youtube")
        self.assertTrue(result["source_player"]["can_embed"])
        self.assertEqual(result["source_player"]["embed_url"], "https://www.youtube.com/embed/abc123XYZ_0")
        self.assertEqual(result["source_player"]["watch_url"], "https://www.youtube.com/watch?v=abc123XYZ_0")

    def test_public_job_includes_bilibili_source_player(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            job = server.create_job({"video_url": "https://www.bilibili.com/video/BV1xx411c7mD/"})

            result = server.public_job(server.load_job(job["job_id"]))

        self.assertEqual(result["source_player"]["provider"], "bilibili")
        self.assertTrue(result["source_player"]["can_embed"])
        self.assertEqual(result["source_player"]["embed_url"], "https://player.bilibili.com/player.html?bvid=BV1xx411c7mD")
        self.assertEqual(result["source_player"]["watch_url"], "https://www.bilibili.com/video/BV1xx411c7mD")

    def test_public_job_marks_unknown_source_as_external_player(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/watch/123"})

            result = server.public_job(server.load_job(job["job_id"]))

        self.assertEqual(result["source_player"]["provider"], "external")
        self.assertFalse(result["source_player"]["can_embed"])
        self.assertEqual(result["source_player"]["watch_url"], "https://example.com/watch/123")

    def test_frame_time_map_indexes_manifest_paths_and_manual_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            frame_path = frames_dir / "frame_003.jpg"
            frame_path.write_bytes(b"fake frame")
            write_frame_manifest([Frame(3, frame_path, 42.5, 0.8)], output_dir, source="local")
            server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
            job = server.create_job({"video_url": "https://www.youtube.com/watch?v=abc123XYZ_0"})
            loaded = server.load_job(job["job_id"])
            loaded["run_dir"] = str(output_dir)
            server.save_job(loaded)

            result = server.frame_time_map(job["job_id"])

        self.assertTrue(result["available"])
        self.assertEqual(result["frames"]["frames/frame_003.jpg"]["timestamp_sec"], 42.5)
        self.assertEqual(result["frames"]["manual_assets/frame_003.jpg"]["timestamp_label"], "0:42")

    def test_public_job_falls_back_to_downloaded_video_for_preview(self):
        video_dir = REPO_ROOT / "downloads" / "url-videos" / "preview-fallback-test"
        video_dir.mkdir(parents=True, exist_ok=True)
        video = video_dir / "video.mp4"
        video.write_bytes(b"fake video")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                server = server_mod.VideoLinkStatusServer(Path(tmp) / "jobs", REPO_ROOT)
                job = server.create_job({"video_url": "https://www.youtube.com/watch?v=preview-fallback-test"})

                result = server.public_job(server.load_job(job["job_id"]))

            self.assertTrue(result["preview"]["video_ready"])
            self.assertEqual(result["preview"]["video_url"], f"/api/video-link/jobs/{job['job_id']}/video")
        finally:
            video.unlink(missing_ok=True)
            try:
                video_dir.rmdir()
            except OSError:
                pass

    def test_preview_video_file_rejects_missing_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})

            with self.assertRaises(server_mod.BridgeError) as raised:
                server.preview_video_file(job["job_id"])

        self.assertEqual(raised.exception.status, server_mod.HTTPStatus.CONFLICT)

    def test_stage_log_can_return_tail_or_full_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            log_path = server.stage_log_path(job["job_id"], "prepare")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

            tail = server.stage_log(job["job_id"], "prepare", limit=2)
            full = server.stage_log(job["job_id"], "prepare", full=True)

        self.assertEqual(tail["lines"], ["two", "three"])
        self.assertEqual(tail["text"], "")
        self.assertEqual(full["lines"], ["one", "two", "three"])
        self.assertEqual(full["text"], "one\ntwo\nthree\n")

    def test_core_progress_parses_current_substep_and_durations(self):
        text = "\n".join(
            [
                "[jetson-ray] existing cluster is ready",
                "2026-05-17 01:12:36,681 - INFO - Extracting audio from video...",
                "2026-05-17 01:12:49,609 - INFO - Transcribing audio...",
                "2026-05-17 01:15:00,000 - INFO - Extracting frames from video using model qwen...",
                "2026-05-17 01:15:30,000 - INFO - Running OCR on extracted frames...",
            ]
        )

        progress = server_mod.parse_core_progress(text, "running")

        by_id = {step["id"]: step for step in progress["steps"]}
        self.assertEqual(progress["current_step"], "ocr")
        self.assertEqual(by_id["audio"]["status"], "succeeded")
        self.assertEqual(by_id["asr"]["status"], "succeeded")
        self.assertEqual(by_id["frames"]["duration_seconds"], 30.0)
        self.assertEqual(by_id["ocr"]["status"], "running")
        self.assertGreater(progress["percent"], 40)
        self.assertLess(progress["percent"], 80)

    def test_core_progress_reports_queued_stale_signals_without_live_current_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "queued"
            loaded["runner"] = {"status": "queued", "current_stage": "analyze-core", "queued_for": "core"}
            loaded["stages"]["probe"] = {"status": "succeeded"}
            loaded["stages"]["prepare"] = {"status": "succeeded"}
            loaded["stages"]["analyze-core"] = {
                "status": "queued",
                "queued_for": "core",
                "log_path": str(server.stage_log_path(job["job_id"], "analyze-core")),
            }
            log_path = server.stage_log_path(job["job_id"], "analyze-core")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "\n".join(
                    [
                        "2026-05-17 01:12:49,609 - INFO - Transcribing audio...",
                        "2026-05-17 01:14:00,000 - INFO - ASR succeeded with provider: vibevoice",
                        "2026-05-17 01:15:00,000 - INFO - Extracting frames from video using model qwen...",
                        "2026-05-17 01:15:30,000 - INFO - Extracted 72 screen keyframes",
                        "2026-05-17 01:15:31,000 - INFO - Running OCR on extracted frames...",
                    ]
                ),
                encoding="utf-8",
            )
            server.save_job(loaded)

            progress = server.public_job(server.load_job(job["job_id"]))["stage_progress"]

        self.assertEqual(progress["status"], "queued")
        self.assertFalse(progress["live"])
        self.assertTrue(progress["stale"])
        self.assertIsNone(progress["current_step"])
        self.assertEqual(progress["last_signal_label"], "OCR关键帧选择/执行")
        self.assertIn("等待 core #1/1", progress["summary"])
        self.assertGreater(progress["percent"], 40)

    def test_core_progress_advances_to_vl_when_vl_signal_appears(self):
        text = "\n".join(
            [
                "2026-05-17 01:12:49,609 - INFO - Transcribing audio...",
                "2026-05-17 01:14:00,000 - INFO - ASR succeeded with provider: vibevoice",
                "2026-05-17 01:15:00,000 - INFO - Extracting frames from video using model qwen...",
                "2026-05-17 01:15:30,000 - INFO - Extracted 72 screen keyframes",
                "2026-05-17 01:15:31,000 - INFO - Running OCR on extracted frames...",
                "2026-05-17 01:17:30,000 - INFO - Selecting and analyzing VL frames...",
            ]
        )

        progress = server_mod.parse_core_progress(text, "running")

        self.assertEqual(progress["current_step"], "vl")
        self.assertGreaterEqual(progress["percent"], 70)

    def test_core_progress_recognizes_resource_lock_wait_signals(self):
        text = "\n".join(
            [
                "2026-05-17 01:12:36,681 - INFO - Extracting audio from video...",
                "2026-05-17 01:12:49,609 - INFO - [resource-lock] waiting resource=asr limit=1 owner=/tmp/run waited=0.000s",
            ]
        )

        progress = server_mod.parse_core_progress(text, "running")

        self.assertEqual(progress["current_step"], "asr")
        by_id = {step["id"]: step for step in progress["steps"]}
        self.assertIn("resource=asr", by_id["asr"]["message"])

    def test_core_progress_recognizes_local_model_lock_wait_signals(self):
        text = "\n".join(
            [
                "2026-05-17 01:12:36,681 - INFO - Extracting audio from video...",
                "2026-05-17 01:12:49,609 - INFO - [local-model-lock] waiting stage=core owner=/tmp/run waited=0.000s",
            ]
        )

        progress = server_mod.parse_core_progress(text, "running")

        self.assertEqual(progress["current_step"], "local_model")
        by_id = {step["id"]: step for step in progress["steps"]}
        self.assertIn("local-model-lock", by_id["local_model"]["message"])

    def test_core_progress_prefers_progress_json_over_stale_asr_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            video_dir = Path(tmp) / "video"
            run_dir = video_dir / "operation-manual"
            run_dir.mkdir(parents=True)
            (run_dir / "progress.json").write_text(
                json.dumps({"current_step": "frames", "status": "running", "message": "extracting candidate frames"}),
                encoding="utf-8",
            )
            loaded["video_dir"] = str(video_dir)
            loaded["status"] = "running"
            loaded["runner"] = {"status": "running", "current_stage": "analyze-core"}
            loaded["stages"]["analyze-core"] = {
                "status": "running",
                "process": {"pid": os.getpid()},
                "log_path": str(server.stage_log_path(job["job_id"], "analyze-core")),
            }
            log_path = server.stage_log_path(job["job_id"], "analyze-core")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("2026-05-17 01:12:49,609 - INFO - Transcribing audio...\n", encoding="utf-8")
            server.save_job(loaded)

            progress = server.public_job(server.load_job(job["job_id"]))["stage_progress"]

        by_id = {step["id"]: step for step in progress["steps"]}
        self.assertEqual(progress["current_step"], "frames")
        self.assertEqual(by_id["asr_done"]["status"], "succeeded")
        self.assertEqual(by_id["frames"]["status"], "running")
        self.assertEqual(progress["source"], "progress_json")

    def test_core_progress_infers_asr_done_from_transcript_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "run_name": "operation-manual"})
            loaded = server.load_job(job["job_id"])
            video_dir = Path(tmp) / "video"
            run_dir = video_dir / "operation-manual"
            run_dir.mkdir(parents=True)
            (run_dir / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
            loaded["video_dir"] = str(video_dir)
            loaded["status"] = "running"
            loaded["runner"] = {"status": "running", "current_stage": "analyze-core"}
            loaded["stages"]["analyze-core"] = {
                "status": "running",
                "process": {"pid": os.getpid()},
                "log_path": str(server.stage_log_path(job["job_id"], "analyze-core")),
            }
            log_path = server.stage_log_path(job["job_id"], "analyze-core")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("2026-05-17 01:12:49,609 - INFO - Transcribing audio...\n", encoding="utf-8")
            server.save_job(loaded)

            progress = server.public_job(server.load_job(job["job_id"]))["stage_progress"]

        by_id = {step["id"]: step for step in progress["steps"]}
        self.assertEqual(progress["current_step"], "asr_done")
        self.assertEqual(by_id["asr"]["status"], "succeeded")
        self.assertEqual(by_id["asr_done"]["status"], "running")

    def test_prepare_progress_parses_download_and_context_substeps(self):
        text = "\n".join(
            [
                "Extracting cookies from chrome",
                "Extracted 966 cookies from chrome",
                "[youtube] Extracting URL: https://www.youtube.com/watch?v=abc",
                "[youtube] abc: Downloading webpage",
                "[youtube] [jsc:node] Solving JS challenges using node",
                "[download]  49.3% of   16.21MiB at   44.97MiB/s ETA 00:00",
                "[Merger] Merging formats into \"download.mp4\"",
                "[download] video: downloads/url-videos/abc/video.mp4",
                "[download] context: downloads/url-videos/abc/page_context.md",
                "[download] subtitle transcript: not available; analyzer will use configured ASR",
            ]
        )

        progress = server_mod.parse_stage_progress("prepare", text, "running")

        by_id = {step["id"]: step for step in progress["steps"]}
        self.assertEqual(progress["current_step"], "transcript")
        self.assertEqual(by_id["cookies"]["status"], "succeeded")
        self.assertEqual(by_id["download"]["status"], "succeeded")
        self.assertEqual(by_id["transcript"]["status"], "running")
        self.assertIn("subtitle transcript", by_id["transcript"]["message"])

    def test_probe_progress_keeps_mode_resolution_as_distinct_substep(self):
        text = "probe stage started\nresolved mode: long-talk-fast\n"

        progress = server_mod.parse_stage_progress("probe", text, "succeeded")

        by_id = {step["id"]: step for step in progress["steps"]}
        self.assertEqual(by_id["probe"]["status"], "succeeded")
        self.assertEqual(by_id["resolve"]["status"], "succeeded")
        self.assertIn("long-talk-fast", by_id["resolve"]["message"])

    def test_later_stage_progress_parses_final_publish_substeps(self):
        text = "\n".join(
            [
                "[images] exists: baoyu_images/final/operation_manual_cover.png",
                "[docs] multidoc",
                "[docs] deep-v2",
                "[export] skipped pdf",
                "[verify] pdf=skipped",
                "[summary] /tmp/run/final_publish_summary.json",
                "[send] skipped",
            ]
        )

        progress = server_mod.parse_stage_progress("final-publish", text, "succeeded")

        by_id = {step["id"]: step for step in progress["steps"]}
        self.assertEqual(by_id["images"]["status"], "succeeded")
        self.assertEqual(by_id["export"]["status"], "succeeded")
        self.assertEqual(by_id["verify"]["status"], "succeeded")
        self.assertEqual(by_id["send"]["status"], "succeeded")

    def test_verify_progress_uses_synthetic_signals_when_stage_has_no_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            loaded = server.load_job(job["job_id"])
            loaded["runner"] = {"status": "running", "current_stage": "verify-core", "server_pid": os.getpid()}
            loaded["stages"]["verify-core"] = {"status": "succeeded"}

            progress = server.public_job(loaded)["stage_progress"]

        by_id = {step["id"]: step for step in progress["steps"]}
        self.assertEqual(progress["stage"], "verify-core")
        self.assertEqual(by_id["check"]["status"], "succeeded")
        self.assertEqual(by_id["complete"]["status"], "succeeded")

    def test_create_page_contains_form_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            html = server_mod.render_create_page(server.options())

        for needle in (
            'id="video_url"',
            'id="analysis_mode"',
            'id="profile"',
            'id="cookies_from_browser"',
            'id="keep_existing"',
            'id="include_subtitles"',
            'id="include_comments"',
            'id="subtitle_langs"',
        ):
            self.assertIn(needle, html)
        self.assertIn("/api/video-link/jobs", html)

    def test_dashboard_contains_retry_run_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            html = server_mod.render_job_dashboard(job)

        self.assertIn('id="runButton"', html)
        self.assertIn("/run", html)
        self.assertIn("/open-run-dir", html)

    def test_dashboard_contains_log_copy_and_core_progress_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            html = server_mod.render_job_dashboard(job)

        self.assertIn('id="copyLogButton"', html)
        self.assertIn('id="corePanel"', html)
        self.assertIn("文档预览", html)
        self.assertIn("renderDocumentPreview", html)
        self.assertIn("Mermaid 源码", html)
        self.assertIn("?full=1", html)
        self.assertIn("copyText", html)
        self.assertIn('document.execCommand("copy")', html)
        self.assertIn("查看日志", html)


if __name__ == "__main__":
    unittest.main()
