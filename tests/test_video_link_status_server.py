import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO_ROOT / "tools" / "video_link_status_server.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


server_mod = load_module(SERVER_PATH, "video_link_status_server")


class VideoLinkStatusServerTests(unittest.TestCase):
    def test_options_include_defaults_and_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            options = server.options()

        self.assertEqual(options["defaults"]["analysis_mode"], "auto")
        self.assertEqual(options["defaults"]["profile"], "deepseek_v4_flash")
        self.assertEqual(options["defaults"]["cookies_from_browser"], "chrome")
        self.assertIn("balanced", options["choices"]["analysis_modes"])
        self.assertIn("deepseek_v4_flash", options["choices"]["profiles"])

    def test_create_job_saves_common_and_collection_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job(
                {
                    "videoUrl": "https://example.com/video",
                    "analysisMode": "deep",
                    "profile": "deepseek_v4_flash",
                    "runName": "../operation manual!",
                    "cookiesFromBrowser": "none",
                    "skipImages": True,
                    "keepExisting": False,
                    "includeSubtitles": False,
                    "preferSubtitleTranscript": True,
                    "includeComments": False,
                    "maxComments": "12",
                    "subtitleLangs": "en,zh",
                    "refreshContext": True,
                }
            )

        self.assertEqual(job["options"]["analysis_mode"], "deep")
        self.assertEqual(job["options"]["run_name"], "operation-manual")
        self.assertEqual(job["options"]["cookies_from_browser"], "")
        self.assertTrue(job["options"]["skip_images"])
        self.assertFalse(job["options"]["keep_existing"])
        self.assertFalse(job["options"]["include_subtitles"])
        self.assertTrue(job["options"]["prefer_subtitle_transcript"])
        self.assertFalse(job["options"]["include_comments"])
        self.assertEqual(job["options"]["max_comments"], 12)
        self.assertEqual(job["options"]["subtitle_langs"], "en,zh")
        self.assertTrue(job["options"]["refresh_context"])

    def test_command_mapping_for_collection_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job(
                {
                    "video_url": "https://example.com/video",
                    "analysis_mode": "fast",
                    "profile": "deepseek_v4_flash",
                    "cookies_from_browser": "none",
                    "keep_existing": False,
                    "include_subtitles": False,
                    "prefer_subtitle_transcript": True,
                    "include_comments": False,
                    "max_comments": 8,
                    "subtitle_langs": "en-US,en",
                    "refresh_context": True,
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
        self.assertNotIn("--keep-existing", command)
        self.assertNotIn("--cookies-from-browser", command)

    def test_command_mapping_defaults_keep_downloads_and_browser_cookies(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "fast"

        command = server.prepare_command(loaded)

        self.assertIn("--download-only", command)
        self.assertIn("--keep-existing", command)
        self.assertIn("--cookies-from-browser", command)
        self.assertIn("chrome", command)
        self.assertIn("--include-subtitles", command)
        self.assertIn("--include-comments", command)

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
        self.assertEqual(command[command.index("--profile") + 1], "deepseek_v4_flash")
        self.assertNotIn("--pipeline-mode", command)

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
        self.assertIn("--jobs", command)
        self.assertEqual(command[command.index("--jobs") + 1], "3")

    def test_failed_stage_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["resolved_mode"] = "fast"
            loaded["stages"]["probe"] = {"status": "succeeded"}
            loaded["stages"]["prepare"] = {"status": "failed", "error": "old error"}
            server.save_job(loaded)

            def fake_prepare(current_job, log_path):
                return {"artifacts": {"video_path": "/tmp/video.mp4"}, "stdout_tail": ["ok"]}

            with patch.object(server, "stage_prepare", side_effect=fake_prepare):
                result = server.run_stage(job["job_id"], "prepare")

        self.assertEqual(result["stages"]["prepare"]["status"], "succeeded")
        self.assertNotIn("old error", json.dumps(result["stages"]["prepare"], ensure_ascii=False))

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

            def fake_locked(job_id, stage):
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

    def test_load_job_marks_orphaned_running_stage_queued_to_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            loaded = server.load_job(job["job_id"])
            loaded["status"] = "running"
            loaded["runner"] = {"status": "running", "current_stage": "final-publish"}
            loaded["stages"]["final-publish"] = {"status": "running"}
            server.save_job(loaded)

            recovered = server.load_job(job["job_id"])

        self.assertEqual(recovered["status"], "queued")
        self.assertEqual(recovered["runner"]["status"], "queued")
        self.assertEqual(recovered["stages"]["final-publish"]["status"], "queued")
        self.assertIn("queued", recovered["runner"]["error"])

    def test_probe_auto_routes_long_video_to_long_talk_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})

            with patch.object(server_mod, "probe_duration_seconds", return_value=3600):
                result = server.run_stage(job["job_id"], "probe")

        self.assertEqual(result["resolved_mode"], "long-talk-fast")
        self.assertEqual(result["stages"]["probe"]["status"], "succeeded")

    def test_analyze_core_stage_parses_run_dir_from_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp) / "jobs"
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            server = server_mod.VideoLinkStatusServer(jobs_dir, REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            server.run_stage(job["job_id"], "probe")

            def fake_run(command, log_path):
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

            def fake_run_stage(job_id, stage):
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

    def test_dashboard_contains_log_copy_and_core_progress_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video"})
            html = server_mod.render_job_dashboard(job)

        self.assertIn('id="copyLogButton"', html)
        self.assertIn('id="corePanel"', html)
        self.assertIn("?full=1", html)
        self.assertIn("查看日志", html)


if __name__ == "__main__":
    unittest.main()
