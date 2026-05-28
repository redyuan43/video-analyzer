import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_options_include_defaults_and_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            options = server.options()

        self.assertEqual(options["defaults"]["analysis_mode"], "auto")
        self.assertEqual(options["defaults"]["profile"], "deepseek_v4_pro")
        self.assertEqual(options["defaults"]["cookies_from_browser"], "chrome")
        self.assertIn("balanced", options["choices"]["analysis_modes"])
        self.assertIn("deepseek_v4_pro", options["choices"]["profiles"])

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
                    "focusPrompt": "重点关注部署参数和失败恢复",
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
        self.assertEqual(job["options"]["focus_prompt"], "重点关注部署参数和失败恢复")

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
        self.assertEqual(command[command.index("--profile") + 1], "deepseek_v4_pro")
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
        self.assertIn("--profile", command)
        self.assertEqual(command[command.index("--profile") + 1], "deepseek_v4_pro")
        self.assertNotIn("--skip-images", command)

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
        self.assertEqual(recovered["artifacts"]["operation_manual"]["value"], str(run_dir.resolve() / "operation_manual.md"))

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
                "multidoc": {"status": "skipped"},
                "deep-v2": {"status": "skipped"},
                "image-prompts": {"status": "skipped"},
            }
            server.save_job(loaded)

            with patch.object(server, "run_command", side_effect=RuntimeError("publisher unavailable")):
                result = server.run_stage(job["job_id"], "final-publish")

        self.assertEqual(result["stages"]["final-publish"]["status"], "skipped")
        self.assertTrue(result["stages"]["final-publish"]["soft_failed"])
        self.assertEqual(result["warnings"][0]["stage"], "final-publish")
        self.assertIn("publisher unavailable", result["warnings"][0]["message"])

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

            def fake_locked(job_id, stage):
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

    def test_load_job_marks_final_publish_succeeded_when_exports_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = server_mod.VideoLinkStatusServer(Path(tmp), REPO_ROOT)
            run_dir = Path(tmp) / "run"
            export_dir = run_dir / "exports"
            export_dir.mkdir(parents=True)
            for name in server_mod.EXPECTED_FINAL_EXPORTS:
                (export_dir / name).write_text("ok", encoding="utf-8")
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

    def test_analyze_core_stage_parses_run_dir_from_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp) / "jobs"
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            server = server_mod.VideoLinkStatusServer(jobs_dir, REPO_ROOT)
            job = server.create_job({"video_url": "https://example.com/video", "analysis_mode": "fast"})
            server.run_stage(job["job_id"], "probe")

            def fake_run(command, log_path, on_start=None):
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
        self.assertEqual(progress["last_signal_label"], "OCR 执行")
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
                "[pdf] operation_manual.md",
                "[verify] pdf=4",
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
        self.assertIn("?full=1", html)
        self.assertIn("查看日志", html)


if __name__ == "__main__":
    unittest.main()
