import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "video-analyzer-ui"
SERVER_PATH = UI_ROOT / "video_analyzer_ui" / "server.py"


def load_ui_module():
    sys.path.insert(0, str(UI_ROOT))
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("video_analyzer_ui_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


ui_mod = load_ui_module()
from tools import video_link_status_server as status_server


class VideoAnalyzerUITests(unittest.TestCase):
    def test_home_page_contains_unified_video_link_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            client = ui.app.test_client()

            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("URL workflow console", html)
        self.assertIn('id="jobForm"', html)
        self.assertIn('id="videoUrlInput"', html)
        self.assertIn('id="addUrlButton"', html)
        self.assertIn('id="urlList"', html)
        self.assertIn('id="videoUrls"', html)
        self.assertIn('id="focusPrompt"', html)
        self.assertIn("关注重点", (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8"))
        self.assertIn('<label hidden>', html)
        self.assertIn('id="globalSummary"', html)
        self.assertIn('id="resourceLanes"', html)
        self.assertIn('id="jobList"', html)
        self.assertIn('id="copyLogButton"', html)
        self.assertIn('id="stageDurationSummary"', html)
        self.assertIn('id="previewView"', html)
        self.assertIn('id="previewGrid"', html)
        self.assertIn('id="previewTab"', html)
        self.assertIn('id="vscodeView"', html)
        self.assertIn('id="vscodeTab"', html)
        self.assertIn('id="vscodeFrame"', html)
        self.assertIn('id="docList"', html)
        self.assertIn('id="docPreviewBody"', html)
        self.assertIn("vendor/markdown-it/markdown-it.min.js", html)
        self.assertIn("vendor/dompurify/purify.min.js", html)
        self.assertIn("vendor/katex/katex.min.css", html)
        self.assertIn("vendor/katex/katex.min.js", html)
        self.assertIn("vendor/katex/contrib/auto-render.min.js", html)
        self.assertLess(html.index('id="jobList"'), html.index('id="globalSummary"'))

    def test_video_link_api_create_list_get_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            client = ui.app.test_client()

            create = client.post(
                "/api/video-link/jobs",
                json={"video_url": "https://example.com/video", "analysis_mode": "fast"},
            )
            job = create.get_json()
            list_response = client.get("/api/video-link/jobs")
            get_response = client.get(f"/api/video-link/jobs/{job['job_id']}")
            log_response = client.get(f"/api/video-link/jobs/{job['job_id']}/logs/probe?full=1")
            delete_response = client.delete(f"/api/video-link/jobs/{job['job_id']}")
            deleted_get_response = client.get(f"/api/video-link/jobs/{job['job_id']}")

        self.assertEqual(create.status_code, 201)
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(log_response.status_code, 200)
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(deleted_get_response.status_code, 404)
        self.assertEqual(list_response.get_json()["total"], 1)
        self.assertIn("resources", list_response.get_json())
        self.assertEqual(get_response.get_json()["dashboard_url"], f"/?job={job['job_id']}")

    def test_video_link_api_batch_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            client = ui.app.test_client()

            create = client.post(
                "/api/video-link/jobs/batch",
                json={
                    "video_urls_text": "https://example.com/one\ninvalid\nhttps://example.com/two",
                    "analysis_mode": "fast",
                    "auto_start": False,
                },
            )
            result = create.get_json()
            list_response = client.get("/api/video-link/jobs")

        self.assertEqual(create.status_code, 201)
        self.assertEqual(result["created"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(list_response.get_json()["total"], 2)

    def test_video_link_api_open_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            client = ui.app.test_client()
            create = client.post(
                "/api/video-link/jobs",
                json={"video_url": "https://example.com/video", "analysis_mode": "fast"},
            )
            job_id = create.get_json()["job_id"]

            with patch.object(ui.video_link, "open_run_dir", return_value={"opened": True, "run_dir": "/tmp/run"}):
                response = client.post(f"/api/video-link/jobs/{job_id}/open-run-dir", json={})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["opened"])

    def test_video_link_resource_route_serves_run_dir_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp) / "jobs", video_link_auto_resume=False)
            client = ui.app.test_client()
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "operation_manual.md").write_text("# 标题\n正文", encoding="utf-8")
            create = client.post(
                "/api/video-link/jobs",
                json={"video_url": "https://example.com/video", "analysis_mode": "fast"},
            )
            job_id = create.get_json()["job_id"]
            loaded = ui.video_link.load_job(job_id)
            loaded["run_dir"] = str(run_dir)
            ui.video_link.save_job(loaded)

            response = client.get(f"/api/video-link/jobs/{job_id}/resource?path=operation_manual.md")
            escaped = client.get(f"/api/video-link/jobs/{job_id}/resource?path=../secret.md")
            body = response.get_data(as_text=True)
            response.close()
            escaped.close()

        self.assertEqual(response.status_code, 200)
        self.assertIn("# 标题", body)
        self.assertEqual(escaped.status_code, 403)

    def test_video_link_api_starts_vscode_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp) / "jobs", video_link_auto_resume=False)
            client = ui.app.test_client()
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            create = client.post(
                "/api/video-link/jobs",
                json={"video_url": "https://example.com/video", "analysis_mode": "fast"},
            )
            job_id = create.get_json()["job_id"]
            loaded = ui.video_link.load_job(job_id)
            loaded["run_dir"] = str(run_dir)
            ui.video_link.save_job(loaded)

            fake_process = type("FakeProcess", (), {"pid": 12345})()
            with patch("tools.video_link_status_server.find_code_server_binary", return_value={"server": "code-server", "command": ["/bin/true"]}), \
                patch("tools.video_link_status_server.allocate_vscode_port", return_value=19000), \
                patch("tools.video_link_status_server.discover_global_vscode_session", return_value=None), \
                patch("tools.video_link_status_server.stop_managed_vscode_sessions", return_value=0), \
                patch("tools.video_link_status_server.subprocess.Popen", return_value=fake_process), \
                patch("tools.video_link_status_server.local_tailscale_host", return_value=None), \
                patch("tools.video_link_status_server.process_alive", return_value=True):
                response = client.post(f"/api/video-link/jobs/{job_id}/vscode-session", json={})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["port"], 19000)
        self.assertTrue(payload["url"].startswith("http://localhost:19000/?folder="))
        self.assertIn("run", payload["url"])
        self.assertEqual(payload["server"], "code-server")

    def test_video_link_job_detail_discovers_global_vscode_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp) / "jobs", video_link_auto_resume=False)
            client = ui.app.test_client()
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            create = client.post(
                "/api/video-link/jobs",
                json={"video_url": "https://example.com/video", "analysis_mode": "fast"},
            )
            job_id = create.get_json()["job_id"]
            loaded = ui.video_link.load_job(job_id)
            loaded["run_dir"] = str(run_dir)
            ui.video_link.save_job(loaded)

            discovered = {
                "job_id": job_id,
                "pid": 12345,
                "port": 19005,
                "run_dir": None,
                "server": "code-server",
                "started_at": "2026-05-29T00:00:00+0800",
            }
            with patch("tools.video_link_status_server.discover_global_vscode_session", return_value=discovered), \
                patch("tools.video_link_status_server.local_tailscale_host", return_value="100.91.42.28"), \
                patch("tools.video_link_status_server.process_alive", return_value=True):
                response = client.get(f"/api/video-link/jobs/{job_id}")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()["vscode_preview"]
        self.assertTrue(payload["ready"])
        self.assertTrue(payload["url"].startswith("http://100.91.42.28:19005/?folder="))
        self.assertIn("run", payload["url"])

    def test_vscode_process_discovery_matches_exact_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "operation-manual"
            sibling_dir = Path(tmp) / "operation-manual-agx"
            run_dir.mkdir()
            sibling_dir.mkdir()
            ps_output = (
                f"101 101 node /bin/code-server --bind-addr 0.0.0.0:19000 {run_dir}\n"
                f"202 202 node /bin/code-server --bind-addr 0.0.0.0:19003 {sibling_dir}\n"
            )
            with patch("tools.video_link_status_server.subprocess.check_output", return_value=ps_output):
                matches = status_server.discover_vscode_processes(run_dir)

        self.assertEqual([{key: matches[0][key] for key in ("pid", "pgid", "port")}], [{"pid": 101, "pgid": 101, "port": 19000}])

    def test_video_link_preview_video_route_streams_ready_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp) / "jobs", video_link_auto_resume=False)
            client = ui.app.test_client()
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"0123456789")
            create = client.post(
                "/api/video-link/jobs",
                json={"video_url": "https://example.com/video", "analysis_mode": "fast"},
            )
            job_id = create.get_json()["job_id"]
            loaded = ui.video_link.load_job(job_id)
            loaded["video_path"] = str(video)
            ui.video_link.save_job(loaded)

            response = client.get(f"/api/video-link/jobs/{job_id}/video", headers={"Range": "bytes=0-3"})
            body = response.get_data()
            response.close()

        self.assertIn(response.status_code, {200, 206})
        self.assertEqual(body[:4], b"0123")

    def test_video_link_preview_video_route_waits_for_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            client = ui.app.test_client()
            create = client.post(
                "/api/video-link/jobs",
                json={"video_url": "https://example.com/video", "analysis_mode": "fast"},
            )
            job_id = create.get_json()["job_id"]

            response = client.get(f"/api/video-link/jobs/{job_id}/video")

        self.assertEqual(response.status_code, 409)

    def test_legacy_job_url_redirects_to_home_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            client = ui.app.test_client()

            response = client.get("/video-link/jobs/0123456789abcdef0123456789abcdef")

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/?job=0123456789abcdef0123456789abcdef")

    def test_static_ui_marks_running_and_pending_states_visually(self):
        js = (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8")
        css = (UI_ROOT / "video_analyzer_ui" / "static" / "css" / "styles.css").read_text(encoding="utf-8")

        self.assertIn("status-spinner", js)
        self.assertIn("pendingUrls", js)
        self.assertIn("addPendingUrls", js)
        self.assertIn("focusPromptMap", js)
        self.assertIn("focus_prompt", js)
        self.assertIn("focus_prompts", js)
        self.assertIn("focusPrompt", js)
        self.assertIn(".url-focus-input", js)
        self.assertIn("renderUrlList", js)
        self.assertIn("stage-progress-meta", js)
        self.assertIn("totalStageDuration", js)
        self.assertIn("durationMinutes", js)
        self.assertIn("stageDurationSummary", js)
        self.assertIn("原视频长度", js)
        self.assertIn(".stage-duration-summary", css)
        self.assertIn("open-run-dir", js)
        self.assertIn("renderPreviewGrid", js)
        self.assertIn("renderVscodePanel", js)
        self.assertIn("ensureVscodeSession", js)
        self.assertIn("vscode-session", js)
        self.assertIn("renderDocPreviewPanel", js)
        self.assertIn("renderMarkdown", js)
        self.assertIn("markdownit", js)
        self.assertIn("DOMPurify", js)
        self.assertIn("renderMathInElement", js)
        self.assertIn("normalizeMarkdownForPreview", js)
        self.assertIn("isPotentialMarkdownTableRow", js)
        self.assertIn(".doc-preview-body", css)
        self.assertIn("td img.markdown-image", css)
        self.assertIn(".doc-list", css)
        self.assertIn("video-seek", js)
        self.assertIn("scan-line", js)
        self.assertIn("preview-success-link", js)
        self.assertIn("openPreviewRunDir", js)
        self.assertIn("成功", js)
        self.assertIn("job.warnings", js)
        self.assertIn("部分环节有警告", js)
        self.assertIn(".status.pending", css)
        self.assertIn(".url-add-row", css)
        self.assertIn(".url-list", css)
        self.assertIn(".url-focus", css)
        self.assertIn(".focus-prompt", css)
        self.assertIn(".option-grid", css)
        self.assertIn(".row-warning", css)
        self.assertIn("button.success-action", css)
        self.assertIn(".job-item.queued", css)
        self.assertIn(".preview-grid", css)
        self.assertIn(".vscode-shell", css)
        self.assertIn(".preview-success-link", css)
        self.assertIn("@keyframes scan-sweep", css)
        self.assertIn("@keyframes status-spin", css)


if __name__ == "__main__":
    unittest.main()
