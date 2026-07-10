import importlib.util
import io
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
        self.assertIn('id="fileSourceTab"', html)
        self.assertIn('id="mediaFile"', html)
        self.assertIn('data-intent="smart"', html)
        self.assertIn('data-intent="transcribe"', html)
        self.assertIn('data-intent="scene"', html)
        self.assertIn('data-intent="tools"', html)
        self.assertIn('id="templatePanel"', html)
        self.assertIn('id="templateSearch"', html)
        self.assertIn('id="templateCategory"', html)
        self.assertIn('id="templateList"', html)
        self.assertIn('id="selectedTemplatePanel"', html)
        self.assertIn('id="focusPrompt"', html)
        main_js = (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8")
        self.assertIn("关注重点", main_js)
        self.assertIn("audio_prompt_templates.json", main_js)
        self.assertIn("applyFocusPromptTemplate", main_js)
        self.assertIn("【模板指令开始】", main_js)
        self.assertTrue((UI_ROOT / "video_analyzer_ui" / "static" / "data" / "audio_prompt_templates.json").is_file())
        self.assertIn("/api/video-link/jobs/upload", main_js)
        self.assertIn("FormData", main_js)
        self.assertIn('<label hidden>', html)
        self.assertIn('id="globalSummary"', html)
        self.assertIn('id="resourceLanes"', html)
        self.assertIn('id="jobList"', html)
        self.assertIn('id="copyLogButton"', html)
        self.assertIn("copyText", (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8"))
        self.assertIn("document.execCommand('copy')", (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8"))
        self.assertIn('id="stageDurationSummary"', html)
        self.assertIn('id="coreDiagnosticsPanel"', html)
        self.assertNotIn('id="previewView"', html)
        self.assertNotIn('id="previewGrid"', html)
        self.assertNotIn('id="previewTab"', html)
        self.assertIn('id="vscodeView"', html)
        self.assertIn('id="vscodeTab"', html)
        self.assertIn('id="vscodeFrame"', html)
        self.assertIn('id="docList"', html)
        self.assertIn('id="docPreviewBody"', html)
        self.assertIn('id="generateSkillButton"', html)
        self.assertIn('id="enableSkillButton"', html)
        self.assertIn('id="qaSourcePlayerBody"', html)
        self.assertIn('id="qaSourcePlayerSummary"', html)
        self.assertIn('id="qaSourcePlayerStopButton"', html)
        self.assertIn('data-resize-pane="qa-source"', html)
        self.assertIn('data-resize-pane="qa-source-height"', html)
        self.assertIn('class="qa-source-player source-player-panel"', html)
        self.assertIn('id="sourcePlayerStopButton"', html)
        self.assertIn('id="sourcePlayerPanel"', html)
        self.assertIn('id="toggleSourcePlayerPanel"', html)
        self.assertIn('data-resize-pane="study"', html)
        self.assertIn('data-resize-pane="doc-list"', html)
        self.assertIn("vendor/markdown-it/markdown-it.min.js", html)
        self.assertIn("vendor/dompurify/purify.min.js", html)
        self.assertIn("vendor/katex/katex.min.css", html)
        self.assertIn("vendor/katex/katex.min.js", html)
        self.assertIn("vendor/katex/contrib/auto-render.min.js", html)
        main_js = (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8")
        styles_css = (UI_ROOT / "video_analyzer_ui" / "static" / "css" / "styles.css").read_text(encoding="utf-8")
        self.assertIn("study-workflow", main_js)
        self.assertIn("study-detail-shell", main_js)
        self.assertIn("representative_frame", main_js)
        self.assertIn("wireStudyGuideInteractions", main_js)
        self.assertIn("bindImageViewer", main_js)
        self.assertIn("bindPaneResizers", main_js)
        self.assertIn("videoAnalyzerPaneLayout", main_js)
        self.assertIn("data-image-viewer-src", main_js)
        self.assertIn("const changed = selectedJobId !== jobId", main_js)
        self.assertIn("if (changed) resetQaMessages()", main_js)
        self.assertIn("loadQaHistory(job.job_id)", main_js)
        self.assertIn("/qa/history?limit=50", main_js)
        self.assertIn("qa-saved-at", main_js)
        self.assertIn("parseMermaidEdge", main_js)
        self.assertIn("normalizeMermaidPreviewLines", main_js)
        self.assertIn("-.->", main_js)
        self.assertIn("==>", main_js)
        self.assertIn("generateSkillCandidate", main_js)
        self.assertIn("enableSkillCandidate", main_js)
        self.assertIn("study-node", styles_css)
        self.assertIn("study-frame", styles_css)
        self.assertIn(".image-viewer-stage", styles_css)
        self.assertIn(".pane-resizer", styles_css)
        self.assertIn(".qa-saved-at", styles_css)
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

    def test_video_link_skill_candidate_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp) / "jobs", video_link_auto_resume=False)
            client = ui.app.test_client()
            repo_root = Path(tmp) / "repo"
            run_dir = Path(tmp) / "run"
            repo_root.mkdir()
            run_dir.mkdir()
            ui.video_link.repo_root = repo_root
            (run_dir / "operation_manual.md").write_text("# Demo Tool Setup\n\n1. 填写 API Token。\n", encoding="utf-8")
            (run_dir / "manual_evidence.md").write_text("# Evidence\n\nframe_001 显示 API Token。\n", encoding="utf-8")
            create = client.post("/api/video-link/jobs", json={"video_url": "https://example.com/video"})
            job = create.get_json()
            loaded = ui.video_link.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)
            ui.video_link.save_job(loaded)

            before = client.get(f"/api/video-link/jobs/{job['job_id']}/skill-candidate")
            generated = client.post(f"/api/video-link/jobs/{job['job_id']}/skill-candidate/generate", json={})
            enabled = client.post(f"/api/video-link/jobs/{job['job_id']}/skill-candidate/enable", json={})

        self.assertEqual(before.status_code, 200)
        self.assertFalse(before.get_json()["available"])
        self.assertEqual(generated.status_code, 200)
        self.assertTrue(generated.get_json()["available"])
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.get_json()["enabled"])

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

    def test_video_link_api_upload_media_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp) / "jobs", video_link_auto_resume=False)
            ui.video_link.repo_root = Path(tmp) / "repo"
            ui.video_link.repo_root.mkdir()
            client = ui.app.test_client()

            create = client.post(
                "/api/video-link/jobs/upload",
                data={
                    "media": (io.BytesIO(b"fake audio"), "sample.mp3"),
                    "analysis_mode": "fast",
                    "auto_start": "false",
                },
                content_type="multipart/form-data",
            )
            result = create.get_json()
            list_response = client.get("/api/video-link/jobs")

        self.assertEqual(create.status_code, 201)
        self.assertEqual(result["source_type"], "upload")
        self.assertEqual(result["source_name"], "sample.mp3")
        self.assertEqual(list_response.get_json()["total"], 1)

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
            path_response = client.get(f"/api/video-link/jobs/{job_id}/resources/operation_manual.md")
            escaped = client.get(f"/api/video-link/jobs/{job_id}/resource?path=../secret.md")
            escaped_path = client.get(f"/api/video-link/jobs/{job_id}/resources/../secret.md")
            body = response.get_data(as_text=True)
            path_body = path_response.get_data(as_text=True)
            response.close()
            path_response.close()
            escaped.close()
            escaped_path.close()

        self.assertEqual(response.status_code, 200)
        self.assertIn("# 标题", body)
        self.assertEqual(path_response.status_code, 200)
        self.assertIn("# 标题", path_body)
        self.assertEqual(escaped.status_code, 403)
        self.assertIn(escaped_path.status_code, {403, 404})

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
        self.assertIn("renderCoreDiagnostics", js)
        self.assertIn("renderGpuDiagnostics", js)
        self.assertIn("core_diagnostics", js)
        self.assertIn("power_draw_w", js)
        self.assertIn("power_limit_w", js)
        self.assertIn("totalStageDuration", js)
        self.assertIn("durationMinutes", js)
        self.assertIn("stageDurationSummary", js)
        self.assertIn("原视频长度", js)
        self.assertIn("stop-action", js)
        self.assertIn("play-action", js)
        self.assertIn("'stop'", js)
        self.assertIn(".stage-duration-summary", css)
        self.assertIn("button.play-action", css)
        self.assertIn("button.stop-action", css)
        self.assertIn(".core-diagnostics-panel", css)
        self.assertIn(".diagnostic-gpu-table", css)
        self.assertIn(".diagnostic-issue", css)
        self.assertIn("open-run-dir", js)
        self.assertNotIn("renderPreviewGrid", js)
        self.assertNotIn("autoplay", js)
        self.assertIn("sourcePlayerState.loaded", js)
        self.assertIn("jobDisplayTitle(job)", js)
        self.assertIn("renderVscodePanel", js)
        self.assertIn("mergeSelectedJobSnapshot", js)
        self.assertIn("renderSelectedJobSnapshot", js)
        self.assertNotIn("if (selectedFromList) renderJob(selectedFromList)", js)
        self.assertIn("ensureVscodeSession", js)
        self.assertIn("vscode-session", js)
        self.assertIn("renderDocPreviewPanel", js)
        self.assertIn("renderMarkdown", js)
        self.assertIn("markdownit", js)
        self.assertIn("DOMPurify", js)
        self.assertIn("renderMathInElement", js)
        self.assertIn("normalizeMarkdownForPreview", js)
        self.assertIn("splitInlineMarkdownTableLine", js)
        self.assertIn("isPotentialMarkdownTableRow", js)
        self.assertIn("standaloneNodes", js)
        self.assertIn("document_preview", js)
        self.assertIn("文档推导脑图", js)
        self.assertIn("Mermaid 预览", js)
        self.assertIn("重点阅读", js)
        self.assertIn("证据审计", js)
        self.assertIn("过程文件", js)
        self.assertIn("DOCUMENT_DERIVATION_PATH", js)
        self.assertIn(".doc-preview-body", css)
        self.assertIn(".mindmap-preview", css)
        self.assertIn(".mindmap-mermaid", css)
        self.assertIn(".doc-group", css)
        self.assertIn("td img.markdown-image", css)
        self.assertIn(".doc-list", css)
        self.assertNotIn("video-seek", js)
        self.assertNotIn('preload="none"', js)
        self.assertNotIn("ensurePreviewVideoSource", js)
        self.assertNotIn("preview-source-link", js)
        self.assertNotIn("scan-line", js)
        self.assertNotIn("preview-success-link", js)
        self.assertIn("sourcePlayerState.loaded = true", js)
        self.assertIn("sourcePlayerTargets", js)
        self.assertIn("qaSourcePlayerBody", js)
        self.assertIn("sourcePlayer: true", js)
        self.assertIn("'sourcePlayer'", js)
        self.assertIn("toggleSourcePlayerPanel", js)
        self.assertIn("pauseSourcePlayer", js)
        self.assertNotIn("sourceLocalVideoUrl", js)
        self.assertNotIn("syncLocalSourceVideos", js)
        self.assertNotIn("dataset.seekSeconds", js)
        self.assertNotIn("video.loop = false", js)
        self.assertNotIn("video.muted = false", js)
        self.assertNotIn("Bilibili 内嵌播放器可能静音或连播", js)
        self.assertIn("dataset.playerKind", js)
        self.assertIn("dataset.playerUrl", js)
        self.assertIn("renderSourcePlayerBody(targets, 'iframe', embedUrl, html)", js)
        self.assertNotIn("source-player-click-catcher", js)
        self.assertNotIn("bindSourcePlayerSurfacePause", js)
        self.assertIn("url.searchParams.set('t', String(value))", js)
        self.assertIn("nodes.sourcePlayerPanel && learningPanelVisibility.sourcePlayer", js)
        self.assertIn("playerVisible && (docListVisible || studyVisible || contentVisible)", js)
        self.assertIn("qaSourceHeight", js)
        self.assertIn("--qa-source-pane-width", css)
        self.assertIn("--qa-source-player-height", css)
        self.assertNotIn(".source-player-body video", css)
        self.assertIn(".source-player-frame-wrap", css)
        self.assertNotIn(".source-player-click-catcher", css)
        self.assertNotIn("--source-player-control-safe-zone", css)
        self.assertIn(".row-resizer", css)
        self.assertIn("if (!['qa', 'vscode'].includes(currentView)) setView('vscode')", js)
        self.assertIn(".qa-source-player", css)
        self.assertIn("不会自动播放", js)
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
        self.assertNotIn(".preview-grid", css)
        self.assertIn(".vscode-shell", css)
        self.assertNotIn(".preview-success-link", css)
        self.assertNotIn(".preview-source-link", css)
        self.assertNotIn("@keyframes scan-sweep", css)
        self.assertIn("@keyframes status-spin", css)


if __name__ == "__main__":
    unittest.main()
