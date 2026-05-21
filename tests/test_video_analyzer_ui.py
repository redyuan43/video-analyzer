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
        self.assertIn('id="videoUrls"', html)
        self.assertIn('id="globalSummary"', html)
        self.assertIn('id="resourceLanes"', html)
        self.assertIn('id="jobList"', html)
        self.assertIn('id="copyLogButton"', html)
        self.assertIn('id="previewView"', html)
        self.assertIn('id="previewGrid"', html)
        self.assertIn('id="previewTab"', html)
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

        self.assertEqual(create.status_code, 201)
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(log_response.status_code, 200)
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
        self.assertIn("stage-progress-meta", js)
        self.assertIn("open-run-dir", js)
        self.assertIn("renderPreviewGrid", js)
        self.assertIn("video-seek", js)
        self.assertIn("scan-line", js)
        self.assertIn("成功", js)
        self.assertIn(".status.pending", css)
        self.assertIn("button.success-action", css)
        self.assertIn(".job-item.queued", css)
        self.assertIn(".preview-grid", css)
        self.assertIn("@keyframes scan-sweep", css)
        self.assertIn("@keyframes status-spin", css)


if __name__ == "__main__":
    unittest.main()
