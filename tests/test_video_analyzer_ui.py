import importlib.util
import io
import sys
import tempfile
import types
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
from video_analyzer_ui.runtime_identity import RuntimeIdentity


class VideoAnalyzerUITests(unittest.TestCase):
    def test_runtime_identity_detects_loaded_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            source = repo_root / "video_analyzer" / "sample_runtime.py"
            source.parent.mkdir()
            source.write_text("VALUE = 1\n", encoding="utf-8")
            module_name = "_video_analyzer_runtime_identity_test"
            module = types.ModuleType(module_name)
            module.__file__ = str(source)
            sys.modules[module_name] = module
            try:
                runtime = RuntimeIdentity(repo_root)
                initial = runtime.payload()
                source.write_text("VALUE = 2\n", encoding="utf-8")
                changed = runtime.payload()
            finally:
                sys.modules.pop(module_name, None)

        self.assertFalse(initial["source_stale"])
        self.assertTrue(changed["source_stale"])
        self.assertIn("video_analyzer/sample_runtime.py", changed["stale_files"])

    def test_health_exposes_runtime_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            response = ui.app.test_client().get("/api/video-link/health")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("runtime", payload)
        self.assertIn("runtime_id", payload["runtime"])
        self.assertIn("current_fingerprint", payload["runtime"])

    def test_settings_routes_expose_and_update_runtime_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            model = {
                "id": "cloud_text",
                "name": "Cloud Text",
                "kind": "text",
                "protocol": "openai_compatible",
            }
            with patch.object(ui.video_link, "settings", return_value={"models": [], "profiles": []}), \
                patch.object(ui.video_link, "save_model_setting", return_value=model) as save:
                client = ui.app.test_client()
                listed = client.get("/api/settings")
                created = client.post("/api/settings/models", json=model)

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.get_json()["id"], "cloud_text")
        save.assert_called_once_with("cloud_text", model)

    def test_profile_test_route_forwards_current_model_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            payload = {
                "profile_name": "draft",
                "mode": "quick",
                "models": {"text": "cloud_text"},
            }
            expected = {"ok": True, "mode": "quick", "results": {}}
            with patch.object(ui.video_link, "test_profile_setting", return_value=expected) as test_profile:
                response = ui.app.test_client().post("/api/settings/profile-test", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected)
        test_profile.assert_called_once_with(payload)

    def test_mobile_audio_upload_route_uses_dedicated_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(
                jobs_dir=Path(tmp) / "jobs",
                video_link_auto_resume=False,
            )
            expected = {
                "job_id": "a" * 32,
                "status": "queued",
                "external_attempt_id": "attempt-1",
            }
            with patch.object(
                ui.video_link,
                "create_mobile_audio_job",
                return_value=expected,
            ) as create:
                response = ui.app.test_client().post(
                    "/api/mobile/audio-jobs/upload",
                    data={
                        "media": (io.BytesIO(b"fake audio"), "sample.mp3"),
                        "external_attempt_id": "attempt-1",
                    },
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json(), expected)
        self.assertEqual(create.call_args.args[2], "sample.mp3")

    def test_mobile_transcript_route_accepts_multipart_without_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(
                jobs_dir=Path(tmp) / "jobs",
                video_link_auto_resume=False,
            )
            expected = {
                "job_id": "c" * 32,
                "status": "queued",
                "external_attempt_id": "provided-1",
                "provided_transcript": True,
            }
            with patch.object(
                ui.video_link,
                "create_mobile_transcript_job",
                return_value=expected,
            ) as create:
                response = ui.app.test_client().post(
                    "/api/mobile/audio-jobs/from-transcript",
                    data={
                        "transcript": (io.BytesIO(b'{"text":"hello"}'), "transcript.json"),
                        "external_attempt_id": "provided-1",
                        "source_sha256": "a" * 64,
                        "source_transcription_id": "tx-1",
                        "source_transcript_sha256": "b" * 64,
                        "template_id": "tmpl-1",
                        "focus_prompt": "focus",
                        "profile": "deepseek_v4_pro",
                    },
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json(), expected)
        self.assertEqual(create.call_args.args[2], "transcript.json")
        self.assertEqual(create.call_args.args[0]["source_transcription_id"], "tx-1")

    def test_mobile_audio_transcription_upload_uses_transcription_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(
                jobs_dir=Path(tmp) / "jobs",
                video_link_auto_resume=False,
            )
            expected = {
                "job_id": "b" * 32,
                "status": "queued",
                "external_attempt_id": "transcription-1",
                "pipeline_kind": "transcription",
            }
            with patch.object(
                ui.video_link,
                "create_mobile_audio_job",
                return_value=expected,
            ) as create:
                response = ui.app.test_client().post(
                    "/api/mobile/audio-transcriptions/upload",
                    data={
                        "media": (io.BytesIO(b"fake audio"), "sample.mp3"),
                        "external_attempt_id": "transcription-1",
                    },
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json(), expected)
        self.assertEqual(create.call_args.args[2], "sample.mp3")
        self.assertEqual(
            create.call_args.kwargs["pipeline_kind"],
            "transcription",
        )

    def test_mobile_audio_attempt_lookup_uses_dedicated_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(
                jobs_dir=Path(tmp) / "jobs",
                video_link_auto_resume=False,
            )
            expected = {
                "job_id": "a" * 32,
                "status": "running",
                "external_attempt_id": "attempt-1",
            }
            with patch.object(
                ui.video_link,
                "get_mobile_audio_job_by_attempt",
                return_value=expected,
            ) as lookup:
                response = ui.app.test_client().get(
                    "/api/mobile/audio-jobs/by-attempt/attempt-1"
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected)
        lookup.assert_called_once_with("attempt-1")

    def test_mobile_audio_routes_honor_pipeline_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ",
                {"VIDEO_ANALYZER_AUDIO_PIPELINE_TOKEN": "secret"},
            ):
                ui = ui_mod.VideoAnalyzerUI(
                    jobs_dir=Path(tmp),
                    video_link_auto_resume=False,
                )
                denied = ui.app.test_client().get("/api/mobile/audio-templates")
                allowed = ui.app.test_client().get(
                    "/api/mobile/audio-templates",
                    headers={"X-Audio-Pipeline-Token": "secret"},
                )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        payload = allowed.get_json()
        self.assertEqual(payload["total"], 382)
        self.assertNotIn("prompt_original", payload["templates"][0])
        self.assertNotIn("prompt", payload["templates"][0])

    def test_stale_runtime_rejects_new_job_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            stale_runtime = {
                "runtime_id": "old-runtime",
                "source_stale": True,
                "stale_files": ["tools/video_link_status_server.py"],
            }
            with patch.object(ui.runtime_identity, "payload", return_value=stale_runtime):
                with patch.object(ui.video_link, "create_job") as create_job:
                    response = ui.app.test_client().post(
                        "/api/video-link/jobs",
                        json={"video_url": "https://example.com/video"},
                    )

        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.get_json()["runtime"]["source_stale"])
        create_job.assert_not_called()

    def test_debug_console_context_falls_back_for_external_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(
                jobs_dir=Path(tmp) / "jobs",
                video_link_auto_resume=False,
            )
            external_run_dir = Path(tmp) / "runs" / "job-1"
            external_run_dir.mkdir(parents=True)
            job = {
                "job_id": "job-1",
                "run_dir": str(external_run_dir),
                "status": "failed",
                "current_stage": "analyze-core",
                "stages": {
                    "analyze-core": {
                        "status": "failed",
                        "error": "worker failed",
                    }
                },
            }

            with patch.object(ui.video_link, "load_job", return_value=job):
                context = ui.debug_console_context("job-1")

        self.assertEqual(context["cwd"], str(ui_mod.VIDEO_LINK_REPO_ROOT))
        self.assertEqual(context["status"], "failed")
        self.assertEqual(context["failed_stage"], "analyze-core")

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
        self.assertIn('id="settingsTab"', html)
        self.assertIn('id="settingsView"', html)
        self.assertIn('id="settingsModelsView"', html)
        self.assertIn('id="settingsProfilesView"', html)
        self.assertIn('id="profileFlowViewport"', html)
        self.assertIn('id="profileFlowEdges"', html)
        self.assertIn('id="profileFlowNodes"', html)
        self.assertIn('id="profileFlowInspector"', html)
        self.assertIn('id="profileWorkflow"', html)
        self.assertIn('id="profileTestMode"', html)
        self.assertIn('id="testProfileButton"', html)
        self.assertIn('id="profileTestAvailability"', html)
        self.assertIn('id="profileTestSummary"', html)
        self.assertIn('id="profile"', html)
        self.assertIn('id="templateSearch"', html)
        self.assertIn('id="templateCategory"', html)
        self.assertIn('id="templateList"', html)
        self.assertIn('id="selectedTemplatePanel"', html)
        self.assertIn('id="focusPrompt"', html)
        main_js = (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8")
        self.assertIn("关注重点", main_js)
        self.assertIn("audio_prompt_templates.json", main_js)
        self.assertIn("applyFocusPromptTemplate", main_js)
        self.assertIn("renderProfileFlow", main_js)
        self.assertIn("drawProfileFlowEdges", main_js)
        self.assertIn("data-flow-model-slot", main_js)
        self.assertIn("selectedWorkflowSchema", main_js)
        self.assertIn("workflow_id: selectedWorkflowId()", main_js)
        self.assertIn("/api/settings/profile-test", main_js)
        self.assertIn("testCurrentProfile", main_js)
        self.assertIn("updateProfileTestAvailability", main_js)
        self.assertIn("后台空闲，可以执行通路测试", main_js)
        self.assertIn("从本机禁用内置运行方案", main_js)
        self.assertNotIn("恢复内置运行方案并清除本地覆盖", main_js)
        self.assertIn("【模板指令开始】", main_js)
        self.assertTrue((UI_ROOT / "video_analyzer_ui" / "static" / "data" / "audio_prompt_templates.json").is_file())
        self.assertIn("/api/video-link/jobs/upload", main_js)
        self.assertIn("FormData", main_js)
        self.assertNotIn('<label hidden>\n                            <span>运行配置</span>', html)
        self.assertIn('id="globalSummary"', html)
        self.assertIn('id="resourceLanes"', html)
        self.assertIn('id="jobList"', html)
        self.assertIn('id="showNonRerunFailures"', html)
        self.assertIn('id="copyLogButton"', html)
        self.assertIn("copyText", (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8"))
        self.assertIn("document.execCommand('copy')", (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8"))
        self.assertIn("failure_disposition?.rerun_recommended", main_js)
        self.assertIn("按当前方案重跑核心分析", main_js)
        self.assertIn("/stages/analyze-core/rerun", main_js)
        self.assertIn("refresh_runtime_profile: true", main_js)
        self.assertNotIn("用 Flash 继续", main_js)
        self.assertNotIn("resumeProfile", main_js)
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
        self.assertIn('id="skillLiveActivity"', html)
        self.assertIn('id="skillStageRail"', html)
        self.assertIn('id="resourceSkillsView"', html)
        self.assertIn('id="resourceSkillsTab"', html)
        self.assertIn('id="skillsResourceDocsTab"', html)
        self.assertIn('id="skillsResourceSkillsTab"', html)
        self.assertIn('id="toggleSkillsFocusButton"', html)
        self.assertIn('id="currentSkillsWorkspace"', html)
        self.assertIn('id="skillProjectsWorkspace"', html)
        self.assertIn('id="skillProjectForm"', html)
        self.assertIn('id="createTargetedSkillProjectButton"', html)
        self.assertIn('id="skillLibraryWorkspace"', html)
        self.assertIn('id="skillEditor"', html)
        self.assertIn('id="openSkillsWorkspaceButton"', html)
        qa_html = html[html.index('id="qaView"'):html.index('id="vscodeView"')]
        self.assertNotIn('id="skillSummary"', qa_html)
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
        self.assertIn('data-resize-pane="skills-current-left"', html)
        self.assertIn('data-resize-pane="skills-current-right"', html)
        self.assertIn('data-resize-pane="skills-current-height"', html)
        self.assertIn('data-resize-pane="skills-library-left"', html)
        self.assertIn('data-resize-pane="skills-library-right"', html)
        self.assertIn('data-resize-pane="skills-library-height"', html)
        self.assertIn("vendor/markdown-it/markdown-it.min.js", html)
        self.assertIn("vendor/dompurify/purify.min.js", html)
        self.assertIn("vendor/katex/katex.min.css", html)
        self.assertIn("vendor/katex/katex.min.js", html)
        self.assertIn("vendor/katex/contrib/auto-render.min.js", html)
        self.assertIn("vendor/mermaid/mermaid.min.js", html)
        self.assertIn('name="web-debug-token"', html)
        self.assertIn("web_debug_console.static", (UI_ROOT / "video_analyzer_ui" / "templates" / "index.html").read_text(encoding="utf-8"))
        self.assertIn("debug-console.js", html)
        self.assertIn("debug-console.css", html)
        main_js = (UI_ROOT / "video_analyzer_ui" / "static" / "js" / "main.js").read_text(encoding="utf-8")
        styles_css = (UI_ROOT / "video_analyzer_ui" / "static" / "css" / "styles.css").read_text(encoding="utf-8")
        self.assertTrue((UI_ROOT / "video_analyzer_ui" / "static" / "vendor" / "mermaid" / "mermaid.min.js").is_file())
        self.assertTrue((UI_ROOT / "video_analyzer_ui" / "static" / "vendor" / "mermaid" / "LICENSE").is_file())
        self.assertIn("study-workflow", main_js)
        self.assertIn("study-detail-shell", main_js)
        self.assertIn("representative_frame", main_js)
        self.assertIn("wireStudyGuideInteractions", main_js)
        self.assertIn("bindImageViewer", main_js)
        self.assertIn("bindPaneResizers", main_js)
        self.assertIn("videoAnalyzerPaneLayout", main_js)
        self.assertIn("resizeSkillsPane", main_js)
        self.assertIn("adjustSkillsPaneHeight", main_js)
        self.assertIn("skillsCurrentLeft", main_js)
        self.assertIn("skillsLibraryHeight", main_js)
        self.assertIn("videoAnalyzerSkillsFocusMode", main_js)
        self.assertIn("toggleSkillsFocusMode", main_js)
        self.assertNotIn("videoAnalyzerChromeCollapse", main_js)
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
        self.assertIn("loadCurrentSkillsWorkspace", main_js)
        self.assertIn("loadSkillProjects", main_js)
        self.assertIn("startSkillProjectDistillation", main_js)
        self.assertIn("syncSkillProjectPolling", main_js)
        self.assertIn("renderSkillProjectCandidateReview", main_js)
        self.assertIn("skillProjectCandidateDrafts", main_js)
        self.assertIn("data-project-candidate-confirm", main_js)
        self.assertIn("loadSkillProjectLog", main_js)
        self.assertIn("data-skill-project-log-stage", main_js)
        self.assertIn("skill-project-log-console", styles_css)
        self.assertIn("skill-project-spinner", styles_css)
        self.assertIn(".skill-project-candidate-review", styles_css)
        self.assertIn("/api/skill-projects", main_js)
        self.assertIn("saveCurrentLibrarySkill", main_js)
        self.assertIn("restoreSkillVersion", main_js)
        self.assertIn("const skillCandidateDrafts = new Map()", main_js)
        self.assertIn("updateSkillCandidateDraft", main_js)
        self.assertIn("draft.selected.add(input.value)", main_js)
        self.assertIn("nodes.skillCandidateList.scrollTop = scrollTop", main_js)
        self.assertIn("generating_tests: '生成压力测试题'", main_js)
        self.assertIn("已运行 ${elapsedSeconds} 秒", main_js)
        self.assertIn("renderSkillLiveActivity", main_js)
        self.assertIn("updateSkillLiveClock", main_js)
        self.assertIn("skillActivityMessage", main_js)
        self.assertIn("study-node", styles_css)
        self.assertIn("study-frame", styles_css)
        self.assertIn(".image-viewer-stage", styles_css)
        self.assertIn(".pane-resizer", styles_css)
        self.assertIn(".qa-saved-at", styles_css)
        self.assertIn("#resourceDocsView", styles_css)
        self.assertIn("grid-template-rows: minmax(0, 1fr) auto", styles_css)
        self.assertIn("#resourceSkillsView", styles_css)
        self.assertIn("height: var(--skills-pane-height, 100%)", styles_css)
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
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            create = client.post("/api/video-link/jobs", json={"video_url": "https://example.com/video"})
            job = create.get_json()
            loaded = ui.video_link.load_job(job["job_id"])
            loaded["run_dir"] = str(run_dir)
            ui.video_link.save_job(loaded)

            started_payload = {
                "available": False,
                "status": "running",
                "profile": "deepseek_v4_pro",
            }
            enabled_payload = {
                "available": True,
                "status": "succeeded",
                "enabled": True,
            }
            with (
                patch.object(
                    ui.video_link,
                    "start_skill_distillation",
                    return_value=started_payload,
                ) as start,
                patch.object(
                    ui.video_link,
                    "enable_skill_distillation",
                    return_value=enabled_payload,
                ) as enable,
            ):
                before = client.get(f"/api/video-link/jobs/{job['job_id']}/skill-distillation")
                generated = client.post(
                    f"/api/video-link/jobs/{job['job_id']}/skill-distillation/start",
                    json={"profile": "deepseek_v4_pro"},
                )
                enabled = client.post(
                    f"/api/video-link/jobs/{job['job_id']}/skill-distillation/enable",
                    json={"overwrite": False},
                )

        self.assertEqual(before.status_code, 200)
        self.assertFalse(before.get_json()["available"])
        self.assertEqual(generated.status_code, 202)
        self.assertEqual(generated.get_json()["profile"], "deepseek_v4_pro")
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.get_json()["enabled"])
        start.assert_called_once_with(job["job_id"], {"profile": "deepseek_v4_pro"})
        enable.assert_called_once_with(job["job_id"], {"overwrite": False})

    def test_skill_project_routes_create_assess_and_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(
                jobs_dir=Path(tmp) / "jobs",
                video_link_auto_resume=False,
            )
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            ui.video_link.repo_root = repo_root
            ui.video_link.skill_projects = status_server.SkillProjectStore(
                repo_root / "var" / "skill-projects"
            )
            client = ui.app.test_client()

            created = client.post(
                "/api/skill-projects",
                json={"title": "测试项目", "goal": "沉淀一个可执行的排错方法"},
            )
            project_id = created.get_json()["id"]
            listed = client.get("/api/skill-projects")
            assessed = client.post(f"/api/skill-projects/{project_id}/assess", json={})
            updated = client.patch(
                f"/api/skill-projects/{project_id}",
                json={"expected_output": "给出检查步骤"},
            )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(listed.get_json()["count"], 1)
        self.assertEqual(assessed.get_json()["assessment"]["verdict"], "needs_materials")
        self.assertEqual(updated.get_json()["brief"]["expected_output"], "给出检查步骤")

    def test_skill_library_routes_cover_edit_and_state_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(
                jobs_dir=Path(tmp) / "jobs",
                video_link_auto_resume=False,
            )
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            ui.video_link.repo_root = repo_root
            skill_dir = repo_root / ".codex" / "skills" / "route-test-skill"
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_dir.joinpath("SKILL.md").write_text(
                "---\nname: route-test-skill\ndescription: route test\n---\n\n# Route Test\n",
                encoding="utf-8",
            )
            client = ui.app.test_client()

            listed = client.get("/api/skills?state=enabled&query=route-test")
            detail = client.get("/api/skills/enabled/route-test-skill")
            original = detail.get_json()
            updated = client.put(
                "/api/skills/enabled/route-test-skill",
                json={
                    "revision": original["revision"],
                    "markdown": original["markdown"] + "\nUpdated.\n",
                },
            )
            disabled = client.post("/api/skills/enabled/route-test-skill/disable")
            restored = client.post("/api/skills/disabled/route-test-skill/restore")
            trashed = client.delete("/api/skills/enabled/route-test-skill", json={})
            trash_payload = trashed.get_json()
            permanent = client.delete(
                f"/api/skills/trash/{trash_payload['id']}",
                json={"confirmation": "route-test-skill"},
            )

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.get_json()["count"], 1)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(len(updated.get_json()["versions"]), 1)
        self.assertEqual(disabled.get_json()["state"], "disabled")
        self.assertEqual(restored.get_json()["state"], "enabled")
        self.assertEqual(trashed.get_json()["state"], "trash")
        self.assertEqual(permanent.get_json()["status"], "deleted")

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

    def test_video_link_rerun_stage_route_restarts_from_requested_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp), video_link_auto_resume=False)
            client = ui.app.test_client()
            job_id = "0123456789abcdef0123456789abcdef"
            expected = {"job_id": job_id, "runner": {"current_stage": "deep-v2"}}

            with patch.object(ui.video_link, "rerun_from_stage", return_value=expected) as rerun:
                response = client.post(f"/api/video-link/jobs/{job_id}/stages/deep-v2/rerun", json={})

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json(), expected)
        rerun.assert_called_once_with(
            job_id,
            "deep-v2",
            profile=None,
            refresh_runtime_profile=False,
        )

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

    def test_video_link_api_upload_media_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui = ui_mod.VideoAnalyzerUI(jobs_dir=Path(tmp) / "jobs", video_link_auto_resume=False)
            ui.video_link.repo_root = Path(tmp) / "repo"
            ui.video_link.repo_root.mkdir()
            client = ui.app.test_client()

            response = client.post(
                "/api/video-link/jobs/upload",
                data={"media": (io.BytesIO(b""), "empty.mp3")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "uploaded media file is empty")

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
        self.assertIn("studyCanResizeWidth", js)
        self.assertIn("studyRightReserve", js)
        self.assertIn("return Boolean(docListVisible || playerVisible || contentVisible)", js)
        self.assertIn("return nodes.studyResizer && !nodes.studyResizer.hidden ? 14 : 0", js)
        self.assertIn("const sourcePlayerNeedsHandle = playerVisible && (docListVisible || studyVisible || contentVisible)", js)
        self.assertIn("normalizeMarkdownForPreview", js)
        self.assertIn("splitInlineMarkdownTableLine", js)
        self.assertIn("isPotentialMarkdownTableRow", js)
        self.assertIn("standaloneNodes", js)
        self.assertIn("document_preview", js)
        self.assertIn("文档推导脑图", js)
        self.assertIn("Mermaid 预览", js)
        self.assertIn("initializeMermaid", js)
        self.assertIn("window.mermaid.render", js)
        self.assertIn("data-mermaid-diagram", js)
        self.assertIn("securityLevel: 'antiscript'", js)
        self.assertIn("重点阅读", js)
        self.assertIn("证据审计", js)
        self.assertIn("过程文件", js)
        self.assertIn("DOCUMENT_DERIVATION_PATH", js)
        self.assertIn(".doc-preview-body", css)
        self.assertIn(".mindmap-preview", css)
        self.assertIn(".mindmap-mermaid", css)
        self.assertIn(".mindmap-mermaid svg", css)
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
