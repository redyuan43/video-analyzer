import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from flask import Flask

from web_debug_console import WebDebugConsole


class FakeCodexAppServer:
    instances = []

    def __init__(
        self,
        cwd,
        context,
        sandbox,
        thread_id=None,
        event_callback=None,
    ):
        self.cwd = cwd
        self.context = context
        self.sandbox = sandbox
        self.resumed_from = thread_id
        self.thread_id = thread_id or "thread-test"
        self.event_callback = event_callback
        self.closed = False
        self.process = Mock()
        self.process.poll.return_value = None
        self.__class__.instances.append(self)

    def start_turn(self, prompt):
        if not prompt:
            raise ValueError("debug message is required")
        if self.event_callback:
            self.event_callback({"type": "assistant", "text": "定位完成"})
        return "turn-test"

    def events(self, after, wait):
        return {
            "sequence": after + 1,
            "events": [{"type": "assistant", "text": "定位完成"}],
            "running": True,
        }

    def close(self):
        self.closed = True


class WebDebugConsoleTests(unittest.TestCase):
    def setUp(self):
        FakeCodexAppServer.instances.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.app = Flask(__name__)
        self.console = WebDebugConsole(
            self.app,
            self.root,
            context_provider=lambda job_id: {
                "cwd": str(self.run_dir),
                "job_id": job_id,
                "status": "failed",
                "failed_stage": "analyze-core",
                "error": "worker failed",
                "log_tail": "first fatal error",
            },
            history_dir=self.root / "history",
        )
        self.client = self.app.test_client()
        self.headers = {"X-Debug-Token": self.console.token}

    def tearDown(self):
        for session in list(self.console.terminals.values()):
            session.terminate()
        for session in list(self.console.debug_sessions.values()):
            session.close()
        self.temp.cleanup()

    def test_config_requires_capability_token_and_restricts_remote_network(self):
        static_asset = self.client.get("/devtools/assets/debug-console.css")
        markdown_asset = self.client.get(
            "/devtools/assets/vendor/markdown-it/markdown-it.min.js"
        )
        purifier_asset = self.client.get(
            "/devtools/assets/vendor/dompurify/purify.min.js"
        )
        missing = self.client.get("/devtools/api/config")
        external = self.client.get(
            "/devtools/api/config",
            headers=self.headers,
            environ_base={"REMOTE_ADDR": "8.8.8.8"},
        )
        allowed = self.client.get(
            "/devtools/api/config?job=job-1",
            headers=self.headers,
            environ_base={"REMOTE_ADDR": "100.91.42.28"},
        )

        self.assertEqual(static_asset.status_code, 200)
        self.assertEqual(markdown_asset.status_code, 200)
        self.assertEqual(purifier_asset.status_code, 200)
        self.assertEqual(missing.status_code, 403)
        self.assertEqual(external.status_code, 403)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.get_json()["cwd"], str(self.run_dir))
        self.assertEqual(allowed.get_json()["context"]["failed_stage"], "analyze-core")
        static_asset.close()
        markdown_asset.close()
        purifier_asset.close()

    def test_debug_console_renders_assistant_messages_as_sanitized_markdown(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "web_debug_console"
            / "static"
            / "debug-console.js"
        ).read_text(encoding="utf-8")

        self.assertIn("markdownit({", script)
        self.assertIn("DOMPurify.sanitize", script)
        self.assertIn("kind === 'assistant'", script)
        self.assertIn("body.innerHTML = render(text)", script)
        self.assertIn("/debug/history?job=", script)
        self.assertIn("result.resumed ? '已恢复'", script)
        self.assertIn("window.sessionStorage.setItem", script)
        self.assertIn("restoreLiveDebugSession", script)

    def test_terminal_session_runs_real_pty_in_context_directory(self):
        created = self.client.post(
            "/devtools/api/terminal/sessions",
            headers=self.headers,
            json={"job_id": "job-1", "tool": "shell", "rows": 20, "cols": 80},
        )
        self.assertEqual(created.status_code, 201)
        session_id = created.get_json()["session_id"]

        written = self.client.post(
            f"/devtools/api/terminal/sessions/{session_id}/input",
            headers=self.headers,
            json={"data": "printf 'PTY_OK:%s\\n' \"$PWD\"; exit\n"},
        )
        self.assertEqual(written.status_code, 200)

        output = ""
        sequence = 0
        running = True
        deadline = time.monotonic() + 10
        while running and time.monotonic() < deadline:
            response = self.client.get(
                f"/devtools/api/terminal/sessions/{session_id}/output"
                f"?after={sequence}&wait=1",
                headers=self.headers,
            )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            sequence = payload["sequence"]
            output += payload["output"]
            running = payload["running"]

        self.assertFalse(running)
        self.assertIn(f"PTY_OK:{self.run_dir}", output)

    def test_terminal_rejects_working_directory_outside_project(self):
        response = self.client.post(
            "/devtools/api/terminal/sessions",
            headers=self.headers,
            json={"cwd": "/tmp", "tool": "shell"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("outside the project root", response.get_json()["error"])

    def test_debug_session_api_lifecycle(self):
        with patch(
            "web_debug_console.console.CodexAppServer", FakeCodexAppServer
        ):
            created = self.client.post(
                "/devtools/api/debug/sessions",
                headers=self.headers,
                json={
                    "job_id": "job-1",
                    "sandbox": "workspace-write",
                },
            )
            self.assertEqual(created.status_code, 201)
            session_id = created.get_json()["session_id"]
            session = self.console.debug_sessions[session_id]
            self.assertEqual(session.context["log_tail"], "first fatal error")
            live = self.client.get(
                f"/devtools/api/debug/sessions/{session_id}",
                headers=self.headers,
            )

            message = self.client.post(
                f"/devtools/api/debug/sessions/{session_id}/messages",
                headers=self.headers,
                json={"message": "定位根因"},
            )
            events = self.client.get(
                f"/devtools/api/debug/sessions/{session_id}/events?after=0&wait=0",
                headers=self.headers,
            )
            stopped = self.client.delete(
                f"/devtools/api/debug/sessions/{session_id}",
                headers=self.headers,
            )

        self.assertEqual(message.status_code, 202)
        self.assertTrue(live.get_json()["running"])
        self.assertEqual(message.get_json()["turn_id"], "turn-test")
        self.assertEqual(events.get_json()["events"][0]["text"], "定位完成")
        self.assertTrue(stopped.get_json()["stopped"])
        self.assertTrue(session.closed)

    def test_debug_history_persists_and_resumes_thread(self):
        with patch(
            "web_debug_console.console.CodexAppServer", FakeCodexAppServer
        ):
            created = self.client.post(
                "/devtools/api/debug/sessions",
                headers=self.headers,
                json={
                    "job_id": "job-1",
                    "sandbox": "workspace-write",
                },
            )
            session_id = created.get_json()["session_id"]
            self.client.post(
                f"/devtools/api/debug/sessions/{session_id}/messages",
                headers=self.headers,
                json={"message": "继续定位"},
            )
            self.client.delete(
                f"/devtools/api/debug/sessions/{session_id}",
                headers=self.headers,
            )

            history = self.client.get(
                "/devtools/api/debug/history?job=job-1",
                headers=self.headers,
            )
            resumed = self.client.post(
                "/devtools/api/debug/sessions",
                headers=self.headers,
                json={
                    "job_id": "job-1",
                    "sandbox": "read-only",
                },
            )

        payload = history.get_json()
        self.assertEqual(payload["thread_id"], "thread-test")
        self.assertEqual(
            [message["type"] for message in payload["messages"]],
            ["user", "assistant"],
        )
        self.assertTrue(resumed.get_json()["resumed"])
        self.assertEqual(FakeCodexAppServer.instances[-1].resumed_from, "thread-test")

        cleared = self.client.delete(
            "/devtools/api/debug/history?job=job-1",
            headers=self.headers,
        )
        empty = self.client.get(
            "/devtools/api/debug/history?job=job-1",
            headers=self.headers,
        )

        self.assertTrue(cleared.get_json()["cleared"])
        self.assertIsNone(empty.get_json()["thread_id"])
        self.assertEqual(empty.get_json()["messages"], [])
