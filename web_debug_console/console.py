from __future__ import annotations

import base64
import hmac
import ipaddress
import json
import os
import pty
import queue
import secrets
import shutil
import signal
import struct
import subprocess
import termios
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import fcntl
from flask import Blueprint, Flask, jsonify, request


ContextProvider = Callable[[str | None], dict[str, Any]]
MAX_BUFFER_BYTES = 2 * 1024 * 1024
TAILSCALE_NETWORK = ipaddress.ip_network("100.64.0.0/10")


class TerminalSession:
    def __init__(self, command: list[str], cwd: Path, rows: int, cols: int) -> None:
        self.session_id = uuid.uuid4().hex
        self.cwd = cwd
        self.command = command
        self.created_at = time.time()
        self.updated_at = self.created_at
        self._chunks: deque[tuple[int, str]] = deque()
        self._buffer_bytes = 0
        self._sequence = 0
        self._condition = threading.Condition()
        self._closed = False
        self._master_fd, slave_fd = pty.openpty()
        self.resize(rows, cols)
        env = os.environ.copy()
        env.update({"TERM": "xterm-256color", "COLORTERM": "truecolor"})
        self.process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        try:
            while True:
                try:
                    data = os.read(self._master_fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                self._append(data.decode("utf-8", errors="replace"))
        finally:
            self._closed = True
            self.updated_at = time.time()
            self._append("")
            try:
                os.close(self._master_fd)
            except OSError:
                pass

    def _append(self, text: str) -> None:
        encoded_size = len(text.encode("utf-8", errors="replace"))
        with self._condition:
            self._sequence += 1
            self._chunks.append((self._sequence, text))
            self._buffer_bytes += encoded_size
            while self._chunks and self._buffer_bytes > MAX_BUFFER_BYTES:
                _, removed = self._chunks.popleft()
                self._buffer_bytes -= len(removed.encode("utf-8", errors="replace"))
            self.updated_at = time.time()
            self._condition.notify_all()

    def read(self, after: int, wait_seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, min(wait_seconds, 25.0))
        with self._condition:
            while self._sequence <= after and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            chunks = [text for sequence, text in self._chunks if sequence > after]
            return {
                "sequence": self._sequence,
                "output": "".join(chunks),
                "running": not self._closed,
                "exit_code": self.process.poll() if self._closed else None,
            }

    def write(self, data: str) -> None:
        if self.process.poll() is not None:
            raise RuntimeError("terminal process has exited")
        os.write(self._master_fd, data.encode("utf-8"))
        self.updated_at = time.time()

    def resize(self, rows: int, cols: int) -> None:
        rows = max(4, min(int(rows), 200))
        cols = max(20, min(int(cols), 400))
        fcntl.ioctl(
            self._master_fd,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", rows, cols, 0, 0),
        )

    def terminate(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._closed = True
        self._append("")


class CodexAppServer:
    def __init__(self, cwd: Path, context: dict[str, Any], sandbox: str) -> None:
        codex = shutil.which("codex")
        if not codex:
            raise RuntimeError("codex CLI is not installed")
        self.cwd = cwd
        self.context = context
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.process = subprocess.Popen(
            [codex, "app-server", "--stdio"],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=1000)
        self._event_sequence = 0
        self._condition = threading.Condition()
        self._closed = False
        self._context_sent = False
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "web-debug-console",
                    "title": "Web Debug Console",
                    "version": "1.0.0",
                },
                "capabilities": {
                    "experimentalApi": True,
                    "optOutNotificationMethods": [
                        "command/exec/outputDelta",
                        "item/agentMessage/delta",
                        "item/plan/delta",
                        "item/fileChange/outputDelta",
                        "item/reasoning/summaryTextDelta",
                        "item/reasoning/textDelta",
                    ],
                },
            },
        )
        self._notify("initialized", {})
        result = self._request(
            "thread/start",
            {
                "cwd": str(cwd),
                "runtimeWorkspaceRoots": [str(cwd)],
                "approvalPolicy": "never",
                "sandbox": sandbox,
                "ephemeral": True,
                "developerInstructions": (
                    "你是嵌入网页的故障诊断助手。使用简体中文，先定位第一处 fatal "
                    "evidence，并继续追踪子服务日志，不要把页面上的 connection refused "
                    "当作根因。只在当前工作目录内读取或修改；不要执行破坏性操作，不要提交或推送。"
                ),
            },
            timeout=30,
        )
        self.thread_id = result["thread"]["id"]

    def _write(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Codex app-server stdin is unavailable")
        with self._write_lock:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()

    def _request(
        self, method: str, params: dict[str, Any], timeout: float = 20
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._pending[request_id] = result_queue
        self._write({"id": request_id, "method": method, "params": params})
        try:
            message = result_queue.get(timeout=timeout)
        except queue.Empty as exc:
            self._pending.pop(request_id, None)
            raise TimeoutError(f"Codex app-server request timed out: {method}") from exc
        if "error" in message:
            raise RuntimeError(str(message["error"]))
        return message.get("result") or {}

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        payload: dict[str, Any] = {"method": method}
        if params:
            payload["params"] = params
        self._write(payload)

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                request_id = message.get("id")
                if request_id and ("result" in message or "error" in message):
                    waiter = self._pending.pop(str(request_id), None)
                    if waiter:
                        waiter.put(message)
                    continue
                if request_id and message.get("method"):
                    self._write(
                        {
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Interactive approval is unavailable in web debug mode",
                            },
                        }
                    )
                    continue
                self._handle_notification(message.get("method"), message.get("params") or {})
        finally:
            self._closed = True
            self._append_event({"type": "closed", "message": "Codex app-server 已退出"})

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            text = line.strip()
            if text:
                self._append_event({"type": "diagnostic", "message": text[:1000]})

    def _handle_notification(self, method: str | None, params: dict[str, Any]) -> None:
        if method == "item/completed":
            item = params.get("item") or {}
            item_type = item.get("type")
            if item_type == "agentMessage":
                self._append_event({"type": "assistant", "text": item.get("text", "")})
            elif item_type == "commandExecution":
                self._append_event(
                    {
                        "type": "command",
                        "command": item.get("command", ""),
                        "status": item.get("status"),
                        "exit_code": item.get("exitCode"),
                        "output": item.get("aggregatedOutput"),
                    }
                )
            elif item_type == "fileChange":
                self._append_event(
                    {
                        "type": "file_change",
                        "status": item.get("status"),
                        "changes": item.get("changes") or [],
                    }
                )
        elif method == "item/started":
            item = params.get("item") or {}
            if item.get("type") in {"commandExecution", "fileChange"}:
                self._append_event(
                    {
                        "type": "activity",
                        "activity": item.get("type"),
                        "label": item.get("command") or "正在应用文件变更",
                    }
                )
        elif method == "turn/completed":
            turn = params.get("turn") or {}
            self._append_event(
                {
                    "type": "done",
                    "status": turn.get("status"),
                    "error": turn.get("error"),
                }
            )
        elif method == "error":
            self._append_event(
                {
                    "type": "error",
                    "message": str((params.get("error") or {}).get("message") or params),
                    "will_retry": bool(params.get("willRetry")),
                }
            )

    def _append_event(self, event: dict[str, Any]) -> None:
        with self._condition:
            self._event_sequence += 1
            self._events.append((self._event_sequence, event))
            self.updated_at = time.time()
            self._condition.notify_all()

    def start_turn(self, prompt: str) -> str:
        text = prompt.strip()
        if not text:
            raise ValueError("debug message is required")
        if not self._context_sent:
            context_json = json.dumps(self.context, ensure_ascii=False, indent=2)
            text = (
                "以下是页面自动采集的当前故障上下文，请先核对真实文件和日志：\n"
                f"```json\n{context_json[:16000]}\n```\n\n用户问题：\n{text}"
            )
            self._context_sent = True
        result = self._request(
            "turn/start",
            {
                "threadId": self.thread_id,
                "input": [{"type": "text", "text": text, "text_elements": []}],
            },
            timeout=30,
        )
        self.updated_at = time.time()
        return result["turn"]["id"]

    def events(self, after: int, wait_seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, min(wait_seconds, 25.0))
        with self._condition:
            while self._event_sequence <= after and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return {
                "sequence": self._event_sequence,
                "events": [
                    event for sequence, event in self._events if sequence > after
                ],
                "running": not self._closed and self.process.poll() is None,
            }

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self._closed = True


class WebDebugConsole:
    def __init__(
        self,
        app: Flask,
        project_root: Path,
        context_provider: ContextProvider | None = None,
        enabled: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.context_provider = context_provider or (
            lambda _job_id: {"cwd": str(self.project_root)}
        )
        self.enabled = enabled
        self.token = secrets.token_urlsafe(32)
        self.terminals: dict[str, TerminalSession] = {}
        self.debug_sessions: dict[str, CodexAppServer] = {}
        self.session_ttl_seconds = max(
            300, int(os.environ.get("WEB_DEBUG_SESSION_TTL_SECONDS", "3600"))
        )
        package_dir = Path(__file__).resolve().parent
        self.blueprint = Blueprint(
            "web_debug_console",
            __name__,
            url_prefix="/devtools",
            static_folder=str(package_dir / "static"),
            static_url_path="/assets",
        )
        self._register_routes()
        app.register_blueprint(self.blueprint)
        app.extensions["web_debug_console"] = self
        threading.Thread(target=self._reap_idle_sessions, daemon=True).start()

    def _register_routes(self) -> None:
        bp = self.blueprint

        @bp.before_request
        def authorize():
            if not self.enabled:
                return jsonify({"error": "web debug console is disabled"}), 404
            if request.endpoint == "web_debug_console.static":
                return None
            if not self._request_allowed():
                return jsonify({"error": "web debug console access denied"}), 403
            return None

        @bp.get("/api/config")
        def config():
            job_id = request.args.get("job")
            context = self._context(job_id)
            return jsonify(
                {
                    "enabled": True,
                    "cwd": context["cwd"],
                    "job_id": job_id,
                    "tools": {
                        "shell": True,
                        "codex": bool(shutil.which("codex")),
                        "claude": bool(shutil.which("claude")),
                        "codex_app_server": bool(shutil.which("codex")),
                    },
                    "context": {
                        key: value
                        for key, value in context.items()
                        if key in {"status", "failed_stage", "error", "log_path"}
                    },
                }
            )

        @bp.post("/api/terminal/sessions")
        def create_terminal():
            payload = request.get_json(silent=True) or {}
            context = self._context(payload.get("job_id"))
            cwd = self._resolve_cwd(payload.get("cwd") or context["cwd"])
            tool = str(payload.get("tool") or "shell")
            command = self._terminal_command(tool, cwd)
            session = TerminalSession(
                command,
                cwd,
                int(payload.get("rows") or 30),
                int(payload.get("cols") or 120),
            )
            self.terminals[session.session_id] = session
            return jsonify(
                {
                    "session_id": session.session_id,
                    "cwd": str(cwd),
                    "tool": tool,
                    "pid": session.process.pid,
                }
            ), 201

        @bp.get("/api/terminal/sessions/<session_id>/output")
        def terminal_output(session_id: str):
            session = self._terminal(session_id)
            return jsonify(
                session.read(
                    request.args.get("after", 0, type=int),
                    request.args.get("wait", 20, type=float),
                )
            )

        @bp.post("/api/terminal/sessions/<session_id>/input")
        def terminal_input(session_id: str):
            payload = request.get_json(silent=True) or {}
            data = payload.get("data")
            if payload.get("encoding") == "base64":
                data = base64.b64decode(str(data or "")).decode(
                    "utf-8", errors="replace"
                )
            self._terminal(session_id).write(str(data or ""))
            return jsonify({"ok": True})

        @bp.post("/api/terminal/sessions/<session_id>/resize")
        def terminal_resize(session_id: str):
            payload = request.get_json(silent=True) or {}
            self._terminal(session_id).resize(
                int(payload.get("rows") or 30), int(payload.get("cols") or 120)
            )
            return jsonify({"ok": True})

        @bp.delete("/api/terminal/sessions/<session_id>")
        def terminal_delete(session_id: str):
            session = self.terminals.pop(session_id, None)
            if session:
                session.terminate()
            return jsonify({"stopped": bool(session)})

        @bp.post("/api/debug/sessions")
        def create_debug():
            payload = request.get_json(silent=True) or {}
            context = self._context(payload.get("job_id"))
            cwd = self._resolve_cwd(payload.get("cwd") or context["cwd"])
            sandbox = str(payload.get("sandbox") or "workspace-write")
            if sandbox not in {"read-only", "workspace-write"}:
                raise ValueError("unsupported debug sandbox")
            session = CodexAppServer(cwd, context, sandbox)
            session_id = uuid.uuid4().hex
            self.debug_sessions[session_id] = session
            return jsonify(
                {
                    "session_id": session_id,
                    "thread_id": session.thread_id,
                    "cwd": str(cwd),
                    "sandbox": sandbox,
                }
            ), 201

        @bp.post("/api/debug/sessions/<session_id>/messages")
        def debug_message(session_id: str):
            payload = request.get_json(silent=True) or {}
            turn_id = self._debug(session_id).start_turn(str(payload.get("message") or ""))
            return jsonify({"turn_id": turn_id}), 202

        @bp.get("/api/debug/sessions/<session_id>/events")
        def debug_events(session_id: str):
            return jsonify(
                self._debug(session_id).events(
                    request.args.get("after", 0, type=int),
                    request.args.get("wait", 20, type=float),
                )
            )

        @bp.delete("/api/debug/sessions/<session_id>")
        def debug_delete(session_id: str):
            session = self.debug_sessions.pop(session_id, None)
            if session:
                session.close()
            return jsonify({"stopped": bool(session)})

        @bp.errorhandler(ValueError)
        @bp.errorhandler(RuntimeError)
        @bp.errorhandler(TimeoutError)
        def handle_console_error(exc):
            return jsonify({"error": str(exc)}), 400

    def _request_allowed(self) -> bool:
        supplied = request.headers.get("X-Debug-Token", "")
        if not hmac.compare_digest(supplied, self.token):
            return False
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            return False
        try:
            address = ipaddress.ip_address(request.remote_addr or "")
        except ValueError:
            return False
        return address.is_loopback or address in TAILSCALE_NETWORK

    def _context(self, job_id: str | None) -> dict[str, Any]:
        context = dict(self.context_provider(job_id) or {})
        context["cwd"] = str(self._resolve_cwd(context.get("cwd") or self.project_root))
        return context

    def _resolve_cwd(self, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("working directory is outside the project root") from exc
        if not path.is_dir():
            raise ValueError("working directory does not exist")
        return path

    def _terminal_command(self, tool: str, cwd: Path) -> list[str]:
        if tool == "shell":
            return [os.environ.get("SHELL", "/bin/bash"), "--login"]
        if tool == "codex":
            executable = shutil.which("codex")
            if executable:
                return [executable, "--no-alt-screen", "-C", str(cwd)]
        if tool == "claude":
            executable = shutil.which("claude")
            if executable:
                return [executable]
        raise ValueError(f"terminal tool is unavailable: {tool}")

    def _terminal(self, session_id: str) -> TerminalSession:
        session = self.terminals.get(session_id)
        if not session:
            raise ValueError("terminal session not found")
        return session

    def _debug(self, session_id: str) -> CodexAppServer:
        session = self.debug_sessions.get(session_id)
        if not session:
            raise ValueError("debug session not found")
        return session

    def _reap_idle_sessions(self) -> None:
        while True:
            time.sleep(60)
            cutoff = time.time() - self.session_ttl_seconds
            for session_id, session in list(self.terminals.items()):
                if session.updated_at < cutoff:
                    self.terminals.pop(session_id, None)
                    session.terminate()
            for session_id, session in list(self.debug_sessions.items()):
                if session.updated_at < cutoff:
                    self.debug_sessions.pop(session_id, None)
                    session.close()
