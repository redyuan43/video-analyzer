"""Bounded incident diagnosis and recovery for video-link jobs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any

import requests

from video_analyzer.config import Config
from video_analyzer.jobengine._shared import iso_from_timestamp, iso_now, normalize_stage_name
from video_analyzer.jobengine.errors import BridgeError


REPAIR_STATUSES = {
    "idle",
    "queued",
    "analyzing",
    "executing",
    "verifying",
    "waiting_approval",
    "succeeded",
    "blocked",
    "exhausted",
    "disabled",
}
AUTOMATIC_ACTIONS = {"wait_and_retry", "retry_failed_stage", "resume_first_incomplete_stage"}
APPROVAL_ACTIONS = {"request_service_restart", "request_code_patch"}
TERMINAL_ACTIONS = {"manual_review"}
ALLOWED_ACTIONS = AUTOMATIC_ACTIONS | APPROVAL_ACTIONS | TERMINAL_ACTIONS
SENSITIVE_KEY_PATTERN = re.compile(
    r"(authorization|api[_-]?key|token|secret|password|cookie|cookies_from_browser)",
    re.IGNORECASE,
)
SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)(cookie:\s*)[^\r\n]+"),
)
FATAL_LINE_PATTERN = re.compile(
    r"(fatal|traceback|exception|error|failed|timeout|timed out|oom|out of memory|cuda)",
    re.IGNORECASE,
)
PATCH_PATH_PREFIXES = ("video_analyzer/", "video-analyzer-ui/", "tools/", "tests/")
PATCH_DENIED_PATHS = {
    "AGENTS.md",
    "config/config.json",
    "video_analyzer/config/default_config.json",
}
PATCH_COPY_IGNORE = {
    ".git",
    ".venv",
    ".cache",
    "downloads",
    "tmp",
    "var",
    "output",
    "node_modules",
    "__pycache__",
}


class RepairMixin:
    """Repair lifecycle mixed into ``VideoLinkStatusServer``."""

    def start_repair_loop(self) -> None:
        if self.repair_thread and self.repair_thread.is_alive():
            return
        self.repair_stop.clear()
        self.repair_thread = threading.Thread(
            target=self._repair_loop,
            daemon=True,
            name="video-link-incident-repair",
        )
        self.repair_thread.start()

    def recover_interrupted_repairs(self) -> None:
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            repair = dict(job.get("repair") or {})
            if job.get("status") != "repairing" or repair.get("status") not in {
                "analyzing",
                "executing",
                "verifying",
            }:
                continue
            repair.update(
                {
                    "status": "queued",
                    "queued_at": iso_now(),
                    "next_attempt_at": iso_now(),
                    "updated_at": iso_now(),
                    "recovered_after_restart": True,
                }
            )
            job["repair"] = repair
            job["updated_at"] = iso_now()
            self.save_job(job)

    def _repair_loop(self) -> None:
        while not self.repair_stop.wait(max(1.0, float(self.repair_config().get("poll_seconds") or 5))):
            try:
                self.repair_queued_jobs_once()
                self.repair_last_error = ""
            except Exception as exc:
                self.repair_last_error = str(exc)

    def repair_queued_jobs_once(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        candidates: list[dict[str, Any]] = []
        for path in sorted(self.jobs_dir.glob("*/job.json")):
            try:
                job = self.load_job(path.parent.name)
            except Exception:
                continue
            repair = job.get("repair") or {}
            if (
                job.get("status") == "repairing"
                and repair.get("status") == "queued"
                and parse_repair_timestamp(repair.get("next_attempt_at")) <= now
            ):
                candidates.append(job)
        candidates.sort(key=lambda item: str((item.get("repair") or {}).get("queued_at") or ""))
        handled: list[str] = []
        for job in candidates:
            if not self.repair_lock.acquire(blocking=False):
                break
            try:
                self.run_repair_cycle(str(job["job_id"]))
                handled.append(str(job["job_id"]))
            finally:
                self.repair_lock.release()
        return handled

    def repair_config(self) -> dict[str, Any]:
        config_loader = getattr(self, "incident_repair_config", None)
        if callable(config_loader):
            config = dict(config_loader())
        else:
            loaded = Config(self.repo_root / "config").config
            raw = loaded.get("incident_repair") or {}
            config = dict(raw) if isinstance(raw, dict) else {}
        endpoint = str(
            config.get("base_url")
            or os.environ.get(str(config.get("base_url_env") or "VIDEO_ANALYZER_REPAIR_LLM_BASE_URL"), "")
        ).strip()
        model = str(
            config.get("model")
            or os.environ.get(str(config.get("model_env") or "VIDEO_ANALYZER_REPAIR_LLM_MODEL"), "")
        ).strip()
        configured = bool(endpoint and model)
        config["base_url"] = endpoint.rstrip("/")
        config["model"] = model
        config["configured"] = configured
        config["enabled"] = bool(config.get("enabled", configured) and configured)
        config["enabled_by_default"] = bool(config.get("enabled_by_default", True) and configured)
        config["max_cycles"] = max(1, min(int(config.get("max_cycles") or 3), 10))
        config["same_fingerprint_limit"] = max(
            1,
            min(int(config.get("same_fingerprint_limit") or 2), 5),
        )
        config["timeout_seconds"] = max(10, min(int(config.get("timeout_seconds") or 300), 1800))
        config["max_log_chars"] = max(4000, min(int(config.get("max_log_chars") or 30000), 120000))
        config["max_evidence_chars"] = max(
            12000,
            min(int(config.get("max_evidence_chars") or 60000), 180000),
        )
        return config

    def repair_default_enabled(self) -> bool:
        config = self.repair_config()
        return bool(config.get("enabled") and config.get("enabled_by_default"))

    def queue_repair_after_failure(self, job: dict[str, Any], error: str) -> bool:
        config = self.repair_config()
        if not config.get("enabled") or not bool((job.get("options") or {}).get("auto_repair")):
            return False
        if "stopped by user" in str(error).lower():
            return False
        failed_stage = self.failed_stage(job)
        if not failed_stage:
            return False
        existing = dict(job.get("repair") or {})
        if existing.get("status") in {"waiting_approval", "exhausted", "disabled"}:
            return False
        now = iso_now()
        job["repair"] = {
            **existing,
            "enabled": True,
            "status": "queued",
            "queued_at": now,
            "updated_at": now,
            "trigger_stage": failed_stage,
            "last_error": str(error),
            "cycle": int(existing.get("cycle") or 0),
            "max_cycles": config["max_cycles"],
            "same_fingerprint_limit": config["same_fingerprint_limit"],
            "next_attempt_at": now,
            "history": list(existing.get("history") or []),
        }
        job["status"] = "repairing"
        runner = dict(job.get("runner") or {})
        runner.update(
            {
                "status": "repairing",
                "current_stage": failed_stage,
                "error": str(error),
                "updated_at": now,
            }
        )
        job["runner"] = runner
        job["updated_at"] = now
        self.save_job(job)
        return True

    def run_repair_cycle(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        repair = dict(job.get("repair") or {})
        if job.get("status") != "repairing" or repair.get("status") != "queued":
            return self.public_job(job)
        config = self.repair_config()
        if not config.get("enabled"):
            return self.block_repair(job, "repair model is not configured or enabled")
        scheduled_action = repair.get("scheduled_action")
        if (
            isinstance(scheduled_action, dict)
            and scheduled_action.get("type") == "wait_and_retry"
        ):
            stage = self.failed_stage(job) or normalize_stage_name(
                str(repair.get("trigger_stage") or "")
            )
            if not stage or stage not in self.stage_order_for_job(job):
                return self.block_repair(
                    job,
                    "scheduled repair did not resolve to a valid stage",
                )
            repair.update(
                {
                    "status": "executing",
                    "executed_action": scheduled_action,
                    "scheduled_action": None,
                    "updated_at": iso_now(),
                }
            )
            job["repair"] = repair
            job["status"] = "failed"
            runner = dict(job.get("runner") or {})
            runner.update({"status": "failed", "current_stage": None})
            job["runner"] = runner
            self.save_job(job)
            return self.rerun_from_stage(job_id, stage, enqueue=True)

        cycle = int(repair.get("cycle") or 0) + 1
        if cycle > config["max_cycles"]:
            return self.exhaust_repair(job, "repair cycle limit reached")

        incident_id = uuid.uuid4().hex
        incident_dir = self.repair_incident_dir(job_id, incident_id)
        incident_dir.mkdir(parents=True, exist_ok=False)
        evidence = self.build_repair_evidence(job, config)
        fingerprint = failure_fingerprint(evidence)
        history = list(repair.get("history") or [])
        same_count = sum(
            1
            for item in history
            if isinstance(item, dict) and item.get("failure_fingerprint") == fingerprint
        ) + 1
        if same_count >= config["same_fingerprint_limit"]:
            return self.exhaust_repair(job, "the same failure fingerprint repeated")

        write_json(incident_dir / "evidence.json", evidence)
        now = iso_now()
        repair.update(
            {
                "status": "analyzing",
                "cycle": cycle,
                "incident_id": incident_id,
                "failure_fingerprint": fingerprint,
                "updated_at": now,
                "model": config["model"],
            }
        )
        job["repair"] = repair
        job["updated_at"] = now
        self.save_job(job)

        try:
            decision, raw_response = self.request_repair_decision(evidence, config)
        except Exception as exc:
            write_json(
                incident_dir / "decision_error.json",
                {"error": str(exc), "created_at": iso_now()},
            )
            self.repair_model_last_error = str(exc)
            return self.block_repair(self.load_job(job_id), f"repair model failed: {exc}")

        self.repair_model_last_error = ""
        self.repair_model_last_success_at = iso_now()
        (incident_dir / "response.txt").write_text(raw_response, encoding="utf-8")
        write_json(incident_dir / "decision.json", decision)
        job = self.load_job(job_id)
        repair = dict(job.get("repair") or {})
        entry = {
            "cycle": cycle,
            "incident_id": incident_id,
            "failure_fingerprint": fingerprint,
            "diagnosis": decision["diagnosis"],
            "action": decision["action"],
            "created_at": iso_now(),
        }
        history = list(repair.get("history") or [])
        history.append(entry)
        repair["history"] = history[-20:]
        repair["diagnosis"] = decision["diagnosis"]
        repair["pending_action"] = decision["action"]
        repair["updated_at"] = iso_now()
        job["repair"] = repair
        self.save_job(job)
        return self.execute_repair_decision(job_id, incident_id, decision)

    def build_repair_evidence(self, job: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        stage = self.failed_stage(job) or normalize_stage_name(
            str((job.get("repair") or {}).get("trigger_stage") or "")
        )
        stage_info = dict((job.get("stages") or {}).get(stage) or {})
        log_path = Path(str(stage_info.get("log_path") or self.stage_log_path(job["job_id"], stage)))
        log_text = ""
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        fatal_lines = [line for line in log_text.splitlines() if FATAL_LINE_PATTERN.search(line)]
        selected_log = "\n".join((fatal_lines[:20] + log_text.splitlines()[-120:]))
        selected_log = redact_text(selected_log)[-config["max_log_chars"] :]
        run_dir = self.discover_run_dir(job)
        artifacts = []
        if run_dir and run_dir.is_dir():
            for relative in (
                "analysis.json",
                "transcript.md",
                "operation_manual.md",
                "operation_manual.quality_failed.md",
                "manual_evidence.md",
                "docs_analysis_chapters/knowledge_notes_v2.md",
                "docs_analysis_chapters/deep_report_v2.md",
                "final_publish_summary.json",
            ):
                path = run_dir / relative
                artifacts.append(
                    {
                        "path": relative,
                        "exists": path.is_file(),
                        "size": path.stat().st_size if path.is_file() else 0,
                    }
                )
        failure = stage_info.get("failure") if isinstance(stage_info.get("failure"), dict) else {}
        evidence = {
            "schema_version": 1,
            "job_id": job.get("job_id"),
            "source_type": job.get("source_type"),
            "stage": stage,
            "next_stage": self.next_stage(job),
            "job_status": job.get("status"),
            "runner_error": redact_text(str((job.get("runner") or {}).get("error") or "")),
            "failure": redact_value(failure),
            "stage_attempt": stage_info.get("attempt"),
            "stage_error": redact_text(str(stage_info.get("error") or "")),
            "stage_log": selected_log,
            "durable_artifacts": artifacts,
            "missing_core_artifacts": self.missing_core_artifacts(run_dir) if run_dir else [],
            "failure_disposition": redact_value(self.failure_disposition({**job, "status": "failed"})),
            "allowed_actions": sorted(ALLOWED_ACTIONS),
            "automatic_actions": sorted(AUTOMATIC_ACTIONS),
            "service_actions": sorted(
                str(name)
                for name in (config.get("service_actions") or {})
            ),
        }
        encoded = json.dumps(evidence, ensure_ascii=False)
        if len(encoded) > config["max_evidence_chars"]:
            evidence["stage_log"] = evidence["stage_log"][-max(4000, config["max_evidence_chars"] // 2) :]
        return evidence

    def request_repair_decision(
        self,
        evidence: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        prompt = repair_prompt(evidence)
        payload = {
            "model": config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You diagnose video pipeline incidents. Return one JSON object only. "
                        "Never invent commands, paths, stages, credentials, or artifacts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": int(config.get("max_tokens") or 1200),
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            session = requests.Session()
            session.trust_env = False
            content = self.call_repair_model(session, payload, config)
            try:
                decision = validate_repair_decision(parse_json_object(content), evidence)
                self.repair_model_last_success_at = iso_now()
                self.repair_model_last_error = ""
                return decision, content
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                correction = dict(payload)
                correction["messages"] = [
                    *payload["messages"],
                    {"role": "assistant", "content": content[:12000]},
                    {
                        "role": "user",
                        "content": (
                            f"The response was rejected: {exc}. Return a corrected JSON object only, "
                            "using the required shape and allowed recovery boundary."
                        ),
                    },
                ]
                corrected = self.call_repair_model(session, correction, config)
                decision = validate_repair_decision(parse_json_object(corrected), evidence)
                self.repair_model_last_success_at = iso_now()
                self.repair_model_last_error = ""
                return decision, corrected
        finally:
            if "session" in locals():
                session.close()

    @staticmethod
    def call_repair_model(
        session: requests.Session,
        payload: dict[str, Any],
        config: dict[str, Any],
    ) -> str:
        response = session.post(
            f"{config['base_url']}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=config["timeout_seconds"],
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        message = choices[0].get("message") if choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("repair model returned no content")
        return content

    def execute_repair_decision(
        self,
        job_id: str,
        incident_id: str,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        job = self.load_job(job_id)
        repair = dict(job.get("repair") or {})
        action = decision["action"]
        action_type = action["type"]
        if action_type in APPROVAL_ACTIONS:
            if action_type == "request_code_patch":
                try:
                    action = self.prepare_code_patch_proposal(
                        job,
                        self.repair_incident_dir(job_id, incident_id),
                        action,
                    )
                except Exception as exc:
                    return self.block_repair(job, f"code patch proposal rejected: {exc}")
            repair.update(
                {
                    "status": "waiting_approval",
                    "pending_action": action,
                    "updated_at": iso_now(),
                }
            )
            job["repair"] = repair
            job["status"] = "repairing"
            self.save_job(job)
            return self.public_job(job)
        if action_type == "manual_review":
            return self.block_repair(job, action.get("reason") or "manual review required")
        if action_type == "wait_and_retry":
            delay = max(1, min(int(action.get("delay_seconds") or 60), 600))
            repair.update(
                {
                    "status": "queued",
                    "scheduled_action": action,
                    "next_attempt_at": iso_from_timestamp(time.time() + delay),
                    "updated_at": iso_now(),
                }
            )
            job["repair"] = repair
            self.save_job(job)
            return self.public_job(job)

        stage = self.failed_stage(job) or normalize_stage_name(str(repair.get("trigger_stage") or ""))
        if action_type == "resume_first_incomplete_stage":
            stage = self.next_stage(job) or stage
        if not stage or stage not in self.stage_order_for_job(job):
            return self.block_repair(job, "repair decision did not resolve to a valid stage")
        repair.update(
            {
                "status": "executing",
                "executed_action": action,
                "updated_at": iso_now(),
                "pending_action": None,
            }
        )
        job["repair"] = repair
        job["status"] = "failed"
        runner = dict(job.get("runner") or {})
        runner["status"] = "failed"
        runner["current_stage"] = None
        job["runner"] = runner
        self.save_job(job)
        return self.rerun_from_stage(job_id, stage, enqueue=True)

    def approve_repair(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        job = self.load_job(job_id)
        repair = dict(job.get("repair") or {})
        if repair.get("status") != "waiting_approval":
            raise BridgeError(HTTPStatus.CONFLICT, "repair is not waiting for approval")
        if str(payload.get("incident_id") or "") != str(repair.get("incident_id") or ""):
            raise BridgeError(HTTPStatus.CONFLICT, "repair incident changed; refresh before approving")
        action = dict(repair.get("pending_action") or {})
        if action.get("type") == "request_code_patch":
            self.apply_approved_code_patch(action)
            repair.update(
                {
                    "status": "queued",
                    "pending_action": None,
                    "approved_action": {
                        "type": "request_code_patch",
                        "files": action.get("files") or [],
                    },
                    "queued_at": iso_now(),
                    "next_attempt_at": iso_now(),
                    "updated_at": iso_now(),
                }
            )
            job["repair"] = repair
            job["status"] = "repairing"
            self.save_job(job)
            return self.public_job(job)
        if action.get("type") == "request_service_restart":
            self.run_approved_service_action(str(action.get("service_id") or ""))
            repair.update(
                {
                    "status": "queued",
                    "pending_action": None,
                    "approved_action": action,
                    "queued_at": iso_now(),
                    "next_attempt_at": iso_now(),
                    "updated_at": iso_now(),
                }
            )
            job["repair"] = repair
            job["status"] = "repairing"
            self.save_job(job)
            return self.public_job(job)
        raise BridgeError(HTTPStatus.BAD_REQUEST, "repair action cannot be approved")

    def prepare_code_patch_proposal(
        self,
        job: dict[str, Any],
        incident_dir: Path,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        patch_text = str(action.get("proposal") or "").strip()
        files, changed_lines = validate_patch_text(patch_text)
        patch_path = incident_dir / "proposal.patch"
        patch_path.write_text(patch_text + "\n", encoding="utf-8")
        file_hashes = {
            relative: file_sha256(self.repo_root / relative)
            for relative in files
        }
        validation = self.validate_patch_in_sandbox(patch_path, files)
        write_json(incident_dir / "patch_validation.json", validation)
        if not validation.get("ok"):
            raise ValueError(validation.get("error") or "sandbox validation failed")
        return {
            **action,
            "proposal": "",
            "patch_path": str(patch_path),
            "files": files,
            "changed_lines": changed_lines,
            "file_hashes": file_hashes,
            "validation": validation,
        }

    def validate_patch_in_sandbox(
        self,
        patch_path: Path,
        files: list[str],
    ) -> dict[str, Any]:
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="video-analyzer-repair-") as tmp:
            sandbox = Path(tmp) / "repo"
            shutil.copytree(
                self.repo_root,
                sandbox,
                symlinks=True,
                ignore=shutil.ignore_patterns(*PATCH_COPY_IGNORE),
            )
            subprocess.run(
                ["git", "init", "-q"],
                cwd=sandbox,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            sandbox_patch = sandbox / ".incident-repair.patch"
            shutil.copy2(patch_path, sandbox_patch)
            try:
                subprocess.run(
                    ["git", "apply", "--check", str(sandbox_patch)],
                    cwd=sandbox,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", str(sandbox_patch)],
                    cwd=sandbox,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                checks = self.patch_validation_commands(files, sandbox)
                results = []
                for command in checks:
                    completed = subprocess.run(
                        command,
                        cwd=sandbox,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        env={
                            **os.environ,
                            "PYTHONPATH": f"{sandbox / 'video-analyzer-ui'}:{sandbox}",
                        },
                    )
                    results.append(
                        {
                            "command": command,
                            "returncode": completed.returncode,
                            "stdout_tail": completed.stdout[-2000:],
                            "stderr_tail": completed.stderr[-2000:],
                        }
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                return {
                    "ok": False,
                    "error": str(exc),
                    "duration_seconds": round(time.time() - started, 3),
                }
        return {
            "ok": True,
            "checks": results,
            "duration_seconds": round(time.time() - started, 3),
        }

    def patch_validation_commands(
        self,
        files: list[str],
        sandbox: Path,
    ) -> list[list[str]]:
        commands: list[list[str]] = []
        python_files = [str(sandbox / path) for path in files if path.endswith(".py")]
        if python_files:
            commands.append([str(self.repo_root / ".venv" / "bin" / "python"), "-m", "py_compile", *python_files])
        for path in files:
            if path.endswith(".js"):
                commands.append(["node", "--check", str(sandbox / path)])
        if any(
            path.startswith(("video_analyzer/jobengine/", "video-analyzer-ui/video_analyzer_ui/"))
            for path in files
        ):
            commands.append(
                [
                    str(self.repo_root / ".venv" / "bin" / "python"),
                    "-m",
                    "unittest",
                    "tests.test_video_link_status_server",
                    "tests.test_video_analyzer_ui",
                ]
            )
        return commands

    def apply_approved_code_patch(self, action: dict[str, Any]) -> None:
        patch_path = Path(str(action.get("patch_path") or ""))
        files = [str(path) for path in (action.get("files") or [])]
        expected_hashes = action.get("file_hashes") or {}
        if not patch_path.is_file() or not files:
            raise BridgeError(HTTPStatus.CONFLICT, "approved repair patch artifact is missing")
        for relative in files:
            if file_sha256(self.repo_root / relative) != str(expected_hashes.get(relative) or ""):
                raise BridgeError(
                    HTTPStatus.CONFLICT,
                    f"repair target changed after validation: {relative}",
                )
        try:
            subprocess.run(
                ["git", "apply", "--check", str(patch_path)],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch_path)],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BridgeError(
                HTTPStatus.CONFLICT,
                f"approved repair patch could not be applied: {exc}",
            ) from exc

    def run_approved_service_action(self, service_id: str) -> None:
        actions = self.repair_config().get("service_actions") or {}
        spec = actions.get(service_id) if isinstance(actions, dict) else None
        if not isinstance(spec, dict):
            raise BridgeError(HTTPStatus.BAD_REQUEST, "repair service action is not allowlisted")
        kind = str(spec.get("kind") or "")
        unit = str(spec.get("unit") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", unit):
            raise BridgeError(HTTPStatus.BAD_REQUEST, "repair service unit is invalid")
        if kind == "ssh_systemd_user":
            host = str(spec.get("host") or "")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", host):
                raise BridgeError(HTTPStatus.BAD_REQUEST, "repair SSH host is invalid")
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                host,
                "systemctl",
                "--user",
                "restart",
                unit,
            ]
        elif kind == "systemd_user":
            command = ["systemctl", "--user", "restart", unit]
        else:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "repair service action kind is not allowed")
        try:
            subprocess.run(
                command,
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(10, min(int(spec.get("timeout_seconds") or 120), 600)),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BridgeError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"approved service restart failed: {exc}",
            ) from exc
        health_url = str(spec.get("health_url") or "")
        if health_url:
            session = requests.Session()
            session.trust_env = False
            try:
                deadline = time.monotonic() + max(
                    10,
                    min(int(spec.get("health_timeout_seconds") or 180), 600),
                )
                last_error = ""
                while time.monotonic() < deadline:
                    try:
                        response = session.get(health_url, timeout=5)
                        if response.ok:
                            return
                        last_error = f"HTTP {response.status_code}"
                    except requests.RequestException as exc:
                        last_error = str(exc)
                    time.sleep(2)
                raise BridgeError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    f"approved service restarted but health did not recover: {last_error}",
                )
            finally:
                session.close()

    def reject_repair(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        job = self.load_job(job_id)
        repair = dict(job.get("repair") or {})
        if repair.get("status") != "waiting_approval":
            raise BridgeError(HTTPStatus.CONFLICT, "repair is not waiting for approval")
        if payload.get("incident_id") and str(payload["incident_id"]) != str(repair.get("incident_id") or ""):
            raise BridgeError(HTTPStatus.CONFLICT, "repair incident changed; refresh before rejecting")
        return self.block_repair(job, str(payload.get("reason") or "repair action rejected by user"))

    def retry_repair(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        repair = dict(job.get("repair") or {})
        if repair.get("status") not in {"blocked", "waiting_approval"}:
            raise BridgeError(HTTPStatus.CONFLICT, "repair cannot be retried from its current state")
        repair.update(
            {
                "status": "queued",
                "queued_at": iso_now(),
                "next_attempt_at": iso_now(),
                "pending_action": None,
                "updated_at": iso_now(),
            }
        )
        job["repair"] = repair
        job["status"] = "repairing"
        self.save_job(job)
        return self.public_job(job)

    def disable_repair(self, job_id: str) -> dict[str, Any]:
        job = self.load_job(job_id)
        job.setdefault("options", {})["auto_repair"] = False
        repair = dict(job.get("repair") or {})
        repair.update({"enabled": False, "status": "disabled", "updated_at": iso_now()})
        job["repair"] = repair
        if job.get("status") == "repairing":
            job["status"] = "failed"
            runner = dict(job.get("runner") or {})
            runner["status"] = "failed"
            job["runner"] = runner
        self.save_job(job)
        if job.get("status") == "failed":
            self.advance_collection_after_job(job_id)
        return self.public_job(job)

    def complete_repair_if_active(self, job: dict[str, Any]) -> None:
        repair = dict(job.get("repair") or {})
        if repair.get("status") not in {"queued", "analyzing", "executing", "verifying"}:
            return
        repair.update(
            {
                "status": "succeeded",
                "pending_action": None,
                "finished_at": iso_now(),
                "updated_at": iso_now(),
            }
        )
        job["repair"] = repair

    def block_repair(self, job: dict[str, Any], reason: str) -> dict[str, Any]:
        return self.finish_repair(job, "blocked", reason)

    def exhaust_repair(self, job: dict[str, Any], reason: str) -> dict[str, Any]:
        return self.finish_repair(job, "exhausted", reason)

    def finish_repair(self, job: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
        repair = dict(job.get("repair") or {})
        repair.update(
            {
                "status": status,
                "reason": str(reason),
                "pending_action": None,
                "finished_at": iso_now(),
                "updated_at": iso_now(),
            }
        )
        job["repair"] = repair
        job["status"] = "failed"
        runner = dict(job.get("runner") or {})
        runner.update({"status": "failed", "error": str(reason), "finished_at": iso_now()})
        job["runner"] = runner
        job["updated_at"] = iso_now()
        self.save_job(job)
        self.advance_collection_after_job(str(job["job_id"]))
        return self.public_job(job)

    def repair_incident_dir(self, job_id: str, incident_id: str) -> Path:
        return self.job_dir(job_id) / "repairs" / incident_id

    def failed_stage(self, job: dict[str, Any]) -> str:
        for stage in self.stage_order_for_job(job):
            if ((job.get("stages") or {}).get(stage) or {}).get("status") == "failed":
                return stage
        return normalize_stage_name(
            str((job.get("runner") or {}).get("current_stage") or "")
        )

    def repair_worker_status(self) -> dict[str, Any]:
        config = self.repair_config()
        queued = 0
        active_job_id = ""
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            repair = job.get("repair") or {}
            if repair.get("status") == "queued":
                queued += 1
            if repair.get("status") in {"analyzing", "executing", "verifying"}:
                active_job_id = str(job.get("job_id") or "")
        return {
            "alive": bool(self.repair_thread and self.repair_thread.is_alive()),
            "configured": bool(config.get("configured")),
            "enabled": bool(config.get("enabled")),
            "model": config.get("model") or "",
            "queue_depth": queued,
            "active_job_id": active_job_id,
            "last_error": self.repair_last_error or self.repair_model_last_error,
            "last_success_at": self.repair_model_last_success_at,
        }


def failure_fingerprint(evidence: dict[str, Any]) -> str:
    failure = evidence.get("failure") or {}
    raw = {
        "stage": evidence.get("stage"),
        "kind": failure.get("kind"),
        "status_code": failure.get("status_code"),
        "provider_code": failure.get("provider_code"),
        "fatal": normalize_failure_text(
            str(evidence.get("stage_error") or evidence.get("runner_error") or evidence.get("stage_log") or "")
        ),
    }
    return hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_patch_text(patch_text: str) -> tuple[list[str], int]:
    if not patch_text.startswith("diff --git "):
        raise ValueError("proposal must be a unified git diff")
    files: list[str] = []
    changed_lines = 0
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            match = re.fullmatch(r"diff --git a/(.+) b/(.+)", line)
            if not match or match.group(1) != match.group(2):
                raise ValueError("file renames are not allowed")
            relative = match.group(1)
            if (
                relative in PATCH_DENIED_PATHS
                or relative.startswith((".", "/"))
                or ".." in Path(relative).parts
                or not relative.startswith(PATCH_PATH_PREFIXES)
            ):
                raise ValueError(f"patch path is not allowed: {relative}")
            if relative not in files:
                files.append(relative)
        elif line.startswith(("new file mode", "deleted file mode", "rename from", "rename to")):
            raise ValueError("creating, deleting, or renaming files is not allowed")
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            changed_lines += 1
    if not files:
        raise ValueError("patch does not modify an allowed source file")
    if len(files) > 3:
        raise ValueError("patch modifies more than 3 files")
    if changed_lines > 300:
        raise ValueError("patch changes more than 300 lines")
    return files, changed_lines


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"repair target does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_failure_text(value: str) -> str:
    text = value.lower()
    text = re.sub(r"\b\d{4}-\d\d-\d\d[ t]\d\d:\d\d:\d\d(?:[.,]\d+)?(?:[+-]\d+)?\b", "<time>", text)
    text = re.sub(r"\bpid[ =:]+\d+\b", "pid=<n>", text)
    text = re.sub(r"\b0x[0-9a-f]+\b", "<hex>", text)
    text = re.sub(r"\b\d+\.\d+\.\d+\.\d+:\d+\b", "<endpoint>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1000]


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): ("<redacted>" if SENSITIVE_KEY_PATTERN.search(str(key)) else redact_value(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(value: str) -> str:
    text = value
    for pattern in SENSITIVE_TEXT_PATTERNS:
        text = pattern.sub(r"\1<redacted>", text)
    return text


def repair_prompt(evidence: dict[str, Any]) -> str:
    schema = {
        "diagnosis": {
            "summary": "short root-cause statement",
            "category": "transient|resource|configuration|code|external|quality|unknown",
            "confidence": 0.0,
            "evidence_refs": ["stage_error", "stage_log"],
        },
        "action": {
            "type": "|".join(sorted(ALLOWED_ACTIONS)),
            "stage": evidence.get("stage"),
            "delay_seconds": 60,
            "reason": "why this is the smallest safe action",
            "service_id": "",
            "proposal": "",
        },
    }
    return (
        "Choose the smallest safe recovery action. Automatic actions may only retry the failed "
        "stage, resume the deterministic first incomplete stage, or wait. Never switch runtime "
        "profiles, delete artifacts, or invent shell commands. Use request_service_restart or "
        "request_code_patch only as proposals requiring approval.\n\n"
        f"Required JSON shape:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Incident evidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("repair decision must be a JSON object")
    return value


def validate_repair_decision(decision: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    diagnosis = decision.get("diagnosis")
    action = decision.get("action")
    if not isinstance(diagnosis, dict) or not isinstance(action, dict):
        raise ValueError("repair decision requires diagnosis and action objects")
    action_type = str(action.get("type") or "")
    if action_type not in ALLOWED_ACTIONS:
        raise ValueError(f"repair action is not allowed: {action_type}")
    stage = normalize_stage_name(str(action.get("stage") or evidence.get("stage") or ""))
    if action_type in {"retry_failed_stage", "resume_first_incomplete_stage"}:
        allowed_stages = {normalize_stage_name(str(evidence.get("stage") or ""))}
        next_stage = normalize_stage_name(str(evidence.get("next_stage") or ""))
        if next_stage:
            allowed_stages.add(next_stage)
        if stage not in allowed_stages:
            raise ValueError("repair action selected a stage outside the deterministic recovery boundary")
    failure_kind = str((evidence.get("failure") or {}).get("kind") or "")
    disposition = evidence.get("failure_disposition") or {}
    if action_type in AUTOMATIC_ACTIONS and (
        failure_kind.startswith("permanent_")
        or disposition.get("category") in {"review_required", "artifacts_complete", "superseded"}
    ):
        raise ValueError("this failure class requires manual review instead of an automatic action")
    if action_type == "request_service_restart":
        service_id = str(action.get("service_id") or "")
        if service_id not in set(evidence.get("service_actions") or []):
            raise ValueError("repair service action is not allowlisted")
    normalized = {
        "diagnosis": {
            "summary": str(diagnosis.get("summary") or "No diagnosis supplied")[:1000],
            "category": str(diagnosis.get("category") or "unknown")[:80],
            "confidence": max(0.0, min(float(diagnosis.get("confidence") or 0), 1.0)),
            "evidence_refs": [
                str(item)[:120]
                for item in (diagnosis.get("evidence_refs") or [])
                if isinstance(item, (str, int, float))
            ][:10],
        },
        "action": {
            "type": action_type,
            "stage": stage,
            "delay_seconds": max(1, min(int(action.get("delay_seconds") or 60), 600)),
            "reason": str(action.get("reason") or "")[:1200],
            "service_id": str(action.get("service_id") or "")[:120],
            "proposal": str(action.get("proposal") or "")[:12000],
        },
    }
    return normalized


def parse_repair_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError:
        return 0.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
