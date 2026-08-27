"""video-link 状态引擎 settings 片：模型/profile 设置读写、通路测试与 TTS 试听。

主类 VideoLinkStatusServer 继承 SettingsMixin，保持对外 API 不变。
"""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any

import requests

from video_analyzer.jobengine.errors import BridgeError
from video_analyzer.model_settings import SettingsValidationError


class SettingsMixin:
    """模型/profile 设置相关方法片（由主类继承）。"""

    def settings(self) -> dict[str, Any]:
        try:
            return self.runtime_settings.public_settings()
        except (OSError, json.JSONDecodeError, SettingsValidationError) as exc:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)) from exc

    def save_model_setting(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.runtime_settings.save_model(model_id, payload)
        except SettingsValidationError as exc:
            raise BridgeError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    def delete_model_setting(self, model_id: str) -> dict[str, Any]:
        try:
            return self.runtime_settings.delete_model(model_id)
        except FileNotFoundError as exc:
            raise BridgeError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except SettingsValidationError as exc:
            raise BridgeError(HTTPStatus.CONFLICT, str(exc)) from exc

    def test_model_setting(self, model_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        self.ensure_settings_test_idle()
        try:
            return self.runtime_settings.test_model(
                model_id,
                str(payload.get("mode") or "quick"),
                force=bool(payload.get("force")),
            )
        except FileNotFoundError as exc:
            raise BridgeError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except SettingsValidationError as exc:
            raise BridgeError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    def test_profile_setting(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_settings_test_idle()
        try:
            return self.runtime_settings.test_profile(payload)
        except SettingsValidationError as exc:
            raise BridgeError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    def preview_tts_setting(self, model_id: str, payload: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        self.ensure_settings_test_idle()
        try:
            return self.runtime_settings.preview_tts(model_id, str(payload.get("text") or ""))
        except FileNotFoundError as exc:
            raise BridgeError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except SettingsValidationError as exc:
            raise BridgeError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except requests.RequestException as exc:
            raise BridgeError(HTTPStatus.BAD_GATEWAY, f"TTS 试听失败：{exc}") from exc

    def settings_test_blockers(self) -> list[dict[str, str]]:
        blockers: list[dict[str, str]] = []
        for job in self.list_jobs(200).get("jobs") or []:
            runner = job.get("runner") or {}
            process = job.get("process") or {}
            status = str(job.get("status") or "")
            runner_status = str(runner.get("status") or "")
            if not (
                status in {"running", "queued"}
                or runner_status in {"running", "queued"}
                or process.get("alive")
            ):
                continue
            blockers.append(
                {
                    "job_id": str(job.get("job_id") or ""),
                    "title": str(
                        job.get("display_title")
                        or job.get("title")
                        or job.get("video_url")
                        or job.get("job_id")
                        or "后台任务"
                    ),
                    "status": runner_status or status or "running",
                    "stage": str(job.get("current_stage") or runner.get("current_stage") or ""),
                }
            )
        return blockers

    def ensure_settings_test_idle(self) -> None:
        blockers = self.settings_test_blockers()
        if not blockers:
            return
        first = blockers[0]
        stage = f"，阶段 {first['stage']}" if first.get("stage") else ""
        raise BridgeError(
            HTTPStatus.CONFLICT,
            f"后台有 {len(blockers)} 个任务正在运行或排队，通路测试暂不可用"
            f"（{first['title']}{stage}）",
        )

    def save_profile_setting(self, profile_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.runtime_settings.save_profile(profile_name, payload)
        except SettingsValidationError as exc:
            raise BridgeError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    def delete_profile_setting(self, profile_name: str) -> dict[str, Any]:
        try:
            return self.runtime_settings.delete_profile(profile_name)
        except FileNotFoundError as exc:
            raise BridgeError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except SettingsValidationError as exc:
            raise BridgeError(HTTPStatus.CONFLICT, str(exc)) from exc

    def activate_profile_setting(self, profile_name: str) -> dict[str, Any]:
        try:
            return self.runtime_settings.activate_profile(profile_name)
        except FileNotFoundError as exc:
            raise BridgeError(HTTPStatus.NOT_FOUND, str(exc)) from exc