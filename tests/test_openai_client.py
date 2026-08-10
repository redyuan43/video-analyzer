import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient, OpenAIAPIError
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature
from video_analyzer.failures import FAILURE_FILE_ENV


class GenericOpenAIAPIClientTests(unittest.TestCase):
    def _operation_manual_args(self, **overrides):
        values = {
            "client": None,
            "profile": None,
            "asr_provider": None,
            "task": "operation_manual",
            "llm_base_url": "http://100.90.114.26:18081/v1",
            "vision_base_url": "http://100.96.79.21:18082/v1",
            "text_base_url": "http://100.90.114.26:18081/v1",
            "vision_model": "minicpm-v-4.5-v100",
            "text_model": "local-text-model",
        }
        values.update(overrides)
        return Namespace(**values)

    def test_local_text_endpoint_drops_default_deepseek_key_env(self):
        config = Config()
        config.update_from_args(self._operation_manual_args())

        manual_config = config.get("operation_manual")

        self.assertEqual(manual_config["text_base_url"], "http://100.90.114.26:18081/v1")
        self.assertNotIn("text_api_key_env", manual_config)
        self.assertEqual(resolve_api_key(api_url=manual_config["text_base_url"]), "0")

    def test_deepseek_text_endpoint_keeps_default_key_env(self):
        config = Config()
        config.update_from_args(
            self._operation_manual_args(
                llm_base_url="https://api.deepseek.com",
                text_base_url="https://api.deepseek.com",
                text_model="deepseek-v4-pro",
            )
        )

        manual_config = config.get("operation_manual")

        self.assertEqual(manual_config["text_api_key_env"], "DEEPSEEK_API_KEY")

    def test_tailscale_cgnat_bypasses_env_proxy(self):
        client = GenericOpenAIAPIClient("0", "http://100.90.114.26:18081/v1")

        self.assertFalse(client.session.trust_env)
        self.assertTrue(client._allows_reasoning_content_fallback())

    def test_public_endpoint_keeps_env_proxy(self):
        client = GenericOpenAIAPIClient("key", "https://openrouter.ai/api/v1")

        self.assertTrue(client.session.trust_env)

    def test_deepseek_endpoint_uses_standard_env_key(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "secret"}, clear=True):
            self.assertEqual(resolve_api_key(api_url="https://api.deepseek.com"), "secret")

    def test_deepseek_endpoint_loads_default_env_file(self):
        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "deepseek.env"
            env_path.write_text("export DEEPSEEK_API_KEY=file-secret\n", encoding="utf-8")

            with patch.dict("os.environ", {"VIDEO_ANALYZER_DEEPSEEK_ENV": str(env_path)}, clear=True):
                self.assertEqual(resolve_api_key(api_url="https://api.deepseek.com"), "file-secret")

    def test_local_endpoint_allows_placeholder_key(self):
        self.assertEqual(resolve_api_key(api_url="http://100.90.114.26:18081/v1"), "0")

    def test_text_timeout_environment_overrides_client_default(self):
        with patch.dict("os.environ", {"VIDEO_ANALYZER_TEXT_TIMEOUT_SECONDS": "3600"}, clear=True):
            client = GenericOpenAIAPIClient("0", "http://127.0.0.1:18081/v1")

        self.assertEqual(client.timeout_seconds, 3600)

    def test_missing_deepseek_key_reports_env_name(self):
        with TemporaryDirectory() as tmpdir:
            missing_env = str(Path(tmpdir) / "missing.env")
            env = {"VIDEO_ANALYZER_DEEPSEEK_ENV": missing_env}
            with patch.dict("os.environ", env, clear=True):
                with self.assertRaises(ValueError) as raised:
                    resolve_api_key(api_url="https://api.deepseek.com")

        self.assertIn("DEEPSEEK_API_KEY", str(raised.exception))

    def test_deepseek_env_file_does_not_override_shell_key(self):
        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "deepseek.env"
            env_path.write_text("DEEPSEEK_API_KEY=file-secret\n", encoding="utf-8")
            env = {"VIDEO_ANALYZER_DEEPSEEK_ENV": str(env_path), "DEEPSEEK_API_KEY": "shell-secret"}

            with patch.dict("os.environ", env, clear=True):
                self.assertEqual(resolve_api_key(api_url="https://api.deepseek.com"), "shell-secret")

    def test_missing_explicit_deepseek_key_reports_env_name(self):
        with TemporaryDirectory() as tmpdir:
            missing_env = str(Path(tmpdir) / "missing.env")
            env = {"VIDEO_ANALYZER_DEEPSEEK_ENV": missing_env}
            with patch.dict("os.environ", env, clear=True):
                with self.assertRaises(ValueError) as raised:
                    resolve_api_key(api_key_env="DEEPSEEK_API_KEY")

        self.assertIn("DEEPSEEK_API_KEY", str(raised.exception))

    def test_missing_non_deepseek_key_reports_env_name(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as raised:
                resolve_api_key(api_key_env="OTHER_API_KEY")

        self.assertIn("OTHER_API_KEY", str(raised.exception))

    def test_deepseek_extra_body_disables_thinking(self):
        extra_body = build_openai_extra_body(
            {"deepseek_thinking": "disabled"},
            "https://api.deepseek.com",
        )

        self.assertEqual(extra_body, {"thinking": {"type": "disabled"}})

    def test_review_extra_body_enables_thinking_with_effort(self):
        extra_body = build_openai_extra_body(
            {"review_deepseek_thinking": "enabled", "review_reasoning_effort": "high"},
            "https://api.deepseek.com",
            prefix="review_",
        )

        self.assertEqual(extra_body, {"thinking": {"type": "enabled"}, "reasoning_effort": "high"})

    def test_non_deepseek_extra_body_does_not_send_thinking(self):
        extra_body = build_openai_extra_body(
            {"deepseek_thinking": "disabled"},
            "http://100.90.114.26:18081/v1",
        )

        self.assertEqual(extra_body, {})

    def test_lmstudio_extra_body_sends_reasoning_effort_without_deepseek_thinking(self):
        extra_body = build_openai_extra_body(
            {
                "deepseek_thinking": "disabled",
                "reasoning_effort": "none",
            },
            "http://100.90.114.26:18081/v1",
        )

        self.assertEqual(extra_body, {"reasoning_effort": "none"})

    def test_text_temperature_uses_profile_value(self):
        self.assertEqual(resolve_temperature({"text_temperature": 1.0}, 0.2), 1.0)

    def test_client_merges_extra_body_into_request(self):
        client = GenericOpenAIAPIClient(
            "key",
            "https://api.deepseek.com",
            max_retries=1,
            extra_body={"thinking": {"type": "disabled"}},
        )
        response = type("Response", (), {})()
        response.status_code = 200
        response.json = lambda: {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(client.session, "post", return_value=response) as post:
            result = client.generate(prompt="hello", model="deepseek-v4-pro")

        self.assertEqual(result["response"], "ok")
        self.assertEqual(post.call_args.kwargs["json"]["thinking"], {"type": "disabled"})

    def test_insufficient_balance_is_permanent_and_not_retried(self):
        client = GenericOpenAIAPIClient("key", "https://api.deepseek.com", max_retries=3)
        response = type("Response", (), {})()
        response.status_code = 402
        response.text = '{"error":{"message":"Insufficient Balance","code":"invalid_request_error"}}'
        response.headers = {}
        response.json = lambda: {
            "error": {
                "message": "Insufficient Balance",
                "code": "invalid_request_error",
            }
        }

        with TemporaryDirectory() as tmpdir:
            failure_path = Path(tmpdir) / "failure.json"
            with patch.dict("os.environ", {FAILURE_FILE_ENV: str(failure_path)}, clear=False):
                with patch.object(client.session, "post", return_value=response) as post:
                    with patch("video_analyzer.clients.generic_openai_api.time.sleep") as sleep:
                        with self.assertRaises(OpenAIAPIError) as raised:
                            client.generate(prompt="hello", model="deepseek-v4-pro")

            payload = json.loads(failure_path.read_text(encoding="utf-8"))

        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(raised.exception.kind, "permanent_billing")
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(payload["status_code"], 402)

    def test_rate_limit_retries_with_retry_after_then_writes_transient_failure(self):
        client = GenericOpenAIAPIClient("key", "https://api.deepseek.com", max_retries=2)
        response = type("Response", (), {})()
        response.status_code = 429
        response.text = '{"error":{"message":"rate limited","code":"rate_limit"}}'
        response.headers = {"Retry-After": "1"}
        response.json = lambda: {"error": {"message": "rate limited", "code": "rate_limit"}}

        with TemporaryDirectory() as tmpdir:
            failure_path = Path(tmpdir) / "failure.json"
            with patch.dict("os.environ", {FAILURE_FILE_ENV: str(failure_path)}, clear=False):
                with patch.object(client.session, "post", return_value=response) as post:
                    with patch("video_analyzer.clients.generic_openai_api.time.sleep") as sleep:
                        with self.assertRaises(OpenAIAPIError) as raised:
                            client.generate(prompt="hello", model="deepseek-v4-pro")

        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1.0)
        self.assertEqual(raised.exception.kind, "transient_http")
        self.assertTrue(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()
