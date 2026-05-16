import unittest
from unittest.mock import patch

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import build_openai_extra_body, resolve_api_key, resolve_temperature


class GenericOpenAIAPIClientTests(unittest.TestCase):
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

    def test_local_endpoint_allows_placeholder_key(self):
        self.assertEqual(resolve_api_key(api_url="http://100.90.114.26:18081/v1"), "0")

    def test_missing_deepseek_key_reports_env_name(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as raised:
                resolve_api_key(api_url="https://api.deepseek.com")

        self.assertIn("DEEPSEEK_API_KEY", str(raised.exception))

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


if __name__ == "__main__":
    unittest.main()
