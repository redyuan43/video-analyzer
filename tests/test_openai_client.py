import unittest

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient


class GenericOpenAIAPIClientTests(unittest.TestCase):
    def test_tailscale_cgnat_bypasses_env_proxy(self):
        client = GenericOpenAIAPIClient("0", "http://100.90.114.26:18081/v1")

        self.assertFalse(client.session.trust_env)
        self.assertTrue(client._allows_reasoning_content_fallback())

    def test_public_endpoint_keeps_env_proxy(self):
        client = GenericOpenAIAPIClient("key", "https://openrouter.ai/api/v1")

        self.assertTrue(client.session.trust_env)


if __name__ == "__main__":
    unittest.main()
