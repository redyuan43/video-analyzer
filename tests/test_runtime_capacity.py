import unittest
from unittest.mock import Mock, patch

from video_analyzer.runtime_capacity import (
    endpoint_worker_capacity,
    health_url_for_endpoint,
    resolve_endpoint_concurrency,
)


class RuntimeCapacityTests(unittest.TestCase):
    def test_health_url_replaces_openai_path(self):
        self.assertEqual(
            health_url_for_endpoint("http://127.0.0.1:18088/v1"),
            "http://127.0.0.1:18088/api/health",
        )

    @patch("video_analyzer.runtime_capacity.requests.Session")
    def test_worker_capacity_uses_proxy_worker_list(self, session_class):
        response = Mock()
        response.json.return_value = {"workers": [{"gpu": 0}, {"gpu": 4}]}
        session_class.return_value.get.return_value = response

        self.assertEqual(endpoint_worker_capacity("http://127.0.0.1:18082/v1"), 2)
        self.assertFalse(session_class.return_value.trust_env)
        session_class.return_value.close.assert_called_once_with()

    @patch("video_analyzer.runtime_capacity.requests.Session")
    def test_worker_capacity_accepts_worker_count(self, session_class):
        response = Mock()
        response.json.return_value = {"worker_count": 5}
        session_class.return_value.get.return_value = response

        self.assertEqual(endpoint_worker_capacity("http://127.0.0.1:18012/v1"), 5)

    @patch("video_analyzer.runtime_capacity.endpoint_worker_capacity")
    def test_auto_concurrency_sums_endpoint_capacity(self, capacity):
        capacity.side_effect = [3, 2]

        self.assertEqual(
            resolve_endpoint_concurrency(
                "auto",
                ["http://127.0.0.1:18088/v1", "http://127.0.0.1:18089/v1"],
            ),
            5,
        )
        self.assertEqual(resolve_endpoint_concurrency(4, []), 4)


if __name__ == "__main__":
    unittest.main()
