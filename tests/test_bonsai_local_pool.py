import unittest
from unittest.mock import patch

from tools.ops.bonsai_local_pool import (
    PoolServer,
    Worker,
    configured_workers,
    worker_command,
)


class BonsaiLocalPoolTests(unittest.TestCase):
    def setUp(self):
        self.workers = [
            Worker("0", "GPU-p40", "Tesla P40", 18110),
            Worker("5", "GPU-p40-5", "Tesla P40", 18114),
        ]

    @patch("tools.ops.bonsai_local_pool.worker_ready", return_value=True)
    def test_health_reports_all_p40_workers(self, _ready):
        response = PoolServer(self.workers).app.test_client().get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["worker_count"], 2)
        self.assertEqual(payload["ready_workers"], 2)
        self.assertNotIn("fallback_to_p40", payload)

    def test_worker_command_uses_qwen38_mtp4_and_full_gpu_offload(self):
        command = worker_command(self.workers[1])

        self.assertEqual(command[command.index("--device") + 1], "CUDA0")
        self.assertEqual(command[command.index("--ctx-size") + 1], "65536")
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "all")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(command[command.index("--cache-type-v") + 1], "q8_0")
        self.assertEqual(command[command.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "4")
        self.assertNotIn("--mmproj", command)

    def test_request_can_fail_immediately_when_pool_is_busy(self):
        server = PoolServer(self.workers)
        reserved = [server.available.get_nowait() for _ in self.workers]
        try:
            response = server.app.test_client().post(
                "/v1/chat/completions",
                headers={"X-Bonsai-Acquire-Timeout": "0"},
                json={"messages": [{"role": "user", "content": "test"}]},
            )
        finally:
            for item in reserved:
                server.available.put(item)

        self.assertEqual(response.status_code, 503)
        self.assertIn("busy", response.get_json()["error"]["message"])

    @patch(
        "tools.ops.bonsai_local_pool.gpu_inventory",
        return_value={"3": ("GPU-v100", "Tesla V100-SXM2-16GB")},
    )
    @patch("tools.ops.bonsai_local_pool.WORKER_COUNT", 1)
    @patch("tools.ops.bonsai_local_pool.GPU_IDS", ("3",))
    def test_configured_workers_accepts_v100(self, _inventory):
        workers = configured_workers()

        self.assertEqual(workers[0].gpu_id, "3")
        self.assertIn("V100", workers[0].name)


if __name__ == "__main__":
    unittest.main()
