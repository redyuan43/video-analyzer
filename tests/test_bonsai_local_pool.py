import unittest
from unittest.mock import patch

from tools.bonsai_local_pool import (
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

    @patch("tools.bonsai_local_pool.worker_ready", return_value=True)
    def test_health_reports_all_p40_workers(self, _ready):
        response = PoolServer(self.workers).app.test_client().get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["worker_count"], 2)
        self.assertEqual(payload["ready_workers"], 2)
        self.assertNotIn("fallback_to_p40", payload)

    def test_worker_command_keeps_f16_kv_and_full_gpu_offload(self):
        command = worker_command(self.workers[1])

        self.assertEqual(command[command.index("--device") + 1], "CUDA0")
        self.assertEqual(command[command.index("--ctx-size") + 1], "128405")
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "999")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "f16")
        self.assertEqual(command[command.index("--cache-type-v") + 1], "f16")

    @patch(
        "tools.bonsai_local_pool.gpu_inventory",
        return_value={"3": ("GPU-v100", "Tesla V100-SXM2-16GB")},
    )
    @patch("tools.bonsai_local_pool.WORKER_COUNT", 1)
    @patch("tools.bonsai_local_pool.GPU_IDS", ("3",))
    def test_configured_workers_rejects_non_p40_gpu(self, _inventory):
        with self.assertRaisesRegex(RuntimeError, "not a Tesla P40"):
            configured_workers()


if __name__ == "__main__":
    unittest.main()
