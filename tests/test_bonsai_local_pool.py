import unittest
from unittest.mock import patch

from tools.ops.qwen38_ray_pool import (
    GPU,
    MODEL_ALIAS,
    P40_CACHE_TYPE,
    TIER_P40,
    TIER_V100_16_P40,
    TIER_V100_32,
    V100_16_P40_CACHE_TYPE,
    V100_32_CACHE_TYPE,
    LocalCoordinator,
    PoolServer,
    configured_workers,
    plan_workers,
    worker_command,
)


class Qwen38RayPoolTests(unittest.TestCase):
    def setUp(self):
        self.gpus = [
            GPU("0", "GPU-v100-16", "Tesla V100-SXM2-16GB", 16384, 16140),
            GPU("1", "GPU-p40-1", "Tesla P40", 24576, 24434),
            GPU("2", "GPU-p40-2", "Tesla P40", 24576, 24433),
            GPU("4", "GPU-v100-32", "Tesla V100-PCIE-32GB", 32768, 32489),
        ]
        self.workers = plan_workers(self.gpus)

    def test_planner_prioritizes_v10032_then_pair_then_p40(self):
        self.assertEqual(
            [worker.tier for worker in self.workers],
            [TIER_V100_32, TIER_V100_16_P40, TIER_P40],
        )
        self.assertEqual(self.workers[0].gpu_ids, ("4",))
        self.assertEqual(self.workers[1].gpu_ids, ("0", "1"))
        self.assertEqual(self.workers[2].gpu_ids, ("2",))

    def test_v10032_worker_uses_f16_and_full_single_gpu_offload(self):
        worker = self.workers[0]
        command = worker_command(worker)

        self.assertEqual(worker.cache_type_k, V100_32_CACHE_TYPE)
        self.assertEqual(command[command.index("--device") + 1], "CUDA0")
        self.assertEqual(command[command.index("--split-mode") + 1], "none")
        self.assertEqual(command[command.index("--ctx-size") + 1], "65536")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "f16")
        self.assertEqual(command[command.index("--spec-type") + 1], "draft-dflash")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "5")

    def test_v10016_p40_worker_uses_q8_layer_split(self):
        worker = self.workers[1]
        command = worker_command(worker)

        self.assertEqual(worker.cache_type_k, V100_16_P40_CACHE_TYPE)
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(command[command.index("--split-mode") + 1], "layer")
        self.assertEqual(command[command.index("--tensor-split") + 1], "2,3")
        self.assertNotIn("--device", command)

    def test_p40_worker_uses_q8_and_single_gpu_offload(self):
        worker = self.workers[2]
        command = worker_command(worker)

        self.assertEqual(worker.cache_type_k, P40_CACHE_TYPE)
        self.assertEqual(command[command.index("--cache-type-v") + 1], "q8_0")
        self.assertEqual(command[command.index("--split-mode") + 1], "none")

    def test_coordinator_leases_workers_by_tier_priority(self):
        coordinator = LocalCoordinator(self.workers)

        leased = [coordinator.acquire(0) for _ in self.workers]

        self.assertEqual(
            [worker.tier for worker in leased if worker is not None],
            [TIER_V100_32, TIER_V100_16_P40, TIER_P40],
        )
        self.assertIsNone(coordinator.acquire(0))

    def test_health_reports_ray_topology_contract(self):
        response = PoolServer(self.workers).app.test_client().get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["model"], MODEL_ALIAS)
        self.assertEqual(payload["worker_count"], 3)
        self.assertEqual(payload["ready_workers"], 3)

    def test_request_can_fail_immediately_when_pool_is_busy(self):
        server = PoolServer(self.workers)
        reserved = [
            server.coordinator.acquire(0)
            for _ in self.workers
        ]
        try:
            response = server.app.test_client().post(
                "/v1/chat/completions",
                headers={"X-Qwen38-Acquire-Timeout": "0"},
                json={"messages": [{"role": "user", "content": "test"}]},
            )
        finally:
            for worker in reserved:
                if worker is not None:
                    server.coordinator.release(worker.worker_id)

        self.assertEqual(response.status_code, 503)
        self.assertIn("busy", response.get_json()["error"]["message"])

    @patch("tools.ops.qwen38_ray_pool.gpu_compute_processes", return_value={})
    @patch("tools.ops.qwen38_ray_pool.gpu_inventory")
    def test_configured_workers_accepts_supported_idle_topology(
        self,
        inventory,
        _processes,
    ):
        inventory.return_value = self.gpus

        workers = configured_workers()

        self.assertEqual(len(workers), 3)
        self.assertEqual(workers[0].tier, TIER_V100_32)


if __name__ == "__main__":
    unittest.main()
