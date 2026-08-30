import unittest

from tools.ops.discover_idle_gpus import (
    GPU,
    parse_compute_processes,
    parse_gpus,
    select_gpus,
)


class GPUResourceDiscoveryTests(unittest.TestCase):
    def test_parser_and_selection_include_idle_p40_and_v100(self):
        gpus = parse_gpus(
            "\n".join(
                [
                    "0, GPU-a, Tesla V100-SXM2-16GB, 16384, 16000",
                    "1, GPU-b, Tesla P40, 24576, 24000",
                    "2, GPU-c, Tesla P40, 24576, 12000",
                ]
            )
        )
        processes = parse_compute_processes(
            "GPU-c, 42, /opt/llama-server, 12000\n"
        )

        payload = select_gpus(
            gpus,
            processes,
            allowed_names=["Tesla P40", "Tesla V100"],
            min_total_mib=12000,
            min_free_mib=12000,
            max_count=None,
        )

        self.assertEqual(payload["selected_gpu_ids"], [0, 1])
        self.assertEqual(payload["worker_count"], 2)
        self.assertEqual(payload["skipped"][0]["reasons"], ["compute_process_present"])

    def test_selection_rejects_incompatible_or_small_gpu(self):
        payload = select_gpus(
            [
                GPU(0, "a", "RTX 4090", 24576, 24576),
                GPU(1, "b", "Tesla V100", 16384, 8000),
            ],
            {},
            allowed_names=["Tesla P40", "Tesla V100"],
            min_total_mib=12000,
            min_free_mib=12000,
            max_count=None,
        )

        self.assertEqual(payload["selected_gpu_ids"], [])
        self.assertEqual(payload["skipped"][0]["reasons"], ["incompatible_model"])
        self.assertEqual(payload["skipped"][1]["reasons"], ["insufficient_free_memory"])

    def test_max_count_caps_auto_discovery(self):
        payload = select_gpus(
            [
                GPU(0, "a", "Tesla P40", 24576, 24576),
                GPU(1, "b", "Tesla V100", 32768, 32768),
            ],
            {},
            allowed_names=["Tesla P40", "Tesla V100"],
            min_total_mib=12000,
            min_free_mib=12000,
            max_count=1,
        )

        self.assertEqual(payload["selected_gpu_ids"], [0])
        self.assertEqual(payload["skipped"][0]["reasons"], ["worker_limit"])

    def test_model_memory_floor_can_exclude_v100_16gb(self):
        payload = select_gpus(
            [
                GPU(0, "a", "Tesla V100-SXM2-16GB", 16384, 16300),
                GPU(1, "b", "Tesla P40", 24576, 24400),
                GPU(2, "c", "Tesla V100-PCIE-32GB", 32768, 32600),
            ],
            {},
            allowed_names=["Tesla P40", "Tesla V100"],
            min_total_mib=20000,
            min_free_mib=12000,
            max_count=None,
        )

        self.assertEqual(payload["selected_gpu_ids"], [1, 2])
        self.assertEqual(
            payload["skipped"][0]["reasons"],
            ["insufficient_total_memory"],
        )


if __name__ == "__main__":
    unittest.main()
