import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "minicpm_p40_proxy.py"
SPEC = importlib.util.spec_from_file_location("minicpm_p40_proxy", MODULE_PATH)
minicpm_p40_proxy = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["minicpm_p40_proxy"] = minicpm_p40_proxy
SPEC.loader.exec_module(minicpm_p40_proxy)


class MiniCpmWorkerPoolTests(TestCase):
    def make_pool(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        workers = [
            minicpm_p40_proxy.WorkerSpec(gpu=0, port=18182, log_path=Path(temp_dir.name) / "w0.log"),
            minicpm_p40_proxy.WorkerSpec(gpu=1, port=18183, log_path=Path(temp_dir.name) / "w1.log"),
            minicpm_p40_proxy.WorkerSpec(gpu=2, port=18184, log_path=Path(temp_dir.name) / "w2.log"),
        ]
        pool = minicpm_p40_proxy.LlamaWorkerPool(
            server_bin=Path("/bin/true"),
            model_path=Path("/tmp/model.gguf"),
            mmproj_path=Path("/tmp/mmproj.gguf"),
            model_alias="minicpm-test",
            host="127.0.0.1",
            workers=workers,
            ctx_size=8192,
            parallel=1,
            startup_timeout=1,
            stop_timeout=1,
            backend_timeout=1,
            extra_args=[],
        )
        pool.ensure_ready = lambda: None
        pool._is_http_ready = lambda _worker: True
        return pool, workers

    def test_choose_worker_prefers_least_inflight(self):
        pool, workers = self.make_pool()

        first = pool.choose_worker()
        second = pool.choose_worker()
        third = pool.choose_worker()
        self.assertEqual([first.port, second.port, third.port], [18182, 18183, 18184])

        pool.release_worker(second, failed=False, latency_sec=0.2)
        fourth = pool.choose_worker()

        self.assertEqual(fourth.port, second.port)

    def test_health_exposes_worker_runtime_stats(self):
        pool, _workers = self.make_pool()

        worker = pool.choose_worker()
        health = pool.health()

        worker_payload = next(item for item in health["workers"] if item["port"] == worker.port)
        self.assertEqual(worker_payload["stats"]["inflight"], 1)
        self.assertEqual(worker_payload["stats"]["assigned"], 1)

        pool.release_worker(worker, failed=True, latency_sec=1.25)
        health = pool.health()
        worker_payload = next(item for item in health["workers"] if item["port"] == worker.port)
        self.assertEqual(worker_payload["stats"]["inflight"], 0)
        self.assertEqual(worker_payload["stats"]["failed"], 1)
        self.assertEqual(worker_payload["stats"]["last_latency_sec"], 1.25)
