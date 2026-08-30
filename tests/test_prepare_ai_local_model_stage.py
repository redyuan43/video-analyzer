import unittest
from pathlib import Path


class PrepareAiLocalModelStageTests(unittest.TestCase):
    def test_bonsai_lifecycle_is_managed_by_systemd(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "ops"
            / "prepare_ai_local_model_stage.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "systemctl --user stop bonsai-local-pool.service",
            script,
        )
        self.assertIn(
            "systemctl --user start bonsai-local-pool.service",
            script,
        )
        self.assertIn(
            "systemctl --user restart bonsai-local-pool.service",
            script,
        )
        self.assertIn("write_bonsai_runtime_config", script)
        self.assertIn("curl --noproxy", script)
        self.assertNotIn('bonsai_local_pool.py" stop', script)
        self.assertNotIn('bonsai_local_pool.py" start', script)

    def test_bonsai_stop_waits_for_service_and_listener_to_exit(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "ops"
            / "prepare_ai_local_model_stage.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "systemctl --user is-active --quiet bonsai-local-pool.service",
            script,
        )
        self.assertIn("fuser -n tcp 18103", script)

    def test_ocr_and_vl_use_runtime_gpu_discovery(self):
        root = Path(__file__).resolve().parents[1]
        stage_script = (root / "tools" / "ops" / "prepare_ai_local_model_stage.sh").read_text(
            encoding="utf-8"
        )
        vision_script = (root / "tools" / "ops" / "start_minicpm_p40_service.sh").read_text(
            encoding="utf-8"
        )
        vision_proxy = (root / "tools" / "ocr_servers" / "minicpm_p40_proxy.py").read_text(
            encoding="utf-8"
        )
        qwen3_asr_script = (
            root / "tools" / "ops" / "start_qwen3_asr_p40_service.sh"
        ).read_text(encoding="utf-8")
        firered_script = (
            root / "tools" / "ops" / "start_firered_asr2_p40_service.sh"
        ).read_text(encoding="utf-8")
        qwen3_asr_proxy = (
            root / "tools" / "asr_servers" / "qwen3_asr_p40_proxy.py"
        ).read_text(encoding="utf-8")

        self.assertIn('UNLIMITED_OCR_WORKER_COUNT:-auto', stage_script)
        self.assertNotIn('UNLIMITED_OCR_GPU_IDS:-0,1,2,4,5', stage_script)
        self.assertIn("reclaim_idle_gpu_services", stage_script)
        self.assertIn("restore_reclaimed_gpu_services", stage_script)
        self.assertIn('VIBEVOICE_WORKER_COUNT:-auto', stage_script)
        self.assertIn("1800[0-5]", stage_script)
        self.assertIn('"${ROOT_DIR}/config/config.json"', stage_script)
        tts_case = stage_script.split("  tts)", 1)[1].split("    ;;", 1)[0]
        self.assertIn("reclaim_idle_gpu_services", tts_case)
        self.assertIn("discover_idle_gpus.py", vision_script)
        self.assertIn('MINICPM_GPU_SELECTION:-auto', vision_script)
        self.assertNotIn("GPU 3 is reserved", vision_script)
        self.assertNotIn("GPU 3 is reserved", vision_proxy)
        self.assertNotIn("GPU 3 is reserved", qwen3_asr_proxy)
        self.assertIn("discover_idle_gpus.py", qwen3_asr_script)
        self.assertIn('QWEN3_ASR_GPU_IDS:-auto', qwen3_asr_script)
        self.assertIn("discover_idle_gpus.py", firered_script)
        self.assertIn('FIRERED_ASR2_GPU_IDS:-auto', firered_script)
        self.assertNotIn("resolve_bonsai_gpu_selection", stage_script)
        self.assertNotIn("reuse_running_bonsai_gpu_selection", stage_script)
        self.assertIn('"BONSAI_LOCAL_GPU_SELECTION"', stage_script)
        self.assertIn('"BONSAI_LOCAL_MODEL"', stage_script)
        self.assertIn('"BONSAI_LOCAL_DRAFT_MODEL"', stage_script)
        self.assertIn('"BONSAI_LOCAL_LLAMA_SERVER"', stage_script)
        self.assertIn('"BONSAI_LOCAL_V100_16_P40_TENSOR_SPLIT"', stage_script)


if __name__ == "__main__":
    unittest.main()
