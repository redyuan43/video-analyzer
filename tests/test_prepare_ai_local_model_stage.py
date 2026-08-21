import unittest
from pathlib import Path


class PrepareAiLocalModelStageTests(unittest.TestCase):
    def test_bonsai_lifecycle_is_managed_by_systemd(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "tools"
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
            / "prepare_ai_local_model_stage.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "systemctl --user is-active --quiet bonsai-local-pool.service",
            script,
        )
        self.assertIn("fuser -n tcp 18103", script)


if __name__ == "__main__":
    unittest.main()
