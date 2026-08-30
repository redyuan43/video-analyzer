import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError

from tools.ops.reclaim_idle_gpu_services import _slots_idle, reclaim, restore


class GPUServiceReclaimTests(unittest.TestCase):
    @patch("tools.ops.reclaim_idle_gpu_services.time.sleep")
    @patch(
        "tools.ops.reclaim_idle_gpu_services._read_json",
        side_effect=[
            URLError("cold start"),
            [{"id": 0, "is_processing": False}],
            [{"id": 0, "is_processing": False}],
        ],
    )
    def test_slots_idle_waits_through_cold_start(self, read_json, sleep):
        self.assertTrue(_slots_idle("http://127.0.0.1:1234/slots", 0, 5))
        self.assertEqual(read_json.call_count, 3)
        sleep.assert_called()

    @patch("tools.ops.reclaim_idle_gpu_services.subprocess.run")
    @patch("tools.ops.reclaim_idle_gpu_services._systemd_active", return_value=True)
    @patch("tools.ops.reclaim_idle_gpu_services._slots_idle", return_value=True)
    def test_idle_systemd_service_is_reclaimed_and_restored(
        self,
        slots_idle,
        systemd_active,
        run,
    ):
        with TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            services = [
                {
                    "name": "model",
                    "kind": "systemd_user",
                    "unit": "model.service",
                    "slots_url": "http://127.0.0.1:1234/slots",
                }
            ]

            self.assertEqual(reclaim(state, services, 0), 0)
            self.assertTrue(state.is_file())
            self.assertEqual(restore(state), 0)
            self.assertFalse(state.exists())

        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(["systemctl", "--user", "stop", "model.service"], commands)
        self.assertIn(["systemctl", "--user", "start", "model.service"], commands)

    @patch("tools.ops.reclaim_idle_gpu_services._systemd_active", return_value=True)
    @patch("tools.ops.reclaim_idle_gpu_services._slots_idle", return_value=False)
    def test_busy_service_is_not_reclaimed(self, slots_idle, systemd_active):
        with TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            services = [
                {
                    "name": "busy",
                    "kind": "systemd_user",
                    "unit": "busy.service",
                    "slots_url": "http://127.0.0.1:1234/slots",
                }
            ]

            self.assertEqual(reclaim(state, services, 0), 0)
            self.assertFalse(state.exists())

    @patch(
        "tools.ops.reclaim_idle_gpu_services._listener_pid",
        return_value=999,
    )
    def test_existing_process_listener_is_not_duplicated_on_restore(self, listener_pid):
        with TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            state.write_text(
                json.dumps(
                    {
                        "services": [
                            {
                                "kind": "process",
                                "name": "model",
                                "port": 18500,
                                "argv": ["/bin/false"],
                                "cwd": tmp,
                                "environment": {},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(restore(state), 0)
            self.assertFalse(state.exists())


if __name__ == "__main__":
    unittest.main()
