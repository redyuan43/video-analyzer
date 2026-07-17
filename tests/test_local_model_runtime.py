import unittest
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_analyzer.local_model_runtime import (
    has_loopback_endpoint,
    is_loopback_endpoint,
    local_model_runtime_session,
    local_model_stage,
    local_model_stage_needed,
    prepare_local_model_stage,
)


class LocalModelRuntimeTests(unittest.TestCase):
    def test_loopback_endpoint_detection(self):
        self.assertTrue(is_loopback_endpoint("http://127.0.0.1:18088/v1"))
        self.assertTrue(is_loopback_endpoint("http://localhost:18082/v1"))
        self.assertFalse(is_loopback_endpoint("http://100.90.114.26:18081/v1"))
        self.assertTrue(has_loopback_endpoint(["http://spark:8000/v1", "http://127.0.0.1:18088/v1"]))

    def test_stage_detection_uses_stage_specific_endpoints(self):
        config = {
            "asr": {"vibevoice": {"deep_remote_urls": ["http://127.0.0.1:18012/api/asr/transcribe"]}},
            "ocr": {"base_urls": ["http://127.0.0.1:18088/v1"]},
            "operation_manual": {
                "vision_base_url": "http://127.0.0.1:18082/v1",
                "text_base_url": "http://127.0.0.1:18081/v1",
            },
        }

        self.assertTrue(local_model_stage_needed("asr", config))
        self.assertTrue(local_model_stage_needed("ocr", config))
        self.assertTrue(local_model_stage_needed("vl", config))
        self.assertTrue(local_model_stage_needed("text", config))

    def test_remote_endpoints_do_not_run_local_stage_switch(self):
        config = {
            "asr": {"vibevoice": {"deep_remote_urls": ["http://edge.taild500c8.ts.net:8012/api/asr/transcribe"]}},
            "ocr": {"base_urls": ["http://spark-31d6.taild500c8.ts.net:8000/v1"]},
            "operation_manual": {"vision_base_url": "http://100.96.79.21:18082/v1"},
        }

        self.assertFalse(local_model_stage_needed("asr", config))
        self.assertFalse(local_model_stage_needed("ocr", config))
        self.assertFalse(local_model_stage_needed("vl", config))
        self.assertFalse(local_model_stage_needed("text", config))

    @patch("video_analyzer.local_model_runtime.subprocess.run")
    def test_local_model_stage_unloads_on_exit_when_enabled(self, run):
        with TemporaryDirectory() as tmp:
            config = {
                "operation_manual": {"text_base_url": "http://127.0.0.1:18081/v1"},
                "local_model_runtime": {
                    "lock_path": str(Path(tmp) / "local.lock"),
                    "stage_commands": {
                        "text": ["/bin/echo", "text"],
                        "stop": ["/bin/echo", "stop"],
                    },
                    "unload_on_stage_exit": True,
                },
            }

            with local_model_stage("text", config, __import__("logging").getLogger(__name__), "text-job"):
                pass

            self.assertEqual(
                [call.args[0] for call in run.call_args_list],
                [["/bin/echo", "text"], ["/bin/echo", "stop"]],
            )

    @patch("video_analyzer.local_model_runtime.subprocess.run")
    def test_prepare_stage_runs_configured_command(self, run):
        config = {
            "ocr": {"base_url": "http://127.0.0.1:18088/v1"},
            "local_model_runtime": {
                "stage_commands": {"ocr": ["/bin/echo", "ocr"]},
                "stage_timeout_seconds": 7,
            },
        }

        prepare_local_model_stage("ocr", config, logger=__import__("logging").getLogger(__name__))

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["/bin/echo", "ocr"])
        self.assertEqual(run.call_args.kwargs["timeout"], 7)

    @patch("video_analyzer.local_model_runtime.subprocess.run")
    def test_local_model_stage_serializes_loopback_stages(self, run):
        with TemporaryDirectory() as tmp:
            config = {
                "ocr": {"base_url": "http://127.0.0.1:18088/v1"},
                "operation_manual": {"vision_base_url": "http://127.0.0.1:18082/v1"},
                "local_model_runtime": {
                    "lock_path": str(Path(tmp) / "local.lock"),
                    "stage_commands": {
                        "ocr": ["/bin/echo", "ocr"],
                        "vl": ["/bin/echo", "vl"],
                    },
                    "poll_seconds": 0.01,
                    "log_interval_seconds": 0.01,
                },
            }
            events: list[str] = []
            logger = __import__("logging").getLogger(__name__)

            def hold_ocr():
                with local_model_stage("ocr", config, logger, "first"):
                    events.append("first-acquired")
                    time.sleep(0.08)
                    events.append("first-releasing")

            thread = threading.Thread(target=hold_ocr)
            thread.start()
            while "first-acquired" not in events:
                time.sleep(0.005)

            with local_model_stage("vl", config, logger, "second"):
                events.append("second-acquired")

            thread.join(timeout=1)
            self.assertEqual(events, ["first-acquired", "first-releasing", "second-acquired"])

    @patch("video_analyzer.local_model_runtime.subprocess.run")
    def test_runtime_session_allows_internal_stage_switch_but_blocks_second_task(self, run):
        with TemporaryDirectory() as tmp:
            config = {
                "ocr": {"base_url": "http://127.0.0.1:18088/v1"},
                "operation_manual": {"vision_base_url": "http://127.0.0.1:18082/v1"},
                "local_model_runtime": {
                    "lock_path": str(Path(tmp) / "local.lock"),
                    "stage_commands": {
                        "ocr": ["/bin/echo", "ocr"],
                        "vl": ["/bin/echo", "vl"],
                    },
                    "poll_seconds": 0.01,
                    "log_interval_seconds": 0.01,
                },
            }
            events: list[str] = []
            logger = __import__("logging").getLogger(__name__)

            def first_core():
                with local_model_runtime_session(config, logger, "first-core"):
                    events.append("first-core-acquired")
                    with local_model_stage("ocr", config, logger, "first-core"):
                        events.append("first-ocr")
                    with local_model_stage("vl", config, logger, "first-core"):
                        events.append("first-vl")
                    time.sleep(0.08)
                    events.append("first-core-releasing")

            thread = threading.Thread(target=first_core)
            thread.start()
            while "first-vl" not in events:
                time.sleep(0.005)

            with local_model_runtime_session(config, logger, "second-core"):
                events.append("second-core-acquired")

            thread.join(timeout=1)
            self.assertEqual(
                events,
                [
                    "first-core-acquired",
                    "first-ocr",
                    "first-vl",
                    "first-core-releasing",
                    "second-core-acquired",
                ],
            )

    @patch("video_analyzer.local_model_runtime.subprocess.run")
    def test_runtime_session_depth_is_thread_local(self, run):
        with TemporaryDirectory() as tmp:
            config = {
                "ocr": {"base_url": "http://127.0.0.1:18088/v1"},
                "operation_manual": {"vision_base_url": "http://127.0.0.1:18082/v1"},
                "local_model_runtime": {
                    "lock_path": str(Path(tmp) / "local.lock"),
                    "stage_commands": {
                        "ocr": ["/bin/echo", "ocr"],
                        "vl": ["/bin/echo", "vl"],
                    },
                    "poll_seconds": 0.01,
                    "log_interval_seconds": 0.01,
                },
            }
            events: list[str] = []
            logger = __import__("logging").getLogger(__name__)

            def first_core():
                with local_model_runtime_session(config, logger, "first-core"):
                    events.append("first-core-acquired")
                    with local_model_stage("ocr", config, logger, "first-core"):
                        events.append("first-ocr")
                    time.sleep(0.08)
                    events.append("first-core-releasing")

            thread = threading.Thread(target=first_core)
            thread.start()
            while "first-ocr" not in events:
                time.sleep(0.005)

            with local_model_stage("vl", config, logger, "second-standalone"):
                events.append("second-vl-acquired")

            thread.join(timeout=1)
            self.assertEqual(
                events,
                [
                    "first-core-acquired",
                    "first-ocr",
                    "first-core-releasing",
                    "second-vl-acquired",
                ],
            )
