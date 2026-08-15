import unittest
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_analyzer.local_model_runtime import (
    has_loopback_endpoint,
    is_loopback_endpoint,
    local_model_runtime_lock,
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
            "tts": {"enabled": True, "base_url": "http://127.0.0.1:8092"},
        }

        self.assertTrue(local_model_stage_needed("asr", config))
        self.assertTrue(local_model_stage_needed("ocr", config))
        self.assertTrue(local_model_stage_needed("vl", config))
        self.assertTrue(local_model_stage_needed("text", config))
        self.assertTrue(local_model_stage_needed("tts", config))

    def test_remote_endpoints_do_not_run_local_stage_switch(self):
        config = {
            "asr": {"vibevoice": {"deep_remote_urls": ["http://edge.taild500c8.ts.net:8012/api/asr/transcribe"]}},
            "ocr": {"base_urls": ["http://spark-31d6.taild500c8.ts.net:8000/v1"]},
            "operation_manual": {"vision_base_url": "http://100.96.79.21:18082/v1"},
            "tts": {"enabled": True, "base_url": "http://ivan.tailnet:8092"},
        }

        self.assertFalse(local_model_stage_needed("asr", config))
        self.assertFalse(local_model_stage_needed("ocr", config))
        self.assertFalse(local_model_stage_needed("vl", config))
        self.assertFalse(local_model_stage_needed("text", config))
        self.assertFalse(local_model_stage_needed("tts", config))

    def test_remote_profile_loopback_text_endpoint_does_not_start_bonsai(self):
        config = {
            "active_runtime_profile": "trae-api",
            "runtime_profiles": {
                "trae-api": {
                    "deployment": "remote",
                    "provider": "trae_local_api",
                    "text_base_url": "http://127.0.0.1:19220/v1",
                }
            },
            "operation_manual": {
                "text_base_url": "http://127.0.0.1:19220/v1",
            },
        }

        self.assertFalse(local_model_stage_needed("text", config))

    def test_remote_profile_does_not_mask_explicit_local_text_override(self):
        config = {
            "active_runtime_profile": "trae-api",
            "runtime_profiles": {
                "trae-api": {
                    "deployment": "remote",
                    "provider": "trae_local_api",
                    "text_base_url": "http://127.0.0.1:19220/v1",
                }
            },
            "operation_manual": {
                "text_base_url": "http://127.0.0.1:18103/v1",
            },
        }

        self.assertTrue(local_model_stage_needed("text", config))

    def test_disabled_tts_does_not_run_local_stage_switch(self):
        config = {"tts": {"enabled": False, "base_url": "http://127.0.0.1:8092"}}

        self.assertFalse(local_model_stage_needed("tts", config))

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
    def test_text_stage_forwards_local_bonsai_topology(self, run):
        config = {
            "operation_manual": {
                "text_base_url": "http://127.0.0.1:18103/v1",
                "text_port": 18103,
                "text_worker_count": 5,
                "text_gpu_ids": [0, 1, 2, 4, 5],
            },
            "local_model_runtime": {
                "stage_commands": {"text": ["/bin/echo", "text"]},
            },
        }

        prepare_local_model_stage(
            "text",
            config,
            logger=__import__("logging").getLogger(__name__),
        )

        env = run.call_args.kwargs["env"]
        self.assertEqual(env["BONSAI_LOCAL_PORT"], "18103")
        self.assertEqual(env["BONSAI_LOCAL_WORKER_COUNT"], "5")
        self.assertEqual(env["BONSAI_LOCAL_GPU_IDS"], "0,1,2,4,5")

    @patch("video_analyzer.local_model_runtime.subprocess.run")
    def test_qwen3_asr_model_id_does_not_override_local_model_path(self, run):
        config = {
            "asr": {
                "provider": "qwen3_asr",
                "vibevoice": {
                    "qwen3_asr_url": "http://127.0.0.1:18013/api/asr/transcribe",
                    "qwen3_asr_model": "Qwen/Qwen3-ASR-1.7B",
                    "qwen3_asr_options": {
                        "worker_count": 5,
                        "single_pass_max_duration_sec": 240,
                        "chunk_duration_sec": 180,
                        "chunk_overlap_sec": 12,
                    },
                },
            },
            "local_model_runtime": {
                "stage_commands": {"asr": ["/bin/echo", "asr"]},
            },
        }

        prepare_local_model_stage("asr", config, logger=__import__("logging").getLogger(__name__))

        env = run.call_args.kwargs["env"]
        self.assertNotIn("QWEN3_ASR_MODEL", env)
        self.assertEqual(env["QWEN3_ASR_WORKER_COUNT"], "5")
        self.assertEqual(env["QWEN3_ASR_SINGLE_PASS_SECONDS"], "240")
        self.assertEqual(env["QWEN3_ASR_CHUNK_SECONDS"], "180")
        self.assertEqual(env["QWEN3_ASR_CHUNK_OVERLAP_SECONDS"], "12")

    @patch("video_analyzer.local_model_runtime.subprocess.run")
    def test_vibevoice_asr_worker_count_takes_precedence_over_chunk_parallelism(self, run):
        config = {
            "asr": {
                "provider": "vibevoice",
                "vibevoice": {
                    "deep_remote_urls": ["http://127.0.0.1:18012/api/asr/transcribe"],
                    "worker_count": 5,
                    "chunk_parallel_workers": 2,
                    "single_pass_max_duration_sec": 240,
                    "chunk_duration_sec": 180,
                    "chunk_overlap_sec": 12,
                },
            },
            "local_model_runtime": {
                "stage_commands": {"asr": ["/bin/echo", "asr"]},
            },
        }

        prepare_local_model_stage("asr", config, logger=__import__("logging").getLogger(__name__))

        env = run.call_args.kwargs["env"]
        self.assertEqual(env["VIBEVOICE_WORKER_COUNT"], "5")
        self.assertEqual(env["VIBEVOICE_SINGLE_PASS_MAX_DURATION_SEC"], "240")
        self.assertEqual(env["VIBEVOICE_CHUNK_DURATION_SEC"], "180")
        self.assertEqual(env["VIBEVOICE_CHUNK_OVERLAP_SEC"], "12")

    @patch("video_analyzer.local_model_runtime.subprocess.run")
    def test_firered_asr_chunk_settings_are_forwarded(self, run):
        config = {
            "asr": {
                "provider": "firered_asr2",
                "vibevoice": {
                    "firered_asr2_url": "http://127.0.0.1:18014/api/asr/transcribe",
                    "firered_asr2_options": {
                        "worker_count": 5,
                        "single_pass_max_duration_sec": 50,
                        "chunk_duration_sec": 30,
                        "chunk_overlap_sec": 3,
                        "segmentation_mode": "vad",
                        "vad_max_segment_sec": 50,
                    },
                },
            },
            "local_model_runtime": {
                "stage_commands": {"asr": ["/bin/echo", "asr"]},
            },
        }

        prepare_local_model_stage("asr", config, logger=__import__("logging").getLogger(__name__))

        env = run.call_args.kwargs["env"]
        self.assertEqual(env["FIRERED_ASR2_WORKER_COUNT"], "5")
        self.assertEqual(env["FIRERED_ASR2_SINGLE_PASS_SECONDS"], "50")
        self.assertEqual(env["FIRERED_ASR2_CHUNK_SECONDS"], "30")
        self.assertEqual(env["FIRERED_ASR2_CHUNK_OVERLAP_SECONDS"], "3")
        self.assertEqual(env["FIRERED_ASR2_SEGMENTATION_MODE"], "vad")
        self.assertEqual(env["FIRERED_VAD_MAX_SEGMENT_SECONDS"], "50")

    @patch("video_analyzer.local_model_runtime.subprocess.run")
    def test_qwen3_asr_explicit_local_model_path_is_forwarded(self, run):
        with TemporaryDirectory() as tmp:
            config = {
                "asr": {
                    "provider": "qwen3_asr",
                    "vibevoice": {
                        "qwen3_asr_url": "http://127.0.0.1:18013/api/asr/transcribe",
                        "qwen3_asr_model": "Qwen/Qwen3-ASR-1.7B",
                        "qwen3_asr_options": {
                            "worker_count": 5,
                            "model_path": tmp,
                        },
                    },
                },
                "local_model_runtime": {
                    "stage_commands": {"asr": ["/bin/echo", "asr"]},
                },
            }

            prepare_local_model_stage("asr", config, logger=__import__("logging").getLogger(__name__))

        self.assertEqual(run.call_args.kwargs["env"]["QWEN3_ASR_MODEL"], tmp)

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

    def test_runtime_session_allows_nested_diarization_lock(self):
        with TemporaryDirectory() as tmp:
            config = {
                "operation_manual": {
                    "vision_base_url": "http://127.0.0.1:18082/v1",
                },
                "local_model_runtime": {
                    "lock_path": str(Path(tmp) / "local.lock"),
                    "poll_seconds": 0.01,
                    "log_interval_seconds": 0.01,
                },
            }
            events: list[str] = []
            logger = __import__("logging").getLogger(__name__)

            def run_nested_lock():
                with local_model_runtime_session(config, logger, "core-job"):
                    events.append("core")
                    with local_model_runtime_lock(
                        config,
                        logger,
                        "speaker-diarization",
                        stage="diarization",
                    ):
                        events.append("diarization")

            thread = threading.Thread(target=run_nested_lock, daemon=True)
            thread.start()
            thread.join(timeout=1)

            self.assertFalse(thread.is_alive())
            self.assertEqual(events, ["core", "diarization"])

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
