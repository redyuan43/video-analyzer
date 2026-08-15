import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from video_analyzer.config import Config, get_runtime_profile
from video_analyzer.model_settings import RuntimeSettingsStore, SettingsValidationError


class ModelSettingsTests(unittest.TestCase):
    def make_repo(self, root: Path) -> RuntimeSettingsStore:
        default_path = root / "video_analyzer" / "config" / "default_config.json"
        default_path.parent.mkdir(parents=True)
        (root / "config").mkdir()
        default_path.write_text(
            json.dumps(
                {
                    "active_runtime_profile": "default",
                    "runtime_profiles": {
                        "default": {
                            "asr_provider": "vibevoice",
                            "vibevoice_urls": ["http://127.0.0.1:18012/api/asr/transcribe"],
                            "ocr_provider": "dots_mocr_vllm",
                            "ocr_base_urls": ["http://127.0.0.1:18088/v1"],
                            "ocr_model": "dots-mocr",
                            "vision_base_url": "http://127.0.0.1:18082/v1",
                            "vision_model": "minicpm",
                            "text_base_url": "https://api.example.com/v1",
                            "text_model": "text-model",
                            "text_api_key_env": "TEXT_API_KEY",
                        }
                    },
                    "speaker_diarization": {"enabled": True, "enable_3dspeaker": True},
                }
            ),
            encoding="utf-8",
        )
        return RuntimeSettingsStore(root)

    def test_legacy_profile_is_exposed_as_reusable_model_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_repo(Path(tmp))
            settings = store.public_settings()

        self.assertEqual(settings["active_runtime_profile"], "default")
        self.assertEqual(len(settings["profiles"]), 1)
        profile = settings["profiles"][0]
        self.assertTrue(profile["asr_model_id"])
        self.assertTrue(profile["diarization_model_id"])
        self.assertTrue(profile["ocr_model_id"])
        self.assertTrue(profile["vision_model_id"])
        self.assertTrue(profile["text_model_id"])

    def test_model_save_keeps_only_api_key_environment_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            saved = store.save_model(
                "cloud_text",
                {
                    "name": "Cloud Text",
                    "kind": "text",
                    "protocol": "openai_compatible",
                    "model": "cloud-model",
                    "endpoints": ["https://api.example.com/v1"],
                    "api_key_env": "CLOUD_API_KEY",
                    "api_key": "must-not-be-saved",
                    "options": {"temperature": 0.2},
                },
            )
            user_config = json.loads((root / "config" / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(saved["api_key_env"], "CLOUD_API_KEY")
        stored = user_config["model_catalog"]["cloud_text"]
        self.assertEqual(stored["api_key_env"], "CLOUD_API_KEY")
        self.assertNotIn("api_key", stored)
        self.assertNotIn("must-not-be-saved", json.dumps(user_config))

    def test_model_catalog_endpoint_placeholders_are_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            default_path = root / "video_analyzer" / "config" / "default_config.json"
            defaults = json.loads(default_path.read_text(encoding="utf-8"))
            defaults["endpoints"] = {
                "hosts": {"local_gpu": "127.0.0.1"},
                "services": {"local_vl": "http://{local_gpu}:18082/v1"},
            }
            defaults["model_catalog"] = {
                "vision-local": {
                    "name": "Local Vision",
                    "kind": "vision",
                    "protocol": "openai_compatible",
                    "model": "vision-model",
                    "endpoints": ["{local_vl}"],
                }
            }
            default_path.write_text(json.dumps(defaults), encoding="utf-8")

            settings = store.public_settings()

        model = next(item for item in settings["models"] if item["id"] == "vision-local")
        self.assertEqual(model["endpoints"], ["http://127.0.0.1:18082/v1"])

    def test_profile_refs_expand_to_existing_runtime_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            models = {
                kind: source[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }
            store.save_profile(
                "custom",
                {
                    "label": "Custom",
                    "models": models,
                    "settings": {
                        "pipeline_mode": "deep",
                        "vl_concurrency": 4,
                        "multidoc_chapter_concurrency": 10,
                    },
                },
            )
            _defaults, _user, merged = store.load()
            profile = get_runtime_profile(merged, "custom")

        self.assertEqual(profile["pipeline_mode"], "deep")
        self.assertEqual(profile["asr_provider"], "vibevoice")
        self.assertEqual(profile["vision_model"], "minicpm")
        self.assertEqual(profile["text_model"], "text-model")
        self.assertEqual(profile["speaker_diarization"]["enabled"], True)
        self.assertEqual(profile["multidoc_chapter_concurrency"], 10)

    def test_profile_rejects_chapter_concurrency_above_ten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            models = {
                kind: source[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }

            with self.assertRaisesRegex(
                SettingsValidationError,
                "multidoc_chapter_concurrency must be between 1 and 10",
            ):
                store.save_profile(
                    "too-wide",
                    {
                        "label": "Too Wide",
                        "models": models,
                        "settings": {"multidoc_chapter_concurrency": 11},
                    },
                )

    def test_inherited_study_and_triage_override_legacy_qwen_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            default_path = root / "video_analyzer" / "config" / "default_config.json"
            defaults = json.loads(default_path.read_text(encoding="utf-8"))
            defaults["study_cards"] = {
                "llm_base_url": "http://legacy.example/v1",
                "model": "qwen3:4b-instruct",
            }
            defaults["runtime_profiles"]["default"].update(
                {
                    "text_model_id": "text-flash",
                    "study_card_model_id": "study-inherit-text",
                    "triage_model_id": "triage-inherit-study",
                    "study_card_llm_base_url": "http://legacy.example/v1",
                    "study_card_model": "qwen3:4b-instruct",
                    "study_card_api_key_env": "",
                }
            )
            defaults["model_catalog"] = {
                "text-flash": {
                    "name": "DeepSeek Flash",
                    "kind": "text",
                    "protocol": "openai_compatible",
                    "model": "deepseek-v4-flash",
                    "endpoints": ["https://api.deepseek.com"],
                    "api_key_env": "DEEPSEEK_API_KEY",
                }
            }
            default_path.write_text(json.dumps(defaults), encoding="utf-8")

            _defaults, _user, merged = store.load()
            profile = get_runtime_profile(merged, "default")

        self.assertEqual(profile["study_card_inherit"], "text")
        self.assertEqual(profile["study_card_llm_base_url"], "https://api.deepseek.com")
        self.assertEqual(profile["study_card_model"], "deepseek-v4-flash")
        self.assertEqual(profile["study_card_api_key_env"], "DEEPSEEK_API_KEY")
        self.assertEqual(profile["triage_inherit"], "study")
        self.assertEqual(profile["triage_llm_base_url"], "https://api.deepseek.com")
        self.assertEqual(profile["triage_model"], "deepseek-v4-flash")
        self.assertEqual(profile["triage_api_key_env"], "DEEPSEEK_API_KEY")

    def test_settings_expose_fixed_branched_profile_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.make_repo(Path(tmp)).public_settings()

        flow = settings["schema"]["profile_flow"]
        node_ids = {item["id"] for item in flow["nodes"]}
        edges = {(item["from"], item["to"]) for item in flow["edges"]}
        models = {item["id"]: item for item in settings["models"]}
        self.assertEqual(flow["version"], 5)
        self.assertTrue(
            {
                "asr",
                "diarization",
                "transcript_merge",
                "frame_audit",
                "ocr",
                "vision",
                "visual_evidence",
                "evidence_merge",
                "text_fallback",
                "deep_report",
                "deep_review",
                "web_evidence",
                "final_publish",
                "tts",
                "narration_audio",
                "operation_manual_doc",
            }
            <= node_ids
        )
        self.assertIn(("input", "prepare"), edges)
        self.assertIn(("prepare", "audio_extract"), edges)
        self.assertIn(("prepare", "frame_extract"), edges)
        self.assertIn(("audio_extract", "asr"), edges)
        self.assertIn(("audio_extract", "diarization"), edges)
        self.assertNotIn(("asr", "diarization"), edges)
        self.assertIn(("asr", "transcript_merge"), edges)
        self.assertIn(("diarization", "transcript_merge"), edges)
        self.assertIn(("frame_extract", "frame_audit"), edges)
        self.assertIn(("frame_audit", "ocr"), edges)
        self.assertIn(("ocr", "vision"), edges)
        self.assertNotIn(("frame_extract", "vision"), edges)
        self.assertIn(("visual_evidence", "evidence_merge"), edges)
        self.assertIn(("text", "text_fallback"), edges)
        self.assertIn(("text_fallback", "core_verify"), edges)
        self.assertIn(("final_publish", "operation_manual_doc"), edges)
        self.assertIn(("final_publish", "tts"), edges)
        self.assertIn(("tts", "narration_audio"), edges)
        self.assertEqual(models["vision-disabled"]["protocol"], "none")
        self.assertEqual(models["text-disabled"]["protocol"], "none")
        self.assertEqual(
            models["text-deepseek-v4-pro"]["model"],
            "deepseek-v4-pro",
        )
        self.assertEqual(models["review-inherit-text"]["protocol"], "inherit_text")
        self.assertEqual(models["image-disabled"]["protocol"], "none")
        self.assertEqual(models["tts-disabled"]["protocol"], "none")

    def test_settings_expose_local_multiworker_model_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.make_repo(Path(tmp)).public_settings()

        models = {item["id"]: item for item in settings["models"]}
        self.assertEqual(models["asr-qwen3-1_7b-local"]["protocol"], "qwen3_asr_http")
        self.assertEqual(models["asr-qwen3-1_7b-local"]["options"]["worker_count"], 5)
        self.assertEqual(models["asr-firered2-local"]["protocol"], "firered_asr2_http")
        self.assertEqual(
            models["asr-firered2-local"]["endpoints"],
            ["http://127.0.0.1:18014/api/asr/transcribe"],
        )
        self.assertEqual(models["asr-firered2-local"]["options"]["deployment"], "local")
        self.assertEqual(
            models["diarization-pyannote-community1-local"]["protocol"],
            "pyannote_community",
        )
        self.assertEqual(
            models["diarization-wespeaker-cn-local"]["protocol"],
            "wespeaker_diarization",
        )
        self.assertEqual(models["ocr-unlimited-local"]["protocol"], "unlimited_ocr_openai")
        self.assertEqual(models["ocr-unlimited-local"]["options"]["max_tokens"], 8192)
        self.assertEqual(models["ocr-unlimited-local"]["options"]["max_image_long_side"], 0)
        self.assertEqual(models["ocr-unlimited-local"]["options"]["image_mode"], "gundam")
        self.assertEqual(models["ocr-dots-local"]["protocol"], "dots_ocr_openai")
        self.assertEqual(models["ocr-unlimited-local"]["model"], "baidu/Unlimited-OCR")
        self.assertEqual(models["ocr-dots-local"]["model"], "rednote-hilab/dots.ocr")
        self.assertEqual(
            models["vision-qwen3-vl-4b-local"]["options"]["engine"],
            "qwen3_vl_4b",
        )
        amd_text = models["text-amd-lmstudio-bonsai-27b"]
        self.assertEqual(amd_text["protocol"], "openai_compatible")
        self.assertEqual(amd_text["model"], "prism-ml/bonsai-27b")
        self.assertEqual(
            amd_text["endpoints"],
            ["http://100.90.114.26:18081/v1"],
        )
        self.assertEqual(amd_text["options"]["runtime"], "lm_studio")
        self.assertEqual(amd_text["options"]["reasoning_effort"], "none")
        local_text = models["text-local-bonsai-27b-6gpu"]
        self.assertEqual(local_text["options"]["text_gpu_ids"], [0, 1, 2, 4, 5])
        self.assertEqual(local_text["options"]["text_worker_count"], 5)
        self.assertEqual(local_text["options"]["text_concurrency"], 5)
        self.assertIn("五张 P40", local_text["name"])
        local_tts = models["tts-indextts25-ray-p40"]
        self.assertEqual(local_tts["protocol"], "openai_speech")
        self.assertEqual(local_tts["endpoints"], ["http://127.0.0.1:8092"])
        self.assertEqual(local_tts["options"]["voice"], "check_boards_sweet")

    def test_indextts_profile_expands_to_runtime_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            models = {
                kind: source[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }
            models["tts"] = "tts-indextts25-ray-p40"
            store.save_profile(
                "with-tts",
                {"label": "With TTS", "models": models, "settings": {}},
            )
            _defaults, _user, merged = store.load()
            profile = get_runtime_profile(merged, "with-tts")

        self.assertTrue(profile["tts_enabled"])
        self.assertEqual(profile["tts_provider"], "openai_speech")
        self.assertEqual(profile["tts_base_url"], "http://127.0.0.1:8092")
        self.assertEqual(profile["tts_model"], "/home/ai/github/indextts-2.5")
        self.assertEqual(profile["tts_voice"], "check_boards_sweet")

    def test_settings_expose_tencent_hy_asr_cloud_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.make_repo(Path(tmp)).public_settings()

        models = {item["id"]: item for item in settings["models"]}
        tencent = models["asr-tencent-hy3-preview-cloud"]
        self.assertEqual(tencent["protocol"], "tencent_hy_asr_ws")
        self.assertEqual(
            tencent["endpoints"],
            ["wss://asr.cloud.tencent.com/asr/v2"],
        )
        self.assertEqual(
            tencent["options"]["secret_key_env"],
            "TENCENTCLOUD_SECRET_KEY",
        )
        self.assertEqual(tencent["options"]["parallel_chunks"], 6)

    def test_amd_lmstudio_text_model_expands_to_runtime_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            models = {
                kind: source[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }
            models["text"] = "text-amd-lmstudio-bonsai-27b"
            store.save_profile(
                "amd-lmstudio",
                {"label": "AMD LM Studio", "models": models, "settings": {}},
            )
            _defaults, _user, merged = store.load()
            profile = get_runtime_profile(merged, "amd-lmstudio")

        self.assertEqual(profile["text_base_url"], "http://100.90.114.26:18081/v1")
        self.assertEqual(profile["text_model"], "prism-ml/bonsai-27b")
        self.assertEqual(profile["reasoning_effort"], "none")
        self.assertEqual(profile["text_timeout_seconds"], 900)

    def test_video_profile_can_enable_or_disable_text_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            models = {
                slot: source[spec["field"]]
                for slot, spec in settings["schema"]["workflows"][
                    "video_operation_manual"
                ]["model_fields"].items()
                if spec["field"] in source
            }
            models["text_fallback"] = "text-deepseek-v4-pro"
            store.save_profile(
                "with-fallback",
                {"label": "With fallback", "models": models, "settings": {}},
            )
            _defaults, _user, merged = store.load()
            enabled = get_runtime_profile(merged, "with-fallback")

            models["text_fallback"] = "text-disabled"
            store.save_profile(
                "without-fallback",
                {"label": "Without fallback", "models": models, "settings": {}},
            )
            _defaults, _user, merged = store.load()
            disabled = get_runtime_profile(merged, "without-fallback")

        self.assertTrue(enabled["text_fallback_enabled"])
        self.assertEqual(enabled["text_fallback_model"], "deepseek-v4-pro")
        self.assertEqual(
            enabled["text_fallback_api_key_env"],
            "DEEPSEEK_API_KEY",
        )
        self.assertFalse(disabled["text_fallback_enabled"])

    def test_local_firered_profile_expands_to_local_runtime_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            models = {
                kind: source[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }
            models["asr"] = "asr-firered2-local"
            store.save_profile(
                "firered-local",
                {"label": "FireRed Local", "models": models, "settings": {}},
            )
            _defaults, _user, merged = store.load()
            profile = get_runtime_profile(merged, "firered-local")

        self.assertEqual(profile["asr_provider"], "firered_asr2")
        self.assertEqual(
            profile["firered_asr2_url"],
            "http://127.0.0.1:18014/api/asr/transcribe",
        )
        self.assertEqual(profile["firered_asr2_options"]["worker_count"], 5)
        self.assertEqual(profile["firered_asr2_options"]["gpu_ids"], [0, 1, 2, 4, 5])
        self.assertEqual(
            profile["firered_asr2_options"]["segmentation_mode"],
            "vad",
        )
        self.assertEqual(
            profile["firered_asr2_options"]["vad_max_segment_sec"],
            50,
        )

    def test_profile_custom_asr_chunk_settings_override_supported_local_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            base_models = {
                kind: source[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }
            cases = (
                ("vibevoice", "asr-vibevoice-local", None, 180),
                ("qwen3", "asr-qwen3-1_7b-local", "qwen3_asr_options", 180),
                ("firered", "asr-firered2-local", "firered_asr2_options", 50),
            )
            for suffix, model_id, options_field, chunk_seconds in cases:
                models = dict(base_models)
                models["asr"] = model_id
                store.save_profile(
                    f"chunk-{suffix}",
                    {
                        "label": f"Chunk {suffix}",
                        "models": models,
                        "settings": {
                            "asr_chunk_mode": "custom",
                            "single_pass_max_duration_sec": chunk_seconds,
                            "chunk_duration_sec": chunk_seconds,
                            "chunk_overlap_sec": 5,
                        },
                    },
                )
                _defaults, _user, merged = store.load()
                profile = get_runtime_profile(merged, f"chunk-{suffix}")
                effective = profile[options_field] if options_field else profile
                self.assertEqual(
                    effective["single_pass_max_duration_sec"],
                    chunk_seconds,
                )
                self.assertEqual(effective["chunk_duration_sec"], chunk_seconds)
                self.assertEqual(effective["chunk_overlap_sec"], 5)
                if suffix == "vibevoice":
                    self.assertEqual(
                        effective["chunk_parallel_workers"],
                        effective["worker_count"],
                    )

    def test_firered_profile_accepts_vad_ray_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            source = store.public_settings()["profiles"][0]
            models = {
                kind: source[field]
                for kind, field in store.public_settings()["schema"][
                    "profile_model_fields"
                ].items()
            }
            models["asr"] = "asr-firered2-local"
            store.save_profile(
                "firered-vad",
                {
                    "label": "FireRed VAD",
                    "models": models,
                    "settings": {
                        "asr_segmentation_mode": "vad",
                        "asr_worker_count": 5,
                        "vad_max_segment_sec": 50,
                    },
                },
            )
            _defaults, _user, merged = store.load()
            profile = get_runtime_profile(merged, "firered-vad")

        options = profile["firered_asr2_options"]
        self.assertEqual(options["segmentation_mode"], "vad")
        self.assertEqual(options["worker_count"], 5)
        self.assertEqual(options["vad_max_segment_sec"], 50)

    def test_profile_rejects_asr_chunk_overlap_not_smaller_than_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_repo(Path(tmp))
            settings = store.public_settings()
            source = settings["profiles"][0]
            models = {
                kind: source[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }
            with self.assertRaisesRegex(
                SettingsValidationError,
                "chunk_overlap_sec must be smaller",
            ):
                store.save_profile(
                    "invalid-chunks",
                    {
                        "label": "Invalid chunks",
                        "models": models,
                        "settings": {
                            "asr_chunk_mode": "custom",
                            "single_pass_max_duration_sec": 240,
                            "chunk_duration_sec": 60,
                            "chunk_overlap_sec": 60,
                        },
                    },
                )

    def test_settings_expose_audio_workflow_and_builtin_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            default_path = root / "video_analyzer" / "config" / "default_config.json"
            defaults = json.loads(default_path.read_text(encoding="utf-8"))
            defaults["runtime_profiles"]["audio_nx1"] = {
                **defaults["runtime_profiles"]["default"],
                "workflow_id": "audio_nx1",
                "asr_fallback_model_id": "asr-disabled",
                "diarization_fallback_model_id": "diarization-disabled",
            }
            default_path.write_text(json.dumps(defaults), encoding="utf-8")
            settings = store.public_settings()

        workflow = settings["schema"]["workflows"]["audio_nx1"]
        edges = {(item["from"], item["to"]) for item in workflow["flow"]["edges"]}
        profile = next(
            item for item in settings["profiles"] if item["name"] == "audio_nx1"
        )
        self.assertEqual(profile["workflow_id"], "audio_nx1")
        self.assertEqual(
            set(workflow["model_fields"]),
            {
                "asr",
                "diarization",
                "asr_fallback",
                "diarization_fallback",
                "selector",
                "text",
            },
        )
        self.assertEqual(profile["asr_fallback_model_id"], "asr-disabled")
        self.assertEqual(
            profile["diarization_fallback_model_id"],
            "diarization-disabled",
        )
        self.assertIn(("audio_input", "asr"), edges)
        self.assertIn(("audio_input", "diarization"), edges)
        self.assertNotIn(("asr", "diarization"), edges)
        self.assertIn(("asr", "transcript_merge"), edges)
        self.assertIn(("diarization", "transcript_merge"), edges)
        self.assertIn(("transcript_merge", "template_selector"), edges)
        self.assertNotIn(("asr", "template_selector"), edges)

    def test_expanded_diarization_options_survive_second_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "active_runtime_profile": "audio_nx1",
                        "runtime_profiles": {
                            "audio_nx1": {
                                "workflow_id": "audio_nx1",
                                "diarization_model_id": "diarization-3dspeaker-local",
                                "speaker_diarization": {
                                    "backend": "3dspeaker",
                                    "diarization_project_root": "/srv/3D-Speaker",
                                    "external_python": "/srv/venv/bin/python",
                                    "enabled": True,
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = Config(str(root))
            expanded = config.get_runtime_profile("audio_nx1")

        self.assertEqual(
            expanded["speaker_diarization"]["diarization_project_root"],
            "/srv/3D-Speaker",
        )
        self.assertEqual(
            expanded["speaker_diarization"]["external_python"],
            "/srv/venv/bin/python",
        )

    def test_audio_profile_can_be_saved_without_video_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            store.save_profile(
                "audio-custom",
                {
                    "label": "Audio Custom",
                    "workflow_id": "audio_nx1",
                    "models": {
                        "asr": source["asr_model_id"],
                        "diarization": source["diarization_model_id"],
                        "asr_fallback": "asr-disabled",
                        "diarization_fallback": "diarization-disabled",
                        "selector": "selector-inherit-text",
                        "text": source["text_model_id"],
                    },
                    "settings": {},
                },
            )
            saved = next(
                item
                for item in store.public_settings()["profiles"]
                if item["name"] == "audio-custom"
            )
            stored_profile = json.loads(
                (root / "config" / "config.json").read_text(encoding="utf-8")
            )["runtime_profiles"]["audio-custom"]

        self.assertEqual(saved["workflow_id"], "audio_nx1")
        self.assertNotIn("ocr_model_id", stored_profile)

    def test_disabled_visual_node_expands_to_no_frame_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            models = {
                kind: source[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }
            models["vision"] = "vision-disabled"
            store.save_profile("no-vision", {"label": "No Vision", "models": models, "settings": {}})
            _defaults, _user, merged = store.load()
            profile = get_runtime_profile(merged, "no-vision")
            user_config = json.loads((root / "config" / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(profile["vl_frame_policy"], "none")
        self.assertEqual(profile["vision_model"], "")
        self.assertNotIn("vision-disabled", user_config.get("model_catalog", {}))

    def test_text_node_cannot_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            settings = store.public_settings()
            source = settings["profiles"][0]
            models = {
                kind: source[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }
            store.save_model(
                "text-disabled",
                {"name": "Disabled Text", "kind": "text", "protocol": "none"},
            )
            models["text"] = "text-disabled"
            with self.assertRaisesRegex(SettingsValidationError, "cannot be disabled"):
                store.save_profile("invalid", {"label": "Invalid", "models": models, "settings": {}})

    def test_quick_test_treats_lazy_sleeping_proxy_as_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_repo(Path(tmp))
            settings = store.public_settings()
            vision_id = settings["profiles"][0]["vision_model_id"]
            response = MagicMock()
            response.status_code = 200
            response.headers = {"Content-Type": "application/json"}
            response.json.return_value = {"status": "sleeping", "ready": False}
            response.raise_for_status.return_value = None
            session = MagicMock()
            session.get.return_value = response
            with patch("video_analyzer.model_settings._test_session", return_value=session):
                result = store.test_model(vision_id, "quick", force=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "sleeping")
        self.assertIn("模型休眠", result["detail"])

    def test_quick_test_treats_stopped_local_on_demand_service_as_sleeping(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_repo(Path(tmp))
            settings = store.public_settings()
            vision_id = settings["profiles"][0]["vision_model_id"]
            session = MagicMock()
            session.get.side_effect = requests.exceptions.ConnectionError("connection refused")
            with patch("video_analyzer.model_settings._test_session", return_value=session):
                result = store.test_model(vision_id, "quick", force=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "sleeping")
        self.assertIn("最小推理会自动冷启动", result["detail"])

    def test_model_inference_prepares_selected_local_ocr_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_repo(Path(tmp))
            settings = store.public_settings()
            ocr_id = settings["profiles"][0]["ocr_model_id"]
            captured = {}

            def stage_context(stage, config, *_args, **_kwargs):
                captured["stage"] = stage
                captured["config"] = config
                return contextlib.nullcontext()

            with (
                patch(
                    "video_analyzer.local_model_runtime.local_model_stage",
                    side_effect=stage_context,
                ),
                patch.object(
                    store,
                    "_inference_test_resource",
                    return_value={"ok": True, "status": "passed", "detail": "test"},
                ),
            ):
                result = store.test_model(ocr_id, "inference", force=True)

        self.assertTrue(result["ok"])
        self.assertEqual(captured["stage"], "ocr")
        self.assertEqual(captured["config"]["ocr"]["base_urls"], ["http://127.0.0.1:18088/v1"])

    def test_tts_preview_prepares_stage_then_returns_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_repo(Path(tmp))
            response = MagicMock()
            response.content = b"RIFF" + b"\0" * 64
            response.raise_for_status.return_value = None
            session = MagicMock()
            session.post.return_value = response
            captured = {}

            def stage_context(stage, config, *_args, **_kwargs):
                captured["stage"] = stage
                captured["config"] = config
                return contextlib.nullcontext()

            with (
                patch(
                    "video_analyzer.local_model_runtime.local_model_stage",
                    side_effect=stage_context,
                ),
                patch("video_analyzer.model_settings._test_session", return_value=session),
            ):
                wav, metadata = store.preview_tts(
                    "tts-indextts25-ray-p40",
                    "这是一段前端语音试听。",
                )

        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertEqual(metadata["voice"], "check_boards_sweet")
        self.assertEqual(captured["stage"], "tts")
        self.assertEqual(captured["config"]["tts"]["base_url"], "http://127.0.0.1:8092")
        request = session.post.call_args
        self.assertEqual(request.args[0], "http://127.0.0.1:8092/v1/audio/speech")
        self.assertEqual(request.kwargs["json"]["input"], "这是一段前端语音试听。")

    def test_profile_test_returns_status_for_every_flow_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_repo(Path(tmp))
            settings = store.public_settings()
            profile = settings["profiles"][0]
            models = {
                kind: profile[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }

            def fake_test(model_id, mode="quick", **_kwargs):
                model = next(item for item in settings["models"] if item["id"] == model_id)
                return {
                    "ok": True,
                    "status": "disabled" if model["protocol"] == "none" else "reachable",
                    "detail": "test",
                    "model_id": model_id,
                    "kind": model["kind"],
                    "mode": mode,
                    "elapsed_ms": 1,
                }

            with patch.object(store, "test_model", side_effect=fake_test):
                result = store.test_profile({"profile_name": "draft", "mode": "quick", "models": models})

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["summary"]["total"],
            len(settings["schema"]["profile_flow"]["nodes"]),
        )
        self.assertEqual(set(result["results"]), {item["id"] for item in settings["schema"]["profile_flow"]["nodes"]})
        self.assertEqual(result["results"]["input"]["status"], "configured")
        self.assertEqual(result["results"]["text"]["status"], "reachable")
        self.assertEqual(result["results"]["documents"]["reused_slot_result"], "text")

    def test_pathway_profile_test_prepares_local_model_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_repo(Path(tmp))
            settings = store.public_settings()
            profile = settings["profiles"][0]
            models = {
                kind: profile[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }

            def fake_test(model_id, mode="quick", **_kwargs):
                model = next(item for item in settings["models"] if item["id"] == model_id)
                return {
                    "ok": True,
                    "status": "passed",
                    "detail": "test",
                    "model_id": model_id,
                    "kind": model["kind"],
                    "mode": mode,
                    "elapsed_ms": 1,
                }

            with (
                patch.object(store, "test_model", side_effect=fake_test),
                patch(
                    "video_analyzer.local_model_runtime.local_model_runtime_session",
                    side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
                ) as runtime_session,
                patch(
                    "video_analyzer.local_model_runtime.local_model_stage",
                    side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
                ) as model_stage,
            ):
                result = store.test_profile(
                    {
                        "profile_name": "default",
                        "mode": "pathway",
                        "models": models,
                    }
                )

        self.assertTrue(result["ok"])
        runtime_session.assert_called_once()
        self.assertEqual(
            [call.args[0] for call in model_stage.call_args_list],
            ["asr", "ocr", "vl", "tts"],
        )

    def test_pathway_profile_test_reports_local_stage_start_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_repo(Path(tmp))
            settings = store.public_settings()
            profile = settings["profiles"][0]
            models = {
                kind: profile[field]
                for kind, field in settings["schema"]["profile_model_fields"].items()
            }

            @contextlib.contextmanager
            def failed_stage():
                raise RuntimeError("vision files missing")
                yield

            def stage_context(stage, *_args, **_kwargs):
                return failed_stage() if stage == "vl" else contextlib.nullcontext()

            with (
                patch.object(
                    store,
                    "test_model",
                    return_value={
                        "ok": True,
                        "status": "passed",
                        "detail": "test",
                        "elapsed_ms": 1,
                    },
                ),
                patch(
                    "video_analyzer.local_model_runtime.local_model_runtime_session",
                    side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
                ),
                patch(
                    "video_analyzer.local_model_runtime.local_model_stage",
                    side_effect=stage_context,
                ),
            ):
                result = store.test_profile(
                    {
                        "profile_name": "default",
                        "mode": "pathway",
                        "models": models,
                    }
                )

        self.assertFalse(result["ok"])
        self.assertEqual(result["results"]["vision"]["status"], "failed")
        self.assertIn("vision files missing", result["results"]["vision"]["detail"])

    def test_deleting_built_in_profile_disables_it_locally_and_save_restores_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_repo(root)
            default_path = root / "video_analyzer" / "config" / "default_config.json"
            defaults = json.loads(default_path.read_text(encoding="utf-8"))
            defaults["runtime_profiles"]["keep"] = dict(defaults["runtime_profiles"]["default"])
            defaults["active_runtime_profile"] = "keep"
            default_path.write_text(json.dumps(defaults), encoding="utf-8")
            before = store.public_settings()
            source = next(item for item in before["profiles"] if item["name"] == "default")
            models = {
                kind: source[field]
                for kind, field in before["schema"]["profile_model_fields"].items()
            }

            deleted = store.delete_profile("default")
            after_delete = store.public_settings()
            user = json.loads((root / "config" / "config.json").read_text(encoding="utf-8"))
            (root / "config" / "default_config.json").write_text(
                json.dumps(defaults),
                encoding="utf-8",
            )
            loaded = Config(str(root / "config")).config

            self.assertEqual(deleted["status"], "disabled")
            self.assertIn("default", user["disabled_runtime_profiles"])
            self.assertNotIn("default", {item["name"] for item in after_delete["profiles"]})
            self.assertNotIn("default", loaded["runtime_profiles"])

            store.save_profile("default", {"label": "Restored", "models": models, "settings": {}})
            restored = store.public_settings()
            user = json.loads((root / "config" / "config.json").read_text(encoding="utf-8"))

        self.assertIn("default", {item["name"] for item in restored["profiles"]})
        self.assertNotIn("default", user.get("disabled_runtime_profiles", []))


if __name__ == "__main__":
    unittest.main()
