import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
                    "settings": {"pipeline_mode": "deep", "vl_concurrency": 4},
                },
            )
            _defaults, _user, merged = store.load()
            profile = get_runtime_profile(merged, "custom")

        self.assertEqual(profile["pipeline_mode"], "deep")
        self.assertEqual(profile["asr_provider"], "vibevoice")
        self.assertEqual(profile["vision_model"], "minicpm")
        self.assertEqual(profile["text_model"], "text-model")
        self.assertEqual(profile["speaker_diarization"]["enabled"], True)

    def test_settings_expose_fixed_branched_profile_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.make_repo(Path(tmp)).public_settings()

        flow = settings["schema"]["profile_flow"]
        node_ids = {item["id"] for item in flow["nodes"]}
        edges = {(item["from"], item["to"]) for item in flow["edges"]}
        models = {item["id"]: item for item in settings["models"]}
        self.assertEqual(flow["version"], 1)
        self.assertTrue({"asr", "diarization", "ocr", "vision", "evidence_merge", "final_publish"} <= node_ids)
        self.assertIn(("input", "audio_extract"), edges)
        self.assertIn(("input", "frame_extract"), edges)
        self.assertIn(("vision", "evidence_merge"), edges)
        self.assertEqual(models["vision-disabled"]["protocol"], "none")
        self.assertEqual(models["review-inherit-text"]["protocol"], "inherit_text")
        self.assertEqual(models["image-disabled"]["protocol"], "none")

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
        self.assertEqual(models["ocr-unlimited-local"]["model"], "baidu/Unlimited-OCR")
        self.assertEqual(models["ocr-dots-local"]["model"], "rednote-hilab/dots.ocr")
        self.assertEqual(
            models["vision-qwen3-vl-4b-local"]["options"]["engine"],
            "qwen3_vl_4b",
        )

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
        self.assertEqual(result["summary"]["total"], 17)
        self.assertEqual(set(result["results"]), {item["id"] for item in settings["schema"]["profile_flow"]["nodes"]})
        self.assertEqual(result["results"]["input"]["status"], "configured")
        self.assertEqual(result["results"]["text"]["status"], "reachable")

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
