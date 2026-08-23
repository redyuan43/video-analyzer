import hashlib
import json
import unittest
from pathlib import Path

from video_analyzer.audio_workflow_snapshot import (
    AudioWorkflowSnapshotError,
    parse_audio_workflow_snapshot,
    resolve_audio_workflow_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def audio_model(
    model_id,
    kind,
    protocol,
    *,
    execution_target="ai",
    deployment="local",
    model="",
    endpoints=None,
    options=None,
):
    return {
        "id": model_id,
        "label": model_id,
        "kind": kind,
        "protocol": protocol,
        "execution_target": execution_target,
        "deployment": deployment,
        "model": model,
        "endpoints": list(endpoints or []),
        "health_endpoint": "",
        "options": dict(options or {}),
        "built_in": True,
    }


def long_audio_snapshot(profile_id="long-default", revision=1):
    snapshot = {
        "schema_version": 1,
        "workflow_id": "audio_long_v1",
        "profile_id": profile_id,
        "profile_revision": revision,
        "node_configs": {
            "asr": {
                "enabled": True,
                "route_policy": "auto",
                "local_model": audio_model(
                    "ai-vibevoice-asr",
                    "asr",
                    "vibevoice_http",
                ),
                "cloud_model": audio_model(
                    "ai-tencent-hy-asr",
                    "asr",
                    "tencent_hy_asr_ws",
                    deployment="cloud",
                    model="hy-asr-3.0-preview",
                ),
            },
            "diarization": {
                "enabled": True,
                "route_policy": "auto",
                "local_model": audio_model(
                    "nano-3dspeaker",
                    "diarization",
                    "three_d_speaker_http",
                    execution_target="nano",
                    endpoints=[
                        "http://100.64.0.1:5021/api/diarization/turns"
                    ],
                    options={
                        "token_env": "NANO_DIARIZATION_TOKEN",
                        "fallback_backend": "3dspeaker",
                        "parallel_with_asr": True,
                    },
                ),
                "cloud_model": audio_model(
                    "ai-asr-embedded-speaker",
                    "diarization",
                    "asr_embedded",
                    deployment="cloud",
                ),
            },
            "selector": {
                "enabled": True,
                "route_policy": "auto",
                "local_model": audio_model(
                    "ai-local-selector",
                    "selector",
                    "inherit_text",
                ),
                "cloud_model": audio_model(
                    "ai-cloud-selector",
                    "selector",
                    "openai_compatible",
                    deployment="cloud",
                    model="deepseek-v4-flash",
                ),
            },
            "summary": {
                "enabled": True,
                "route_policy": "auto",
                "local_model": audio_model(
                    "ai-local-text",
                    "text",
                    "openai_compatible",
                    model="prism-ml/bonsai-27b",
                ),
                "cloud_model": audio_model(
                    "ai-cloud-text",
                    "text",
                    "openai_compatible",
                    deployment="cloud",
                    model="deepseek-v4-flash",
                ),
            },
            "tts": {
                "enabled": True,
                "route_policy": "background_local",
                "local_model": audio_model(
                    "ai-indextts",
                    "tts",
                    "openai_speech",
                ),
                "cloud_model": audio_model(
                    "ai-xiaomi-mimo-tts",
                    "tts",
                    "xiaomi_mimo_tts",
                    deployment="cloud",
                    model="mimo-v2.5-tts",
                    options={"voice": "冰糖"},
                ),
            },
        },
    }
    snapshot["fingerprint"] = hashlib.sha256(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return snapshot


class AudioWorkflowSnapshotTests(unittest.TestCase):
    def config(self):
        return json.loads(
            (
                REPO_ROOT
                / "video_analyzer"
                / "config"
                / "default_config.json"
            ).read_text(encoding="utf-8")
        )

    def test_snapshot_drives_runtime_models_and_routes(self):
        snapshot = parse_audio_workflow_snapshot(
            json.dumps(long_audio_snapshot(), ensure_ascii=False)
        )

        profile, metadata, runtime_models = resolve_audio_workflow_profile(
            self.config(),
            "audio_nx1",
            snapshot,
        )

        self.assertEqual(profile["asr_provider"], "vibevoice")
        self.assertEqual(
            profile["speaker_diarization"]["backend"],
            "remote_3dspeaker_http",
        )
        self.assertEqual(
            profile["speaker_diarization"]["endpoint"],
            "http://100.64.0.1:5021/api/diarization/turns",
        )
        self.assertEqual(profile["text_model"], "prism-ml/bonsai-27b")
        self.assertEqual(profile["text_fallback_model"], "deepseek-v4-flash")
        self.assertTrue(profile["template_selector_fallback_enabled"])
        self.assertEqual(
            profile["template_selector_fallback_model"],
            "deepseek-v4-flash",
        )
        self.assertTrue(profile["audio_cloud_fallback"]["enabled"])
        self.assertTrue(profile["tts_fallback_enabled"])
        self.assertEqual(profile["tts_fallback_provider"], "xiaomi_mimo_tts")
        self.assertEqual(profile["tts_fallback_model"], "mimo-v2.5-tts")
        self.assertEqual(
            profile["audio_cloud_fallback"]["asr"]["protocol"],
            "tencent_hy_asr_ws",
        )
        self.assertEqual(metadata["source"], "nano_workflow_snapshot")
        self.assertEqual(metadata["profile_revision"], 1)
        self.assertIn("audio-snapshot-summary-local", runtime_models)
        self.assertIn("audio-snapshot-summary-cloud", runtime_models)

    def test_cloud_only_uses_cloud_model_as_primary(self):
        payload = long_audio_snapshot()
        payload["node_configs"]["summary"]["route_policy"] = "cloud_only"
        payload_without_fingerprint = dict(payload)
        payload_without_fingerprint.pop("fingerprint")
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(
                payload_without_fingerprint,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        profile, _metadata, _runtime_models = resolve_audio_workflow_profile(
            self.config(),
            "audio_nx1",
            parse_audio_workflow_snapshot(payload),
        )

        self.assertEqual(profile["text_model"], "deepseek-v4-flash")
        self.assertFalse(profile["text_fallback_enabled"])

    def test_rejects_modified_snapshot_fingerprint(self):
        payload = long_audio_snapshot()
        payload["profile_revision"] = 2

        with self.assertRaisesRegex(
            AudioWorkflowSnapshotError,
            "fingerprint mismatch",
        ):
            parse_audio_workflow_snapshot(payload)

    def test_rejects_secret_material(self):
        payload = long_audio_snapshot()
        payload["node_configs"]["summary"]["cloud_model"]["api_key"] = "secret"
        payload_without_fingerprint = dict(payload)
        payload_without_fingerprint.pop("fingerprint")
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(
                payload_without_fingerprint,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        with self.assertRaisesRegex(AudioWorkflowSnapshotError, "not allowed"):
            parse_audio_workflow_snapshot(payload)

    def test_rejects_public_or_metadata_endpoint(self):
        for endpoint in (
            "https://8.8.8.8/v1",
            "http://169.254.169.254/latest/meta-data",
        ):
            with self.subTest(endpoint=endpoint):
                payload = long_audio_snapshot()
                payload["node_configs"]["diarization"]["local_model"][
                    "endpoints"
                ] = [endpoint]
                payload_without_fingerprint = dict(payload)
                payload_without_fingerprint.pop("fingerprint")
                payload["fingerprint"] = hashlib.sha256(
                    json.dumps(
                        payload_without_fingerprint,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                with self.assertRaises(AudioWorkflowSnapshotError):
                    parse_audio_workflow_snapshot(payload)


if __name__ == "__main__":
    unittest.main()
