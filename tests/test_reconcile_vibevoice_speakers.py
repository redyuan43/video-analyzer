import unittest

from tools.reconcile_vibevoice_speakers import find_vibevoice_metadata, update_vibevoice_metadata


class ReconcileVibeVoiceSpeakersToolTests(unittest.TestCase):
    def test_finds_top_level_vibevoice_chunk_results(self):
        metadata = {"chunk_results": [{"chunk": {"chunk_index": 0}}], "provider": "vibevoice"}

        found, source = find_vibevoice_metadata(metadata)

        self.assertEqual(source, "metadata")
        self.assertEqual(found["provider"], "vibevoice")

    def test_finds_nested_deep_transcript_vibevoice_chunk_results(self):
        metadata = {
            "source": "merged_remote_http_vibevoice",
            "fast_transcript_metadata": {"provider": "fast"},
            "deep_transcript_metadata": {
                "chunk_results": [{"chunk": {"chunk_index": 0}}],
                "provider": "vibevoice",
            },
        }

        found, source = find_vibevoice_metadata(metadata)

        self.assertEqual(source, "metadata.deep_transcript_metadata")
        self.assertEqual(found["provider"], "vibevoice")

    def test_updates_nested_vibevoice_metadata_without_dropping_merged_metadata(self):
        metadata = {
            "source": "merged_remote_http_vibevoice",
            "fast_transcript_metadata": {"provider": "fast"},
            "deep_transcript_metadata": {"chunk_results": [], "provider": "vibevoice"},
        }
        source_metadata = {
            "chunk_results": [],
            "provider": "vibevoice",
            "quality_report": {"global_speaker_count": 2},
            "mode": "offline_ray_chunk_reconcile",
        }

        updated = update_vibevoice_metadata(metadata, source_metadata, "metadata.deep_transcript_metadata")

        self.assertEqual(updated["source"], "merged_remote_http_vibevoice")
        self.assertEqual(updated["fast_transcript_metadata"]["provider"], "fast")
        self.assertEqual(updated["deep_transcript_metadata"]["quality_report"]["global_speaker_count"], 2)
        self.assertEqual(updated["quality_report"]["global_speaker_count"], 2)
        self.assertEqual(updated["offline_reconcile_source"], "metadata.deep_transcript_metadata")


if __name__ == "__main__":
    unittest.main()
