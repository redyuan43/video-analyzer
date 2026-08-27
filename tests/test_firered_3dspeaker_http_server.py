import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tools import firered_3dspeaker_http_server as server


class FireRed3DSpeakerServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache_dir = Path(self.temporary.name) / "cache"
        self.audio_path = Path(self.temporary.name) / "audio.wav"
        self.audio_path.write_bytes(b"audio fixture")
        self.cache_patch = patch.object(server, "CACHE_DIR", self.cache_dir)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def pipeline_patches(self, transcribe, diarize):
        return (
            patch.object(server, "transcribe_with_firered", side_effect=transcribe),
            patch.object(server, "run_3dspeaker_assignment", side_effect=diarize),
            patch.object(
                server,
                "assign_speakers_by_overlap",
                side_effect=lambda transcript, _turns: (
                    transcript,
                    {"final_speaker_count": 1},
                ),
            ),
        )

    def test_pipeline_runs_firered_and_diarization_concurrently(self):
        barrier = threading.Barrier(2, timeout=1)

        def transcribe(_path):
            barrier.wait()
            time.sleep(0.08)
            return {"text": "hello", "segments": [], "language": "en"}

        def diarize(_path, _config):
            barrier.wait()
            time.sleep(0.08)
            return ([{"start": 0, "end": 1, "speaker": 0}], {})

        transcribe_patch, diarize_patch, assign_patch = self.pipeline_patches(
            transcribe, diarize
        )
        with transcribe_patch, diarize_patch, assign_patch:
            payload = server.run_pipeline(self.audio_path)

        self.assertFalse(payload["cache_hit"])
        self.assertIn("queue_seconds", payload["stage_timings"]["firered"])

    def test_stage_cache_hits_without_rerunning_providers(self):
        transcript = {"text": "hello", "segments": [], "language": "en"}
        turns = ([{"start": 0, "end": 1, "speaker": 0}], {})
        transcribe_patch, diarize_patch, assign_patch = self.pipeline_patches(
            lambda _path: transcript,
            lambda _path, _config: turns,
        )
        with (
            transcribe_patch as transcribe_mock,
            diarize_patch as diarize_mock,
            assign_patch as assign_mock,
        ):
            first = server.run_pipeline(self.audio_path)
            second = server.run_pipeline(self.audio_path)

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertEqual(diarize_mock.call_count, 1)
        self.assertEqual(assign_mock.call_count, 1)
        self.assertTrue(second["stage_timings"]["alignment"]["cache_hit"])

    def test_corrupt_stage_cache_is_ignored_and_replaced(self):
        audio_hash = server.sha256_file(self.audio_path)
        fingerprint = server.firered_fingerprint()
        corrupt_path = server.cache_path(audio_hash, fingerprint, "firered")
        corrupt_path.parent.mkdir(parents=True)
        corrupt_path.write_text("{}", encoding="utf-8")
        transcript = {"text": "recovered", "segments": [], "language": "en"}
        transcribe_patch, diarize_patch, assign_patch = self.pipeline_patches(
            lambda _path: transcript,
            lambda _path, _config: ([{"start": 0, "end": 1, "speaker": 0}], {}),
        )
        with transcribe_patch as transcribe_mock, diarize_patch, assign_patch:
            payload = server.run_pipeline(self.audio_path)

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertFalse(payload["stage_timings"]["firered"]["cache_hit"])
        self.assertEqual(json.loads(corrupt_path.read_text())["text"], "recovered")

    def test_same_stage_key_is_single_flight_when_concurrency_is_two(self):
        calls = 0
        barrier = threading.Barrier(2, timeout=1)

        def operation():
            nonlocal calls
            calls += 1
            time.sleep(0.08)
            return {"text": "once", "segments": []}

        def invoke(_index):
            barrier.wait()
            return server.run_cached_stage(
                "firered", "same-audio", "same-config", threading.Semaphore(2),
                operation, lambda payload: payload.get("text") == "once",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(invoke, range(2)))

        self.assertEqual(calls, 1)
        self.assertEqual(sum(not timing["cache_hit"] for _payload, timing in results), 1)
        self.assertEqual(server._SINGLE_FLIGHTS, {})

    def test_alignment_cache_is_single_flight_for_concurrent_pipelines(self):
        transcript = {"text": "hello", "segments": [], "language": "en"}
        turns = ([{"start": 0, "end": 1, "speaker": 0}], {})
        calls = 0
        calls_lock = threading.Lock()

        def assign(source, _turns):
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.08)
            return source, {"final_speaker_count": 1}

        transcribe_patch, diarize_patch, _assign_patch = self.pipeline_patches(
            lambda _path: transcript,
            lambda _path, _config: turns,
        )
        with (
            transcribe_patch,
            diarize_patch,
            patch.object(server, "assign_speakers_by_overlap", side_effect=assign),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            payloads = list(
                executor.map(
                    lambda _index: server.run_pipeline(self.audio_path),
                    range(2),
                )
            )

        self.assertEqual(calls, 1)
        self.assertEqual(
            sum(
                not payload["stage_timings"]["alignment"]["cache_hit"]
                for payload in payloads
            ),
            1,
        )
        self.assertEqual(server._SINGLE_FLIGHTS, {})

    def test_alignment_algorithm_change_only_invalidates_alignment(self):
        transcript = {"text": "hello", "segments": [], "language": "en"}
        turns = ([{"start": 0, "end": 1, "speaker": 0}], {})
        transcribe_patch, diarize_patch, assign_patch = self.pipeline_patches(
            lambda _path: transcript,
            lambda _path, _config: turns,
        )
        with (
            transcribe_patch as transcribe_mock,
            diarize_patch as diarize_mock,
            assign_patch as assign_mock,
        ):
            first = server.run_pipeline(self.audio_path)
            with patch.object(
                server,
                "alignment_fingerprint",
                return_value="changed-alignment",
            ):
                changed = server.run_pipeline(self.audio_path)
                changed_again = server.run_pipeline(self.audio_path)

        self.assertFalse(first["cache_hit"])
        self.assertTrue(changed["stage_timings"]["firered"]["cache_hit"])
        self.assertTrue(changed["stage_timings"]["3dspeaker"]["cache_hit"])
        self.assertFalse(changed["stage_timings"]["alignment"]["cache_hit"])
        self.assertFalse(changed["cache_hit"])
        self.assertTrue(changed_again["cache_hit"])
        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertEqual(diarize_mock.call_count, 1)
        self.assertEqual(assign_mock.call_count, 2)

    def test_alignment_input_fingerprint_tracks_both_artifacts_and_algorithm(self):
        baseline = server.alignment_input_fingerprint(
            {"text": "hello", "segments": []},
            {"turns": [{"speaker": 0}], "report": {}},
            "algorithm-a",
        )
        changed_firered = server.alignment_input_fingerprint(
            {"text": "changed", "segments": []},
            {"turns": [{"speaker": 0}], "report": {}},
            "algorithm-a",
        )
        changed_diarization = server.alignment_input_fingerprint(
            {"text": "hello", "segments": []},
            {"turns": [{"speaker": 1}], "report": {}},
            "algorithm-a",
        )
        changed_algorithm = server.alignment_input_fingerprint(
            {"text": "hello", "segments": []},
            {"turns": [{"speaker": 0}], "report": {}},
            "algorithm-b",
        )

        self.assertEqual(
            len({baseline, changed_firered, changed_diarization, changed_algorithm}),
            4,
        )

    def test_stage_fingerprints_are_isolated(self):
        original = (server.firered_fingerprint(), server.diarization_fingerprint(), server.alignment_fingerprint())
        with patch.object(server, "EDGE_URL", server.EDGE_URL + "/changed"):
            changed_firered = (server.firered_fingerprint(), server.diarization_fingerprint(), server.alignment_fingerprint())
        with patch.object(server, "THREED_ROOT", server.THREED_ROOT + "-changed"):
            changed_diarization = (server.firered_fingerprint(), server.diarization_fingerprint(), server.alignment_fingerprint())

        self.assertNotEqual(changed_firered[0], original[0])
        self.assertEqual(changed_firered[1:], original[1:])
        self.assertEqual(changed_diarization[0], original[0])
        self.assertNotEqual(changed_diarization[1], original[1])
        self.assertEqual(changed_diarization[2], original[2])


if __name__ == "__main__":
    unittest.main()
