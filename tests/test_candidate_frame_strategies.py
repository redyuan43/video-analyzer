from unittest import TestCase
from unittest.mock import patch

from video_analyzer.candidate_frame_strategies import (
    _sklearn_mean_shift_modes,
    classify_video_profile,
    select_candidate_frames_with_strategy,
)


def candidate(index, timestamp, visual, textness, gffv=None):
    item = {
        "timestamp": timestamp,
        "score": visual + textness * 30.0,
        "visual_score": visual,
        "textness_score": textness,
        "gffv": gffv or [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    return (index, {"item": item}, timestamp, item["score"])


class CandidateFrameStrategyTests(TestCase):
    def test_classifies_generic_visual_change_video(self):
        rows = [candidate(index, index * 10.0, 30.0 if index % 2 else 18.0, 0.05) for index in range(12)]

        profile = classify_video_profile(rows, video_duration_seconds=120.0)

        self.assertEqual(profile["profile"], "generic")
        self.assertGreater(profile["confidence"], 0.5)

    def test_classifies_lecture_stable_text_video(self):
        rows = [candidate(index, index * 10.0, 4.0, 0.45) for index in range(12)]

        profile = classify_video_profile(rows, video_duration_seconds=120.0, transcript_segments=[{"start": 0, "end": 15}])

        self.assertEqual(profile["profile"], "lecture")
        self.assertGreater(profile["confidence"], 0.6)

    def test_lecture_strategy_prefers_stable_text_candidates(self):
        rows = [
            candidate(0, 0.0, 30.0, 0.05),
            candidate(1, 10.0, 4.0, 0.50),
            candidate(2, 20.0, 5.0, 0.52),
            candidate(3, 30.0, 28.0, 0.08),
            candidate(4, 40.0, 4.0, 0.48),
            candidate(5, 50.0, 5.0, 0.46),
        ]

        result = select_candidate_frames_with_strategy(
            rows,
            candidate_budget=3,
            video_duration_seconds=60.0,
            pipeline_mode="balanced",
            strategy="lecture",
        )

        selected_textness = [item[1]["item"]["textness_score"] for item in result.selected]
        self.assertGreaterEqual(sum(value >= 0.28 for value in selected_textness), 2)
        self.assertEqual(result.metadata["paper_algorithm"], "sspa_spatio_temporal_subtitle_projection")

    def test_operation_strategy_uses_gffv_modes_and_text_anchors(self):
        rows = [
            candidate(0, 0.0, 8.0, 0.35, [1, 0, 0, 0, 0, 0, 0, 0]),
            candidate(1, 8.0, 9.0, 0.36, [1, 0, 0, 0, 0, 0, 0, 0]),
            candidate(2, 16.0, 14.0, 0.40, [0, 1, 0, 0, 0, 0, 0, 0]),
            candidate(3, 24.0, 12.0, 0.38, [0, 1, 0, 0, 0, 0, 0, 0]),
            candidate(4, 32.0, 10.0, 0.42, [0, 0, 1, 0, 0, 0, 0, 0]),
            candidate(5, 40.0, 11.0, 0.41, [0, 0, 1, 0, 0, 0, 0, 0]),
        ]

        result = select_candidate_frames_with_strategy(
            rows,
            candidate_budget=3,
            video_duration_seconds=45.0,
            pipeline_mode="deep",
            strategy="operation",
            transcript_segments=[{"start": 15, "end": 20}],
        )

        self.assertEqual(len(result.selected), 3)
        self.assertEqual(result.metadata["paper_algorithm"], "mskvs_gffv_adaptive_mean_shift_with_sspa")
        self.assertEqual(
            result.metadata["strategy_observations"]["paper_algorithm_trace"]["mskvs"]["feature"],
            "gffv_global_orientation",
        )

    def test_operation_strategy_falls_back_when_sklearn_mean_shift_errors(self):
        rows = [
            candidate(0, 0.0, 8.0, 0.35, [1, 0, 0, 0, 0, 0, 0, 0]),
            candidate(1, 8.0, 9.0, 0.36, [1, 0, 0, 0, 0, 0, 0, 0]),
            candidate(2, 16.0, 14.0, 0.40, [0, 1, 0, 0, 0, 0, 0, 0]),
            candidate(3, 24.0, 12.0, 0.38, [0, 1, 0, 0, 0, 0, 0, 0]),
        ]

        with patch(
            "video_analyzer.candidate_frame_strategies._sklearn_mean_shift_modes",
            return_value=[],
        ):
            result = select_candidate_frames_with_strategy(
                rows,
                candidate_budget=2,
                video_duration_seconds=30.0,
                pipeline_mode="deep",
                strategy="operation",
            )

        self.assertEqual(len(result.selected), 2)
        self.assertEqual(result.metadata["paper_algorithm"], "mskvs_gffv_adaptive_mean_shift_with_sspa")

    def test_sklearn_mean_shift_value_error_is_treated_as_unavailable(self):
        rows = [
            candidate(0, 0.0, 8.0, 0.35),
            candidate(1, 8.0, 9.0, 0.36),
        ]

        class FailingMeanShift:
            def __init__(self, *args, **kwargs):
                pass

            def fit_predict(self, matrix):
                raise ValueError("No point was within bandwidth")

        with patch("sklearn.cluster.MeanShift", FailingMeanShift):
            selected = _sklearn_mean_shift_modes(
                rows,
                vectors=[[0.0, 1.0], [1.0, 0.0]],
                budget=1,
                duration=8.0,
            )

        self.assertEqual(selected, [])

    def test_lmske_uses_clip_embeddings_and_transnet_boundaries_when_present(self):
        rows = [
            candidate(0, 0.0, 3.0, 0.01, [1, 0, 0, 0, 0, 0, 0, 0]),
            candidate(1, 10.0, 4.0, 0.01, [1, 0, 0, 0, 0, 0, 0, 0]),
            candidate(2, 20.0, 5.0, 0.01, [0, 1, 0, 0, 0, 0, 0, 0]),
            candidate(3, 30.0, 4.0, 0.01, [0, 1, 0, 0, 0, 0, 0, 0]),
        ]
        rows[0][1]["item"]["clip_embedding"] = [1.0, 0.0]
        rows[1][1]["item"]["clip_embedding"] = [0.9, 0.1]
        rows[2][1]["item"]["clip_embedding"] = [0.0, 1.0]
        rows[2][1]["item"]["shot_boundary"] = True
        rows[3][1]["item"]["clip_embedding"] = [0.1, 0.9]

        result = select_candidate_frames_with_strategy(
            rows,
            candidate_budget=2,
            video_duration_seconds=40.0,
            pipeline_mode="deep",
            strategy="generic",
        )

        trace = result.metadata["strategy_observations"]["paper_algorithm_trace"]["lmske"]
        self.assertEqual(trace["embedding_source"], "clip_embedding")
        self.assertEqual(trace["shot_detector"], "transnetv2")
        self.assertEqual(len(result.selected), 2)

    def test_auto_low_confidence_defaults_to_operation(self):
        rows = [
            candidate(0, 0.0, 4.0, 0.02),
            candidate(1, 20.0, 5.0, 0.02),
            candidate(2, 40.0, 4.0, 0.02),
        ]

        result = select_candidate_frames_with_strategy(
            rows,
            candidate_budget=2,
            video_duration_seconds=60.0,
            pipeline_mode="balanced",
            strategy="auto",
        )

        self.assertEqual(result.metadata["selected_candidate_strategy"], "operation")

    def test_deep_auto_adds_secondary_strategy(self):
        rows = [candidate(index, index * 10.0, 4.0, 0.45) for index in range(8)]

        result = select_candidate_frames_with_strategy(
            rows,
            candidate_budget=4,
            video_duration_seconds=80.0,
            pipeline_mode="deep",
            strategy="auto",
            transcript_segments=[{"start": 10, "end": 12}],
        )

        trace = result.metadata["strategy_observations"]["paper_algorithm_trace"]
        self.assertEqual(result.metadata["selected_candidate_strategy"], "lecture")
        self.assertEqual(trace["secondary_strategy"], "operation")

    def test_low_quality_strategy_falls_back_to_legacy_union_or_legacy(self):
        rows = [
            candidate(0, 0.0, 20.0, 0.01),
            candidate(1, 1.0, 19.0, 0.01),
            candidate(2, 2.0, 18.0, 0.01),
            candidate(3, 90.0, 17.0, 0.01),
        ]

        result = select_candidate_frames_with_strategy(
            rows,
            candidate_budget=3,
            video_duration_seconds=120.0,
            pipeline_mode="balanced",
            strategy="lecture",
        )

        self.assertIn(result.metadata["paper_algorithm_status"], {"ok", "guarded_union", "guarded_legacy", "degraded_backend_unavailable"})
        self.assertEqual(len(result.selected), 3)
