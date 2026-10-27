import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.predicates.extract_predicates import (  # noqa: E402
    build_predicate_config,
    extract_coordination,
    extract_finger_motion,
    extract_motion,
    extract_predicates,
    trend_consistency_ratio,
    validate_trend_consistency,
)
from src_neurosymbolic.events.build_events_phase_v2 import build_events_phase_v2  # noqa: E402
from src_neurosymbolic.kg.build_instance_graph_phase_v2 import build_instance_graph_phase_v2  # noqa: E402


def keypoint(x: float, y: float, score: float = 1.0) -> dict:
    return {"xy": [x, y], "score": score}


def motion_frames() -> list[dict]:
    frames = []
    for frame_id in range(6):
        shift = frame_id * 0.10
        progress = frame_id / 5.0
        root_x = shift
        points = {
            "right_hand_root": keypoint(root_x, 0.0),
            "right_wrist": keypoint(root_x, -0.5),
            "right_forefinger1": keypoint(root_x + 0.2, -0.1),
            "right_pinky_finger1": keypoint(root_x - 0.2, -0.1),
            "right_middle_finger1": keypoint(root_x, -0.4),
            "right_thumb4": keypoint(root_x - 0.1, -0.2),
            # Both tips move down relative to the translating hand; the index
            # also moves away from the thumb to produce an opening pinch.
            "right_forefinger4": keypoint(root_x + 0.1 + 0.2 * progress, -0.4 + 0.3 * progress),
            "right_middle_finger4": keypoint(root_x, -0.4 + 0.3 * progress),
        }
        frames.append({"frame_id": frame_id, "valid": True, "keypoints": points})
    return frames


class PredicateV2Tests(unittest.TestCase):
    def test_temporal_factor_v3_is_opt_in_and_preserves_distinct_click_ids(self) -> None:
        def make_frame(frame_id: int, closed: bool) -> dict:
            tip_offset = 0.25 if closed else 1.5
            points = {
                "right_hand_root": keypoint(0.0, 0.0),
                "right_wrist": keypoint(0.0, -1.0),
                "right_forefinger1": keypoint(-0.5, -0.5),
                "right_pinky_finger1": keypoint(0.5, -0.5),
                "right_middle_finger1": keypoint(0.0, -1.0),
                "right_forefinger4": keypoint(-0.5, -0.5 + tip_offset),
                "right_middle_finger4": keypoint(0.0, -1.0 + tip_offset),
                "right_thumb4": keypoint(-0.8, -0.5),
            }
            return {"frame_id": frame_id, "valid": True, "keypoints": points}

        data = {
            "video_id": "two-close-clicks",
            "frames": [
                make_frame(0, False),
                make_frame(1, True),
                make_frame(2, False),
                make_frame(3, True),
                make_frame(4, False),
            ],
        }
        legacy = extract_predicates(data)
        self.assertEqual(legacy["predicate_config"]["version"], "ipn_predicate_v2")
        self.assertNotIn(
            "index_click_cycle",
            {predicate["name"] for predicate in legacy["predicates"]},
        )

        v3 = extract_predicates(
            data,
            enable_temporal_factors=True,
            click_active_threshold=0.6,
            click_inactive_threshold=0.45,
            pinch_transition_excursion=10.0,
        )
        index_cycles = [
            predicate
            for predicate in v3["predicates"]
            if predicate["name"] == "index_click_cycle"
        ]
        two_finger_cycles = [
            predicate
            for predicate in v3["predicates"]
            if predicate["name"] == "two_finger_click_cycle"
        ]
        self.assertEqual(v3["predicate_config"]["version"], "ipn_predicate_v3")
        self.assertEqual(len(index_cycles), 2)
        self.assertEqual(len(two_finger_cycles), 2)
        self.assertEqual(len({item["occurrence_id"] for item in index_cycles}), 2)
        events = build_events_phase_v2(
            v3,
            min_phase_duration=1,
            min_overlap_frames=1,
        )
        graph = build_instance_graph_phase_v2(events)
        factor_occurrences = {
            occurrence["occurrence_id"]
            for node in graph["nodes"]
            if node["type"] == "Event"
            for occurrence in node["properties"].get("factor_occurrences", [])
            if occurrence.get("predicate_name") == "index_click_cycle"
        }
        self.assertEqual(
            factor_occurrences,
            {item["occurrence_id"] for item in index_cycles},
        )

    def test_hand_and_coordination_trends_skip_windows_with_missing_samples(self) -> None:
        hand_motion_frames = [
            {
                "frame_id": 0,
                "valid": True,
                "keypoints": {
                    "right_hand_root": keypoint(0.0, 0.0),
                    "right_wrist": keypoint(0.0, -0.5),
                },
            },
            {"frame_id": 1, "valid": False, "keypoints": {}},
            {
                "frame_id": 2,
                "valid": True,
                "keypoints": {
                    "right_hand_root": keypoint(0.0, 0.3),
                    "right_wrist": keypoint(0.0, -0.2),
                },
            },
        ]
        hand_predicates, _, _ = extract_motion(
            hand_motion_frames,
            min_score=0.3,
            window_size=2,
            motion_threshold=0.08,
        )
        self.assertNotIn("hand_moves_down", {predicate["name"] for predicate in hand_predicates})

        coordination_frames = [
            {
                "frame_id": 0,
                "valid": True,
                "keypoints": {
                    "left_hand_root": keypoint(-0.5, 0.0),
                    "left_wrist": keypoint(-0.5, -0.5),
                    "right_hand_root": keypoint(0.5, 0.0),
                    "right_wrist": keypoint(0.5, -0.5),
                },
            },
            {
                "frame_id": 1,
                "valid": True,
                "keypoints": {
                    "left_hand_root": keypoint(-0.4, 0.0),
                    "left_wrist": keypoint(-0.4, -0.5),
                    # The right-hand sample is missing inside the trend window.
                },
            },
            {
                "frame_id": 2,
                "valid": True,
                "keypoints": {
                    "left_hand_root": keypoint(-0.2, 0.0),
                    "left_wrist": keypoint(-0.2, -0.5),
                    "right_hand_root": keypoint(0.8, 0.0),
                    "right_wrist": keypoint(0.8, -0.5),
                },
            },
        ]
        coordination_predicates = extract_coordination(
            coordination_frames,
            min_score=0.3,
            window_size=2,
            motion_threshold=0.08,
        )
        self.assertEqual(coordination_predicates, [])

    def test_hand_motion_rejects_alternating_root_and_wrist_anchors(self) -> None:
        frames = []
        for frame_id in range(3):
            points = (
                {"right_hand_root": keypoint(0.0, 0.0)}
                if frame_id % 2 == 0
                else {"right_wrist": keypoint(0.8, 0.0)}
            )
            frames.append({"frame_id": frame_id, "valid": True, "keypoints": points})
        predicates, _, _ = extract_motion(
            frames,
            min_score=0.3,
            window_size=2,
            motion_threshold=0.08,
        )
        self.assertEqual(predicates, [])

    def test_trend_consistency_keeps_monotonic_windows_and_rejects_reversal(self) -> None:
        def make_frames(xs: list[float]) -> list[dict]:
            return [
                {
                    "frame_id": frame_id,
                    "valid": True,
                    "keypoints": {
                        "right_hand_root": keypoint(x, 0.0),
                        "right_wrist": keypoint(x, -0.5),
                    },
                }
                for frame_id, x in enumerate(xs)
            ]

        monotonic, _, _ = extract_motion(
            make_frames([0.0, 0.12, 0.24, 0.36, 0.48]),
            min_score=0.3,
            window_size=4,
            motion_threshold=0.08,
            trend_consistency=0.75,
        )
        self.assertIn("hand_moves_right", {predicate["name"] for predicate in monotonic})

        # The endpoint still moves right by enough to pass the old test, but
        # the return/reversal consumes enough total variation to fail the gate.
        reversing, _, _ = extract_motion(
            make_frames([0.0, 0.24, 0.48, 0.28, 0.60]),
            min_score=0.3,
            window_size=4,
            motion_threshold=0.08,
            trend_consistency=0.75,
        )
        self.assertNotIn("hand_moves_right", {predicate["name"] for predicate in reversing})
        self.assertAlmostEqual(trend_consistency_ratio([0.0, 0.12, 0.24, 0.36], 1), 1.0)
        self.assertLess(trend_consistency_ratio([0.0, 0.24, 0.48, 0.28, 0.60], 1), 0.75)

    def test_trend_consistency_is_recorded_and_range_checked(self) -> None:
        config = build_predicate_config(trend_consistency=0.75)
        self.assertEqual(config["trend_consistency"], 0.75)
        self.assertEqual(config["trend_consistency_formula"], "signed_net_movement_over_total_variation")
        with self.assertRaises(ValueError):
            validate_trend_consistency(-0.01)
        with self.assertRaises(ValueError):
            validate_trend_consistency(1.01)

    def test_phase_v2_resolves_joint_angle_conflict_like_legacy_builder(self) -> None:
        result = build_events_phase_v2(
            {
                "video_id": "joint-angle-conflict",
                "predicates": [
                    {
                        "name": "elbow_bent",
                        "subject": "right_arm",
                        "object": None,
                        "start_frame": 0,
                        "end_frame": 5,
                        "confidence": 0.9,
                        "evidence_keypoints": [],
                    },
                    {
                        "name": "elbow_fully_extended",
                        "subject": "right_arm",
                        "object": None,
                        "start_frame": 0,
                        "end_frame": 5,
                        "confidence": 0.4,
                        "evidence_keypoints": [],
                    },
                ],
            },
            min_phase_duration=1,
            min_overlap_frames=1,
        )
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["joint_angles"], ["elbow_bent"])
        self.assertNotIn("factor_occurrences", result["events"][0])
        graph = build_instance_graph_phase_v2(result)
        for node in graph["nodes"]:
            if node["type"] in {"Event", "Evidence"}:
                self.assertNotIn("factor_occurrences", node["properties"])

    def test_relative_finger_and_pinch_motion_survive_global_motion(self) -> None:
        frames = motion_frames()
        moving = {("right", frame_id): True for frame_id in range(len(frames))}
        predicates = extract_finger_motion(
            frames,
            min_score=0.3,
            window_size=5,
            motion_threshold=0.08,
            hand_moving=moving,
            hand_fast=moving,
            finger_motion_threshold=0.04,
            pinch_threshold=0.08,
        )
        names = {predicate["name"] for predicate in predicates}
        self.assertIn("index_finger_moves_down", names)
        self.assertIn("middle_finger_moves_down", names)
        self.assertIn("pinch_distance_increasing", names)

    def test_config_records_independent_thresholds(self) -> None:
        config = build_predicate_config(
            motion_threshold=0.08,
            hand_motion_threshold=0.12,
            finger_motion_threshold=0.04,
            pinch_threshold=0.07,
        )
        self.assertEqual(config["version"], "ipn_predicate_v2")
        self.assertEqual(
            config["thresholds"],
            {
                "legacy_motion_threshold": 0.08,
                "hand_motion_threshold": 0.12,
                "finger_motion_threshold": 0.04,
                "pinch_threshold": 0.07,
            },
        )


if __name__ == "__main__":
    unittest.main()
