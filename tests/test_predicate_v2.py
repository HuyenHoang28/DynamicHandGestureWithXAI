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
)
from src_neurosymbolic.events.build_events_phase_v2 import build_events_phase_v2  # noqa: E402


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
