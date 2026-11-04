import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "keypoint_extractor"))

from extract_mediapipe_holistic import frame_keypoints, swap_name  # noqa: E402


def landmarks(count, x0=0.5, y0=0.5):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x0 + i * 0.001, y=y0, visibility=1.0) for i in range(count)])


def fake_results(left_hand=True, right_hand=False):
    return SimpleNamespace(
        pose_landmarks=landmarks(33),
        left_hand_landmarks=landmarks(21, 0.25, 0.75) if left_hand else None,
        right_hand_landmarks=landmarks(21, 0.75, 0.75) if right_hand else None,
    )


def test_swap_name_exchanges_sides_and_keeps_midline_points():
    assert swap_name("left_shoulder") == "right_shoulder"
    assert swap_name("right_thumb3") == "left_thumb3"
    assert swap_name("nose") == "nose"
    assert swap_name("face-8") == "face-8"


def test_full_layout_without_swap_has_52_names_when_both_hands_present():
    kp = frame_keypoints(fake_results(True, True), mirror_swap=False)
    assert len(kp) == 52
    assert kp["left_hand_root"]["xy"] == [0.25 * 640.0, 0.75 * 480.0]
    assert kp["left_wrist"]["xy"] == kp["left_hand_root"]["xy"]
    assert all(item["score"] == 1.0 for item in kp.values())
    assert kp["face-8"]["xy"][0] == (0.5 + 9 * 0.001 + 0.5 + 10 * 0.001) / 2 * 640.0


def test_mirror_swap_exchanges_the_detected_hand_side():
    plain = frame_keypoints(fake_results(True, False), mirror_swap=False)
    swapped = frame_keypoints(fake_results(True, False), mirror_swap=True)
    assert "left_thumb1" in plain and "right_thumb1" not in plain
    assert "right_thumb1" in swapped and "left_thumb1" not in swapped
    assert swapped["right_thumb1"] == plain["left_thumb1"]


def test_missing_hand_uses_pose_wrist_unless_fallback_disabled():
    results = fake_results(False, False)
    with_fallback = frame_keypoints(results, mirror_swap=False, wrist_fallback=True)
    without = frame_keypoints(results, mirror_swap=False, wrist_fallback=False)
    assert with_fallback["left_wrist"]["xy"] == [(0.5 + 15 * 0.001) * 640.0, 0.5 * 480.0]
    assert "left_wrist" not in without and "right_wrist" not in without
    assert not any(name.startswith("left_thumb") for name in with_fallback)


def test_no_detection_returns_empty():
    results = SimpleNamespace(pose_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None)
    assert frame_keypoints(results) == {}
