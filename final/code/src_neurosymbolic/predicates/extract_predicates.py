import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median


SIDES = ["left", "right"]
FINGER_TIPS = ["thumb4", "forefinger4", "middle_finger4", "ring_finger4", "pinky_finger4"]

# ``motion_threshold`` remains the legacy public knob.  The IPN phase-v2
# precompute entry point uses the more conservative hand threshold and the
# finer-grained finger/pinch thresholds below, while direct callers that only
# pass the legacy argument retain the old behaviour.
DEFAULT_MOTION_THRESHOLD = 0.08
IPN_HAND_MOTION_THRESHOLD = 0.12
IPN_FINGER_MOTION_THRESHOLD = 0.04
IPN_PINCH_THRESHOLD = 0.08
# Descriptive aliases keep the configuration easy to discover for callers
# that do not need to distinguish the IPN preset name.
DEFAULT_HAND_MOTION_THRESHOLD = IPN_HAND_MOTION_THRESHOLD
DEFAULT_FINGER_MOTION_THRESHOLD = IPN_FINGER_MOTION_THRESHOLD
DEFAULT_PINCH_THRESHOLD = IPN_PINCH_THRESHOLD
# A value of zero preserves the pre-trend-consistency behaviour for direct
# Python callers.  The phase-v2 precompute entry point opts into the more
# conservative IPN preset below unless a caller supplies its own value.
DEFAULT_TREND_CONSISTENCY = 0.0
IPN_TREND_CONSISTENCY = 0.6
# Explicit aliases make the range-constrained knob discoverable to callers
# that use ``*_threshold`` terminology for the other motion settings.
DEFAULT_TREND_CONSISTENCY_THRESHOLD = DEFAULT_TREND_CONSISTENCY
IPN_TREND_CONSISTENCY_THRESHOLD = IPN_TREND_CONSISTENCY
FINGERS = {
    "thumb": ["thumb1", "thumb2", "thumb3", "thumb4"],
    "forefinger": ["forefinger1", "forefinger2", "forefinger3", "forefinger4"],
    "middle_finger": ["middle_finger1", "middle_finger2", "middle_finger3", "middle_finger4"],
    "ring_finger": ["ring_finger1", "ring_finger2", "ring_finger3", "ring_finger4"],
    "pinky_finger": ["pinky_finger1", "pinky_finger2", "pinky_finger3", "pinky_finger4"],
}


@dataclass
class Point:
    x: float
    y: float
    score: float


def load_normalized(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "frames" not in data:
        raise ValueError(f"Expected normalized keypoint object with frames: {path}")
    return data


def point(frame: dict, name: str, min_score: float) -> Point | None:
    item = (frame.get("keypoints") or {}).get(name) or {}
    xy = item.get("xy")
    score = max(0.0, min(1.0, float(item.get("score") or 0.0)))
    if not xy or len(xy) != 2 or score < min_score:
        return None
    return Point(float(xy[0]), float(xy[1]), score)


def avg_point(points: list[Point]) -> Point | None:
    good = [p for p in points if p is not None]
    if not good:
        return None
    return Point(
        sum(p.x for p in good) / len(good),
        sum(p.y for p in good) / len(good),
        sum(p.score for p in good) / len(good),
    )


def dist(a: Point, b: Point) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def angle_degrees(a: Point, b: Point, c: Point) -> float | None:
    bax = a.x - b.x
    bay = a.y - b.y
    bcx = c.x - b.x
    bcy = c.y - b.y
    ba_len = (bax * bax + bay * bay) ** 0.5
    bc_len = (bcx * bcx + bcy * bcy) ** 0.5
    if ba_len <= 1e-9 or bc_len <= 1e-9:
        return None
    cos_v = (bax * bcx + bay * bcy) / (ba_len * bc_len)
    cos_v = max(-1.0, min(1.0, cos_v))
    import math

    return math.degrees(math.acos(cos_v))


def hand_center(frame: dict, side: str, min_score: float) -> Point | None:
    root = point(frame, f"{side}_hand_root", min_score)
    wrist = point(frame, f"{side}_wrist", min_score)
    # A window must use the same anchor definition at every frame. Falling
    # back from the root/wrist midpoint to either landmark alone can turn the
    # anatomical offset between them into artificial hand motion.
    if root is None or wrist is None:
        return None
    return avg_point([root, wrist])


def face_center(frame: dict, min_score: float) -> Point | None:
    return avg_point(
        [
            point(frame, "nose", min_score),
            point(frame, "left_eye", min_score),
            point(frame, "right_eye", min_score),
            point(frame, "face-8", min_score),
        ]
    )


def add_frame_predicate(predicates: list[dict], name: str, subject: str, frame_id: int, confidence: float, evidence: list[str], obj=None) -> None:
    predicates.append(
        {
            "name": name,
            "subject": subject,
            "object": obj,
            "start_frame": frame_id,
            "end_frame": frame_id,
            "confidence": round(float(confidence), 6),
            "evidence_keypoints": evidence,
        }
    )


def confidence(*points: Point | None) -> float:
    good = [p.score for p in points if p is not None]
    if not good:
        return 0.0
    return sum(good) / len(good)


def extract_spatial_and_shape(frames: list[dict], min_score: float) -> list[dict]:
    predicates = []
    for frame in frames:
        if not frame.get("valid", False):
            continue
        frame_id = int(frame.get("frame_id", 0))
        face = face_center(frame, min_score)
        left_hand = hand_center(frame, "left", min_score)
        right_hand = hand_center(frame, "right", min_score)

        if left_hand and right_hand:
            d = dist(left_hand, right_hand)
            conf = confidence(left_hand, right_hand)
            if d < 0.30:
                add_frame_predicate(predicates, "hands_touch", "both_hands", frame_id, conf, ["left_hand_root", "right_hand_root", "left_wrist", "right_wrist"])
            if d < 0.65:
                add_frame_predicate(predicates, "hands_close", "both_hands", frame_id, conf, ["left_hand_root", "right_hand_root", "left_wrist", "right_wrist"])
            if d > 1.40:
                add_frame_predicate(predicates, "hands_far", "both_hands", frame_id, conf, ["left_hand_root", "right_hand_root", "left_wrist", "right_wrist"])

        for side in SIDES:
            hand = hand_center(frame, side, min_score)
            shoulder = point(frame, f"{side}_shoulder", min_score)
            elbow = point(frame, f"{side}_elbow", min_score)
            wrist = point(frame, f"{side}_wrist", min_score)
            other_shoulder = point(frame, f"{'right' if side == 'left' else 'left'}_shoulder", min_score)
            if not hand:
                continue

            subject = f"{side}_hand"
            if shoulder:
                if hand.y < shoulder.y - 0.15:
                    add_frame_predicate(predicates, "hand_above_shoulder", subject, frame_id, confidence(hand, shoulder), [f"{side}_hand_root", f"{side}_wrist", f"{side}_shoulder"])
                if hand.y > shoulder.y + 0.15:
                    add_frame_predicate(predicates, "hand_below_shoulder", subject, frame_id, confidence(hand, shoulder), [f"{side}_hand_root", f"{side}_wrist", f"{side}_shoulder"])
            if elbow:
                if hand.y < elbow.y - 0.10:
                    add_frame_predicate(predicates, "hand_above_elbow", subject, frame_id, confidence(hand, elbow), [f"{side}_hand_root", f"{side}_wrist", f"{side}_elbow"])
                if hand.y > elbow.y + 0.10:
                    add_frame_predicate(predicates, "hand_below_elbow", subject, frame_id, confidence(hand, elbow), [f"{side}_hand_root", f"{side}_wrist", f"{side}_elbow"])
            if face and dist(hand, face) < 0.85:
                add_frame_predicate(predicates, "hand_near_face", subject, frame_id, confidence(hand, face), [f"{side}_hand_root", f"{side}_wrist", "nose", "left_eye", "right_eye", "face-8"])
            if face and hand.y > face.y + 0.20:
                add_frame_predicate(predicates, "hand_below_face", subject, frame_id, confidence(hand, face), [f"{side}_hand_root", f"{side}_wrist", "nose", "left_eye", "right_eye", "face-8"])
            if shoulder and other_shoulder:
                body_x = (shoulder.x + other_shoulder.x) / 2.0
                if hand.x < body_x - 0.10:
                    add_frame_predicate(predicates, "hand_left_of_body_center", subject, frame_id, confidence(hand, shoulder, other_shoulder), [f"{side}_hand_root", f"{side}_wrist", "left_shoulder", "right_shoulder"])
                if hand.x > body_x + 0.10:
                    add_frame_predicate(predicates, "hand_right_of_body_center", subject, frame_id, confidence(hand, shoulder, other_shoulder), [f"{side}_hand_root", f"{side}_wrist", "left_shoulder", "right_shoulder"])

            if shoulder and elbow and wrist:
                angle = angle_degrees(shoulder, elbow, wrist)
                if angle is not None:
                    arm_subject = f"{side}_arm"
                    ev = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
                    conf = confidence(shoulder, elbow, wrist)
                    if angle < 70:
                        add_frame_predicate(predicates, "elbow_very_bent", arm_subject, frame_id, conf, ev)
                    elif angle < 120:
                        add_frame_predicate(predicates, "elbow_bent", arm_subject, frame_id, conf, ev)
                    elif angle < 155:
                        add_frame_predicate(predicates, "elbow_half_extended", arm_subject, frame_id, conf, ev)
                    else:
                        add_frame_predicate(predicates, "elbow_fully_extended", arm_subject, frame_id, conf, ev)

            add_hand_shape_predicates(predicates, frame, side, frame_id, min_score)
    return predicates


def add_hand_shape_predicates(predicates: list[dict], frame: dict, side: str, frame_id: int, min_score: float) -> None:
    root = point(frame, f"{side}_hand_root", min_score)
    if root is None:
        return

    extension_ratios = []
    extension_points = []
    for joints in FINGERS.values():
        pts = [root] + [point(frame, f"{side}_{joint}", min_score) for joint in joints]
        if any(p is None for p in pts):
            continue
        path_len = sum(dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        if path_len <= 1e-9:
            continue
        extension_ratios.append(dist(root, pts[-1]) / path_len)
        extension_points.extend(pts[1:])

    if len(extension_ratios) < 3:
        return

    mean_extension = sum(extension_ratios) / len(extension_ratios)
    conf = confidence(root, *extension_points)
    subject = f"{side}_hand"
    if mean_extension > 0.86:
        add_frame_predicate(predicates, "hand_open", subject, frame_id, conf, [f"{side}_hand_root"] + [f"{side}_{tip}" for tip in FINGER_TIPS])
    if mean_extension < 0.45:
        add_frame_predicate(predicates, "hand_closed", subject, frame_id, conf, [f"{side}_hand_root"] + [f"{side}_{tip}" for tip in FINGER_TIPS])
    if mean_extension < 0.75:
        add_frame_predicate(predicates, "hand_fist_like", subject, frame_id, conf, [f"{side}_hand_root"] + [f"{side}_{tip}" for tip in FINGER_TIPS])
    if 0.75 <= mean_extension < 0.86:
        add_frame_predicate(predicates, "fingers_slightly_curled", subject, frame_id, conf, [f"{side}_hand_root"] + [f"{side}_{tip}" for tip in FINGER_TIPS])

    add_single_finger_extension(predicates, frame, side, "forefinger", "index_extended", frame_id, min_score, 0.65)
    add_single_finger_extension(predicates, frame, side, "thumb", "thumb_extended", frame_id, min_score, 0.55)
    add_index_middle_extension(predicates, frame, side, frame_id, min_score)


def finger_extension_ratio(frame: dict, side: str, finger: str, min_score: float) -> tuple[float, float, list[str]] | None:
    root = point(frame, f"{side}_hand_root", min_score)
    joints = FINGERS[finger]
    pts = [root] + [point(frame, f"{side}_{joint}", min_score) for joint in joints]
    if any(p is None for p in pts):
        return None
    path_len = sum(dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    if path_len <= 1e-9:
        return None
    evidence = [f"{side}_hand_root"] + [f"{side}_{joint}" for joint in joints]
    return dist(root, pts[-1]) / path_len, confidence(*pts), evidence


def add_index_middle_extension(predicates: list[dict], frame: dict, side: str, frame_id: int, min_score: float) -> None:
    index_ratio = finger_extension_ratio(frame, side, "forefinger", min_score)
    middle_ratio = finger_extension_ratio(frame, side, "middle_finger", min_score)
    ring_ratio = finger_extension_ratio(frame, side, "ring_finger", min_score)
    pinky_ratio = finger_extension_ratio(frame, side, "pinky_finger", min_score)
    if not index_ratio or not middle_ratio:
        return

    other_ratios = [item[0] for item in [ring_ratio, pinky_ratio] if item is not None]
    other_fingers_not_dominant = not other_ratios or max(other_ratios) < 0.75
    if index_ratio[0] > 0.62 and middle_ratio[0] > 0.62 and other_fingers_not_dominant:
        evidence = sorted(set(index_ratio[2] + middle_ratio[2]))
        conf = (index_ratio[1] + middle_ratio[1]) / 2.0
        add_frame_predicate(predicates, "index_middle_extended", f"{side}_hand", frame_id, conf, evidence)


def add_single_finger_extension(
    predicates: list[dict],
    frame: dict,
    side: str,
    finger: str,
    predicate_name: str,
    frame_id: int,
    min_score: float,
    threshold: float,
) -> None:
    root = point(frame, f"{side}_hand_root", min_score)
    joints = FINGERS[finger]
    pts = [root] + [point(frame, f"{side}_{joint}", min_score) for joint in joints]
    if any(p is None for p in pts):
        return
    path_len = sum(dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    if path_len <= 1e-9:
        return
    extension_ratio = dist(root, pts[-1]) / path_len
    if extension_ratio > threshold:
        subject = f"{side}_hand"
        add_frame_predicate(
            predicates,
            predicate_name,
            subject,
            frame_id,
            confidence(*pts),
            [f"{side}_hand_root"] + [f"{side}_{joint}" for joint in joints],
        )


def delta(values: list[Point | None], start: int, end: int, axis: str) -> float | None:
    a = values[start]
    b = values[end]
    if a is None or b is None:
        return None
    return (b.y - a.y) if axis == "y" else (b.x - a.x)


def window_visible(values: list[object | None], start: int, end: int) -> bool:
    """Require every sample in a trend window to pass visibility checks."""

    return not any(value is None for value in values[start : end + 1])


def rel_tip(frame: dict, side: str, tip_name: str, min_score: float) -> Point | None:
    tip = point(frame, f"{side}_{tip_name}", min_score)
    root = point(frame, f"{side}_hand_root", min_score)
    if tip is None or root is None:
        return None
    return Point(tip.x - root.x, tip.y - root.y, min(tip.score, root.score))


def local_hand_scale(frame: dict, side: str, min_score: float) -> tuple[float, float, list[str]] | None:
    """Return a robust palm/hand scale and its visibility evidence.

    The shoulder-width normalization used by the pose pipeline is useful for
    body-relative predicates, but it is not a sufficiently local reference
    for a thumb/index pinch.  Use only relatively rigid palm landmarks here;
    fingertip landmarks are deliberately excluded so finger motion cannot
    change the denominator.  A median over the visible anchor-pair lengths
    makes the estimate tolerant of one noisy or missing palm keypoint.
    """

    pair_names = [
        ("hand_root", "wrist"),
        ("forefinger1", "pinky_finger1"),
        ("wrist", "middle_finger1"),
        ("hand_root", "middle_finger1"),
    ]
    distances: list[float] = []
    anchor_points: list[Point] = []
    evidence: list[str] = []
    for first_name, second_name in pair_names:
        first = point(frame, f"{side}_{first_name}", min_score)
        second = point(frame, f"{side}_{second_name}", min_score)
        if first is None or second is None:
            continue
        pair_distance = dist(first, second)
        if pair_distance <= 1e-6:
            continue
        distances.append(pair_distance)
        anchor_points.extend([first, second])
        evidence.extend([f"{side}_{first_name}", f"{side}_{second_name}"])

    if not distances:
        return None
    scale = float(median(distances))
    if scale <= 1e-6:
        return None
    return scale, confidence(*anchor_points), sorted(set(evidence))


def normalized_pinch_distance(frame: dict, side: str, min_score: float) -> tuple[float, float, list[str]] | None:
    """Return thumb/index distance normalized by the local palm scale."""

    thumb = point(frame, f"{side}_thumb4", min_score)
    index = point(frame, f"{side}_forefinger4", min_score)
    scale_data = local_hand_scale(frame, side, min_score)
    if thumb is None or index is None or scale_data is None:
        return None
    scale, scale_confidence, scale_evidence = scale_data
    normalized_distance = dist(thumb, index) / scale
    # Include endpoint visibility and palm-scale visibility in the trend
    # confidence.  A missing/low-score anchor never reaches this point.
    trend_confidence = (confidence(thumb, index) + scale_confidence) / 2.0
    evidence = sorted(
        set(scale_evidence + [f"{side}_thumb4", f"{side}_forefinger4"])
    )
    return normalized_distance, trend_confidence, evidence


def direction_changes(series: list[float], min_abs_delta: float) -> int:
    signs = []
    for value in series:
        if abs(value) < min_abs_delta:
            continue
        signs.append(1 if value > 0 else -1)
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def validate_trend_consistency(value: float | None, default: float = DEFAULT_TREND_CONSISTENCY) -> float:
    """Validate and resolve the directional-trend consistency threshold.

    The threshold is the minimum ratio of signed net movement to total
    variation in a candidate window.  ``0`` disables the additional gate,
    preserving the legacy endpoint-only behaviour.  Values in ``[0, 1]`` are
    valid; rejecting non-finite values here keeps CLI and Python callers
    consistent.
    """

    resolved = default if value is None else value
    try:
        resolved_float = float(resolved)
    except (TypeError, ValueError) as exc:
        raise ValueError("trend_consistency must be a finite value in [0, 1]") from exc
    if not math.isfinite(resolved_float) or not 0.0 <= resolved_float <= 1.0:
        raise ValueError("trend_consistency must be a finite value in [0, 1]")
    return resolved_float


def trend_consistency_ratio(values: list[float], direction: int = 1) -> float:
    """Return directional signed movement divided by total variation.

    ``direction`` is ``1`` for an increasing trend and ``-1`` for a
    decreasing trend.  A monotonic window scores ``1``; a window that reaches
    the same endpoint after substantial reversal scores lower.  A flat or
    empty window scores ``0`` because it provides no directional evidence.
    """

    if direction not in (-1, 1):
        raise ValueError("direction must be 1 or -1")
    if len(values) < 2:
        return 0.0
    steps = [float(values[index + 1]) - float(values[index]) for index in range(len(values) - 1)]
    total_variation = sum(abs(step) for step in steps)
    if total_variation <= 1e-12:
        return 0.0
    signed_movement = direction * sum(steps)
    # A candidate's endpoint displacement already establishes its direction;
    # clamping makes this helper safe and interpretable when used directly.
    ratio = max(0.0, min(1.0, signed_movement / total_variation))
    return 1.0 if ratio >= 1.0 - 1e-12 else ratio


def _window_scalar_trend_consistent(
    values: list[float | None],
    start: int,
    end: int,
    direction: int,
    threshold: float,
) -> bool:
    if threshold <= 0.0:
        return True
    window = values[start : end + 1]
    if any(value is None for value in window):
        return False
    return trend_consistency_ratio([float(value) for value in window], direction) + 1e-12 >= threshold


def _window_point_axis_trend_consistent(
    values: list[Point | None],
    start: int,
    end: int,
    axis: str,
    direction: int,
    threshold: float,
) -> bool:
    if threshold <= 0.0:
        return True
    window = values[start : end + 1]
    if any(value is None for value in window):
        return False
    coordinates = [float(getattr(value, axis)) for value in window]
    return trend_consistency_ratio(coordinates, direction) + 1e-12 >= threshold


def add_window_predicate(predicates: list[dict], name: str, subject: str, start_frame: int, end_frame: int, confidence_value: float, evidence: list[str]) -> None:
    predicates.append(
        {
            "name": name,
            "subject": subject,
            "object": None,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "confidence": round(float(confidence_value), 6),
            "evidence_keypoints": evidence,
        }
    )


def extract_motion(
    frames: list[dict],
    min_score: float,
    window_size: int,
    motion_threshold: float,
    hand_motion_threshold: float | None = None,
    trend_consistency: float | None = DEFAULT_TREND_CONSISTENCY,
) -> tuple[list[dict], dict[tuple[str, int], bool], dict[tuple[str, int], bool]]:
    if hand_motion_threshold is not None:
        motion_threshold = hand_motion_threshold
    trend_consistency = validate_trend_consistency(trend_consistency)
    predicates = []
    hand_moving = {}
    hand_fast = {}
    if len(frames) <= window_size:
        return predicates, hand_moving, hand_fast

    for side in SIDES:
        centers = [hand_center(frame, side, min_score) if frame.get("valid", False) else None for frame in frames]
        for start in range(0, len(frames) - window_size):
            end = start + window_size
            if not window_visible(centers, start, end):
                continue
            start_frame = int(frames[start].get("frame_id", start))
            end_frame = int(frames[end].get("frame_id", end))
            dx = delta(centers, start, end, "x")
            dy = delta(centers, start, end, "y")
            if dx is None or dy is None:
                continue
            displacement = (dx * dx + dy * dy) ** 0.5
            speed = displacement / max(window_size, 1)
            subject = f"{side}_hand"
            ev = [f"{side}_hand_root", f"{side}_wrist"]
            conf = confidence(centers[start], centers[end])
            moving = False
            if dy < -motion_threshold and _window_point_axis_trend_consistent(
                centers, start, end, "y", -1, trend_consistency
            ):
                add_window_predicate(predicates, "hand_moves_up", subject, start_frame, end_frame, conf, ev)
                moving = True
            if dy > motion_threshold and _window_point_axis_trend_consistent(
                centers, start, end, "y", 1, trend_consistency
            ):
                add_window_predicate(predicates, "hand_moves_down", subject, start_frame, end_frame, conf, ev)
                moving = True
            if dx < -motion_threshold and _window_point_axis_trend_consistent(
                centers, start, end, "x", -1, trend_consistency
            ):
                add_window_predicate(predicates, "hand_moves_left", subject, start_frame, end_frame, conf, ev)
                moving = True
            if dx > motion_threshold and _window_point_axis_trend_consistent(
                centers, start, end, "x", 1, trend_consistency
            ):
                add_window_predicate(predicates, "hand_moves_right", subject, start_frame, end_frame, conf, ev)
                moving = True
            diagonal_trend = (
                abs(dx) > motion_threshold
                and abs(dy) > motion_threshold
                and _window_point_axis_trend_consistent(
                    centers, start, end, "x", -1 if dx < 0 else 1, trend_consistency
                )
                and _window_point_axis_trend_consistent(
                    centers, start, end, "y", -1 if dy < 0 else 1, trend_consistency
                )
            )
            if diagonal_trend:
                vertical = "up" if dy < 0 else "down"
                horizontal = "left" if dx < 0 else "right"
                add_window_predicate(predicates, f"hand_moves_diagonal_{vertical}_{horizontal}", subject, start_frame, end_frame, conf, ev)
                moving = True

            start_frame_data = frames[start]
            end_frame_data = frames[end]
            start_shoulder = point(start_frame_data, f"{side}_shoulder", min_score)
            start_other_shoulder = point(start_frame_data, f"{'right' if side == 'left' else 'left'}_shoulder", min_score)
            end_shoulder = point(end_frame_data, f"{side}_shoulder", min_score)
            end_other_shoulder = point(end_frame_data, f"{'right' if side == 'left' else 'left'}_shoulder", min_score)
            if centers[start] and centers[end] and start_shoulder and start_other_shoulder and end_shoulder and end_other_shoulder:
                start_body_x = (start_shoulder.x + start_other_shoulder.x) / 2.0
                end_body_x = (end_shoulder.x + end_other_shoulder.x) / 2.0
                start_side = signed_direction(centers[start].x - start_body_x, motion_threshold)
                end_side = signed_direction(centers[end].x - end_body_x, motion_threshold)
                if start_side != 0 and end_side != 0 and start_side != end_side:
                    add_window_predicate(predicates, "hand_crosses_body_center", subject, start_frame, end_frame, conf, ev + ["left_shoulder", "right_shoulder"])
                    moving = True
            if displacement < motion_threshold:
                add_window_predicate(predicates, "hand_stationary", subject, start_frame, end_frame, conf, ev)
            if displacement > motion_threshold * 2.0:
                add_window_predicate(predicates, "hand_fast_motion", subject, start_frame, end_frame, conf, ev)
                moving = True
                for idx in range(start, end + 1):
                    hand_fast[(side, idx)] = True
            if moving:
                for idx in range(start, end + 1):
                    hand_moving[(side, idx)] = True

        xs = [p.x for p in centers if p is not None]
        ys = [p.y for p in centers if p is not None]
        if len(xs) > window_size:
            # Window-level oscillation is intentionally conservative for the first version.
            for start in range(0, len(frames) - window_size):
                end = start + window_size
                local = centers[start : end + 1]
                if any(p is None for p in local):
                    continue
                dxs = [local[i + 1].x - local[i].x for i in range(len(local) - 1)]
                dys = [local[i + 1].y - local[i].y for i in range(len(local) - 1)]
                start_frame = int(frames[start].get("frame_id", start))
                end_frame = int(frames[end].get("frame_id", end))
                subject = f"{side}_hand"
                ev = [f"{side}_hand_root", f"{side}_wrist"]
                conf = confidence(local[0], local[-1])
                if direction_changes(dxs, motion_threshold / 2.0) >= 2:
                    add_window_predicate(predicates, "hand_oscillates_horizontal", subject, start_frame, end_frame, conf, ev)
                if direction_changes(dys, motion_threshold / 2.0) >= 2:
                    add_window_predicate(predicates, "hand_oscillates_vertical", subject, start_frame, end_frame, conf, ev)

    return predicates, hand_moving, hand_fast


def mean_pairwise_distance(points: list[Point]) -> float | None:
    if len(points) < 2:
        return None
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(dist(points[i], points[j]))
    return sum(distances) / len(distances) if distances else None


def fingertip_spread(frame: dict, side: str, min_score: float) -> tuple[float, float] | None:
    root = point(frame, f"{side}_hand_root", min_score)
    if root is None:
        return None
    rel_tips = []
    scores = [root.score]
    for tip_name in FINGER_TIPS:
        tip = rel_tip(frame, side, tip_name, min_score)
        if tip is None:
            continue
        rel_tips.append(tip)
        scores.append(tip.score)
    if len(rel_tips) < 3:
        return None
    spread = mean_pairwise_distance(rel_tips)
    if spread is None:
        return None
    return spread, sum(scores) / len(scores)


def extract_finger_motion(
    frames: list[dict],
    min_score: float,
    window_size: int,
    motion_threshold: float,
    hand_moving: dict[tuple[str, int], bool],
    hand_fast: dict[tuple[str, int], bool],
    finger_motion_threshold: float | None = None,
    pinch_threshold: float | None = None,
    trend_consistency: float | None = DEFAULT_TREND_CONSISTENCY,
) -> list[dict]:
    """Extract motion in hand-relative coordinates.

    ``hand_moving`` and ``hand_fast`` remain accepted for compatibility with
    earlier callers.  They are intentionally not used as suppression masks:
    the tip-minus-root coordinates below separate finger motion from a moving
    hand, and visibility/confidence checks on each endpoint decide whether a
    predicate is emitted.
    """

    if finger_motion_threshold is None:
        finger_motion_threshold = motion_threshold
    if pinch_threshold is None:
        pinch_threshold = finger_motion_threshold
    trend_consistency = validate_trend_consistency(trend_consistency)
    # Keep the legacy parameters in the signature without allowing global
    # motion flags to hide otherwise observable relative finger motion.
    _ = hand_moving, hand_fast
    predicates = []
    if len(frames) <= window_size:
        return predicates

    for side in SIDES:
        index_rel = [rel_tip(frame, side, "forefinger4", min_score) if frame.get("valid", False) else None for frame in frames]
        middle_rel = [rel_tip(frame, side, "middle_finger4", min_score) if frame.get("valid", False) else None for frame in frames]
        thumb_rel = [rel_tip(frame, side, "thumb4", min_score) if frame.get("valid", False) else None for frame in frames]
        pinch_distances = [
            normalized_pinch_distance(frame, side, min_score) if frame.get("valid", False) else None
            for frame in frames
        ]
        subject = f"{side}_hand"
        for start in range(0, len(frames) - window_size):
            end = start + window_size
            start_frame = int(frames[start].get("frame_id", start))
            end_frame = int(frames[end].get("frame_id", end))

            idx_dx = delta(index_rel, start, end, "x")
            idx_dy = delta(index_rel, start, end, "y")
            if idx_dx is not None and idx_dy is not None and window_visible(index_rel, start, end):
                conf = confidence(index_rel[start], index_rel[end])
                ev = [f"{side}_hand_root", f"{side}_forefinger4"]
                if idx_dy < -finger_motion_threshold and _window_point_axis_trend_consistent(
                    index_rel, start, end, "y", -1, trend_consistency
                ):
                    add_window_predicate(predicates, "index_finger_moves_up", subject, start_frame, end_frame, conf, ev)
                if idx_dy > finger_motion_threshold and _window_point_axis_trend_consistent(
                    index_rel, start, end, "y", 1, trend_consistency
                ):
                    add_window_predicate(predicates, "index_finger_moves_down", subject, start_frame, end_frame, conf, ev)
                if idx_dx < -finger_motion_threshold and _window_point_axis_trend_consistent(
                    index_rel, start, end, "x", -1, trend_consistency
                ):
                    add_window_predicate(predicates, "index_finger_moves_left", subject, start_frame, end_frame, conf, ev)
                if idx_dx > finger_motion_threshold and _window_point_axis_trend_consistent(
                    index_rel, start, end, "x", 1, trend_consistency
                ):
                    add_window_predicate(predicates, "index_finger_moves_right", subject, start_frame, end_frame, conf, ev)
                diagonal_trend = (
                    abs(idx_dx) > finger_motion_threshold
                    and abs(idx_dy) > finger_motion_threshold
                    and _window_point_axis_trend_consistent(
                        index_rel, start, end, "x", -1 if idx_dx < 0 else 1, trend_consistency
                    )
                    and _window_point_axis_trend_consistent(
                        index_rel, start, end, "y", -1 if idx_dy < 0 else 1, trend_consistency
                    )
                )
                if diagonal_trend:
                    vertical = "up" if idx_dy < 0 else "down"
                    horizontal = "left" if idx_dx < 0 else "right"
                    add_window_predicate(predicates, f"index_finger_moves_diagonal_{vertical}_{horizontal}", subject, start_frame, end_frame, conf, ev)

                local = index_rel[start : end + 1]
                if not any(p is None for p in local):
                    dxs = [local[i + 1].x - local[i].x for i in range(len(local) - 1)]
                    dys = [local[i + 1].y - local[i].y for i in range(len(local) - 1)]
                    if (
                        direction_changes(dxs, finger_motion_threshold / 2.0) >= 2
                        or direction_changes(dys, finger_motion_threshold / 2.0) >= 2
                    ):
                        add_window_predicate(predicates, "index_finger_oscillates", subject, start_frame, end_frame, conf, ev)

            middle_dy = delta(middle_rel, start, end, "y")
            if middle_dy is not None and window_visible(middle_rel, start, end):
                conf = confidence(middle_rel[start], middle_rel[end])
                ev = [f"{side}_hand_root", f"{side}_middle_finger4"]
                if middle_dy < -finger_motion_threshold and _window_point_axis_trend_consistent(
                    middle_rel, start, end, "y", -1, trend_consistency
                ):
                    add_window_predicate(predicates, "middle_finger_moves_up", subject, start_frame, end_frame, conf, ev)
                if middle_dy > finger_motion_threshold and _window_point_axis_trend_consistent(
                    middle_rel, start, end, "y", 1, trend_consistency
                ):
                    add_window_predicate(predicates, "middle_finger_moves_down", subject, start_frame, end_frame, conf, ev)

            th_dy = delta(thumb_rel, start, end, "y")
            if th_dy is not None and window_visible(thumb_rel, start, end):
                conf = confidence(thumb_rel[start], thumb_rel[end])
                ev = [f"{side}_hand_root", f"{side}_thumb4"]
                if th_dy < -finger_motion_threshold and _window_point_axis_trend_consistent(
                    thumb_rel, start, end, "y", -1, trend_consistency
                ):
                    add_window_predicate(predicates, "thumb_moves_up", subject, start_frame, end_frame, conf, ev)
                if th_dy > finger_motion_threshold and _window_point_axis_trend_consistent(
                    thumb_rel, start, end, "y", 1, trend_consistency
                ):
                    add_window_predicate(predicates, "thumb_moves_down", subject, start_frame, end_frame, conf, ev)

            start_pinch = pinch_distances[start]
            end_pinch = pinch_distances[end]
            if (
                start_pinch is not None
                and end_pinch is not None
                and window_visible(pinch_distances, start, end)
            ):
                pinch_delta = end_pinch[0] - start_pinch[0]
                conf = (start_pinch[1] + end_pinch[1]) / 2.0
                ev = sorted(set(start_pinch[2] + end_pinch[2]))
                pinch_values = [value[0] for value in pinch_distances[start : end + 1]]
                if pinch_delta > pinch_threshold and _window_scalar_trend_consistent(
                    pinch_values, 0, len(pinch_values) - 1, 1, trend_consistency
                ):
                    add_window_predicate(predicates, "pinch_distance_increasing", subject, start_frame, end_frame, conf, ev)
                if pinch_delta < -pinch_threshold and _window_scalar_trend_consistent(
                    pinch_values, 0, len(pinch_values) - 1, -1, trend_consistency
                ):
                    add_window_predicate(predicates, "pinch_distance_decreasing", subject, start_frame, end_frame, conf, ev)

        spreads = [fingertip_spread(frame, side, min_score) if frame.get("valid", False) else None for frame in frames]
        for start in range(0, len(frames) - window_size):
            end = start + window_size
            start_spread = spreads[start]
            end_spread = spreads[end]
            if (
                start_spread is None
                or end_spread is None
                or not window_visible(spreads, start, end)
            ):
                continue
            delta_spread = end_spread[0] - start_spread[0]
            conf = (start_spread[1] + end_spread[1]) / 2.0
            start_frame = int(frames[start].get("frame_id", start))
            end_frame = int(frames[end].get("frame_id", end))
            ev = [f"{side}_{tip}" for tip in FINGER_TIPS]
            spread_values = [value[0] for value in spreads[start : end + 1]]
            if delta_spread > finger_motion_threshold and _window_scalar_trend_consistent(
                spread_values, 0, len(spread_values) - 1, 1, trend_consistency
            ):
                add_window_predicate(predicates, "fingers_spread_increasing", subject, start_frame, end_frame, conf, ev)
            if delta_spread < -finger_motion_threshold and _window_scalar_trend_consistent(
                spread_values, 0, len(spread_values) - 1, -1, trend_consistency
            ):
                add_window_predicate(predicates, "fingers_spread_decreasing", subject, start_frame, end_frame, conf, ev)

    return predicates


def signed_direction(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def extract_coordination(
    frames: list[dict],
    min_score: float,
    window_size: int,
    motion_threshold: float,
    hand_motion_threshold: float | None = None,
    trend_consistency: float | None = DEFAULT_TREND_CONSISTENCY,
) -> list[dict]:
    if hand_motion_threshold is not None:
        motion_threshold = hand_motion_threshold
    trend_consistency = validate_trend_consistency(trend_consistency)
    predicates = []
    if len(frames) <= window_size:
        return predicates

    left_centers = [hand_center(frame, "left", min_score) if frame.get("valid", False) else None for frame in frames]
    right_centers = [hand_center(frame, "right", min_score) if frame.get("valid", False) else None for frame in frames]

    for start in range(0, len(frames) - window_size):
        end = start + window_size
        if not window_visible(left_centers, start, end) or not window_visible(right_centers, start, end):
            continue
        left_dx = delta(left_centers, start, end, "x")
        right_dx = delta(right_centers, start, end, "x")
        left_dy = delta(left_centers, start, end, "y")
        right_dy = delta(right_centers, start, end, "y")

        if left_dx is None or right_dx is None or left_dy is None or right_dy is None:
            continue

        start_frame = int(frames[start].get("frame_id", start))
        end_frame = int(frames[end].get("frame_id", end))
        conf = confidence(left_centers[start], left_centers[end], right_centers[start], right_centers[end])
        evidence = ["left_hand_root", "left_wrist", "right_hand_root", "right_wrist"]

        left_sign = signed_direction(left_dx, motion_threshold)
        right_sign = signed_direction(right_dx, motion_threshold)
        if left_sign and not _window_point_axis_trend_consistent(
            left_centers, start, end, "x", left_sign, trend_consistency
        ):
            left_sign = 0
        if right_sign and not _window_point_axis_trend_consistent(
            right_centers, start, end, "x", right_sign, trend_consistency
        ):
            right_sign = 0
        if left_sign != 0 and right_sign != 0:
            name = (
                "hands_move_same_direction_horizontal"
                if left_sign == right_sign
                else "hands_move_opposite_direction_horizontal"
            )
            add_window_predicate(predicates, name, "both_hands", start_frame, end_frame, conf, evidence)
            if left_sign < 0 and right_sign > 0:
                add_window_predicate(predicates, "hands_move_apart_horizontal", "both_hands", start_frame, end_frame, conf, evidence)
            if left_sign > 0 and right_sign < 0:
                add_window_predicate(predicates, "hands_move_together_horizontal", "both_hands", start_frame, end_frame, conf, evidence)

        left_vsign = signed_direction(left_dy, motion_threshold)
        right_vsign = signed_direction(right_dy, motion_threshold)
        if left_vsign and not _window_point_axis_trend_consistent(
            left_centers, start, end, "y", left_vsign, trend_consistency
        ):
            left_vsign = 0
        if right_vsign and not _window_point_axis_trend_consistent(
            right_centers, start, end, "y", right_vsign, trend_consistency
        ):
            right_vsign = 0
        if left_vsign != 0 and left_vsign == right_vsign:
            add_window_predicate(predicates, "hands_move_same_direction_vertical", "both_hands", start_frame, end_frame, conf, evidence)
        if left_vsign != 0 and right_vsign != 0 and left_vsign == -right_vsign:
            add_window_predicate(predicates, "hands_move_opposite_direction_vertical", "both_hands", start_frame, end_frame, conf, evidence)

    return predicates


def extract_sweep(frames: list[dict], min_score: float, sweep_threshold: float = 2.0) -> list[dict]:
    predicates = []
    hand_ranges = {}
    hand_confidences = {}

    for side in SIDES:
        centers = [hand_center(frame, side, min_score) if frame.get("valid", False) else None for frame in frames]
        valid = [(idx, point_value) for idx, point_value in enumerate(centers) if point_value is not None]
        if len(valid) < 2:
            continue
        xs = [point_value.x for _, point_value in valid]
        hand_range = max(xs) - min(xs)
        hand_ranges[side] = hand_range
        hand_confidences[side] = sum(point_value.score for _, point_value in valid) / len(valid)
        if hand_range > sweep_threshold:
            start_frame = int(frames[valid[0][0]].get("frame_id", valid[0][0]))
            end_frame = int(frames[valid[-1][0]].get("frame_id", valid[-1][0]))
            add_window_predicate(
                predicates,
                "hand_large_horizontal_sweep",
                f"{side}_hand",
                start_frame,
                end_frame,
                hand_confidences[side],
                [f"{side}_hand_root", f"{side}_wrist"],
            )

    if "left" in hand_ranges and "right" in hand_ranges:
        mean_range = (hand_ranges["left"] + hand_ranges["right"]) / 2.0
        if mean_range > sweep_threshold:
            valid_frames = [
                idx
                for idx, frame in enumerate(frames)
                if frame.get("valid", False)
                and hand_center(frame, "left", min_score) is not None
                and hand_center(frame, "right", min_score) is not None
            ]
            if valid_frames:
                start = valid_frames[0]
                end = valid_frames[-1]
                conf = (hand_confidences["left"] + hand_confidences["right"]) / 2.0
                add_window_predicate(
                    predicates,
                    "hands_large_horizontal_sweep",
                    "both_hands",
                    int(frames[start].get("frame_id", start)),
                    int(frames[end].get("frame_id", end)),
                    conf,
                    ["left_hand_root", "left_wrist", "right_hand_root", "right_wrist"],
                )

    return predicates


def merge_adjacent(predicates: list[dict]) -> list[dict]:
    predicates = sorted(predicates, key=lambda p: (p["name"], p["subject"], p.get("object") or "", p["start_frame"], p["end_frame"]))
    merged = []
    for pred in predicates:
        if (
            merged
            and merged[-1]["name"] == pred["name"]
            and merged[-1]["subject"] == pred["subject"]
            and merged[-1].get("object") == pred.get("object")
            and pred["start_frame"] <= merged[-1]["end_frame"] + 1
        ):
            merged[-1]["end_frame"] = max(merged[-1]["end_frame"], pred["end_frame"])
            merged[-1]["confidence"] = round((merged[-1]["confidence"] + pred["confidence"]) / 2.0, 6)
            merged[-1]["evidence_keypoints"] = sorted(set(merged[-1]["evidence_keypoints"]) | set(pred["evidence_keypoints"]))
        else:
            merged.append(dict(pred))
    return sorted(merged, key=lambda p: (p["start_frame"], p["end_frame"], p["name"], p["subject"]))


def build_predicate_config(
    min_score: float = 0.3,
    window_size: int = 5,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    hand_motion_threshold: float | None = None,
    finger_motion_threshold: float | None = None,
    pinch_threshold: float | None = None,
    trend_consistency: float | None = DEFAULT_TREND_CONSISTENCY,
    trend_consistency_threshold: float | None = None,
) -> dict:
    resolved_hand_threshold = motion_threshold if hand_motion_threshold is None else hand_motion_threshold
    resolved_finger_threshold = motion_threshold if finger_motion_threshold is None else finger_motion_threshold
    resolved_pinch_threshold = resolved_finger_threshold if pinch_threshold is None else pinch_threshold
    if trend_consistency_threshold is not None:
        if trend_consistency is not None and trend_consistency != DEFAULT_TREND_CONSISTENCY and float(trend_consistency) != float(trend_consistency_threshold):
            raise ValueError("specify only one of trend_consistency and trend_consistency_threshold")
        trend_consistency = trend_consistency_threshold
    resolved_trend_consistency = validate_trend_consistency(trend_consistency)
    return {
        "version": "ipn_predicate_v2",
        "min_score": float(min_score),
        "window_size": int(window_size),
        "thresholds": {
            "legacy_motion_threshold": float(motion_threshold),
            "hand_motion_threshold": float(resolved_hand_threshold),
            "finger_motion_threshold": float(resolved_finger_threshold),
            "pinch_threshold": float(resolved_pinch_threshold),
        },
        "trend_consistency": resolved_trend_consistency,
        "trend_consistency_threshold": resolved_trend_consistency,
        "trend_consistency_formula": "signed_net_movement_over_total_variation",
        "finger_coordinates": "tip_minus_hand_root",
        "pinch_distance": "thumb4_forefinger4_distance_divided_by_median_local_palm_scale",
        "visibility": "keypoint_score_must_meet_min_score",
    }


def extract_predicates(
    data: dict,
    min_score: float = 0.3,
    window_size: int = 5,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    hand_motion_threshold: float | None = None,
    finger_motion_threshold: float | None = None,
    pinch_threshold: float | None = None,
    trend_consistency: float | None = DEFAULT_TREND_CONSISTENCY,
    trend_consistency_threshold: float | None = None,
) -> dict:
    """Extract predicates while retaining the legacy ``motion_threshold`` API.

    When the new thresholds are omitted, all motion families use the legacy
    value.  This keeps existing Python callers deterministic; IPN precompute
    supplies its tuned values explicitly.  ``trend_consistency`` is a
    ``[0, 1]`` minimum ratio of signed per-step movement to total variation;
    its default of ``0`` disables the extra gate for legacy callers.
    """

    predicate_config = build_predicate_config(
        min_score=min_score,
        window_size=window_size,
        motion_threshold=motion_threshold,
        hand_motion_threshold=hand_motion_threshold,
        finger_motion_threshold=finger_motion_threshold,
        pinch_threshold=pinch_threshold,
        trend_consistency=trend_consistency,
        trend_consistency_threshold=trend_consistency_threshold,
    )
    thresholds = predicate_config["thresholds"]
    resolved_hand_threshold = thresholds["hand_motion_threshold"]
    resolved_finger_threshold = thresholds["finger_motion_threshold"]
    resolved_pinch_threshold = thresholds["pinch_threshold"]
    frames = data.get("frames") or []
    spatial = extract_spatial_and_shape(frames, min_score)
    motion, hand_moving, hand_fast = extract_motion(
        frames,
        min_score,
        window_size,
        resolved_hand_threshold,
        trend_consistency=predicate_config["trend_consistency"],
    )
    finger_motion = extract_finger_motion(
        frames,
        min_score,
        window_size,
        resolved_finger_threshold,
        hand_moving,
        hand_fast,
        finger_motion_threshold=resolved_finger_threshold,
        pinch_threshold=resolved_pinch_threshold,
        trend_consistency=predicate_config["trend_consistency"],
    )
    coordination = extract_coordination(
        frames,
        min_score,
        window_size,
        resolved_hand_threshold,
        trend_consistency=predicate_config["trend_consistency"],
    )
    sweep = extract_sweep(frames, min_score)
    predicates = merge_adjacent(spatial + motion + finger_motion + coordination + sweep)
    return {
        "video_id": data.get("video_id", ""),
        "source_path": data.get("source_path", ""),
        "predicates": predicates,
        "predicate_config": predicate_config,
    }


def write_json(data: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract symbolic predicates from normalized 52-keypoint JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Input normalized keypoint JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output predicate JSON.")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD, help="Legacy threshold fallback for all motion families.")
    parser.add_argument("--hand-motion-threshold", type=float, default=None)
    parser.add_argument("--finger-motion-threshold", type=float, default=None)
    parser.add_argument("--pinch-threshold", "--pinch-motion-threshold", dest="pinch_threshold", type=float, default=None)
    parser.add_argument(
        "--trend-consistency",
        "--trend-consistency-threshold",
        dest="trend_consistency",
        type=float,
        default=DEFAULT_TREND_CONSISTENCY,
        help="Minimum signed movement / total variation ratio in [0, 1]; 0 preserves endpoint-only trends.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_normalized(args.input)
    out = extract_predicates(
        data,
        args.min_score,
        args.window_size,
        args.motion_threshold,
        hand_motion_threshold=args.hand_motion_threshold,
        finger_motion_threshold=args.finger_motion_threshold,
        pinch_threshold=args.pinch_threshold,
        trend_consistency=args.trend_consistency,
    )
    write_json(out, args.output, args.pretty)
    print(f"Saved predicates: {args.output}")
    print(f"Predicates: {len(out['predicates'])}")


if __name__ == "__main__":
    main()
