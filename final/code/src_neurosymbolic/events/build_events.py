import argparse
import json
from collections import defaultdict
from pathlib import Path


MOTION_PREFIXES = (
    "hand_moves_",
    "hand_oscillates_",
    "hand_fast_motion",
    "hand_stationary",
)
FINGER_MOTION_NAMES = {
    "index_finger_moves_up",
    "index_finger_moves_down",
    "index_finger_moves_left",
    "index_finger_moves_right",
    "index_finger_moves_diagonal_up_left",
    "index_finger_moves_diagonal_up_right",
    "index_finger_moves_diagonal_down_left",
    "index_finger_moves_diagonal_down_right",
    "index_finger_oscillates",
    "thumb_moves_up",
    "thumb_moves_down",
    "fingers_spread_increasing",
    "fingers_spread_decreasing",
}
COORDINATION_NAMES = {
    "hands_move_same_direction_horizontal",
    "hands_move_opposite_direction_horizontal",
    "hands_move_apart_horizontal",
    "hands_move_together_horizontal",
    "hands_move_same_direction_vertical",
    "hands_move_opposite_direction_vertical",
}
SWEEP_NAMES = {
    "hand_large_horizontal_sweep",
    "hands_large_horizontal_sweep",
}
CONFLICT_GROUPS = [
    {
        "hand_above_shoulder",
        "hand_below_shoulder",
    },
    {
        "hand_above_elbow",
        "hand_below_elbow",
    },
    {
        "hand_left_of_body_center",
        "hand_right_of_body_center",
    },
    {
        "hands_touch",
        "hands_close",
        "hands_far",
    },
    {
        "elbow_very_bent",
        "elbow_bent",
        "elbow_half_extended",
        "elbow_fully_extended",
    },
    {
        "hand_open",
        "hand_closed",
        "hand_fist_like",
        "fingers_slightly_curled",
    },
    {
        "hand_moves_up",
        "hand_moves_down",
    },
    {
        "hand_moves_left",
        "hand_moves_right",
    },
    {
        "hand_moves_diagonal_up_left",
        "hand_moves_diagonal_up_right",
        "hand_moves_diagonal_down_left",
        "hand_moves_diagonal_down_right",
    },
]


def load_predicates(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "predicates" not in data:
        raise ValueError(f"Expected predicate object with predicates: {path}")
    return data


def predicate_category(name: str) -> str:
    if name in SWEEP_NAMES:
        return "sweep"
    if name in COORDINATION_NAMES:
        return "coordination"
    if name in FINGER_MOTION_NAMES:
        return "finger_motion"
    if name.startswith(MOTION_PREFIXES):
        return "motion"
    if name.startswith("elbow_"):
        return "joint_angle"
    if name in {"hand_open", "hand_closed", "hand_fist_like", "fingers_slightly_curled", "index_extended", "thumb_extended", "index_middle_extended"}:
        return "hand_shape"
    return "spatial"


def composite_field_for(category: str) -> str:
    return {
        "motion": "motions",
        "spatial": "positions",
        "hand_shape": "hand_shapes",
        "finger_motion": "finger_motions",
        "coordination": "coordinations",
        "sweep": "sweeps",
        "joint_angle": "joint_angles",
    }.get(category, "attributes")


def min_duration_for(category: str) -> int:
    return {
        "spatial": 2,
        "motion": 4,
        "hand_shape": 2,
        "finger_motion": 3,
        "coordination": 3,
        "sweep": 5,
        "joint_angle": 2,
    }.get(category, 2)


def duration(pred: dict) -> int:
    return int(pred["end_frame"]) - int(pred["start_frame"]) + 1


def weighted_confidence(predicates: list[dict]) -> float:
    total_weight = 0
    total = 0.0
    for pred in predicates:
        weight = max(1, duration(pred))
        total_weight += weight
        total += float(pred.get("confidence", 0.0)) * weight
    if total_weight <= 0:
        return 0.0
    return total / total_weight


def merge_group(predicates: list[dict], max_gap: int) -> list[dict]:
    ordered = sorted(predicates, key=lambda p: (int(p["start_frame"]), int(p["end_frame"])))
    merged = []
    current = []
    current_end = None

    for pred in ordered:
        start = int(pred["start_frame"])
        end = int(pred["end_frame"])
        if not current:
            current = [pred]
            current_end = end
            continue
        if start <= current_end + max_gap:
            current.append(pred)
            current_end = max(current_end, end)
        else:
            merged.append(make_event_candidate(current))
            current = [pred]
            current_end = end

    if current:
        merged.append(make_event_candidate(current))

    return merged


def make_event_candidate(predicates: list[dict]) -> dict:
    first = predicates[0]
    evidence = sorted({kp for pred in predicates for kp in pred.get("evidence_keypoints", [])})
    start = min(int(pred["start_frame"]) for pred in predicates)
    end = max(int(pred["end_frame"]) for pred in predicates)
    category = predicate_category(first["name"])
    return {
        "predicate_name": first["name"],
        "category": category,
        "subject": first["subject"],
        "object": first.get("object"),
        "start_frame": start,
        "end_frame": end,
        "duration": end - start + 1,
        "confidence": round(weighted_confidence(predicates), 6),
        "evidence_keypoints": evidence,
        "source_predicate_count": len(predicates),
    }


def group_composite_events(atomic_events: list[dict], max_gap: int) -> list[dict]:
    groups = defaultdict(list)
    for event in atomic_events:
        groups[event["subject"]].append(event)

    composites = []
    for subject, events in groups.items():
        ordered = sorted(events, key=lambda e: (e["start_frame"], e["end_frame"], e["predicate_name"]))
        current = []
        current_end = None

        for event in ordered:
            if not current:
                current = [event]
                current_end = event["end_frame"]
                continue
            if event["start_frame"] <= current_end + max_gap and not conflicts_with_current(event, current):
                current.append(event)
                current_end = max(current_end, event["end_frame"])
            else:
                composites.append(make_composite_candidate(subject, current))
                current = [event]
                current_end = event["end_frame"]

        if current:
            composites.append(make_composite_candidate(subject, current))

    composites.sort(key=lambda e: (e["start_frame"], e["end_frame"], e["subject"]))
    for idx, event in enumerate(composites, start=1):
        event["event_id"] = f"event_{idx:06d}"
    return composites


def conflicts_with_current(event: dict, current_events: list[dict]) -> bool:
    names = {item["predicate_name"] for item in current_events}
    name = event["predicate_name"]
    for group in CONFLICT_GROUPS:
        if name in group and names.intersection(group - {name}):
            return True
    return False


def unique_sorted(values: list[str]) -> list[str]:
    return sorted(set(values))


def make_composite_candidate(subject: str, atomic_events: list[dict]) -> dict:
    start = min(event["start_frame"] for event in atomic_events)
    end = max(event["end_frame"] for event in atomic_events)
    evidence = sorted({kp for event in atomic_events for kp in event.get("evidence_keypoints", [])})
    fields = {
        "motions": [],
        "positions": [],
        "hand_shapes": [],
        "finger_motions": [],
        "coordinations": [],
        "sweeps": [],
        "joint_angles": [],
        "attributes": [],
    }

    for event in atomic_events:
        field = composite_field_for(event["category"])
        fields.setdefault(field, []).append(event["predicate_name"])

    total_weight = 0
    total_conf = 0.0
    for event in atomic_events:
        weight = max(1, event["duration"])
        total_weight += weight
        total_conf += event["confidence"] * weight

    return {
        "subject": subject,
        "start_frame": start,
        "end_frame": end,
        "duration": end - start + 1,
        "confidence": round(total_conf / total_weight, 6) if total_weight else 0.0,
        "motions": unique_sorted(fields["motions"]),
        "positions": unique_sorted(fields["positions"]),
        "hand_shapes": unique_sorted(fields["hand_shapes"]),
        "finger_motions": unique_sorted(fields["finger_motions"]),
        "coordinations": unique_sorted(fields["coordinations"]),
        "sweeps": unique_sorted(fields["sweeps"]),
        "joint_angles": unique_sorted(fields["joint_angles"]),
        "attributes": unique_sorted(fields["attributes"]),
        "evidence_keypoints": evidence,
        "atomic_event_ids": [event["atomic_event_id"] for event in atomic_events],
        "source_atomic_event_count": len(atomic_events),
    }


def build_events(data: dict, max_gap: int = 2, min_confidence: float = 0.3, composite_gap: int = 4) -> dict:
    groups = defaultdict(list)
    for pred in data.get("predicates", []):
        key = (pred.get("name"), pred.get("subject"), pred.get("object"))
        groups[key].append(pred)

    candidates = []
    for group in groups.values():
        candidates.extend(merge_group(group, max_gap))

    filtered = []
    for event in candidates:
        if event["confidence"] < min_confidence:
            continue
        if event["duration"] < min_duration_for(event["category"]):
            continue
        filtered.append(event)

    filtered.sort(key=lambda e: (e["start_frame"], e["end_frame"], e["predicate_name"], e["subject"]))
    for idx, event in enumerate(filtered, start=1):
        event["atomic_event_id"] = f"atomic_event_{idx:06d}"

    composite_events = group_composite_events(filtered, composite_gap)

    return {
        "video_id": data.get("video_id", ""),
        "source_path": data.get("source_path", ""),
        "events": composite_events,
        "atomic_events": filtered,
        "summary": {
            "input_predicates": len(data.get("predicates", [])),
            "atomic_event_count": len(filtered),
            "event_count": len(composite_events),
            "max_gap": max_gap,
            "composite_gap": composite_gap,
            "min_confidence": min_confidence,
        },
    }


def write_json(data: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal events from predicate JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Input predicate JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output event JSON.")
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--composite-gap", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_predicates(args.input)
    out = build_events(data, args.max_gap, args.min_confidence, args.composite_gap)
    write_json(out, args.output, args.pretty)
    print(f"Saved events: {args.output}")
    print(f"Input predicates: {out['summary']['input_predicates']}")
    print(f"Atomic events: {out['summary']['atomic_event_count']}")
    print(f"Composite events: {out['summary']['event_count']}")


if __name__ == "__main__":
    main()
