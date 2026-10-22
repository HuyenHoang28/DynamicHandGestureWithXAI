import argparse
import json
from collections import defaultdict
from pathlib import Path

from src_neurosymbolic.events.build_events import (
    composite_field_for,
    duration,
    load_predicates,
    merge_group,
    min_duration_for,
    predicate_category,
)


CONFLICT_GROUPS = [
    {"hand_above_shoulder", "hand_below_shoulder"},
    {"hand_above_elbow", "hand_below_elbow"},
    {"hand_left_of_body_center", "hand_right_of_body_center"},
    {"hand_open", "hand_closed", "hand_fist_like", "fingers_slightly_curled"},
    {
        "elbow_very_bent",
        "elbow_bent",
        "elbow_half_extended",
        "elbow_fully_extended",
    },
    {"hand_moves_up", "hand_moves_down"},
    {"hand_moves_left", "hand_moves_right"},
    {
        "hand_moves_diagonal_up_left",
        "hand_moves_diagonal_up_right",
        "hand_moves_diagonal_down_left",
        "hand_moves_diagonal_down_right",
    },
    {"hand_fast_motion", "hand_stationary"},
    {"index_finger_moves_up", "index_finger_moves_down"},
    {"middle_finger_moves_up", "middle_finger_moves_down"},
    {"fingers_spread_increasing", "fingers_spread_decreasing"},
    {"pinch_distance_increasing", "pinch_distance_decreasing"},
    {"hands_move_apart_horizontal", "hands_move_together_horizontal"},
    {"hands_move_same_direction_horizontal", "hands_move_opposite_direction_horizontal"},
    {"hands_move_same_direction_vertical", "hands_move_opposite_direction_vertical"},
]

PROXIMITY_NAMES = {"hands_touch", "hands_close", "hands_far"}
PHASE_FIELDS = (
    "motions",
    "positions",
    "hand_shapes",
    "finger_motions",
    "coordinations",
    "sweeps",
    "joint_angles",
    "attributes",
)

# Only direct directional opposites participate. The first non-overlapping run
# is the performed action; a following opposite run is its return. Repetition
# restarts the cycle, so down-up-down keeps both down runs active.
DIRECTIONAL_OPPOSITES = {
    "hand_moves_up": "hand_moves_down", "hand_moves_down": "hand_moves_up",
    "hand_moves_left": "hand_moves_right", "hand_moves_right": "hand_moves_left",
    "hand_moves_diagonal_up_left": "hand_moves_diagonal_down_right",
    "hand_moves_diagonal_down_right": "hand_moves_diagonal_up_left",
    "hand_moves_diagonal_up_right": "hand_moves_diagonal_down_left",
    "hand_moves_diagonal_down_left": "hand_moves_diagonal_up_right",
    "index_finger_moves_up": "index_finger_moves_down", "index_finger_moves_down": "index_finger_moves_up",
    "index_finger_moves_left": "index_finger_moves_right", "index_finger_moves_right": "index_finger_moves_left",
    "index_finger_moves_diagonal_up_left": "index_finger_moves_diagonal_down_right",
    "index_finger_moves_diagonal_down_right": "index_finger_moves_diagonal_up_left",
    "index_finger_moves_diagonal_up_right": "index_finger_moves_diagonal_down_left",
    "index_finger_moves_diagonal_down_left": "index_finger_moves_diagonal_up_right",
    "middle_finger_moves_up": "middle_finger_moves_down", "middle_finger_moves_down": "middle_finger_moves_up",
    "thumb_moves_up": "thumb_moves_down", "thumb_moves_down": "thumb_moves_up",
    "pinch_distance_increasing": "pinch_distance_decreasing", "pinch_distance_decreasing": "pinch_distance_increasing",
    "fingers_spread_increasing": "fingers_spread_decreasing", "fingers_spread_decreasing": "fingers_spread_increasing",
    "hands_move_apart_horizontal": "hands_move_together_horizontal",
    "hands_move_together_horizontal": "hands_move_apart_horizontal",
    "hands_move_same_direction_horizontal": "hands_move_opposite_direction_horizontal",
    "hands_move_opposite_direction_horizontal": "hands_move_same_direction_horizontal",
    "hands_move_same_direction_vertical": "hands_move_opposite_direction_vertical",
    "hands_move_opposite_direction_vertical": "hands_move_same_direction_vertical",
}

ACTION_ROLE_CENTER_TOLERANCE = 1


def classify_action_roles(atomic_events: list[dict]) -> None:
    """Annotate clear directional reversals without changing atomic identity."""
    by_subject_and_family = defaultdict(list)
    for event in atomic_events:
        name = event.get("predicate_name")
        if name in DIRECTIONAL_OPPOSITES:
            family = tuple(sorted((name, DIRECTIONAL_OPPOSITES[name])))
            by_subject_and_family[(event["subject"], family)].append(event)
    for events in by_subject_and_family.values():
        events.sort(
            key=lambda item: (
                int(item["start_frame"]) + int(item["end_frame"]),
                item["start_frame"],
                item["end_frame"],
                item["atomic_event_id"],
            )
        )
        for event in events:
            event["action_role"] = "active"
        for current, following in zip(events, events[1:]):
            current_center = (int(current["start_frame"]) + int(current["end_frame"])) / 2
            following_center = (int(following["start_frame"]) + int(following["end_frame"])) / 2
            are_opposites = (
                DIRECTIONAL_OPPOSITES.get(current["predicate_name"])
                == following["predicate_name"]
            )
            if not are_opposites:
                continue
            if abs(current_center - following_center) <= ACTION_ROLE_CENTER_TOLERANCE:
                current["action_role"] = "ambiguous"
                following["action_role"] = "ambiguous"
                continue
            following["action_role"] = (
                "return" if current["action_role"] == "active" else "active"
            )


def build_atomic_events(data: dict, max_gap: int, min_confidence: float) -> list[dict]:
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
    classify_action_roles(filtered)
    return filtered


def overlap_duration(event: dict, start: int, end: int) -> int:
    left = max(int(event["start_frame"]), start)
    right = min(int(event["end_frame"]), end)
    return max(0, right - left + 1)


def merge_short_intervals(
    intervals: list[tuple[int, int]],
    min_phase_duration: int,
    max_phase_duration: int,
) -> list[tuple[int, int]]:
    if min_phase_duration <= 1 or not intervals:
        return intervals

    merged = []
    current_start = None
    current_end = None

    for start, end in intervals:
        if current_start is None:
            current_start = start
        current_end = end
        if current_end - current_start + 1 >= min_phase_duration:
            merged.append((current_start, current_end))
            current_start = None
            current_end = None

    if current_start is not None:
        if merged and current_end - merged[-1][0] + 1 <= max_phase_duration:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, current_end)
        else:
            merged.append((current_start, current_end))

    return merged


def phase_intervals(
    events: list[dict],
    max_phase_duration: int,
    min_phase_duration: int,
) -> list[tuple[int, int]]:
    if not events:
        return []
    breakpoints = set()
    for event in events:
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        breakpoints.add(start)
        breakpoints.add(end + 1)
        cursor = start + max_phase_duration
        while cursor <= end:
            breakpoints.add(cursor)
            cursor += max_phase_duration

    ordered = sorted(breakpoints)
    intervals = []
    for left, right in zip(ordered, ordered[1:]):
        start = left
        end = right - 1
        if start <= end:
            intervals.append((start, end))
    return merge_short_intervals(intervals, min_phase_duration, max_phase_duration)


def event_phase_score(
    event: dict,
    start: int,
    end: int,
    min_overlap_ratio: float,
    min_overlap_frames: int,
) -> float:
    overlap = overlap_duration(event, start, end)
    if overlap <= 0:
        return 0.0
    if overlap < min_overlap_frames:
        return 0.0
    phase_len = end - start + 1
    event_len = max(1, duration(event))
    if overlap / max(1, phase_len) < min_overlap_ratio and overlap / event_len < min_overlap_ratio:
        return 0.0
    return float(event.get("confidence", 0.0)) * overlap


def suppress_stationary_when_moving(fields: dict[str, list[str]]) -> None:
    motions = set(fields.get("motions", []))
    has_directional_motion = any(name.startswith("hand_moves_") for name in motions)
    has_fast_motion = "hand_fast_motion" in motions
    if "hand_stationary" in motions and (has_directional_motion or has_fast_motion):
        fields["motions"] = [name for name in fields["motions"] if name != "hand_stationary"]


def choose_best(names: set[str], scores: dict[str, float]) -> set[str]:
    if len(names) <= 1:
        return set(names)
    best = max(names, key=lambda name: (scores.get(name, 0.0), name))
    return {best}


def resolve_proximity(names: set[str], scores: dict[str, float]) -> set[str]:
    present = names & PROXIMITY_NAMES
    if len(present) <= 1:
        return present
    near_score = max(scores.get("hands_touch", 0.0), scores.get("hands_close", 0.0))
    far_score = scores.get("hands_far", 0.0)
    if far_score > near_score:
        return {"hands_far"}
    resolved = set()
    if "hands_touch" in present:
        resolved.add("hands_touch")
        if "hands_close" in present:
            resolved.add("hands_close")
    elif "hands_close" in present:
        resolved.add("hands_close")
    return resolved


def resolve_conflicts(names: set[str], scores: dict[str, float]) -> set[str]:
    resolved = set(names)
    proximity_present = resolved & PROXIMITY_NAMES
    if proximity_present:
        resolved -= PROXIMITY_NAMES
        resolved |= resolve_proximity(proximity_present, scores)

    for group in CONFLICT_GROUPS:
        present = resolved & group
        if len(present) >= 2:
            resolved -= present
            resolved |= choose_best(present, scores)
    return resolved


def make_phase_event(
    subject: str,
    phase_index: int,
    start: int,
    end: int,
    active_events: list[tuple[dict, float]],
) -> dict | None:
    if not active_events:
        return None

    field_scores: dict[str, dict[str, float]] = {field: defaultdict(float) for field in PHASE_FIELDS}
    field_sources: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in PHASE_FIELDS
    }
    evidence_keypoints = set()
    total_score = 0.0
    total_overlap = 0.0
    atomic_ids = []
    concept_action_roles: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for event, score in active_events:
        name = event["predicate_name"]
        field = composite_field_for(event["category"])
        field_scores.setdefault(field, defaultdict(float))[name] += score
        field_sources.setdefault(field, defaultdict(set))[name].add(event["atomic_event_id"])
        if event.get("action_role"):
            concept_action_roles[field][name].add(event["action_role"])
        evidence_keypoints.update(event.get("evidence_keypoints", []))
        atomic_ids.append(event["atomic_event_id"])
        total_score += score
        total_overlap += max(1.0, score / max(float(event.get("confidence", 0.0)), 1e-6))

    fields = {field: [] for field in PHASE_FIELDS}
    for field, scores in field_scores.items():
        names = resolve_conflicts(set(scores), scores)
        fields[field] = sorted(names)
    suppress_stationary_when_moving(fields)
    concept_atomic_event_ids = {
        field: {
            name: sorted(field_sources.get(field, {}).get(name, set()))
            for name in names
        }
        for field, names in fields.items()
        if names
    }
    concept_action_roles_out = {
        field: {name: sorted(roles) for name, roles in concepts.items() if name in fields.get(field, [])}
        for field, concepts in concept_action_roles.items()
    }

    if not any(fields.values()):
        return None

    confidence = total_score / total_overlap if total_overlap > 0 else 0.0
    return {
        "event_id": f"event_{phase_index:06d}",
        "subject": subject,
        "start_frame": start,
        "end_frame": end,
        "duration": end - start + 1,
        "confidence": round(confidence, 6),
        **fields,
        "evidence_keypoints": sorted(evidence_keypoints),
        "atomic_event_ids": sorted(set(atomic_ids)),
        "concept_atomic_event_ids": concept_atomic_event_ids,
        "concept_action_roles": concept_action_roles_out,
        "source_atomic_event_count": len(set(atomic_ids)),
        "phase_event": True,
    }


def signature(event: dict) -> tuple:
    semantic = tuple((field, tuple(event.get(field, []))) for field in PHASE_FIELDS)
    roles = tuple(
        (field, tuple((name, tuple(values)) for name, values in sorted(concepts.items())))
        for field, concepts in sorted(event.get("concept_action_roles", {}).items())
    )
    return semantic, roles


def merge_adjacent_phase_events(events: list[dict], max_phase_duration: int) -> list[dict]:
    if not events:
        return []
    merged = []
    current = dict(events[0])
    for event in events[1:]:
        same_semantics = (
            current["subject"] == event["subject"]
            and signature(current) == signature(event)
            and event["start_frame"] <= current["end_frame"] + 1
            and event["end_frame"] - current["start_frame"] + 1 <= max_phase_duration
        )
        if not same_semantics:
            merged.append(current)
            current = dict(event)
            continue
        duration_left = current["duration"]
        duration_right = event["duration"]
        total_duration = duration_left + duration_right
        current["end_frame"] = event["end_frame"]
        current["duration"] = current["end_frame"] - current["start_frame"] + 1
        current["confidence"] = round(
            (
                float(current["confidence"]) * duration_left
                + float(event["confidence"]) * duration_right
            )
            / max(1, total_duration),
            6,
        )
        current["evidence_keypoints"] = sorted(
            set(current.get("evidence_keypoints", [])) | set(event.get("evidence_keypoints", []))
        )
        current["atomic_event_ids"] = sorted(
            set(current.get("atomic_event_ids", [])) | set(event.get("atomic_event_ids", []))
        )
        current["source_atomic_event_count"] = len(current["atomic_event_ids"])
        merged_sources = current.setdefault("concept_atomic_event_ids", {})
        for field, names in event.get("concept_atomic_event_ids", {}).items():
            field_sources = merged_sources.setdefault(field, {})
            for name, atomic_event_ids in names.items():
                field_sources[name] = sorted(
                    set(field_sources.get(name, [])) | set(atomic_event_ids)
                )
        merged_roles = current.setdefault("concept_action_roles", {})
        for field, concepts in event.get("concept_action_roles", {}).items():
            field_roles = merged_roles.setdefault(field, {})
            for name, roles in concepts.items():
                field_roles[name] = sorted(set(field_roles.get(name, [])) | set(roles))
    merged.append(current)
    for idx, event in enumerate(merged, start=1):
        event["event_id"] = f"event_{idx:06d}"
    return merged


def group_phase_events(
    atomic_events: list[dict],
    max_phase_duration: int,
    min_phase_duration: int,
    min_overlap_ratio: float,
    min_overlap_frames: int,
    merge_adjacent: bool,
) -> list[dict]:
    by_subject = defaultdict(list)
    for event in atomic_events:
        by_subject[event["subject"]].append(event)

    phase_events = []
    phase_index = 1
    for subject, events in sorted(by_subject.items()):
        events = sorted(events, key=lambda e: (e["start_frame"], e["end_frame"], e["predicate_name"]))
        subject_phase_events = []
        for start, end in phase_intervals(events, max_phase_duration, min_phase_duration):
            active = []
            for event in events:
                score = event_phase_score(
                    event,
                    start,
                    end,
                    min_overlap_ratio,
                    min_overlap_frames,
                )
                if score > 0:
                    active.append((event, score))
            phase_event = make_phase_event(subject, phase_index, start, end, active)
            if phase_event is not None:
                subject_phase_events.append(phase_event)
                phase_index += 1
        if merge_adjacent:
            subject_phase_events = merge_adjacent_phase_events(
                subject_phase_events,
                max_phase_duration,
            )
        phase_events.extend(subject_phase_events)

    phase_events.sort(key=lambda e: (e["start_frame"], e["end_frame"], e["subject"]))
    for idx, event in enumerate(phase_events, start=1):
        event["event_id"] = f"event_{idx:06d}"
    return phase_events


def build_events_phase_v2(
    data: dict,
    max_gap: int = 2,
    min_confidence: float = 0.3,
    max_phase_duration: int = 16,
    min_phase_duration: int = 4,
    min_overlap_ratio: float = 0.4,
    min_overlap_frames: int = 2,
    merge_adjacent: bool = True,
) -> dict:
    atomic_events = build_atomic_events(data, max_gap, min_confidence)
    phase_events = group_phase_events(
        atomic_events,
        max_phase_duration=max_phase_duration,
        min_phase_duration=min_phase_duration,
        min_overlap_ratio=min_overlap_ratio,
        min_overlap_frames=min_overlap_frames,
        merge_adjacent=merge_adjacent,
    )
    return {
        "video_id": data.get("video_id", ""),
        "source_path": data.get("source_path", ""),
        "events": phase_events,
        "atomic_events": atomic_events,
        "predicate_config": data.get("predicate_config", {}),
        "summary": {
            "input_predicates": len(data.get("predicates", [])),
            "atomic_event_count": len(atomic_events),
            "event_count": len(phase_events),
            "max_gap": max_gap,
            "min_confidence": min_confidence,
            "max_phase_duration": max_phase_duration,
            "min_phase_duration": min_phase_duration,
            "min_overlap_ratio": min_overlap_ratio,
            "min_overlap_frames": min_overlap_frames,
            "merge_adjacent": merge_adjacent,
            "builder": "phase_v4_action_role_audit",
            "action_role_algorithm": "directional_reversal_v2",
            "predicate_config": data.get("predicate_config", {}),
        },
    }


def write_json(data: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase-aware composite events.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--max-phase-duration", type=int, default=16)
    parser.add_argument("--min-phase-duration", type=int, default=4)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.4)
    parser.add_argument("--min-overlap-frames", type=int, default=2)
    parser.add_argument("--no-merge-adjacent", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_predicates(args.input)
    events = build_events_phase_v2(
        data,
        max_gap=args.max_gap,
        min_confidence=args.min_confidence,
        max_phase_duration=args.max_phase_duration,
        min_phase_duration=args.min_phase_duration,
        min_overlap_ratio=args.min_overlap_ratio,
        min_overlap_frames=args.min_overlap_frames,
        merge_adjacent=not args.no_merge_adjacent,
    )
    write_json(events, args.output, args.pretty)
    print(f"Saved phase-aware events: {args.output}")
    print(events["summary"])


if __name__ == "__main__":
    main()
