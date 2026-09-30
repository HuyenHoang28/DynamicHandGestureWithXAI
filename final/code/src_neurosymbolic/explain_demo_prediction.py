import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


IMPORTANT_FIELDS = [
    "coordinations",
    "motions",
    "positions",
    "hand_shapes",
    "finger_motions",
    "sweeps",
    "joint_angles",
]

FIELD_LABELS = {
    "coordinations": "two-hand coordination",
    "motions": "hand motion",
    "positions": "hand position",
    "hand_shapes": "hand shape",
    "finger_motions": "finger motion",
    "sweeps": "sweep pattern",
    "joint_angles": "joint angle",
}

CATEGORY_TO_FIELD = {
    "coordination": "coordinations",
    "motion": "motions",
    "position": "positions",
    "spatial": "positions",
    "hand_shape": "hand_shapes",
    "finger_motion": "finger_motions",
    "sweep": "sweeps",
    "joint_angle": "joint_angles",
}

ACTION_CATEGORIES = {"motion", "coordination", "finger_motion", "sweep"}

CONCEPT_TEXT = {
    "hand_moves_left": "moves left",
    "hand_moves_right": "moves right",
    "hand_moves_up": "moves up",
    "hand_moves_down": "moves down",
    "hand_moves_diagonal_up_left": "moves diagonally up-left",
    "hand_moves_diagonal_up_right": "moves diagonally up-right",
    "hand_moves_diagonal_down_left": "moves diagonally down-left",
    "hand_moves_diagonal_down_right": "moves diagonally down-right",
    "hand_stationary": "stays almost still",
    "hand_fast_motion": "moves fast",
    "hand_oscillates_horizontal": "oscillates horizontally",
    "hand_oscillates_vertical": "oscillates vertically",
    "hands_move_same_direction_horizontal": "both hands move in the same horizontal direction",
    "hands_move_opposite_direction_horizontal": "both hands move in opposite horizontal directions",
    "hands_move_same_direction_vertical": "both hands move in the same vertical direction",
    "hands_move_opposite_direction_vertical": "both hands move in opposite vertical directions",
    "hands_move_apart_horizontal": "two hands move apart",
    "hands_move_together_horizontal": "two hands move closer",
    "hands_close": "two hands are close",
    "hands_far": "two hands are far apart",
    "hand_above_shoulder": "is above shoulder",
    "hand_below_shoulder": "is below shoulder",
    "hand_above_elbow": "is above elbow",
    "hand_below_elbow": "is below elbow",
    "hand_near_face": "is near face",
    "hand_below_face": "is below face",
    "hand_left_of_body_center": "is left of body center",
    "hand_right_of_body_center": "is right of body center",
    "hand_open": "is open",
    "hand_closed": "is closed",
    "forefinger_extended": "has index finger extended",
    "thumb_extended": "has thumb extended",
    "fingers_spread_increasing": "spreads fingers wider",
    "fingers_spread_decreasing": "closes finger spread",
    "index_finger_moves_up": "index finger moves up",
    "index_finger_moves_down": "index finger moves down",
    "index_finger_moves_left": "index finger moves left",
    "index_finger_moves_right": "index finger moves right",
    "index_finger_oscillates": "index finger oscillates",
}

SUBJECT_TEXT = {
    "left_hand": "left hand",
    "right_hand": "right hand",
    "both_hands": "both hands",
}


def concept_to_text(concept: str) -> str:
    return CONCEPT_TEXT.get(concept, concept.replace("_", " "))


def subject_to_text(subject: str | None) -> str:
    return SUBJECT_TEXT.get(str(subject), str(subject).replace("_", " "))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def event_score(event: dict) -> float:
    confidence = float(event.get("confidence") or 0.0)
    duration = max(1, int(event.get("duration") or 1))
    concept_count = sum(len(event.get(field, [])) for field in IMPORTANT_FIELDS)
    return round(confidence * duration * max(1, concept_count), 6)


def event_concepts(event: dict) -> list[dict]:
    concepts = []
    for field in IMPORTANT_FIELDS:
        for concept in event.get(field, []):
            concepts.append(
                {
                    "type": field,
                    "type_text": FIELD_LABELS.get(field, field),
                    "concept": concept,
                    "text": concept_to_text(concept),
                }
            )
    return concepts


def flatten_scored_actions(events: dict) -> list[dict]:
    actions = []
    for event in events.get("events", []):
        concepts = event_concepts(event)
        if not concepts:
            continue
        score = event_score(event)
        actions.append(
            {
                "subject": event.get("subject"),
                "subject_text": subject_to_text(event.get("subject")),
                "start_frame": event.get("start_frame"),
                "end_frame": event.get("end_frame"),
                "duration": event.get("duration"),
                "confidence": event.get("confidence"),
                "score": score,
                "concepts": concepts,
                "description": describe_event(event, concepts),
            }
        )
    return actions


def atomic_action_score(event: dict) -> float:
    confidence = float(event.get("confidence") or 0.0)
    duration = min(32, max(1, int(event.get("duration") or 1)))
    category = str(event.get("category") or "")
    category_weight = 1.5 if category in ACTION_CATEGORIES else 0.45
    return round(confidence * duration * category_weight, 6)


def flatten_atomic_actions(events: dict, action_only: bool = False) -> list[dict]:
    actions = []
    for event in events.get("atomic_events", []):
        concept = str(event.get("predicate_name") or "")
        if not concept:
            continue
        category = str(event.get("category") or "")
        if action_only and category not in ACTION_CATEGORIES:
            continue
        field = CATEGORY_TO_FIELD.get(category, category)
        subject = event.get("subject")
        actions.append(
            {
                "subject": subject,
                "subject_text": subject_to_text(subject),
                "start_frame": event.get("start_frame"),
                "end_frame": event.get("end_frame"),
                "duration": event.get("duration"),
                "confidence": event.get("confidence"),
                "score": atomic_action_score(event),
                "concept": concept,
                "category": category,
                "type": field,
                "description": f"{subject_to_text(subject)}: {concept_to_text(concept)}",
            }
        )
    return actions


def describe_event(event: dict, concepts: list[dict] | None = None) -> str:
    if concepts is None:
        concepts = event_concepts(event)
    subject = subject_to_text(event.get("subject"))
    priority = [
        "coordinations",
        "motions",
        "sweeps",
        "finger_motions",
        "hand_shapes",
        "joint_angles",
        "positions",
    ]
    ordered = sorted(concepts, key=lambda item: priority.index(item["type"]) if item["type"] in priority else 99)
    phrases = [item["text"] for item in ordered[:5]]
    if not phrases:
        return f"{subject}: no strong symbolic action"
    return f"{subject}: " + "; ".join(phrases)


def phase_name(start_frame: int, end_frame: int, total_frames: int) -> str:
    midpoint = (start_frame + end_frame) / 2.0
    if midpoint < total_frames / 3:
        return "early"
    if midpoint < total_frames * 2 / 3:
        return "middle"
    return "late"


def build_timeline(events: dict, max_items: int) -> list[dict]:
    all_events = events.get("events", [])
    if not all_events:
        return []
    total_frames = max(int(event.get("end_frame") or 0) for event in all_events) + 1
    actions = flatten_atomic_actions(events, action_only=True) or flatten_atomic_actions(events) or flatten_scored_actions(events)
    actions.sort(key=lambda item: (-item["score"], item["start_frame"] or 0, item["subject_text"]))

    selected = []
    phase_counts = Counter()
    seen = set()
    for action in actions:
        start = int(action["start_frame"] or 0)
        end = int(action["end_frame"] or start)
        phase = phase_name(start, end, total_frames)
        if action.get("concept"):
            concept_key = action["concept"]
        else:
            concept_key = tuple(item["concept"] for item in action.get("concepts", [])[:3])
        key = (phase, action["subject"], concept_key)
        if key in seen or phase_counts[phase] >= 3:
            continue
        seen.add(key)
        phase_counts[phase] += 1
        selected.append({**action, "phase": phase})
        if len(selected) >= max_items:
            break

    selected.sort(key=lambda item: (item["start_frame"] or 0, item["end_frame"] or 0, -item["score"]))
    return selected


def top_scored_actions(events: dict, max_items: int) -> list[dict]:
    actions = flatten_atomic_actions(events, action_only=True) or flatten_atomic_actions(events) or flatten_scored_actions(events)
    actions.sort(key=lambda item: (-item["score"], item["start_frame"] or 0, item["subject_text"]))
    return actions[:max_items]


def concept_counts(events: dict, max_items: int) -> list[dict]:
    counts = Counter()
    for event in events.get("events", []):
        for field in IMPORTANT_FIELDS:
            for concept in event.get(field, []):
                counts[(field, concept)] += 1
    return [{"type": field, "concept": concept, "count": count} for (field, concept), count in counts.most_common(max_items)]


def symbolic_summary(prediction: dict) -> dict | None:
    symbolic = prediction.get("symbolic")
    if not symbolic:
        return None
    top = symbolic.get("top_k", [])
    best = top[0] if top else {}
    matched = best.get("matched", [])
    required = [item for item in matched if item.get("level") == "required"]
    preferred = [item for item in matched if item.get("level") == "preferred"]
    return {
        "prediction": symbolic.get("prediction"),
        "prediction_label_id": symbolic.get("prediction_label_id"),
        "score": best.get("score"),
        "required_hit": best.get("required_hit"),
        "required_total": best.get("required_total"),
        "matched_required": [
            {
                "subject": item.get("subject"),
                "relation": item.get("relation"),
                "concept": item.get("concept"),
                "strength": item.get("match_strength"),
            }
            for item in required[:8]
        ],
        "matched_preferred": [
            {
                "subject": item.get("subject"),
                "relation": item.get("relation"),
                "concept": item.get("concept"),
                "strength": item.get("match_strength"),
            }
            for item in preferred[:5]
        ],
    }


def make_markdown(explanation: dict) -> str:
    neural = explanation["neural"]
    lines = [
        "# Temporal Gesture Explanation",
        "",
        f"Predicted gesture: **{neural['prediction_label']}**",
        f"Neural confidence: **{neural['confidence']:.4f}**",
        "",
        "## Neural Top-K",
    ]

    for item in neural.get("top_k", [])[:5]:
        lines.append(f"- `{item['label_name']}`: {item['probability']:.4f}")

    lines.extend(
        [
            "",
            "## Temporal Evidence",
            "",
            "The events below are sorted by frame order. In each phase, the explanation keeps the strongest symbolic actions.",
            "",
            "Action score is computed from confidence, duration, and the number of meaningful predicates in the event.",
        ]
    )
    for event in explanation["timeline"]:
        lines.append(
            f"- `{event['phase']}` frames `{event['start_frame']}-{event['end_frame']}`: "
            f"{event['description']} "
            f"(score={event['score']:.3f}, conf={event['confidence']}, duration={event['duration']})"
        )

    lines.extend(
        [
            "",
            "## Highest-Scoring Actions",
            "",
            "These are the strongest actions regardless of time order.",
        ]
    )
    for event in explanation["top_scored_actions"]:
        lines.append(
            f"- frames `{event['start_frame']}-{event['end_frame']}`: "
            f"{event['description']} "
            f"(score={event['score']:.3f})"
        )

    symbolic = explanation.get("symbolic")
    if symbolic:
        lines.extend(
            [
                "",
                "## Template KG Match",
                f"Symbolic top label: **{symbolic['prediction']}**",
                f"Template score: `{symbolic['score']}`",
                f"Required matched: `{symbolic['required_hit']}/{symbolic['required_total']}`",
                "",
                "Matched required template facts:",
            ]
        )
        if symbolic["matched_required"]:
            for item in symbolic["matched_required"]:
                lines.append(f"- `{item['subject']} {item['relation']} {item['concept']}` strength={item['strength']}")
        else:
            lines.append("- No required fact details available in top symbolic match.")

    return "\n".join(lines) + "\n"


def explain(args: argparse.Namespace) -> None:
    prediction = load_json(args.demo_dir / "prediction.json")
    events = load_json(args.demo_dir / "events.json")
    neural = prediction["neural"]
    explanation = {
        "neural": {
            "prediction_label": neural["prediction_label"],
            "prediction_id": neural["prediction_id"],
            "confidence": neural["confidence"],
            "top_k": neural["top_k"],
        },
        "timeline": build_timeline(events, args.max_events),
        "top_scored_actions": top_scored_actions(events, args.max_actions),
        "frequent_concepts": concept_counts(events, args.max_concepts),
        "symbolic": symbolic_summary(prediction),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(explanation, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(make_markdown(explanation), encoding="utf-8")
    print(f"Saved short explanation JSON: {args.output_json}")
    print(f"Saved short explanation MD: {args.output_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a short explanation from demo pipeline artifacts.")
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--max-concepts", type=int, default=8)
    args = parser.parse_args()
    if args.output_json is None:
        args.output_json = args.demo_dir / "explanation_short.json"
    if args.output_md is None:
        args.output_md = args.demo_dir / "explanation_short.md"
    return args


if __name__ == "__main__":
    explain(parse_args())
