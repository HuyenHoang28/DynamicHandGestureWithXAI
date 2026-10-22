import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml


FIELD_TO_NODE_TYPE_AND_RELATION = {
    "motions": ("Motion", "HAS_MOTION"),
    "positions": ("Position", "HAS_POSITION"),
    "hand_shapes": ("HandShape", "HAS_HAND_SHAPE"),
    "finger_motions": ("FingerMotion", "HAS_FINGER_MOTION"),
    "coordinations": ("Coordination", "HAS_COORDINATION"),
    "sweeps": ("Sweep", "HAS_SWEEP"),
    "joint_angles": ("JointAngle", "HAS_JOINT_ANGLE"),
}
REQUIREMENT_LEVELS = {"required", "preferred", "optional"}
DEFAULT_WEIGHTS = {
    "required": 1.0,
    "preferred": 0.6,
    "optional": 0.3,
    "negative": -1.0,
}


def add_node(nodes: dict[str, dict], node_id: str, node_type: str, properties: dict | None = None) -> None:
    if node_id in nodes:
        if properties:
            nodes[node_id].setdefault("properties", {}).update(properties)
        return
    nodes[node_id] = {
        "id": node_id,
        "type": node_type,
        "properties": properties or {},
    }


def add_edge(edges: list[dict], source: str, relation: str, target: str, properties: dict | None = None) -> None:
    edges.append(
        {
            "source": source,
            "relation": relation,
            "target": target,
            "properties": properties or {},
        }
    )


def parse_template_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Template file must contain a mapping: {path}")
    gestures = raw.get("gestures")
    if not isinstance(gestures, dict) or not gestures:
        raise ValueError(f"Template file has no gestures: {path}")
    version = raw.setdefault("version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version not in {1, 2}:
        raise ValueError("Template version must be integer 1 or 2")

    policy = raw.setdefault("template_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("template_policy must be a mapping")
    matching = policy.get("matching")
    if matching is None:
        matching = {}
        policy["matching"] = matching
    if not isinstance(matching, dict):
        raise ValueError("template_policy.matching must be a mapping")
    for level, default in DEFAULT_WEIGHTS.items():
        canonical = f"{level}_weight"
        value = matching.get(
            canonical,
            matching.get(level, policy.get(canonical, policy.get(level, default))),
        )
        if isinstance(value, bool):
            raise ValueError(f"{canonical} must be a finite number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{canonical} must be a finite number")
        matching[canonical] = value

    role_alignment_floor = matching.get("role_alignment_floor", 1.0)
    if isinstance(role_alignment_floor, bool):
        raise ValueError("role_alignment_floor must be a finite number in [0, 1]")
    try:
        role_alignment_floor = float(role_alignment_floor)
    except (TypeError, ValueError) as exc:
        raise ValueError("role_alignment_floor must be a finite number in [0, 1]") from exc
    if not math.isfinite(role_alignment_floor) or not 0.0 <= role_alignment_floor <= 1.0:
        raise ValueError("role_alignment_floor must be a finite number in [0, 1]")
    if version < 2 and role_alignment_floor != 1.0:
        raise ValueError("role_alignment_floor requires template version 2")
    matching["role_alignment_floor"] = role_alignment_floor

    allowed_event_keys = {"subject", "min_occurrences", "max_occurrences", "action_role", *FIELD_TO_NODE_TYPE_AND_RELATION}
    for label, gesture in gestures.items():
        if not isinstance(gesture, dict):
            raise ValueError(f"Gesture {label} must be a mapping")
        events = gesture.setdefault("expected_events", [])
        if not isinstance(events, list) or not events:
            raise ValueError(f"Gesture {label} must define expected_events")
        for event_index, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                raise ValueError(f"Gesture {label} event {event_index} must be a mapping")
            unknown = set(event) - allowed_event_keys
            if unknown:
                raise ValueError(f"Gesture {label} event {event_index} has unsupported keys: {sorted(unknown)}")
            event.setdefault("subject", "any")
            event.setdefault("action_role", "any")
            if not isinstance(event["action_role"], str) or event["action_role"] not in {"any", "active", "return"}:
                raise ValueError(f"Gesture {label} event {event_index} action_role must be any, active, or return")
            min_occurrences = event.get("min_occurrences", 1)
            if isinstance(min_occurrences, bool):
                raise ValueError(f"Gesture {label} event {event_index} min_occurrences must be an integer")
            event["min_occurrences"] = int(min_occurrences)
            max_occurrences = event.get("max_occurrences")
            if isinstance(max_occurrences, bool):
                raise ValueError(f"Gesture {label} event {event_index} max_occurrences must be an integer")
            event["max_occurrences"] = int(max_occurrences) if max_occurrences is not None else None
            if event["min_occurrences"] < 1:
                raise ValueError(f"Gesture {label} event {event_index} min_occurrences must be >= 1")
            if event["max_occurrences"] is not None and event["max_occurrences"] < event["min_occurrences"]:
                raise ValueError(f"Gesture {label} event {event_index} max_occurrences must be >= min_occurrences")

            requirements = defaultdict(lambda: defaultdict(list))
            for field in FIELD_TO_NODE_TYPE_AND_RELATION:
                groups = event.get(field, {})
                if groups is None:
                    continue
                if not isinstance(groups, dict):
                    raise ValueError(f"Gesture {label} event {event_index} field {field} must be a mapping")
                for level, values in groups.items():
                    if level not in REQUIREMENT_LEVELS:
                        raise ValueError(f"Gesture {label} uses unsupported requirement level: {level}")
                    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                        raise ValueError(f"Gesture {label} {field}.{level} must be a list of names")
                    requirements[field][level].extend(values)
            event["requirements"] = requirements
            if version >= 2 and not any(
                requirements[field].get("required")
                for field in FIELD_TO_NODE_TYPE_AND_RELATION
            ):
                raise ValueError(
                    f"Gesture {label} event {event_index} must define at least one required concept"
                )

        negative = gesture.setdefault("negative_evidence", {})
        if not isinstance(negative, dict):
            raise ValueError(f"Gesture {label} negative_evidence must be a mapping")
        unknown_negative = set(negative) - set(FIELD_TO_NODE_TYPE_AND_RELATION)
        if unknown_negative:
            raise ValueError(f"Gesture {label} has unsupported negative fields: {sorted(unknown_negative)}")
        for field, values in negative.items():
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"Gesture {label} negative_evidence.{field} must be a list of names")
        gesture.setdefault("description", "")
        gesture.setdefault("notes", [])

    raw.setdefault("description", "")
    return raw


def load_label_ids(labels_path: Path | None, list_files: list[Path]) -> dict[str, int]:
    label_ids = {}
    if labels_path and labels_path.exists():
        labels_data = yaml.safe_load(labels_path.read_text(encoding="utf-8"))
        if isinstance(labels_data, dict):
            labels_data = labels_data.get("labels", labels_data)
        if isinstance(labels_data, dict):
            for name, label_id in labels_data.items():
                label_ids[str(name)] = int(label_id)
        elif labels_data is not None:
            raise ValueError("labels file must map label names to integer IDs")

    for path in list_files:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=2)
            if len(parts) != 3:
                continue
            video_path, _, label_id_text = parts
            try:
                label_id = int(label_id_text)
            except ValueError:
                continue
            segments = video_path.replace("\\", "/").split("/")
            label = None
            for split_name in ("train", "val", "test"):
                if split_name in segments:
                    idx = segments.index(split_name)
                    if idx + 1 < len(segments):
                        label = segments[idx + 1]
                    break
            if label is not None:
                existing = label_ids.get(label)
                if existing is not None and existing != label_id:
                    raise ValueError(f"Conflicting label IDs for {label}: {existing} and {label_id}")
                label_ids[label] = label_id
    by_id = defaultdict(list)
    for label, label_id in label_ids.items():
        by_id[label_id].append(label)
    conflicts = {label_id: names for label_id, names in by_id.items() if len(names) > 1}
    if conflicts:
        raise ValueError(f"Multiple labels share an ID: {conflicts}")
    return label_ids


def concept_node_id(node_type: str, value: str, event_id: str | None = None) -> str:
    base = f"{node_type.lower()}:{value}"
    return f"{event_id}:{base}" if event_id else base


def build_template_graph(
    label_name: str,
    label_id: int,
    template: dict,
    weights: dict[str, float],
    schema_version: int = 2,
    role_alignment_floor: float = 1.0,
) -> dict:
    if isinstance(role_alignment_floor, bool):
        raise ValueError("role_alignment_floor must be a finite number in [0, 1]")
    try:
        role_alignment_floor = float(role_alignment_floor)
    except (TypeError, ValueError) as exc:
        raise ValueError("role_alignment_floor must be a finite number in [0, 1]") from exc
    if not math.isfinite(role_alignment_floor) or not 0.0 <= role_alignment_floor <= 1.0:
        raise ValueError("role_alignment_floor must be a finite number in [0, 1]")
    if schema_version < 2 and role_alignment_floor != 1.0:
        raise ValueError("role_alignment_floor requires template version 2")
    gesture_id = f"gesture:{label_name}"
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    add_node(
        nodes,
        gesture_id,
        "Gesture",
        {
            "label_id": label_id,
            "label_name": label_name,
            "description": template.get("description", ""),
            "notes": template.get("notes", []),
        },
    )

    event_count = 0
    requirement_edge_count = 0
    positive_event_ids = []
    for idx, event_template in enumerate(template.get("expected_events", []), start=1):
        event_count += 1
        event_id = f"template_event:{label_name}:{idx:03d}"
        positive_event_ids.append(event_id)
        subject = event_template["subject"]
        add_node(
            nodes,
            event_id,
            "Event",
            {
                "event_id": f"{label_name}_template_event_{idx:03d}",
                "template": True,
                "subject": subject,
                "ordinal": idx,
                "min_occurrences": int(event_template.get("min_occurrences", 1)),
                "max_occurrences": event_template.get("max_occurrences"),
                "action_role": event_template.get("action_role", "any"),
            },
        )
        add_edge(
            edges,
            gesture_id,
            "HAS_EVENT",
            event_id,
            {
                "template": True,
                "ordinal": idx,
            },
        )

        subject_id = f"bodypart:{subject}"
        add_node(nodes, subject_id, "BodyPart", {"name": subject})
        add_edge(edges, event_id, "INVOLVES", subject_id, {"template": True})

        for field, groups in event_template.get("requirements", {}).items():
            node_type, relation = FIELD_TO_NODE_TYPE_AND_RELATION[field]
            for level, values in groups.items():
                for value in values:
                    node_id = concept_node_id(node_type, value, event_id)
                    add_node(
                        nodes,
                        node_id,
                        node_type,
                        {"name": value, "template_event": event_id, "ordinal": idx},
                    )
                    add_edge(
                        edges,
                        event_id,
                        relation,
                        node_id,
                        {
                            "template": True,
                            "requirement_level": level,
                            "weight": weights[level],
                        },
                    )
                    requirement_edge_count += 1

    for previous, current in zip(positive_event_ids, positive_event_ids[1:]):
        add_edge(
            edges,
            previous,
            "BEFORE",
            current,
            {"template": True, "required": True},
        )

    negative_count = 0
    for field, values in template.get("negative_evidence", {}).items():
        node_type, relation = FIELD_TO_NODE_TYPE_AND_RELATION[field]
        event_count += 1
        event_id = f"template_event:{label_name}:negative:{negative_count + 1:03d}"
        add_node(
            nodes,
            event_id,
            "Event",
            {
                "event_id": f"{label_name}_negative_event_{negative_count + 1:03d}",
                "template": True,
                "negative": True,
                "subject": "any",
                "ordinal": event_count,
            },
        )
        add_edge(edges, gesture_id, "HAS_EVENT", event_id, {"template": True, "negative": True})
        for value in values:
            node_id = concept_node_id(node_type, value, event_id)
            add_node(nodes, node_id, node_type, {"name": value, "template_event": event_id})
            add_edge(
                edges,
                event_id,
                relation,
                node_id,
                {
                    "template": True,
                    "requirement_level": "negative",
                    "weight": weights["negative"],
                },
            )
            requirement_edge_count += 1
            negative_count += 1

    return {
        "graph_type": "template_graph",
        "schema_version": schema_version,
        "matching_weights": dict(weights),
        "matching_policy": {"role_alignment_floor": float(role_alignment_floor)},
        "label_id": label_id,
        "label_name": label_name,
        "nodes": list(nodes.values()),
        "edges": edges,
        "summary": {
            "template_event_count": event_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "requirement_edge_count": requirement_edge_count,
            "negative_requirement_count": negative_count,
        },
    }


def write_json(data: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build template KG JSON files from gesture_templates.yaml.")
    parser.add_argument("--templates", type=Path, default=Path("configs/ipn/gesture_templates.yaml"))
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.yaml"))
    parser.add_argument("--list-file", type=Path, action="append", default=[Path("train.txt"), Path("val.txt"), Path("test.txt")])
    parser.add_argument("--output-dir", type=Path, default=Path("data/template_graphs"))
    parser.add_argument("--combined-output", type=Path, default=None, help="Optional JSON file containing all template graphs.")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed = parse_template_yaml(args.templates)
    gestures = parsed["gestures"]
    weights = {
        level: float(parsed["template_policy"]["matching"][f"{level}_weight"])
        for level in DEFAULT_WEIGHTS
    }
    matching_policy = {
        "role_alignment_floor": float(
            parsed["template_policy"]["matching"]["role_alignment_floor"]
        )
    }
    label_ids = load_label_ids(args.labels, args.list_file)
    fallback_label_ids = {label: idx for idx, label in enumerate(sorted(gestures))}

    graphs = []
    for label_name in sorted(gestures, key=lambda name: label_ids.get(name, fallback_label_ids[name])):
        label_id = label_ids.get(label_name, fallback_label_ids[label_name])
        graph = build_template_graph(
            label_name,
            label_id,
            gestures[label_name],
            weights,
            schema_version=int(parsed["version"]),
            role_alignment_floor=matching_policy["role_alignment_floor"],
        )
        graphs.append(graph)
        write_json(graph, args.output_dir / f"{label_name}.json", args.pretty)

    index = {
        "graph_type": "template_graph_index",
        "schema_version": int(parsed["version"]),
        "template_source": str(args.templates.as_posix()),
        "template_sha256": hashlib.sha256(args.templates.read_bytes()).hexdigest(),
        "matching_weights": weights,
        "matching_policy": matching_policy,
        "template_count": len(graphs),
        "templates": [
            {
                "label_id": graph["label_id"],
                "label_name": graph["label_name"],
                "path": str((args.output_dir / f"{graph['label_name']}.json").as_posix()),
                "summary": graph["summary"],
            }
            for graph in graphs
        ],
    }
    write_json(index, args.output_dir / "index.json", args.pretty)
    if args.combined_output:
        write_json({"graph_type": "template_graph_collection", "templates": graphs}, args.combined_output, args.pretty)

    print(f"Saved template graphs: {args.output_dir}")
    print(f"Templates: {len(graphs)}")
    print(f"Total nodes: {sum(graph['summary']['node_count'] for graph in graphs)}")
    print(f"Total edges: {sum(graph['summary']['edge_count'] for graph in graphs)}")


if __name__ == "__main__":
    main()
