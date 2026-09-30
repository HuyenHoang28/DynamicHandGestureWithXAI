import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


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


def parse_scalar(value: str):
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_template_yaml(path: Path) -> dict:
    # This intentionally parses only the subset used by configs/gesture_templates.yaml.
    lines = path.read_text(encoding="utf-8").splitlines()
    data = {
        "version": None,
        "description": "",
        "template_policy": {"matching": dict(DEFAULT_WEIGHTS)},
        "gestures": {},
    }

    current_gesture = None
    current_event = None
    current_category = None
    current_level = None
    in_policy_matching = False
    in_gestures = False
    in_expected_events = False
    in_negative = False
    in_notes = False
    negative_category = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if re.match(r"^version:\s*", line):
            data["version"] = parse_scalar(line.split(":", 1)[1])
            continue
        if re.match(r"^description:\s*", line):
            data["description"] = line.split(":", 1)[1].strip()
            continue
        if line == "template_policy:":
            in_policy_matching = False
            in_gestures = False
            continue
        if line == "    matching:":
            in_policy_matching = True
            continue
        if line == "gestures:":
            in_gestures = True
            in_policy_matching = False
            continue

        if in_policy_matching:
            match = re.match(r"^    ([A-Za-z0-9_]+):\s*(.+)$", line)
            if match:
                data["template_policy"]["matching"][match.group(1)] = parse_scalar(match.group(2))
            continue

        if not in_gestures:
            continue

        gesture_match = re.match(r"^  ([A-Za-z0-9_]+):\s*$", line)
        if gesture_match:
            current_gesture = gesture_match.group(1)
            data["gestures"][current_gesture] = {
                "description": "",
                "expected_events": [],
                "negative_evidence": defaultdict(list),
                "notes": [],
            }
            current_event = None
            current_category = None
            current_level = None
            in_expected_events = False
            in_negative = False
            in_notes = False
            negative_category = None
            continue

        if current_gesture is None:
            continue

        gesture = data["gestures"][current_gesture]
        desc_match = re.match(r"^    description:\s*(.*)$", line)
        if desc_match:
            gesture["description"] = desc_match.group(1).strip()
            continue
        if line == "    expected_events:":
            in_expected_events = True
            in_negative = False
            in_notes = False
            continue
        if line == "    negative_evidence:":
            in_expected_events = False
            in_negative = True
            in_notes = False
            current_event = None
            continue
        if line == "    notes:":
            in_expected_events = False
            in_negative = False
            in_notes = True
            continue

        if in_expected_events:
            subject_match = re.match(r"^      - subject:\s*([A-Za-z0-9_]+)\s*$", line)
            if subject_match:
                current_event = {
                    "subject": subject_match.group(1),
                    "requirements": defaultdict(lambda: defaultdict(list)),
                }
                gesture["expected_events"].append(current_event)
                current_category = None
                current_level = None
                continue
            category_match = re.match(r"^        ([A-Za-z0-9_]+):\s*$", line)
            if category_match and category_match.group(1) in FIELD_TO_NODE_TYPE_AND_RELATION:
                current_category = category_match.group(1)
                current_level = None
                continue
            level_match = re.match(r"^          ([A-Za-z0-9_]+):\s*$", line)
            if level_match and level_match.group(1) in REQUIREMENT_LEVELS:
                current_level = level_match.group(1)
                continue
            item_match = re.match(r"^            - ([A-Za-z0-9_]+)\s*$", line)
            if item_match and current_event and current_category and current_level:
                current_event["requirements"][current_category][current_level].append(item_match.group(1))
                continue

        if in_negative:
            category_match = re.match(r"^      ([A-Za-z0-9_]+):\s*$", line)
            if category_match and category_match.group(1) in FIELD_TO_NODE_TYPE_AND_RELATION:
                negative_category = category_match.group(1)
                continue
            item_match = re.match(r"^        - ([A-Za-z0-9_]+)\s*$", line)
            if item_match and negative_category:
                gesture["negative_evidence"][negative_category].append(item_match.group(1))
                continue

        if in_notes:
            note_match = re.match(r"^      - (.*)$", line)
            if note_match:
                gesture["notes"].append(note_match.group(1).strip())

    for gesture in data["gestures"].values():
        gesture["negative_evidence"] = dict(gesture["negative_evidence"])
    return data


def load_label_ids(labels_path: Path | None, list_files: list[Path]) -> dict[str, int]:
    label_ids = {}
    if labels_path and labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            # labels.yaml is currently sparse, but support simple "- id/name" pairs if added later.
            pass

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
                label_ids.setdefault(label, label_id)
    return label_ids


def concept_node_id(node_type: str, value: str) -> str:
    return f"{node_type.lower()}:{value}"


def build_template_graph(label_name: str, label_id: int, template: dict, weights: dict[str, float]) -> dict:
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
    for idx, event_template in enumerate(template.get("expected_events", []), start=1):
        event_count += 1
        event_id = f"template_event:{label_name}:{idx:03d}"
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
                    node_id = concept_node_id(node_type, value)
                    add_node(nodes, node_id, node_type, {"name": value})
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
            node_id = concept_node_id(node_type, value)
            add_node(nodes, node_id, node_type, {"name": value})
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
    parser.add_argument("--templates", type=Path, default=Path("configs/gesture_templates.yaml"))
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
        "required": float(parsed["template_policy"]["matching"].get("required_weight", DEFAULT_WEIGHTS["required"])),
        "preferred": float(parsed["template_policy"]["matching"].get("preferred_weight", DEFAULT_WEIGHTS["preferred"])),
        "optional": float(parsed["template_policy"]["matching"].get("optional_weight", DEFAULT_WEIGHTS["optional"])),
        "negative": float(parsed["template_policy"]["matching"].get("negative_weight", DEFAULT_WEIGHTS["negative"])),
    }
    label_ids = load_label_ids(args.labels, args.list_file)
    fallback_label_ids = {label: idx for idx, label in enumerate(sorted(gestures))}

    graphs = []
    for label_name in sorted(gestures, key=lambda name: label_ids.get(name, fallback_label_ids[name])):
        label_id = label_ids.get(label_name, fallback_label_ids[label_name])
        graph = build_template_graph(label_name, label_id, gestures[label_name], weights)
        graphs.append(graph)
        write_json(graph, args.output_dir / f"{label_name}.json", args.pretty)

    index = {
        "graph_type": "template_graph_index",
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
