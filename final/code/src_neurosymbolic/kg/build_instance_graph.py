import argparse
import json
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


def load_events(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "events" not in data:
        raise ValueError(f"Expected event object with events: {path}")
    return data


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


def evidence_node_id(event_id: str) -> str:
    return f"evidence_{event_id}"


def build_instance_graph(data: dict) -> dict:
    video_id = data.get("video_id") or "unknown_video"
    video_node_id = f"video:{video_id}"
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    add_node(
        nodes,
        video_node_id,
        "Video",
        {
            "video_id": video_id,
            "source_path": data.get("source_path", ""),
        },
    )

    events = sorted(data.get("events", []), key=lambda e: (e["start_frame"], e["end_frame"], e["event_id"]))
    for event in events:
        event_id = f"event:{event['event_id']}"
        add_node(
            nodes,
            event_id,
            "Event",
            {
                "event_id": event["event_id"],
                "subject": event["subject"],
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "duration": event["duration"],
                "confidence": event["confidence"],
                "source_atomic_event_count": event.get("source_atomic_event_count", 0),
                "atomic_event_ids": event.get("atomic_event_ids", []),
            },
        )
        add_edge(
            edges,
            video_node_id,
            "HAS_EVENT",
            event_id,
            {"confidence": event["confidence"]},
        )

        subject_id = f"bodypart:{event['subject']}"
        add_node(nodes, subject_id, "BodyPart", {"name": event["subject"]})
        add_edge(edges, event_id, "INVOLVES", subject_id)

        for field, (node_type, relation) in FIELD_TO_NODE_TYPE_AND_RELATION.items():
            for value in event.get(field, []):
                concept_id = f"{node_type.lower()}:{value}"
                add_node(nodes, concept_id, node_type, {"name": value})
                add_edge(edges, event_id, relation, concept_id, {"confidence": event["confidence"]})

        ev_node_id = f"evidence:{event['event_id']}"
        add_node(
            nodes,
            ev_node_id,
            "Evidence",
            {
                "source": "composite_event",
                "confidence": event["confidence"],
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "evidence_keypoints": event.get("evidence_keypoints", []),
                "atomic_event_ids": event.get("atomic_event_ids", []),
            },
        )
        add_edge(edges, event_id, "HAS_EVIDENCE", ev_node_id, {"confidence": event["confidence"]})

    add_temporal_edges(events, edges)

    return {
        "graph_type": "instance_graph",
        "video_id": video_id,
        "source_path": data.get("source_path", ""),
        "nodes": list(nodes.values()),
        "edges": edges,
        "summary": {
            "event_count": len(events),
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }


def add_temporal_edges(events: list[dict], edges: list[dict]) -> None:
    ordered = sorted(events, key=lambda e: (e["start_frame"], e["end_frame"], e["event_id"]))
    for prev, curr in zip(ordered, ordered[1:]):
        prev_id = f"event:{prev['event_id']}"
        curr_id = f"event:{curr['event_id']}"
        if prev["end_frame"] < curr["start_frame"]:
            relation = "BEFORE"
        else:
            relation = "OVERLAPS"
        add_edge(
            edges,
            prev_id,
            relation,
            curr_id,
            {
                "gap": curr["start_frame"] - prev["end_frame"],
                "prev_end_frame": prev["end_frame"],
                "next_start_frame": curr["start_frame"],
            },
        )


def write_json(data: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an instance KG from composite event JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Input composite event JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output instance graph JSON.")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_events(args.input)
    graph = build_instance_graph(data)
    write_json(graph, args.output, args.pretty)
    print(f"Saved instance graph: {args.output}")
    print(f"Events: {graph['summary']['event_count']}")
    print(f"Nodes: {graph['summary']['node_count']}")
    print(f"Edges: {graph['summary']['edge_count']}")


if __name__ == "__main__":
    main()
