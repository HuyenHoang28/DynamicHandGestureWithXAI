import argparse
import json
from collections import defaultdict
from pathlib import Path

from src_neurosymbolic.kg.build_instance_graph import (
    FIELD_TO_NODE_TYPE_AND_RELATION,
    add_edge,
    add_node,
    load_events,
)


def build_instance_graph_phase_v2(data: dict) -> dict:
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
            "event_builder": data.get("summary", {}).get("builder", "phase_v2"),
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
                "phase_event": bool(event.get("phase_event", False)),
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
                "source": "phase_composite_event",
                "confidence": event["confidence"],
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "evidence_keypoints": event.get("evidence_keypoints", []),
                "atomic_event_ids": event.get("atomic_event_ids", []),
            },
        )
        add_edge(edges, event_id, "HAS_EVIDENCE", ev_node_id, {"confidence": event["confidence"]})

    add_temporal_edges_by_subject(events, edges)

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
            "event_builder": data.get("summary", {}).get("builder", "phase_v2"),
            "temporal_scope": "per_subject",
        },
    }


def add_temporal_edges_by_subject(events: list[dict], edges: list[dict]) -> None:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_subject[str(event.get("subject", ""))].append(event)

    for subject, subject_events in by_subject.items():
        ordered = sorted(subject_events, key=lambda e: (e["start_frame"], e["end_frame"], e["event_id"]))
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
                    "temporal_scope": "subject",
                    "subject": subject,
                },
            )


def write_json(data: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a phase-aware instance graph from phase events.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_events(args.input)
    graph = build_instance_graph_phase_v2(data)
    write_json(graph, args.output, args.pretty)
    print(f"Saved phase-aware instance graph: {args.output}")
    print(graph["summary"])


if __name__ == "__main__":
    main()
