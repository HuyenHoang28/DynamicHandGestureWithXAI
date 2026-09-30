import argparse
import json
from pathlib import Path


CONCEPT_RELATIONS = {
    "HAS_MOTION",
    "HAS_POSITION",
    "HAS_HAND_SHAPE",
    "HAS_FINGER_MOTION",
    "HAS_COORDINATION",
    "HAS_SWEEP",
    "HAS_JOINT_ANGLE",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def node_map(graph: dict) -> dict[str, dict]:
    return {node["id"]: node for node in graph.get("nodes", [])}


def node_name(nodes: dict[str, dict], node_id: str) -> str:
    node = nodes.get(node_id, {})
    props = node.get("properties", {})
    return props.get("name") or props.get("label_name") or node_id.split(":", 1)[-1]


def event_subjects(graph: dict) -> dict[str, str]:
    nodes = node_map(graph)
    subjects = {}
    for node in graph.get("nodes", []):
        if node.get("type") == "Event":
            props = node.get("properties", {})
            if props.get("subject"):
                subjects[node["id"]] = props["subject"]
    for edge in graph.get("edges", []):
        if edge.get("relation") != "INVOLVES":
            continue
        source = edge["source"]
        target = edge["target"]
        if source in subjects:
            subjects[source] = node_name(nodes, target)
    return subjects


def instance_facts(graph: dict) -> tuple[set[tuple[str, str, str]], dict[tuple[str, str, str], list[dict]]]:
    nodes = node_map(graph)
    subjects = event_subjects(graph)
    max_end_frame = 0
    for node in graph.get("nodes", []):
        if node.get("type") == "Event":
            max_end_frame = max(max_end_frame, int(node.get("properties", {}).get("end_frame") or 0))
    total_frames = max(1, max_end_frame + 1)
    facts = set()
    evidence = {}
    for edge in graph.get("edges", []):
        relation = edge.get("relation")
        if relation not in CONCEPT_RELATIONS:
            continue
        event_id = edge["source"]
        subject = subjects.get(event_id)
        if not subject:
            continue
        concept = node_name(nodes, edge["target"])
        fact = (subject, relation, concept)
        facts.add(fact)
        event_node = nodes.get(event_id, {})
        event_props = event_node.get("properties", {})
        duration = int(event_props.get("duration") or 1)
        confidence = edge.get("properties", {}).get("confidence")
        if confidence is None:
            confidence = event_props.get("confidence", 1.0)
        duration_ratio = max(0.0, min(1.0, duration / total_frames))
        duration_factor = 0.35 + 0.65 * duration_ratio
        strength = max(0.0, min(1.0, duration_factor * float(confidence or 0.0)))
        evidence.setdefault(fact, []).append(
            {
                "event_id": event_props.get("event_id", event_id),
                "event_node": event_id,
                "subject": subject,
                "relation": relation,
                "concept": concept,
                "confidence": confidence,
                "duration": duration,
                "strength": round(strength, 6),
                "start_frame": event_props.get("start_frame"),
                "end_frame": event_props.get("end_frame"),
            }
        )
    return facts, evidence


def template_requirements(graph: dict) -> list[dict]:
    nodes = node_map(graph)
    subjects = event_subjects(graph)
    requirements = []
    for edge in graph.get("edges", []):
        relation = edge.get("relation")
        props = edge.get("properties", {})
        level = props.get("requirement_level")
        if relation not in CONCEPT_RELATIONS or not level:
            continue
        event_id = edge["source"]
        subject = subjects.get(event_id, "any")
        concept = node_name(nodes, edge["target"])
        requirements.append(
            {
                "template_event": event_id,
                "subject": subject,
                "relation": relation,
                "concept": concept,
                "level": level,
                "weight": float(props.get("weight", 0.0)),
                "fact": (subject, relation, concept),
            }
        )
    return requirements


def fact_matches(fact: tuple[str, str, str], instance_fact_set: set[tuple[str, str, str]]) -> bool:
    subject, relation, concept = fact
    if fact in instance_fact_set:
        return True
    if subject == "any":
        return any(inst_relation == relation and inst_concept == concept for _, inst_relation, inst_concept in instance_fact_set)
    return False


def evidence_for_fact(fact: tuple[str, str, str], evidence: dict[tuple[str, str, str], list[dict]]) -> list[dict]:
    subject, relation, concept = fact
    if subject != "any":
        return evidence.get(fact, [])
    matched = []
    for inst_fact, items in evidence.items():
        if inst_fact[1] == relation and inst_fact[2] == concept:
            matched.extend(items)
    return matched


def fact_strength(fact: tuple[str, str, str], evidence: dict[tuple[str, str, str], list[dict]]) -> float:
    items = evidence_for_fact(fact, evidence)
    if not items:
        return 0.0
    return max(float(item.get("strength") or 0.0) for item in items)


def score_template(instance_graph: dict, template_graph: dict) -> dict:
    facts, evidence = instance_facts(instance_graph)
    requirements = template_requirements(template_graph)

    positive_total = sum(req["weight"] for req in requirements if req["weight"] > 0)
    positive_score = 0.0
    negative_penalty = 0.0
    matched = []
    missing_required = []
    negative_hits = []

    for req in requirements:
        hit = fact_matches(req["fact"], facts)
        if hit and req["weight"] > 0:
            strength = fact_strength(req["fact"], evidence)
            positive_score += req["weight"] * strength
            matched.append({**req, "match_strength": round(strength, 6), "evidence": evidence_for_fact(req["fact"], evidence)})
        elif hit and req["weight"] < 0:
            strength = fact_strength(req["fact"], evidence)
            negative_penalty += abs(req["weight"]) * strength
            negative_hits.append({**req, "match_strength": round(strength, 6), "evidence": evidence_for_fact(req["fact"], evidence)})
        elif req["level"] == "required":
            missing_required.append(req)

    required_total = sum(1 for req in requirements if req["level"] == "required")
    required_hit = required_total - len(missing_required)
    required_ratio = (required_hit / required_total) if required_total else 1.0
    raw_score = positive_score - negative_penalty
    normalized_score = raw_score / positive_total if positive_total > 0 else 0.0
    normalized_score = max(0.0, min(1.0, normalized_score)) * required_ratio

    return {
        "label_id": template_graph.get("label_id"),
        "label_name": template_graph.get("label_name"),
        "score": round(normalized_score, 6),
        "raw_score": round(raw_score, 6),
        "positive_score": round(positive_score, 6),
        "negative_penalty": round(negative_penalty, 6),
        "positive_total": round(positive_total, 6),
        "required_hit": required_hit,
        "required_total": required_total,
        "matched_count": len(matched),
        "missing_required_count": len(missing_required),
        "negative_hit_count": len(negative_hits),
        "matched": serialize_requirements(matched),
        "missing_required": serialize_requirements(missing_required),
        "negative_hits": serialize_requirements(negative_hits),
    }


def serialize_requirements(requirements: list[dict]) -> list[dict]:
    serialized = []
    for req in requirements:
        item = {
            "subject": req["subject"],
            "relation": req["relation"],
            "concept": req["concept"],
            "level": req["level"],
            "weight": req["weight"],
            "template_event": req["template_event"],
        }
        if "match_strength" in req:
            item["match_strength"] = req["match_strength"]
        if "evidence" in req:
            item["evidence"] = req["evidence"]
        serialized.append(item)
    return serialized


def load_template_graphs(template_dir: Path) -> list[dict]:
    graphs = []
    for path in sorted(template_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        graph = load_json(path)
        if graph.get("graph_type") == "template_graph":
            graphs.append(graph)
    return sorted(graphs, key=lambda graph: graph.get("label_id", 10**9))


def match_instance_to_templates(instance_graph: dict, template_graphs: list[dict], top_k: int = 5) -> dict:
    scores = [score_template(instance_graph, template) for template in template_graphs]
    scores.sort(key=lambda item: (item["score"], item["required_hit"], item["positive_score"]), reverse=True)
    return {
        "video_id": instance_graph.get("video_id", ""),
        "source_path": instance_graph.get("source_path", ""),
        "prediction": scores[0]["label_name"] if scores else None,
        "prediction_label_id": scores[0]["label_id"] if scores else None,
        "top_k": scores[:top_k],
        "scores": [
            {
                "label_id": item["label_id"],
                "label_name": item["label_name"],
                "score": item["score"],
                "raw_score": item["raw_score"],
                "required_hit": item["required_hit"],
                "required_total": item["required_total"],
                "missing_required_count": item["missing_required_count"],
                "negative_hit_count": item["negative_hit_count"],
            }
            for item in scores
        ],
    }


def write_json(data: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an instance KG against gesture template KGs.")
    parser.add_argument("--instance-graph", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, default=Path("data/template_graphs"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance_graph = load_json(args.instance_graph)
    template_graphs = load_template_graphs(args.template_dir)
    result = match_instance_to_templates(instance_graph, template_graphs, args.top_k)
    if args.output:
        write_json(result, args.output, args.pretty)
    print(f"Video: {result['video_id']}")
    print(f"Prediction: {result['prediction']} ({result['prediction_label_id']})")
    print("Top scores:")
    for item in result["top_k"]:
        print(
            f"  {item['label_id']:2d} {item['label_name']:<20} "
            f"score={item['score']:.4f} required={item['required_hit']}/{item['required_total']} "
            f"missing={item['missing_required_count']} negative={item['negative_hit_count']}"
        )


if __name__ == "__main__":
    main()
