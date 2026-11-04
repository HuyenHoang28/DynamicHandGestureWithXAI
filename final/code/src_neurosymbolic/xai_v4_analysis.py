"""Faithfulness analysis of the V4 template-matching explanations (read-only, no training).

For a trained V4 checkpoint this reports, on one split:

* ``presence`` agreement: does the presence head of each template requirement agree with the
  (subject, relation, concept) facts that are actually present in the clip's instance graph, and does the
  weighted satisfaction of the predicted class's requirements separate correct from wrong predictions (AUROC);
* reliance on the symbolic inputs, by perturbing them and re-running the model:
  ``empty_graph`` (instance graph removed), ``shuffled_template`` (class templates assigned to the wrong
  classes), ``delete_supporting`` (events that carry a fact required by the predicted class are removed) against
  ``delete_random`` (the same number of random events removed).

A perturbation that does not change the predictions or the probability of the predicted class shows that the
corresponding explanation component is not what the model relies on.  The split is read once per condition; use
it with a validation split when choosing anything, and a test split only for a final report.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.kg.graph_matching import CONCEPT_RELATIONS, event_subjects, node_map, node_name
from src_neurosymbolic.models.cross_attention_v2 import REQUIREMENT_LEVELS, load_checkpoint_compat
from src_neurosymbolic.models.cross_attention_v4 import DualCrossAttentionV4NeuroSymbolicModel
from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset
from src_neurosymbolic.train_cross_attention_v2 import collate_batch

EMPTY_GRAPH = '{"nodes": [], "edges": []}'
POSITIVE_LEVELS = (
    REQUIREMENT_LEVELS["required"],
    REQUIREMENT_LEVELS["preferred"],
    REQUIREMENT_LEVELS["optional"],
)


def event_fact_map(graph: dict) -> dict[str, set[tuple[str, str, str]]]:
    """``{event_id: {(subject, relation, concept)}}`` for the concept edges of each event."""

    nodes = node_map(graph)
    subjects = event_subjects(graph)
    facts: dict[str, set[tuple[str, str, str]]] = {}
    for edge in graph.get("edges", []):
        if edge.get("relation") not in CONCEPT_RELATIONS:
            continue
        event_id = edge["source"]
        facts.setdefault(event_id, set()).add(
            (subjects.get(event_id, "any"), edge["relation"], node_name(nodes, edge["target"]))
        )
    return facts


def fact_matches(requirement: tuple[str, str, str], fact: tuple[str, str, str]) -> bool:
    subject, relation, concept = requirement
    if subject == "any":
        return fact[1] == relation and fact[2] == concept
    return fact == requirement


def supporting_events(graph: dict, requirements: list[tuple[str, str, str]]) -> list[str]:
    """Events that carry at least one fact required by a class template."""

    return [
        event_id
        for event_id, facts in event_fact_map(graph).items()
        if any(fact_matches(requirement, fact) for requirement in requirements for fact in facts)
    ]


def delete_events(graph: dict, event_ids: set[str]) -> dict:
    """Copy of ``graph`` without the given event nodes and every edge touching them."""

    return {
        **graph,
        "nodes": [node for node in graph.get("nodes", []) if node.get("id") not in event_ids],
        "edges": [
            edge
            for edge in graph.get("edges", [])
            if edge.get("source") not in event_ids and edge.get("target") not in event_ids
        ],
    }


def auroc(scores: list[float], positives: list[bool]) -> float | None:
    """Rank-based AUROC (ties get half credit); None when one class is missing."""

    pos = [score for score, flag in zip(scores, positives) if flag]
    neg = [score for score, flag in zip(scores, positives) if not flag]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def build_v4(checkpoint: dict, template_dir: Path, device: torch.device) -> DualCrossAttentionV4NeuroSymbolicModel:
    if checkpoint.get("model_version") != "v4_structured_requirement_dual_cross_attention":
        raise ValueError(f"Only V4 checkpoints are supported, got {checkpoint.get('model_version')!r}")
    saved = checkpoint["args"]
    model = DualCrossAttentionV4NeuroSymbolicModel(
        template_dir=template_dir,
        num_classes=int(saved["num_classes"]),
        num_frames=int(saved["num_frames"]),
        d_model=int(saved["d_model"]),
        layers=int(saved["layers"]),
        heads=int(saved["heads"]),
        graph_layers=int(saved["graph_layers"]),
        decoder_layers=int(saved["decoder_layers"]),
        instance_tokens=int(saved["instance_tokens"]),
        dropout=float(saved["dropout"]),
        directed_temporal_edges=str(saved.get("directed_temporal_edges", "False")).lower() == "true",
        use_motion_features=str(saved.get("neural_motion_features", "False")).lower() == "true",
        neural_pooling=str(saved.get("neural_pooling", "mean")),
        hand_features=str(saved.get("neural_hand_features", "False")).lower() == "true",
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def class_requirements(model: DualCrossAttentionV4NeuroSymbolicModel) -> list[list[tuple[str, str, str]]]:
    """Positive (required/preferred/optional) facts of every class template."""

    encoder = model.graph_encoder
    levels = encoder.template_requirements.cpu().tolist()
    result = []
    for class_facts, class_levels in zip(encoder.template_requirement_facts, levels):
        result.append(
            [
                fact
                for fact, level in zip(class_facts, class_levels)
                if fact is not None and level in POSITIVE_LEVELS
            ]
        )
    return result


def shuffled_templates(model: DualCrossAttentionV4NeuroSymbolicModel, seed: int):
    """Context-free patch that assigns every class the template of another class."""

    classes = model.graph_encoder.template_padding_mask.shape[0]
    generator = random.Random(seed)
    perm = list(range(classes))
    for _ in range(1000):
        generator.shuffle(perm)
        if all(index != value for index, value in enumerate(perm)):
            break
    index = torch.tensor(perm, dtype=torch.long, device=model.graph_encoder.template_padding_mask.device)
    original = model.graph_encoder.encode_templates

    def patched():
        out = dict(original())
        for key in ("tokens", "mask", "requirements", "requirement_weights"):
            out[key] = out[key][index]
        return out

    model.graph_encoder.encode_templates = patched
    return original


@torch.no_grad()
def forward_batches(model, batches: list[dict], graphs: list[list[str]], device: torch.device) -> list[dict]:
    outputs = []
    for batch, graph_json in zip(batches, graphs):
        out = model(batch["keypoints"].to(device), graph_json)
        outputs.append({key: value.detach().cpu() for key, value in out.items() if isinstance(value, torch.Tensor)})
    return outputs


def cat(outputs: list[dict], key: str) -> torch.Tensor:
    return torch.cat([out[key] for out in outputs], dim=0)


def branch_summary(outputs: list[dict], labels: torch.Tensor, reference_pred: torch.Tensor) -> dict:
    result = {}
    for name, key in (("final", "logits"), ("neural", "neural_logits"), ("kg", "kg_logits")):
        logits = cat(outputs, key)
        probs = logits.softmax(dim=1)
        pred = logits.argmax(dim=1)
        result[name] = {
            "accuracy": float((pred == labels).float().mean()),
            "mean_prob_of_reference_class": float(probs.gather(1, reference_pred.unsqueeze(1)).mean()),
            "prediction_changed_vs_reference": float((pred != reference_pred).float().mean()),
        }
    return result


def presence_report(model, outputs: list[dict], labels: torch.Tensor, pred: torch.Tensor) -> dict:
    presence = cat(outputs, "presence")
    targets = cat(outputs, "requirement_targets")
    levels = cat(outputs, "requirement_ids")
    weights = cat(outputs, "requirement_weights")
    pad = model.graph_encoder.template_padding_mask.cpu().unsqueeze(0).expand_as(presence)
    positive = torch.zeros_like(presence, dtype=torch.bool)
    for level in POSITIVE_LEVELS:
        positive |= levels == level
    positive &= ~pad
    agree = ((presence > 0.5) == (targets > 0.5))

    def agreement(mask: torch.Tensor) -> float | None:
        return float(agree[mask].float().mean()) if bool(mask.any()) else None

    index = torch.arange(presence.size(0))
    pred_mask = torch.zeros_like(positive)
    pred_mask[index, pred] = True
    true_mask = torch.zeros_like(positive)
    true_mask[index, labels] = True

    def satisfaction(source: torch.Tensor, selector: torch.Tensor) -> list[float]:
        chosen = selector & positive
        w = weights * chosen.float()
        return ((source * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp(min=1e-6)).tolist()

    correct = (pred == labels).tolist()
    predicted_presence = satisfaction(presence, pred_mask)
    predicted_fact_support = satisfaction(targets, pred_mask)
    return {
        "presence_vs_fact_agreement_all_classes": agreement(positive),
        "presence_vs_fact_agreement_predicted_class": agreement(positive & pred_mask),
        "presence_vs_fact_agreement_true_class": agreement(positive & true_mask),
        "auroc_correct_from_predicted_class_presence": auroc(predicted_presence, correct),
        "auroc_correct_from_predicted_class_fact_support": auroc(predicted_fact_support, correct),
        "mean_presence_predicted_class_correct": _mean_where(predicted_presence, correct, True),
        "mean_presence_predicted_class_wrong": _mean_where(predicted_presence, correct, False),
    }


def _mean_where(values: list[float], flags: list[bool], wanted: bool) -> float | None:
    chosen = [value for value, flag in zip(values, flags) if flag == wanted]
    return sum(chosen) / len(chosen) if chosen else None


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = load_checkpoint_compat(args.checkpoint)
    model = build_v4(checkpoint, args.template_dir, device)
    saved = checkpoint["args"]
    dataset = KeypointSequenceDataset(
        args.test_file, args.keypoint_root, args.split, int(saved["num_frames"]), args.limit,
        args.cache_root, args.instance_graph_root,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    batches = [batch for batch in loader]
    labels = torch.cat([batch["label"] for batch in batches])
    graphs = [list(batch["instance_graph_json"]) for batch in batches]

    baseline = forward_batches(model, batches, graphs, device)
    pred = cat(baseline, "logits").argmax(dim=1)
    report: dict = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "clips": int(labels.numel()),
        "baseline": branch_summary(baseline, labels, pred),
        "presence": presence_report(model, baseline, labels, pred),
        "perturbations": {},
    }

    report["perturbations"]["empty_graph"] = branch_summary(
        forward_batches(model, batches, [[EMPTY_GRAPH] * len(g) for g in graphs], device), labels, pred
    )

    original = shuffled_templates(model, args.seed)
    try:
        report["perturbations"]["shuffled_template"] = branch_summary(
            forward_batches(model, batches, graphs, device), labels, pred
        )
    finally:
        model.graph_encoder.encode_templates = original

    requirements = class_requirements(model)
    rng = random.Random(args.seed)
    supporting_graphs, random_graphs, removed = [], [], []
    flat_pred = pred.tolist()
    cursor = 0
    for graph_json in graphs:
        support_batch, random_batch = [], []
        for text in graph_json:
            graph = json.loads(text)
            events = [node["id"] for node in graph.get("nodes", []) if node.get("type") == "Event"]
            support = supporting_events(graph, requirements[flat_pred[cursor]])
            others = rng.sample(events, min(len(support), len(events)))
            support_batch.append(json.dumps(delete_events(graph, set(support))))
            random_batch.append(json.dumps(delete_events(graph, set(others))))
            removed.append((len(support), len(events)))
            cursor += 1
        supporting_graphs.append(support_batch)
        random_graphs.append(random_batch)
    report["events_removed"] = {
        "mean_supporting_per_clip": sum(r[0] for r in removed) / max(len(removed), 1),
        "mean_events_per_clip": sum(r[1] for r in removed) / max(len(removed), 1),
    }
    report["perturbations"]["delete_supporting"] = branch_summary(
        forward_batches(model, batches, supporting_graphs, device), labels, pred
    )
    report["perturbations"]["delete_random"] = branch_summary(
        forward_batches(model, batches, random_graphs, device), labels, pred
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--instance-graph-root", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
