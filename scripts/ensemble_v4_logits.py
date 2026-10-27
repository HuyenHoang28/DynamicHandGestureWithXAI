"""Average the logits of several V4 checkpoints (saved with ``test_cross_attention_v2.py --save-logits``).

Each member's ``test_logits.pt`` holds raw ``final``, ``neural`` and ``kg`` logits for the same
clips. This averages softmax probabilities of one branch across members, which is the same
rule ``ensemble_neural_only.py`` uses, so a V4 ensemble (symbolic head included) can be compared
with a neural-only ensemble of the checkpoints it was initialised from.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code" / "neural_only_experiments"))
sys.path.insert(0, str(ROOT / "final" / "code"))

from test_neural_only import classification_metrics  # noqa: E402
from src_neurosymbolic.pair_residual import LABEL_IDS  # noqa: E402

BRANCHES = ("final", "neural", "kg")


def average_branch_probabilities(members: list[dict], branch: str, weights: list[float] | None = None) -> torch.Tensor:
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}")
    if not members:
        raise ValueError("at least one member is required")
    reference = members[0]
    for member in members[1:]:
        if member["paths"] != reference["paths"] or member["labels"] != reference["labels"]:
            raise ValueError("members were evaluated on different clips or in a different order")
    if weights is None:
        weights = [1.0] * len(members)
    if len(weights) != len(members) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative, match the members, and sum to a positive number")
    total = float(sum(weights))
    return sum(m["logits"][branch].softmax(dim=-1) * (w / total) for m, w in zip(members, weights))


def accuracy(predictions: list[int], labels: list[int]) -> float:
    return sum(p == t for p, t in zip(predictions, labels)) / len(labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, nargs="+", required=True, help="test_logits.pt files, one per checkpoint.")
    parser.add_argument("--branch", choices=BRANCHES, default="final")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    members = [torch.load(path, map_location="cpu", weights_only=False) for path in args.logits]
    probabilities = average_branch_probabilities(members, args.branch, args.weights)
    labels = members[0]["labels"]
    num_classes = probabilities.shape[1]
    names = {value: key for key, value in LABEL_IDS.items()} if num_classes == len(LABEL_IDS) else None
    predictions = probabilities.argmax(dim=-1).tolist()
    metrics = classification_metrics(predictions, labels, num_classes, names)
    metrics["accuracy"] = accuracy(predictions, labels)
    metrics["total"] = len(labels)
    metrics["branch"] = args.branch
    metrics["method"] = "mean_of_softmax_probabilities"
    metrics["members"] = [
        {
            "logits": str(path),
            "accuracy": accuracy(member["logits"][args.branch].argmax(dim=-1).tolist(), labels),
            "macro_f1": classification_metrics(member["logits"][args.branch].argmax(dim=-1).tolist(), labels, num_classes, names)["macro_f1"],
        }
        for path, member in zip(args.logits, members)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"members={[round(m['accuracy'], 4) for m in metrics['members']]} "
          f"ensemble[{args.branch}] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
