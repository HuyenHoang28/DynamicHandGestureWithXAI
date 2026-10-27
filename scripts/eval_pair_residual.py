"""Evaluate the sparse pair residual on validation predictions and report net corrections.

Inputs are a neural predictions CSV (``test_neural_only.py`` format, with
``top_labels``/``top_probs``) and the factor CSV from ``extract_pair_factors.py``.
Only ``train``/``val`` factor rows are accepted, so this cannot be run on the
IPN test split by accident. It sweeps ``alpha``, the neural margin and the
quality floor, and reports beneficial, harmful and neutral changes plus
accuracy and macro-F1 before and after. Choose the setting on validation and
report it as frozen before any later test look.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.pair_residual import (  # noqa: E402
    LABEL_IDS,
    NUM_CLASSES,
    PAIR_FAMILIES,
    FactorDecision,
    apply_pair_residual,
    correction_effect,
)

ALLOWED_SPLITS = {"train", "val"}


def clip_key(path: str) -> str:
    return path.replace(chr(92), "/").rsplit("/", 1)[-1].lower()


def load_predictions(path: Path) -> tuple[list[str], torch.Tensor, list[int]]:
    keys, rows, labels = [], [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            probs = torch.full((NUM_CLASSES,), 1e-9)
            for name, value in zip(row["top_labels"].split(), row["top_probs"].split()):
                probs[LABEL_IDS[name]] = max(float(value), 1e-9)
            keys.append(clip_key(row["path"]))
            rows.append(probs.log())
            labels.append(int(row["true_id"]))
    if not keys:
        raise ValueError(f"no predictions in {path}")
    return keys, torch.stack(rows), labels


def load_factors(path: Path) -> dict[str, dict[str, FactorDecision]]:
    result: dict[str, dict[str, FactorDecision]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in ALLOWED_SPLITS:
                raise ValueError(f"refusing factor row from split {row['split']!r}; only train/val are allowed")
            decisions = {}
            for family in PAIR_FAMILIES:
                guess = row[f"{family}_guess"]
                decisions[family] = FactorDecision(
                    None if guess == "unknown" else LABEL_IDS[guess], float(row[f"{family}_quality"])
                )
            result[clip_key(row["source"])] = decisions
    return result


def macro_f1(predictions: list[int], labels: list[int]) -> float:
    scores = []
    for cls in range(NUM_CLASSES):
        tp = sum(p == cls and t == cls for p, t in zip(predictions, labels))
        pred = sum(p == cls for p in predictions)
        true = sum(t == cls for t in labels)
        precision = tp / pred if pred else 0.0
        recall = tp / true if true else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / NUM_CLASSES


def evaluate(logits: torch.Tensor, labels: list[int], decisions: list[dict[str, FactorDecision]], alpha: float, margin: float, quality: float) -> dict:
    before = logits.argmax(dim=-1).tolist()
    new_logits, applied, pair_index = apply_pair_residual(
        logits, decisions, alpha=alpha, max_neural_margin=margin, min_quality=quality
    )
    after = new_logits.argmax(dim=-1).tolist()
    effect = correction_effect(before, after, labels)
    per_pair = {}
    for index, family in enumerate(PAIR_FAMILIES):
        members = [i for i in range(len(labels)) if int(pair_index[i]) == index]
        per_pair[family] = {"applied": len(members), **correction_effect(
            [before[i] for i in members], [after[i] for i in members], [labels[i] for i in members]
        )}
    return {
        "alpha": alpha, "max_neural_margin": margin, "min_quality": quality,
        "applied": int(applied.sum()), **effect, "per_pair": per_pair,
        "accuracy_before": sum(p == t for p, t in zip(before, labels)) / len(labels),
        "accuracy_after": sum(p == t for p, t in zip(after, labels)) / len(labels),
        "macro_f1_before": macro_f1(before, labels), "macro_f1_after": macro_f1(after, labels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Neural validation predictions CSV (top_labels/top_probs).")
    parser.add_argument("--factors", type=Path, required=True, help="CSV from extract_pair_factors.py (train/val rows only).")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--margins", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6, 1.0])
    parser.add_argument("--qualities", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    keys, logits, labels = load_predictions(args.predictions)
    factors = load_factors(args.factors)
    missing = [key for key in keys if key not in factors]
    if missing:
        raise ValueError(f"{len(missing)} prediction clips have no factor row, e.g. {missing[:3]}")
    decisions = [factors[key] for key in keys]
    grid = [
        evaluate(logits, labels, decisions, alpha, margin, quality)
        for alpha, margin, quality in itertools.product(args.alphas, args.margins, args.qualities)
    ]
    grid.sort(key=lambda row: (-row["net"], row["harmful"], row["alpha"], row["max_neural_margin"], row["min_quality"]))
    report = {
        "clips": len(labels),
        "baseline": evaluate(logits, labels, decisions, 0.0, 0.0, 0.0),
        "best_by_net": grid[0],
        "grid": grid,
        "note": "Choose the setting on validation, freeze it, and report beneficial/harmful/net. A positive net on one split is not evidence until it holds across seeds and subjects.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = grid[0]
    print(f"clips={len(labels)} baseline acc={report['baseline']['accuracy_before']:.4f}")
    print(
        f"best: alpha={best['alpha']} margin={best['max_neural_margin']} min_quality={best['min_quality']} "
        f"applied={best['applied']} beneficial={best['beneficial']} harmful={best['harmful']} net={best['net']} "
        f"acc {best['accuracy_before']:.4f}->{best['accuracy_after']:.4f} macroF1 {best['macro_f1_before']:.4f}->{best['macro_f1_after']:.4f}"
    )
    if best["net"] <= 0:
        print("No setting gives a positive net correction; leave the residual off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
