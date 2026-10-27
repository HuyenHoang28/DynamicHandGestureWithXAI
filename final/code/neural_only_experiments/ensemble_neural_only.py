"""Average several neural-only checkpoints and score the ensemble on one split.

Different seeds make different mistakes, so averaging their probabilities usually
beats any single seed when single-run validation swings by several points (as it
does on IPN, where each gesture has only 28 validation clips). Use it with the
checkpoints of one recipe trained with different ``--seed`` values, chosen and
weighted on validation only.

Example:
    python ensemble_neural_only.py --checkpoints a/best.pt b/best.pt c/best.pt \
        --cache-root <cache> --test-file val.txt --split val --output ens/metrics.json
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import NeuralOnlyKeypointDataset, collate_batch, label_names_from_split
from models import NeuralOnlyClassifier
from test_neural_only import classification_metrics, validate_evaluation_label_contract


def build_model(model_args: argparse.Namespace) -> NeuralOnlyClassifier:
    """Rebuild a model from the CLI args stored in its checkpoint (same rules as test_neural_only)."""
    return NeuralOnlyClassifier(
        backbone=model_args.backbone,
        num_classes=model_args.num_classes,
        num_frames=model_args.num_frames,
        d_model=model_args.d_model,
        layers=model_args.layers,
        heads=model_args.heads,
        dropout=model_args.dropout,
        use_motion_features=getattr(model_args, "disable_motion_features", True) is False,
        pooling=getattr(model_args, "pooling", "mean"),
        hand_features=bool(getattr(model_args, "hand_features", False)),
    )


def average_probabilities(per_model_logits: list[torch.Tensor], weights: list[float] | None = None) -> torch.Tensor:
    """Weighted mean of softmax probabilities; ``per_model_logits`` is a list of ``[N, C]`` tensors."""
    if not per_model_logits:
        raise ValueError("At least one model is required")
    shapes = {tuple(item.shape) for item in per_model_logits}
    if len(shapes) != 1:
        raise ValueError(f"Models disagree on output shape: {sorted(shapes)}")
    if weights is None:
        weights = [1.0] * len(per_model_logits)
    if len(weights) != len(per_model_logits) or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be non-negative, match the models, and sum to a positive number")
    total = float(sum(weights))
    stacked = torch.stack([item.softmax(dim=-1) * (weight / total) for item, weight in zip(per_model_logits, weights)])
    return stacked.sum(dim=0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, list[int], list[str]]:
    model.eval()
    logits, labels, paths = [], [], []
    for batch in loader:
        logits.append(model(batch["keypoints"].to(device)).cpu())
        labels.extend(int(value) for value in batch["label"])
        paths.extend(batch["path"])
    return torch.cat(logits), labels, paths


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    per_model_logits: list[torch.Tensor] = []
    reference_labels: list[int] | None = None
    reference_paths: list[str] | None = None
    summaries = []
    label_names: dict[int, str] = {}
    num_classes = None
    evaluation_names = label_names_from_split(args.test_file)
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_args = argparse.Namespace(**checkpoint["args"])
        names = {int(key): value for key, value in checkpoint.get("label_names", {}).items()}
        if num_classes is None:
            num_classes, label_names = model_args.num_classes, names
        elif model_args.num_classes != num_classes or names != label_names:
            raise ValueError(f"{checkpoint_path} has a different class contract from the first checkpoint")
        dataset = NeuralOnlyKeypointDataset(args.test_file, args.cache_root, args.split, model_args.num_frames)
        validate_evaluation_label_contract(dataset.samples, model_args.num_classes, names, evaluation_names)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch
        )
        model = build_model(model_args)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        logits, labels, paths = collect_logits(model, loader, device)
        if reference_labels is None:
            reference_labels, reference_paths = labels, paths
        elif labels != reference_labels or paths != reference_paths:
            raise ValueError("Checkpoints were evaluated on different clip orders")
        single = classification_metrics(logits.argmax(dim=-1).tolist(), labels, num_classes, label_names)
        summaries.append(
            {
                "checkpoint": str(checkpoint_path),
                "sha256": sha256_file(Path(checkpoint_path)),
                "seed": getattr(model_args, "seed", None),
                "accuracy": sum(p == t for p, t in zip(logits.argmax(dim=-1).tolist(), labels)) / len(labels),
                "macro_f1": single["macro_f1"],
            }
        )
        per_model_logits.append(logits)

    probabilities = average_probabilities(per_model_logits, args.weights)
    predictions = probabilities.argmax(dim=-1).tolist()
    metrics = classification_metrics(predictions, reference_labels, num_classes, label_names)
    metrics["accuracy"] = sum(p == t for p, t in zip(predictions, reference_labels)) / len(reference_labels)
    metrics["total"] = len(reference_labels)
    metrics["members"] = summaries
    metrics["weights"] = args.weights or [1.0] * len(summaries)
    metrics["split"] = args.split
    metrics["method"] = "mean_of_softmax_probabilities"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with args.output.with_name("ensemble_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        # top_labels/top_probs match test_neural_only.py so downstream tools (eval_pair_residual.py) read both formats.
        writer.writerow(["path", "true_id", "true_label", "pred_id", "pred_label", "confidence", "top_labels", "top_probs"])
        for path, true_id, pred_id, row in zip(reference_paths, reference_labels, predictions, probabilities):
            top = row.topk(min(5, row.numel()))
            writer.writerow(
                [
                    path, true_id, label_names.get(true_id, str(true_id)), pred_id, label_names.get(pred_id, str(pred_id)),
                    f"{float(row[pred_id]):.6f}",
                    " ".join(label_names.get(int(index), str(int(index))) for index in top.indices),
                    " ".join(f"{float(value):.6f}" for value in top.values),
                ]
            )
    if args.save_probabilities:
        torch.save(
            {"paths": reference_paths, "labels": reference_labels, "per_model_logits": per_model_logits},
            args.output.with_name("member_logits.pt"),
        )
    print(f"members: {[round(item['accuracy'], 4) for item in summaries]}")
    print(f"ensemble accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble neural-only checkpoints by averaging probabilities.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True, help="Manifest of the split to score.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="Optional per-checkpoint weights.")
    parser.add_argument("--save-probabilities", action="store_true", help="Also store each member's raw logits.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
