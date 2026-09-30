import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.models.cross_attention_fusion import NeuralKeypointTemplateCrossAttention
from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset, load_label_names
from src_neurosymbolic.train_keypoint_cross_attention import collate_batch


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, top_k: int) -> dict:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    rows = []
    num_classes = None

    for batch in loader:
        x = batch["keypoints"].to(device)
        y = batch["label"].to(device)
        logits = model(x, instance_graph_json=batch.get("instance_graph_json"))
        probs = logits.softmax(dim=1)
        loss = criterion(logits, y)
        total_loss += float(loss.item())
        if num_classes is None:
            num_classes = logits.shape[1]

        values, indices = probs.topk(k=min(top_k, logits.shape[1]), dim=1)
        preds = indices[:, 0]
        for i in range(y.numel()):
            rows.append(
                {
                    "path": batch["path"][i],
                    "true_id": int(y[i].item()),
                    "pred_id": int(preds[i].item()),
                    "confidence": float(values[i, 0].item()),
                    "top_ids": [int(v) for v in indices[i].tolist()],
                    "top_probs": [float(v) for v in values[i].tolist()],
                }
            )

    return {
        "rows": rows,
        "loss": total_loss / max(len(rows), 1),
        "num_classes": num_classes or 0,
    }


def compute_metrics(rows: list[dict], num_classes: int) -> tuple[dict, list[list[int]], list[dict]]:
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    correct = 0
    for row in rows:
        true_id = row["true_id"]
        pred_id = row["pred_id"]
        confusion[true_id][pred_id] += 1
        correct += int(true_id == pred_id)

    per_class = []
    for class_id in range(num_classes):
        tp = confusion[class_id][class_id]
        support = sum(confusion[class_id])
        predicted = sum(confusion[true_id][class_id] for true_id in range(num_classes))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "label_id": class_id,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_f1 = sum(item["f1"] for item in per_class) / max(num_classes, 1)
    metrics = {
        "accuracy": correct / max(len(rows), 1),
        "macro_f1": macro_f1,
        "total": len(rows),
        "correct": correct,
        "wrong": len(rows) - correct,
    }
    return metrics, confusion, per_class


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = argparse.Namespace(**checkpoint["args"])

    keypoint_root = args.keypoint_root if args.keypoint_root is not None else model_args.keypoint_root
    cache_root = args.cache_root if args.cache_root is not None else model_args.cache_root
    label_names = checkpoint.get("label_names") or load_label_names(args.test_file)
    label_names = {int(k): v for k, v in label_names.items()}

    instance_graph_root = args.instance_graph_root if args.instance_graph_root is not None else getattr(model_args, "instance_graph_root", None)
    dataset = KeypointSequenceDataset(args.test_file, keypoint_root, args.split, model_args.num_frames, None, cache_root, instance_graph_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = NeuralKeypointTemplateCrossAttention(
        num_classes=model_args.num_classes,
        d_model=model_args.d_model,
        num_layers=model_args.layers,
        num_heads=model_args.heads,
        template_tokens=model_args.template_tokens,
        dropout=model_args.dropout,
        max_frames=model_args.num_frames,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    out = collect_predictions(model, loader, device, args.top_k)
    metrics, confusion, per_class = compute_metrics(out["rows"], out["num_classes"])
    metrics["loss"] = out["loss"]
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["test_file"] = str(args.test_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "confusion_matrix.json").write_text(json.dumps(confusion), encoding="utf-8")

    per_class_rows = [
        {
            **item,
            "label_name": label_names.get(item["label_id"], str(item["label_id"])),
        }
        for item in per_class
    ]
    write_csv(
        args.output_dir / "per_class_metrics.csv",
        per_class_rows,
        ["label_id", "label_name", "support", "predicted", "precision", "recall", "f1"],
    )

    prediction_rows = []
    wrong_rows = []
    for row in out["rows"]:
        item = {
            "path": row["path"],
            "true_id": row["true_id"],
            "true_label": label_names.get(row["true_id"], str(row["true_id"])),
            "pred_id": row["pred_id"],
            "pred_label": label_names.get(row["pred_id"], str(row["pred_id"])),
            "confidence": row["confidence"],
            "top_ids": " ".join(map(str, row["top_ids"])),
            "top_labels": " ".join(label_names.get(i, str(i)) for i in row["top_ids"]),
            "top_probs": " ".join(f"{p:.6f}" for p in row["top_probs"]),
        }
        prediction_rows.append(item)
        if row["true_id"] != row["pred_id"]:
            wrong_rows.append(item)

    fields = ["path", "true_id", "true_label", "pred_id", "pred_label", "confidence", "top_ids", "top_labels", "top_probs"]
    write_csv(args.output_dir / "predictions.csv", prediction_rows, fields)
    write_csv(args.output_dir / "wrong_predictions.csv", wrong_rows, fields)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Wrong: {metrics['wrong']}/{metrics['total']}")
    print(f"Saved report: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed evaluation for keypoint cross-attention model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=Path("test.txt"))
    parser.add_argument("--keypoint-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--instance-graph-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/keypoint_cross_attention_cached/test_report"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
