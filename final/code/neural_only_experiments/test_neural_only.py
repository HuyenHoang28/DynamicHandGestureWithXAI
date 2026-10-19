import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import NeuralOnlyKeypointDataset, collate_batch
from models import NeuralOnlyClassifier


def classification_metrics(predictions: list[int], labels: list[int], num_classes: int) -> dict:
    per_class = []
    f1_values = []
    for class_id in range(num_classes):
        support = sum(label == class_id for label in labels)
        predicted = sum(pred == class_id for pred in predictions)
        true_positive = sum(
            pred == class_id and label == class_id for pred, label in zip(predictions, labels)
        )
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "class_id": class_id,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {"macro_f1": sum(f1_values) / max(len(f1_values), 1), "per_class": per_class}


@torch.no_grad()
def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = argparse.Namespace(**checkpoint["args"])
    label_names = {int(k): v for k, v in checkpoint.get("label_names", {}).items()}
    test_set = NeuralOnlyKeypointDataset(args.test_file, args.cache_root, "test", model_args.num_frames)
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    model = NeuralOnlyClassifier(
        backbone=model_args.backbone,
        num_classes=model_args.num_classes,
        num_frames=model_args.num_frames,
        d_model=model_args.d_model,
        layers=model_args.layers,
        heads=model_args.heads,
        dropout=model_args.dropout,
        use_motion_features=getattr(model_args, "disable_motion_features", True) is False,
        pooling=getattr(model_args, "pooling", "mean"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    rows = []
    inference_seconds = 0.0
    timed_samples = 0
    predictions = []
    targets = []
    for batch_index, batch in enumerate(loader):
        x = batch["keypoints"].to(device)
        y = batch["label"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.timing_repeats):
            logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if batch_index >= args.timing_warmup_batches:
            inference_seconds += elapsed
            timed_samples += y.numel() * args.timing_repeats
        loss = criterion(logits, y)
        probs = logits.softmax(dim=1)
        values, indices = probs.topk(k=min(5, model_args.num_classes), dim=1)
        pred = indices[:, 0]
        predictions.extend(pred.tolist())
        targets.extend(y.tolist())
        total += y.numel()
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * y.numel()
        for i in range(y.numel()):
            true_id = int(y[i].item())
            pred_id = int(pred[i].item())
            rows.append(
                {
                    "path": batch["path"][i],
                    "true_id": true_id,
                    "true_label": label_names.get(true_id, str(true_id)),
                    "pred_id": pred_id,
                    "pred_label": label_names.get(pred_id, str(pred_id)),
                    "confidence": float(values[i, 0].item()),
                    "top_labels": " ".join(label_names.get(int(idx), str(int(idx))) for idx in indices[i].tolist()),
                    "top_probs": " ".join(f"{float(value):.6f}" for value in values[i].tolist()),
                }
            )
    detailed_metrics = classification_metrics(predictions, targets, model_args.num_classes)
    metrics = {
        "backbone": model_args.backbone,
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        **detailed_metrics,
        "correct": correct,
        "wrong": total - correct,
        "total": total,
        "inference_timing": {
            "scope": "model_forward_only",
            "device": str(device),
            "batch_size": args.batch_size,
            "warmup_batches": min(args.timing_warmup_batches, len(loader)),
            "repeats_per_batch": args.timing_repeats,
            "timed_samples": timed_samples,
            "total_seconds": inference_seconds,
            "latency_ms_per_sample": 1000.0 * inference_seconds / max(timed_samples, 1),
            "throughput_samples_per_second": timed_samples / max(inference_seconds, 1e-12),
        },
    }
    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        pred_path = args.output.with_name("predictions.csv")
        if rows:
            with pred_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"saved metrics: {args.output}")
        print(f"saved predictions: {pred_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=Path("test.txt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--timing-warmup-batches", type=int, default=2)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
