"""Train/evaluate the neural-only Transformer on variable-length windows (step 6).

Training uses ``WindowedVariableKeypointDataset`` (window 256, stride 128) with a
sample-weighted loss so every clip contributes weight 1. Validation/test metrics
are computed once per clip after mean-logit aggregation. The fixed-64 pipeline in
``train_neural_only.py`` is untouched and remains the control.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import label_names_from_split
from models import NeuralOnlyClassifier
from test_neural_only import classification_metrics
from train_neural_only import augment_keypoints, class_weights, validate_no_split_overlap
from variable_dataset import (
    WindowedVariableKeypointDataset,
    aggregate_window_logits,
    collate_variable_batch,
)

LENGTH_BINS = (("lt64", 0, 64), ("64_127", 64, 128), ("128_255", 128, 256), ("ge256", 256, 10**9))
WINDOW_METADATA = ("label", "source", "split", "clip_index", "window_index", "window_count")


def weighted_window_loss(per_sample: torch.Tensor, sample_weight: torch.Tensor,
                         class_weight: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
    """Weighted mean matching ``CrossEntropyLoss(weight=...)`` but with window weights.

    ``per_sample`` is the unreduced loss that already contains the class weight.
    """
    denominator = sample_weight * (class_weight[labels] if class_weight is not None else 1.0)
    return (sample_weight * per_sample).sum() / denominator.sum().clamp_min(1e-12)


def length_group_metrics(predictions: list[int], labels: list[int], lengths: list[int]) -> dict:
    groups = {}
    for name, low, high in LENGTH_BINS:
        index = [i for i, n in enumerate(lengths) if low <= n < high]
        if not index:
            groups[name] = {"clips": 0}
            continue
        correct = sum(predictions[i] == labels[i] for i in index)
        groups[name] = {"clips": len(index), "accuracy": correct / len(index)}
    return groups


@torch.no_grad()
def evaluate_clips(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int,
                   label_names: dict[int, str] | None = None) -> dict:
    """Evaluate on aggregated clip logits; each clip is counted exactly once."""
    model.eval()
    all_logits, metadata, clip_end = [], {key: [] for key in WINDOW_METADATA}, {}
    forward_seconds, windows, peak = 0.0, 0, 0
    for batch in loader:
        x = batch["keypoints"].to(device)
        mask = batch["padding_mask"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = model(x, padding_mask=mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = max(peak, torch.cuda.max_memory_allocated(device))
        forward_seconds += time.perf_counter() - start
        windows += len(batch["label"])
        all_logits.append(logits.float().cpu())
        metadata["label"].extend(batch["label"].tolist())
        for key in WINDOW_METADATA[1:]:
            metadata[key].extend(batch[key])
        for source, split, end in zip(batch["source"], batch["split"], batch["end"]):
            clip_end[(split, source)] = max(clip_end.get((split, source), 0), int(end))
    clips = aggregate_window_logits(torch.cat(all_logits), metadata)
    predictions = clips["logits"].argmax(dim=1).tolist()
    labels = clips["labels"].tolist()
    lengths = [clip_end[(split, path)] for split, path in zip(clips["splits"], clips["paths"])]
    loss = nn.functional.cross_entropy(clips["logits"], clips["labels"]).item()
    detailed = classification_metrics(predictions, labels, num_classes, label_names)
    return {
        "loss": loss,
        "accuracy": sum(p == y for p, y in zip(predictions, labels)) / max(len(labels), 1),
        "total": len(labels),
        **detailed,
        "length_groups": length_group_metrics(predictions, labels, lengths),
        "windows": windows,
        "timing": {"forward_seconds": forward_seconds,
                   "ms_per_clip": 1000.0 * forward_seconds / max(len(labels), 1),
                   "peak_cuda_bytes": peak},
        "_clips": clips,
    }


def build_model(args: argparse.Namespace) -> nn.Module:
    return NeuralOnlyClassifier(
        backbone="transformer", num_classes=args.num_classes, num_frames=args.max_frames,
        d_model=args.d_model, layers=args.layers, heads=args.heads, dropout=args.dropout,
        use_motion_features=not args.disable_motion_features, pooling=args.pooling,
        hand_features=args.hand_features,
    )


def make_loader(path: Path, cache_root: Path, split: str, args: argparse.Namespace, shuffle: bool,
                batch_size: int, generator: torch.Generator | None = None, limit: int | None = None):
    dataset = WindowedVariableKeypointDataset(path, cache_root, split, args.max_frames, args.stride, limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers,
                        collate_fn=collate_variable_batch, generator=generator)
    return dataset, loader


def save_metrics(metrics: dict, output: Path) -> None:
    clips = metrics.pop("_clips")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"logits": clips["logits"], "labels": clips["labels"], "paths": clips["paths"]},
               output.with_name("clip_logits.pt"))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_set, train_loader = make_loader(args.train_file, args.cache_root, "train", args, True,
                                          args.batch_size, generator, args.limit_train)
    val_set, val_loader = make_loader(args.val_file, args.cache_root, "val", args, False,
                                      args.eval_batch_size, None, args.limit_val)
    validate_no_split_overlap(train_set.samples, val_set.samples)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.scheduler_factor, patience=args.scheduler_patience,
        min_lr=args.min_lr)
    weights = class_weights(train_set.samples, args.num_classes, args.class_weight_power)
    weights = weights.to(device) if weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing, reduction="none")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_names = label_names_from_split(args.train_file)
    best_value, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, weight_sum = 0.0, 0.0
        for step, batch in enumerate(train_loader, start=1):
            x = batch["keypoints"].to(device)
            y = batch["label"].to(device)
            mask = batch["padding_mask"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            x = augment_keypoints(x, args.jitter_std, args.frame_mask_probability,
                                  args.keypoint_dropout_probability)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, padding_mask=mask)
            loss = weighted_window_loss(criterion(logits, y), sample_weight, weights, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += float(loss.item()) * float(sample_weight.sum().item())
            weight_sum += float(sample_weight.sum().item())
            if args.log_every and step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss_sum / weight_sum:.4f}")
        val = evaluate_clips(model, val_loader, device, args.num_classes, label_names)
        val.pop("_clips")
        score = float(val[args.select_metric])
        scheduler.step(score)
        improved = score > best_value + args.early_stop_min_delta
        if improved:
            best_value, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": loss_sum / max(weight_sum, 1e-12),
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                        "val_loss": val["loss"], "improved": improved, "best_epoch": best_epoch})
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.4f} val_acc={val['accuracy']:.4f} "
              f"val_macro_f1={val['macro_f1']:.4f} patience={stale}/{args.early_stop_patience}")
        checkpoint = {"model": model.state_dict(), "args": vars(args), "label_names": label_names,
                      "epoch": epoch, "best_epoch": best_epoch, "best_metric_value": best_value,
                      "select_metric": args.select_metric, "val": {k: val[k] for k in ("accuracy", "macro_f1")}}
        torch.save(checkpoint, args.output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break
    print(f"best epoch: {best_epoch} best val {args.select_metric}: {best_value:.4f}")


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**checkpoint["args"])
    saved.num_workers = args.num_workers
    _, loader = make_loader(args.eval_file, args.cache_root, args.split, saved, False, args.eval_batch_size)
    model = build_model(saved).to(device)
    model.load_state_dict(checkpoint["model"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    label_names = {int(k): v for k, v in checkpoint.get("label_names", {}).items()}
    metrics = evaluate_clips(model, loader, device, saved.num_classes, label_names)
    save_metrics(metrics, args.output)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "total", "length_groups")}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--num-classes", type=int, default=13)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--pooling", choices=["mean", "masked_mean", "attention"], default="masked_mean")
    p.add_argument("--disable-motion-features", action="store_true")
    p.add_argument("--hand-features", action="store_true")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--jitter-std", type=float, default=0.005)
    p.add_argument("--frame-mask-probability", type=float, default=0.0)
    p.add_argument("--keypoint-dropout-probability", type=float, default=0.02)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-patience", type=int, default=2)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--early-stop-patience", type=int, default=12)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--select-metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-val", type=int)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--train-file", type=Path, default=Path("train.txt"))
    p.add_argument("--val-file", type=Path, default=Path("val.txt"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--checkpoint", type=Path, help="Evaluate this checkpoint instead of training.")
    p.add_argument("--eval-file", type=Path)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--output", type=Path, help="metrics.json path for --checkpoint evaluation.")
    return p.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.checkpoint is not None:
        if arguments.eval_file is None or arguments.output is None:
            raise SystemExit("--checkpoint requires --eval-file and --output")
        evaluate_checkpoint(arguments)
    else:
        if arguments.output_dir is None:
            raise SystemExit("training requires --output-dir")
        train(arguments)
