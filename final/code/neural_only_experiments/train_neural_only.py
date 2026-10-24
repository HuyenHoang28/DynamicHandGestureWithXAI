import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import NeuralOnlyKeypointDataset, collate_batch, label_names_from_split
from models import NeuralOnlyClassifier


def macro_f1(predictions: list[int], labels: list[int], num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        support = sum(label == class_id for label in labels)
        true_positive = sum(
            pred == class_id and label == class_id for pred, label in zip(predictions, labels)
        )
        predicted = sum(pred == class_id for pred in predictions)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / max(len(scores), 1)


def temporal_crop_resample(keypoints: torch.Tensor, min_fraction: float) -> torch.Tensor:
    """Random crop of each clip, resampled back to the original length.

    The kept window is between ``min_fraction`` and 100% of the clip, so events
    land at different speeds and positions. Frames are copied (nearest index),
    never blended, so missing joints stay missing.
    """
    batch, frames = keypoints.shape[:2]
    min_length = max(2, int(round(min_fraction * frames)))
    lengths = torch.randint(min_length, frames + 1, (batch,), device=keypoints.device)
    starts = (torch.rand(batch, device=keypoints.device) * (frames - lengths + 1).float()).floor().long()
    steps = torch.linspace(0.0, 1.0, frames, device=keypoints.device)
    index = starts[:, None] + (steps[None, :] * (lengths[:, None] - 1).float()).round().long()
    index = index.clamp_(max=frames - 1)
    return keypoints[torch.arange(batch, device=keypoints.device)[:, None], index]


def augment_keypoints(
    keypoints: torch.Tensor,
    jitter_std: float,
    frame_mask_probability: float,
    keypoint_dropout_probability: float,
    time_crop_min: float = 1.0,
    scale_jitter: float = 0.0,
    shift_std: float = 0.0,
) -> torch.Tensor:
    """Apply label-preserving noise without turning missing joints into fake joints.

    ``time_crop_min`` (<1 enables), ``scale_jitter`` and ``shift_std`` are opt-in
    and default to off so existing recipes are unchanged. Scale and shift act on
    the shoulder-normalised coordinates (a per-clip zoom and camera offset).
    """
    output = keypoints.clone()
    if time_crop_min < 1.0:
        output = temporal_crop_resample(output, time_crop_min)
    point_valid = output[..., 2] > 0 if output.shape[-1] >= 3 else torch.ones_like(
        output[..., 0], dtype=torch.bool
    )
    if scale_jitter > 0 or shift_std > 0:
        batch = output.shape[0]
        scale = 1.0 + (torch.rand(batch, 1, 1, 1, device=output.device) * 2.0 - 1.0) * scale_jitter
        shift = torch.randn(batch, 1, 1, 2, device=output.device) * shift_std
        moved = output[..., :2] * scale + shift
        output[..., :2] = torch.where(point_valid.unsqueeze(-1), moved, output[..., :2])
    if jitter_std > 0:
        noise = torch.randn_like(output[..., :2]) * jitter_std
        output[..., :2] += noise * point_valid.unsqueeze(-1)
    if frame_mask_probability > 0:
        frame_mask = torch.rand(output.shape[:2], device=output.device) < frame_mask_probability
        output = output.masked_fill(frame_mask[:, :, None, None], 0.0)
    if keypoint_dropout_probability > 0:
        point_mask = torch.rand(output.shape[:3], device=output.device) < keypoint_dropout_probability
        output = output.masked_fill(point_mask.unsqueeze(-1), 0.0)
    return output


def class_weights(samples: list[tuple[str, int]], num_classes: int, power: float) -> torch.Tensor | None:
    labels = [label for _, label in samples]
    invalid = sorted({label for label in labels if label < 0 or label >= num_classes})
    if invalid:
        raise ValueError(f"Labels outside [0, {num_classes - 1}]: {invalid}")
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    missing = torch.where(counts == 0)[0].tolist()
    if missing:
        raise ValueError(f"Training split has no samples for class IDs: {missing}")
    if power <= 0:
        return None
    weights = counts.mean().div(counts).pow(power)
    return weights / weights.mean()


def validate_no_split_overlap(
    train_samples: list[tuple[str, int]], val_samples: list[tuple[str, int]]
) -> None:
    train_labels = {source.replace("\\", "/").casefold(): label for source, label in train_samples}
    overlap = []
    for source, label in val_samples:
        key = source.replace("\\", "/").casefold()
        if key in train_labels:
            overlap.append((source, train_labels[key], label))
    if overlap:
        raise ValueError(f"Train/validation overlap detected ({len(overlap)} samples), e.g. {overlap[:3]}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    predictions = []
    targets = []
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        x = batch["keypoints"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        predictions.extend(pred.tolist())
        targets.extend(y.tolist())
        total += y.numel()
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * y.numel()
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1(predictions, targets, num_classes),
        "total": total,
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    train_set = NeuralOnlyKeypointDataset(args.train_file, args.cache_root, "train", args.num_frames, args.limit_train)
    val_set = NeuralOnlyKeypointDataset(args.val_file, args.cache_root, "val", args.num_frames, args.limit_val)
    validate_no_split_overlap(train_set.samples, val_set.samples)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    model = NeuralOnlyClassifier(
        backbone=args.backbone,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        use_motion_features=not args.disable_motion_features,
        pooling=args.pooling,
        hand_features=args.hand_features,
    ).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.data_parallel:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    weights = class_weights(train_set.samples, args.num_classes, args.class_weight_power)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=args.label_smoothing,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_names = label_names_from_split(args.train_file)
    best_metric_value = -1.0
    best_accuracy = -1.0
    max_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x = batch["keypoints"].to(device)
            y = batch["label"].to(device)
            x = augment_keypoints(
                x,
                args.jitter_std,
                args.frame_mask_probability,
                args.keypoint_dropout_probability,
                time_crop_min=args.time_crop_min,
                scale_jitter=args.scale_jitter,
                shift_std=args.shift_std,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            pred = logits.argmax(dim=1)
            total += y.numel()
            correct += int((pred == y).sum().item())
            loss_sum += float(loss.item()) * y.numel()
            if args.log_every and step % args.log_every == 0:
                print(
                    f"backbone={args.backbone} epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss_sum / max(total, 1):.4f} acc={correct / max(total, 1):.4f}"
                )
        train_metrics = {
            "loss": loss_sum / max(total, 1),
            "accuracy": correct / max(total, 1),
            "total": total,
        }
        val_metrics = evaluate(model, val_loader, device, args.num_classes)
        val_score = float(val_metrics[args.select_metric])
        scheduler.step(val_score)
        improved = val_score > best_metric_value + args.early_stop_min_delta
        max_val_accuracy = max(max_val_accuracy, float(val_metrics["accuracy"]))
        if improved:
            best_metric_value = val_score
            best_accuracy = float(val_metrics["accuracy"])
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "learning_rates": learning_rates,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "max_val_accuracy": max_val_accuracy,
                "best_metric_value": best_metric_value,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        print(
            f"backbone={args.backbone} epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} select={args.select_metric} "
            f"lr={','.join(f'{value:.2e}' for value in learning_rates)} "
            f"patience={epochs_without_improvement}/{args.early_stop_patience}"
        )
        checkpoint = {
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "args": vars(args),
            "label_names": label_names,
            "epoch": epoch,
            "val": val_metrics,
            "best_epoch": best_epoch,
            "best_accuracy": best_accuracy,
            "max_val_accuracy": max_val_accuracy,
            "select_metric": args.select_metric,
            "best_metric_value": best_metric_value,
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}; best_epoch={best_epoch} "
                f"best_val_{args.select_metric}={best_metric_value:.4f}"
            )
            break
    print(f"best epoch: {best_epoch}")
    print(f"best val {args.select_metric}: {best_metric_value:.4f}")
    print(f"saved: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["transformer", "tcn", "bigru", "mlp_mixer"], default="transformer")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-file", type=Path, default=Path("val.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=27)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["mean", "masked_mean", "attention"], default="attention")
    parser.add_argument("--disable-motion-features", action="store_true")
    parser.add_argument(
        "--hand-features",
        action="store_true",
        help="Append scale-invariant finger flexion / pinch geometry and its frame-to-frame change.",
    )
    parser.add_argument(
        "--time-crop-min",
        type=float,
        default=1.0,
        help="<1 enables a random temporal crop keeping this fraction to 100% of each clip.",
    )
    parser.add_argument("--scale-jitter", type=float, default=0.0, help="Per-clip zoom, uniform in 1 +/- value.")
    parser.add_argument("--shift-std", type=float, default=0.0, help="Per-clip xy offset std (shoulder-width units).")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--jitter-std", type=float, default=0.005)
    parser.add_argument("--frame-mask-probability", type=float, default=0.05)
    parser.add_argument("--keypoint-dropout-probability", type=float, default=0.03)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--select-metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
