import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import NeuralOnlyKeypointDataset, collate_batch, label_names_from_split
from models import NeuralOnlyClassifier


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        x = batch["keypoints"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * y.numel()
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
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
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_names = label_names_from_split(args.train_file)
    best_acc = -1.0
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
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(float(val_metrics["accuracy"]))
        improved = val_metrics["accuracy"] > best_acc + args.early_stop_min_delta
        if improved:
            best_acc = val_metrics["accuracy"]
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
                "best_accuracy": best_acc,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        print(
            f"backbone={args.backbone} epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
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
            "best_accuracy": best_acc,
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch} best_val_acc={best_acc:.4f}")
            break
    print(f"best epoch: {best_epoch}")
    print(f"best val acc: {best_acc:.4f}")
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
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
