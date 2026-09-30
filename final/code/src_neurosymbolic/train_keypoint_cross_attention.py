import argparse
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


def collate_batch(items: list[dict]) -> dict:
    batch = {
        "keypoints": torch.stack([item["keypoints"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "path": [item["path"] for item in items],
    }
    if "instance_graph_json" in items[0]:
        batch["instance_graph_json"] = [item["instance_graph_json"] for item in items]
    return batch


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
        ig_json = batch.get("instance_graph_json")
        logits = model(x, instance_graph_json=ig_json)
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
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    train_set = KeypointSequenceDataset(args.train_file, args.keypoint_root, "train", args.num_frames, args.limit_train, args.cache_root, args.instance_graph_root)
    val_set = KeypointSequenceDataset(args.val_file, args.keypoint_root, "val", args.num_frames, args.limit_val, args.cache_root, args.instance_graph_root)
    label_names = load_label_names(args.train_file)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = NeuralKeypointTemplateCrossAttention(
        num_classes=args.num_classes,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        template_tokens=args.template_tokens,
        dropout=args.dropout,
        max_frames=args.num_frames,
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.instance_graph_root is None:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.instance_graph_root is not None:
        print("Instance KG Branch is enabled; using a single GPU because DataParallel does not split graph JSON lists safely.")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x = batch["keypoints"].to(device)
            y = batch["label"].to(device)
            ig_json = batch.get("instance_graph_json")
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, instance_graph_json=ig_json)
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
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss_sum / max(total, 1):.4f} acc={correct / max(total, 1):.4f}"
                )

        train_metrics = {
            "loss": loss_sum / max(total, 1),
            "accuracy": correct / max(total, 1),
            "total": total,
        }
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        checkpoint = {
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "args": vars(args),
            "label_names": label_names,
            "epoch": epoch,
            "val": val_metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(checkpoint, args.output_dir / "best.pt")

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoints to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural keypoint + template cross-attention baseline.")
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--instance-graph-root", type=Path, help="Path to instance graphs JSON")
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-file", type=Path, default=Path("val.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/keypoint_cross_attention"))
    parser.add_argument("--num-classes", type=int, default=27)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--template-tokens", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
