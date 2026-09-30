import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.models.cross_attention_v2 import (
    REQUIREMENT_LEVELS,
    TemplateQueryNeuroSymbolicModel,
    load_checkpoint_compat,
)
from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset, load_label_names


def collate_batch(items: list[dict]) -> dict:
    return {
        "keypoints": torch.stack([item["keypoints"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "path": [item["path"] for item in items],
        "instance_graph_json": [item["instance_graph_json"] for item in items],
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == labels).sum().item())


def compute_loss(outputs: dict, labels: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    ce = nn.functional.cross_entropy
    final_loss = ce(outputs["logits"], labels)
    neural_loss = ce(outputs["neural_logits"], labels)
    kg_loss = ce(outputs["kg_logits"], labels)

    requirement_ids = outputs["requirement_ids"]
    requirement_mask = (
        (requirement_ids >= REQUIREMENT_LEVELS["required"])
        & (requirement_ids <= REQUIREMENT_LEVELS["negative"])
        & outputs["requirement_weights"].gt(0)
    )
    requirement_raw = nn.functional.binary_cross_entropy(
        outputs["presence"],
        outputs["requirement_targets"],
        reduction="none",
    )
    requirement_loss = (
        requirement_raw * requirement_mask.float()
    ).sum() / requirement_mask.float().sum().clamp(min=1.0)
    gate_loss = outputs["gate"].mean()

    loss = (
        final_loss
        + args.lambda_neural * neural_loss
        + args.lambda_kg * kg_loss
        + args.lambda_requirement * requirement_loss
        + args.lambda_gate * gate_loss
    )
    return loss, {
        "final": float(final_loss.detach()),
        "neural": float(neural_loss.detach()),
        "kg": float(kg_loss.detach()),
        "requirement": float(requirement_loss.detach()),
        "gate": float(gate_loss.detach()),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    totals = {"loss": 0.0, "final_correct": 0, "neural_correct": 0, "kg_correct": 0, "count": 0}
    for batch in loader:
        keypoints = batch["keypoints"].to(device)
        labels = batch["label"].to(device)
        outputs = model(keypoints, batch["instance_graph_json"])
        batch_size = labels.numel()
        totals["loss"] += float(nn.functional.cross_entropy(outputs["logits"], labels)) * batch_size
        totals["final_correct"] += accuracy(outputs["logits"], labels)
        totals["neural_correct"] += accuracy(outputs["neural_logits"], labels)
        totals["kg_correct"] += accuracy(outputs["kg_logits"], labels)
        totals["count"] += batch_size
    count = max(totals["count"], 1)
    return {
        "loss": totals["loss"] / count,
        "accuracy": totals["final_correct"] / count,
        "neural_accuracy": totals["neural_correct"] / count,
        "kg_accuracy": totals["kg_correct"] / count,
        "total": totals["count"],
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    train_set = KeypointSequenceDataset(
        args.train_file,
        args.keypoint_root,
        "train",
        args.num_frames,
        args.limit_train,
        args.cache_root,
        args.instance_graph_root,
    )
    val_set = KeypointSequenceDataset(
        args.val_file,
        args.keypoint_root,
        "val",
        args.num_frames,
        args.limit_val,
        args.cache_root,
        args.instance_graph_root,
    )
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

    model = TemplateQueryNeuroSymbolicModel(
        template_dir=args.template_dir,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        graph_layers=args.graph_layers,
        decoder_layers=args.decoder_layers,
        instance_tokens=args.instance_tokens,
        dropout=args.dropout,
        fusion_mode=args.fusion_mode,
    )
    if args.neural_checkpoint is not None and args.resume is None:
        model.load_neural_checkpoint(args.neural_checkpoint)
        print(f"Loaded neural checkpoint: {args.neural_checkpoint}")
    elif args.resume is None:
        print("Training the keypoint encoder from scratch.")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.neural.parameters(), "lr": args.neural_lr},
            {
                "params": [parameter for name, parameter in model.named_parameters() if not name.startswith("neural.")],
                "lr": args.kg_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.json"
    history = []
    best_accuracy = -1.0
    start_epoch = 1
    if args.resume is not None:
        checkpoint = load_checkpoint_compat(args.resume)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Restored model and optimizer from {args.resume}")
        else:
            print(f"Restored model from {args.resume}; optimizer state was not present and starts fresh.")
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                best_accuracy = max(float(row["val"]["accuracy"]) for row in history)
        best_accuracy = max(best_accuracy, float(checkpoint.get("val", {}).get("accuracy", -1.0)))
        print(f"Continuing at epoch {start_epoch}; target final epoch is {args.epochs}.")
        if start_epoch > args.epochs:
            print("Nothing to train: checkpoint epoch already reached --epochs.")
            return
    label_names = load_label_names(args.train_file)
    for epoch in range(start_epoch, args.epochs + 1):
        neural_trainable = epoch > args.freeze_neural_epochs
        model.set_neural_trainable(neural_trainable)
        model.train()
        if not neural_trainable:
            model.neural.eval()
        total = correct = 0
        loss_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            keypoints = batch["keypoints"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(keypoints, batch["instance_graph_json"])
            loss, parts = compute_loss(outputs, labels, args)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total += labels.numel()
            correct += accuracy(outputs["logits"], labels)
            loss_sum += float(loss.detach()) * labels.numel()
            if args.log_every and step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss_sum / total:.4f} acc={correct / total:.4f} "
                    f"kg_loss={parts['kg']:.4f} req_loss={parts['requirement']:.4f} "
                    f"gate={parts['gate']:.4f}"
                )

        train_metrics = {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} neural_acc={val_metrics['neural_accuracy']:.4f} "
            f"kg_acc={val_metrics['kg_accuracy']:.4f}"
        )
        checkpoint = {
            "model": model.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "label_names": label_names,
            "concept_to_id": model.graph_encoder.concept_to_id,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val": val_metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Saved V2 checkpoints to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train template-query cross-attention V2.")
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--instance-graph-root", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, default=Path("data/template_graphs"))
    parser.add_argument("--neural-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue from a V2 checkpoint; --epochs is the final epoch.")
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-file", type=Path, default=Path("val.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_attention_v2"))
    parser.add_argument("--fusion-mode", choices=["kg_only", "residual", "convex", "concat"], default="residual")
    parser.add_argument("--num-classes", type=int, default=27)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-neural-epochs", type=int, default=0)
    parser.add_argument("--neural-lr", type=float, default=3e-5)
    parser.add_argument("--kg-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--instance-tokens", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-neural", type=float, default=0.2)
    parser.add_argument("--lambda-kg", type=float, default=0.3)
    parser.add_argument("--lambda-requirement", type=float, default=0.1)
    parser.add_argument("--lambda-gate", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
