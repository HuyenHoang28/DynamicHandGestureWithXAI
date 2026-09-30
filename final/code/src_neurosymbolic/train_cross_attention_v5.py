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

from src_neurosymbolic.models.cross_attention_v2 import REQUIREMENT_LEVELS, load_checkpoint_compat
from src_neurosymbolic.models.cross_attention_v5 import DualCrossAttentionV5NeuroSymbolicModel
from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset, load_label_names
from src_neurosymbolic.train_cross_attention_v2 import collate_batch


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def augment_keypoints(
    keypoints: torch.Tensor,
    jitter_std: float,
    frame_mask_probability: float,
    keypoint_dropout_probability: float,
) -> torch.Tensor:
    output = keypoints.clone()
    if jitter_std > 0:
        output[..., :2] += torch.randn_like(output[..., :2]) * jitter_std
    if frame_mask_probability > 0:
        frame_mask = torch.rand(
            output.shape[:2],
            device=output.device,
        ) < frame_mask_probability
        output = output.masked_fill(frame_mask[:, :, None, None], 0.0)
    if keypoint_dropout_probability > 0:
        keypoint_mask = torch.rand(
            output.shape[:3],
            device=output.device,
        ) < keypoint_dropout_probability
        output = output.masked_fill(keypoint_mask.unsqueeze(-1), 0.0)
    return output


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == labels).sum().item())


def compute_loss(
    outputs: dict,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    final_loss = nn.functional.cross_entropy(
        outputs["logits"],
        labels,
        label_smoothing=args.label_smoothing,
    )
    neural_loss = nn.functional.cross_entropy(
        outputs["neural_logits"],
        labels,
        label_smoothing=args.label_smoothing,
    )
    kg_loss = nn.functional.cross_entropy(
        outputs["kg_logits"],
        labels,
        label_smoothing=args.label_smoothing,
    )
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
    gate_regularization = ((outputs["gate"] - args.initial_neural_gate) ** 2).mean()
    loss = (
        final_loss
        + args.lambda_neural * neural_loss
        + args.lambda_kg * kg_loss
        + args.lambda_requirement * requirement_loss
        + args.lambda_gate * gate_regularization
    )
    return loss, {
        "final": float(final_loss.detach()),
        "neural": float(neural_loss.detach()),
        "kg": float(kg_loss.detach()),
        "requirement": float(requirement_loss.detach()),
        "gate_regularization": float(gate_regularization.detach()),
        "gate_mean": float(outputs["gate"].detach().mean()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    totals = {
        "loss": 0.0,
        "final_correct": 0,
        "neural_correct": 0,
        "kg_correct": 0,
        "gate_sum": 0.0,
        "gate_count": 0,
        "count": 0,
    }
    for batch in loader:
        keypoints = batch["keypoints"].to(device)
        labels = batch["label"].to(device)
        outputs = model(keypoints, batch["instance_graph_json"])
        batch_size = labels.numel()
        totals["loss"] += float(nn.functional.cross_entropy(outputs["logits"], labels)) * batch_size
        totals["final_correct"] += accuracy(outputs["logits"], labels)
        totals["neural_correct"] += accuracy(outputs["neural_logits"], labels)
        totals["kg_correct"] += accuracy(outputs["kg_logits"], labels)
        totals["gate_sum"] += float(outputs["gate"].sum())
        totals["gate_count"] += outputs["gate"].numel()
        totals["count"] += batch_size
    count = max(int(totals["count"]), 1)
    return {
        "loss": totals["loss"] / count,
        "accuracy": totals["final_correct"] / count,
        "neural_accuracy": totals["neural_correct"] / count,
        "kg_accuracy": totals["kg_correct"] / count,
        "gate_mean": totals["gate_sum"] / max(int(totals["gate_count"]), 1),
        "total": int(totals["count"]),
    }


def train(args: argparse.Namespace) -> None:
    loader_generator = set_seed(args.seed)
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
        generator=loader_generator,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = DualCrossAttentionV5NeuroSymbolicModel(
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
    )
    if args.neural_checkpoint is not None and args.resume is None:
        model.load_neural_checkpoint(args.neural_checkpoint)
        print(f"Loaded neural checkpoint: {args.neural_checkpoint}")
    elif args.resume is None:
        print("Training V5 template-tuned structured-requirement model end-to-end from scratch.")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.neural.parameters(), "lr": args.neural_lr},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("neural.")
                ],
                "lr": args.kg_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.json"
    history = []
    best_accuracy = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    if args.resume is not None:
        checkpoint = load_checkpoint_compat(args.resume)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        best_accuracy = float(checkpoint.get("best_accuracy", checkpoint.get("val", {}).get("accuracy", -1.0)))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"Resumed V5 from epoch {start_epoch}.")

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
            keypoints = augment_keypoints(
                batch["keypoints"].to(device),
                args.jitter_std,
                args.frame_mask_probability,
                args.keypoint_dropout_probability,
            )
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(keypoints, batch["instance_graph_json"])
            loss, parts = compute_loss(outputs, labels, args)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}")
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
                    f"neural_loss={parts['neural']:.4f} kg_loss={parts['kg']:.4f} "
                    f"req_loss={parts['requirement']:.4f} gate={parts['gate_mean']:.4f}"
                )

        train_metrics = {
            "loss": loss_sum / max(total, 1),
            "accuracy": correct / max(total, 1),
        }
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(float(val_metrics["accuracy"]))
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        improved = float(val_metrics["accuracy"]) > best_accuracy + args.early_stop_min_delta
        if improved:
            best_accuracy = float(val_metrics["accuracy"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rates": learning_rates,
            "improved": improved,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} neural_acc={val_metrics['neural_accuracy']:.4f} "
            f"kg_acc={val_metrics['kg_accuracy']:.4f} gate={val_metrics['gate_mean']:.4f} "
            f"lr_neural={learning_rates[0]:.2e} lr_kg={learning_rates[1]:.2e} "
            f"patience={epochs_without_improvement}/{args.early_stop_patience}"
        )
        checkpoint = {
            "model_version": "v5_template_tuned_dual_cross_attention",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "label_names": label_names,
            "concept_to_id": model.graph_encoder.concept_to_id,
            "epoch": epoch,
            "val": val_metrics,
            "best_accuracy": best_accuracy,
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pt")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if (not args.disable_early_stop) and epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best_val_acc={best_accuracy:.4f}")
            break
    print(f"Saved V5 checkpoints to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V5 template-tuned structured-requirement dual cross-attention model.")
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--instance-graph-root", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, default=Path("data/template_graphs_v5"))
    parser.add_argument("--neural-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-file", type=Path, default=Path("val.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_attention_v5_template_tuned"))
    parser.add_argument("--num-classes", type=int, default=27)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-neural-epochs", type=int, default=0)
    parser.add_argument("--neural-lr", type=float, default=2e-4)
    parser.add_argument("--kg-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--instance-tokens", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--lambda-neural", type=float, default=0.3)
    parser.add_argument("--lambda-kg", type=float, default=0.3)
    parser.add_argument("--lambda-requirement", type=float, default=0.05)
    parser.add_argument("--lambda-gate", type=float, default=0.01)
    parser.add_argument("--initial-neural-gate", type=float, default=0.8)
    parser.add_argument("--jitter-std", type=float, default=0.005)
    parser.add_argument("--frame-mask-probability", type=float, default=0.05)
    parser.add_argument("--keypoint-dropout-probability", type=float, default=0.03)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
