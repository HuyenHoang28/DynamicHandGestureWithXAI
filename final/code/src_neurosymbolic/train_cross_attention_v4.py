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
    NeuralBaselineAdapter,
    REQUIREMENT_LEVELS,
    load_checkpoint_compat,
)
from src_neurosymbolic.models.cross_attention_v4 import DualCrossAttentionV4NeuroSymbolicModel
from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset, load_label_names
from src_neurosymbolic.train_cross_attention_v2 import collate_batch


def collate_optional_graph_batch(items: list[dict]) -> dict:
    batch = collate_batch(
        [
            {
                **item,
                "instance_graph_json": item.get("instance_graph_json", '{"nodes": [], "edges": []}'),
            }
            for item in items
        ]
    )
    return batch


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


def macro_f1(predictions: list[int], labels: list[int], num_classes: int) -> float:
    # Mean F1 over classes with support > 0, same formula as the neural-only baseline report.
    scores = []
    for class_id in range(num_classes):
        support = sum(1 for label in labels if label == class_id)
        if support == 0:
            continue
        tp = sum(1 for pred, label in zip(predictions, labels) if pred == class_id and label == class_id)
        predicted = sum(1 for pred in predictions if pred == class_id)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / max(len(scores), 1)


def branch_logits(outputs: dict, branch: str) -> torch.Tensor:
    if branch == "final":
        return outputs["logits"]
    if branch == "neural":
        return outputs["neural_logits"]
    if branch == "kg":
        return outputs["kg_logits"]
    raise ValueError(f"Unsupported logit branch: {branch}")


def neural_tokens_without_logits(model: nn.Module, keypoints: torch.Tensor) -> torch.Tensor:
    neural = model.neural
    batch, frames, keypoints_count, dims = keypoints.shape
    x = keypoints.reshape(batch, frames, keypoints_count * dims)
    x = neural.input_proj(x)
    return neural.encoder.norm(neural.encoder.encoder(neural.encoder.pos(x)))


def kg_forward_without_neural_logits(
    model: nn.Module,
    keypoints: torch.Tensor,
    instance_graph_json: list[str],
) -> dict:
    tokens = neural_tokens_without_logits(model, keypoints)
    batch, frames, _ = tokens.shape
    neural_memory = model.neural_projector(tokens)
    neural_mask = torch.zeros((batch, frames), dtype=torch.bool, device=keypoints.device)

    instance = model.graph_encoder.encode_instances(instance_graph_json, keypoints.device)
    graph_tokens, graph_mask = model.instance_compressor(instance["tokens"], instance["mask"])
    graph_memory = model.graph_projector(graph_tokens)

    templates = model.graph_encoder.encode_templates()
    classes, template_nodes, dim = templates["tokens"].shape
    template_tokens = model.template_projector(templates["tokens"])
    query = template_tokens.unsqueeze(0).expand(
        batch,
        classes,
        template_nodes,
        dim,
    ).reshape(batch * classes, template_nodes, dim)
    query_mask = templates["mask"].unsqueeze(0).expand(
        batch,
        classes,
        template_nodes,
    ).reshape(batch * classes, template_nodes)
    expanded_neural = model._expand_memory(neural_memory, classes)
    expanded_graph = model._expand_memory(graph_memory, classes)
    expanded_neural_mask = model._expand_mask(neural_mask, classes)
    expanded_graph_mask = model._expand_mask(graph_mask, classes)

    for decoder_layer in model.decoder:
        query, _, _, _ = decoder_layer(
            query,
            expanded_neural,
            expanded_graph,
            query_mask,
            expanded_neural_mask,
            expanded_graph_mask,
            False,
        )
    matched_tokens = query.reshape(batch, classes, template_nodes, dim)
    class_mask = templates["mask"].unsqueeze(0).expand(batch, -1, -1)
    requirements = templates["requirements"].unsqueeze(0).expand(batch, -1, -1)
    requirement_weights = templates["requirement_weights"].unsqueeze(0).expand(batch, -1, -1)
    scored = model.scorer(
        matched_tokens,
        class_mask,
        requirements,
        requirement_weights,
    )
    kg_logits = scored["logits"]
    requirement_targets = model.graph_encoder.requirement_targets(
        instance["evidence_facts"],
        keypoints.device,
    )
    return {
        "logits": kg_logits,
        "neural_logits": kg_logits.detach(),
        "kg_logits": kg_logits,
        "gate": torch.zeros_like(kg_logits),
        "presence": scored["presence"],
        "requirement_targets": requirement_targets,
        "requirement_ids": requirements,
        "requirement_weights": requirement_weights,
    }


def forward_for_branch(
    model: nn.Module,
    keypoints: torch.Tensor,
    instance_graph_json: list[str],
    branch: str,
) -> dict:
    if branch == "neural":
        neural = model.neural(keypoints) if hasattr(model, "neural") else model(keypoints)
        logits = neural["logits"]
        return {
            "logits": logits,
            "neural_logits": logits,
            "kg_logits": logits.detach(),
            "gate": torch.ones_like(logits),
        }
    if branch == "kg":
        return kg_forward_without_neural_logits(model, keypoints, instance_graph_json)
    return model(keypoints, instance_graph_json)


def compute_loss(
    outputs: dict,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    primary_loss = nn.functional.cross_entropy(
        branch_logits(outputs, args.logit_branch),
        labels,
        label_smoothing=args.label_smoothing,
    )
    if args.logit_branch == "neural":
        return primary_loss, {
            "primary": float(primary_loss.detach()),
            "final": float(primary_loss.detach()),
            "neural": float(primary_loss.detach()),
            "kg": 0.0,
            "requirement": 0.0,
            "gate_regularization": 0.0,
            "gate_mean": 1.0,
        }
    final_loss = nn.functional.cross_entropy(
        outputs["logits"],
        labels,
        label_smoothing=args.label_smoothing,
    )
    if args.logit_branch == "kg":
        neural_loss = torch.zeros((), dtype=primary_loss.dtype, device=primary_loss.device)
    else:
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
    loss = primary_loss
    if args.logit_branch == "final":
        loss = (
            loss
            + args.lambda_neural * neural_loss
            + args.lambda_kg * kg_loss
            + args.lambda_requirement * requirement_loss
            + args.lambda_gate * gate_regularization
        )
    elif args.logit_branch == "kg":
        loss = (
            loss
            + args.lambda_requirement * requirement_loss
        )
    return loss, {
        "primary": float(primary_loss.detach()),
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
    branch: str = "final",
    num_classes: int | None = None,
) -> dict[str, float | int]:
    model.eval()
    predictions = {"final": [], "neural": [], "kg": []}
    targets = []
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
        outputs = forward_for_branch(model, keypoints, batch["instance_graph_json"], branch)
        primary_logits = branch_logits(outputs, branch)
        batch_size = labels.numel()
        totals["loss"] += float(nn.functional.cross_entropy(primary_logits, labels)) * batch_size
        totals["final_correct"] += accuracy(outputs["logits"], labels)
        totals["neural_correct"] += accuracy(outputs["neural_logits"], labels)
        totals["kg_correct"] += accuracy(outputs["kg_logits"], labels)
        for name, key in (("final", "logits"), ("neural", "neural_logits"), ("kg", "kg_logits")):
            predictions[name].extend(outputs[key].argmax(dim=1).tolist())
        targets.extend(labels.tolist())
        totals["gate_sum"] += float(outputs["gate"].sum())
        totals["gate_count"] += outputs["gate"].numel()
        totals["count"] += batch_size
    count = max(int(totals["count"]), 1)
    classes = num_classes or (max(targets, default=0) + 1)
    return {
        "loss": totals["loss"] / count,
        "accuracy": totals[f"{branch}_correct"] / count,
        "macro_f1": macro_f1(predictions[branch], targets, classes),
        "final_macro_f1": macro_f1(predictions["final"], targets, classes),
        "neural_macro_f1": macro_f1(predictions["neural"], targets, classes),
        "kg_macro_f1": macro_f1(predictions["kg"], targets, classes),
        "logit_branch": branch,
        "final_accuracy": totals["final_correct"] / count,
        "neural_accuracy": totals["neural_correct"] / count,
        "kg_accuracy": totals["kg_correct"] / count,
        "gate_mean": totals["gate_sum"] / max(int(totals["gate_count"]), 1),
        "total": int(totals["count"]),
    }


def train(args: argparse.Namespace) -> None:
    loader_generator = set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dataset_instance_graph_root = None if args.logit_branch == "neural" else args.instance_graph_root
    train_set = KeypointSequenceDataset(
        args.train_file,
        args.keypoint_root,
        "train",
        args.num_frames,
        args.limit_train,
        args.cache_root,
        dataset_instance_graph_root,
    )
    val_set = KeypointSequenceDataset(
        args.val_file,
        args.keypoint_root,
        "val",
        args.num_frames,
        args.limit_val,
        args.cache_root,
        dataset_instance_graph_root,
    )
    test_set = None
    test_loader = None
    if args.test_file is not None:
        test_set = KeypointSequenceDataset(
            args.test_file,
            args.keypoint_root,
            "test",
            args.num_frames,
            args.limit_test,
            args.cache_root,
            dataset_instance_graph_root,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        collate_fn=collate_optional_graph_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_optional_graph_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_optional_graph_batch,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
    if args.logit_branch == "neural":
        model = NeuralBaselineAdapter(
            num_classes=args.num_classes,
            num_frames=args.num_frames,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
        )
    else:
        model = DualCrossAttentionV4NeuroSymbolicModel(
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
            directed_temporal_edges=args.directed_temporal_edges,
        )
    if args.neural_checkpoint is not None and args.resume is None:
        if hasattr(model, "load_neural_checkpoint"):
            model.load_neural_checkpoint(args.neural_checkpoint)
        else:
            model.load_baseline_checkpoint(args.neural_checkpoint)
        print(f"Loaded neural checkpoint: {args.neural_checkpoint}")
    elif args.resume is None:
        print(f"Training V4 structured-requirement model with {args.logit_branch} logits.")
    if args.logit_branch == "neural":
        for parameter in model.parameters():
            parameter.requires_grad = True
    model = model.to(device)
    if args.logit_branch == "neural":
        optimizer_params = [{"params": model.parameters(), "lr": args.neural_lr}]
    elif args.logit_branch == "kg":
        optimizer_params = []
        if not args.freeze_neural_for_kg:
            optimizer_params.append({"params": model.neural.parameters(), "lr": args.neural_lr})
        optimizer_params.append(
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("neural.")
                ],
                "lr": args.kg_lr,
            }
        )
    else:
        optimizer_params = [
            {"params": model.neural.parameters(), "lr": args.neural_lr},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("neural.")
                ],
                "lr": args.kg_lr,
            },
        ]
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=args.weight_decay)
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
    best_test_accuracy = -1.0
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
        best_test_accuracy = float(checkpoint.get("best_test_accuracy", -1.0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"Resumed V4 from epoch {start_epoch}.")

    label_names = load_label_names(args.train_file)
    for epoch in range(start_epoch, args.epochs + 1):
        neural_trainable = args.logit_branch == "neural" or (
            args.logit_branch == "kg" and not args.freeze_neural_for_kg
        ) or (
            args.logit_branch == "final" and epoch > args.freeze_neural_epochs
        )
        if hasattr(model, "set_neural_trainable"):
            model.set_neural_trainable(neural_trainable)
        else:
            for parameter in model.parameters():
                parameter.requires_grad = neural_trainable
        model.train()
        if not neural_trainable:
            (model.neural if hasattr(model, "neural") else model).eval()
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
            outputs = forward_for_branch(model, keypoints, batch["instance_graph_json"], args.logit_branch)
            loss, parts = compute_loss(outputs, labels, args)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            primary_logits = branch_logits(outputs, args.logit_branch)
            total += labels.numel()
            correct += accuracy(primary_logits, labels)
            loss_sum += float(loss.detach()) * labels.numel()
            if args.log_every and step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss_sum / total:.4f} acc={correct / total:.4f} "
                    f"primary_loss={parts['primary']:.4f} branch={args.logit_branch} "
                    f"neural_loss={parts['neural']:.4f} kg_loss={parts['kg']:.4f} "
                    f"req_loss={parts['requirement']:.4f} gate={parts['gate_mean']:.4f}"
                )

        train_metrics = {
            "loss": loss_sum / max(total, 1),
            "accuracy": correct / max(total, 1),
        }
        val_metrics = evaluate(model, val_loader, device, args.logit_branch, args.num_classes)
        test_metrics = None
        test_improved = False
        if test_loader is not None:
            test_metrics = evaluate(model, test_loader, device, args.logit_branch, args.num_classes)
            test_improved = float(test_metrics["accuracy"]) > best_test_accuracy + args.early_stop_min_delta
            if test_improved:
                best_test_accuracy = float(test_metrics["accuracy"])
        val_score = float(val_metrics[args.select_metric])
        scheduler.step(val_score)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        improved = val_score > best_accuracy + args.early_stop_min_delta
        if improved:
            best_accuracy = val_score
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
        if test_metrics is not None:
            row["test"] = test_metrics
            row["test_improved"] = test_improved
        history.append(row)
        message = (
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} branch={args.logit_branch} "
            f"final_acc={val_metrics['final_accuracy']:.4f} neural_acc={val_metrics['neural_accuracy']:.4f} "
            f"kg_acc={val_metrics['kg_accuracy']:.4f} gate={val_metrics['gate_mean']:.4f} "
            f"lr={','.join(f'{value:.2e}' for value in learning_rates)} "
            f"patience={epochs_without_improvement}/{args.early_stop_patience}"
        )
        if test_metrics is not None:
            message += (
                f" test_acc={test_metrics['accuracy']:.4f}"
                f" best_test_acc={best_test_accuracy:.4f}"
            )
        print(message)
        checkpoint = {
            "model_version": (
                "v4_neural_logits_only"
                if args.logit_branch == "neural"
                else "v4_structured_requirement_dual_cross_attention"
            ),
            "logit_branch": args.logit_branch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "label_names": label_names,
            "concept_to_id": getattr(getattr(model, "graph_encoder", None), "concept_to_id", {}),
            "epoch": epoch,
            "val": val_metrics,
            "test": test_metrics,
            "best_accuracy": best_accuracy,
            "select_metric": args.select_metric,
            "best_test_accuracy": best_test_accuracy,
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pt")
        if args.save_best_test and test_improved:
            torch.save(checkpoint, args.output_dir / "best_test.pt")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best_val_{args.select_metric}={best_accuracy:.4f}")
            break
    print(f"Saved V4 checkpoints to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V4 structured-requirement dual cross-attention model.")
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--instance-graph-root", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, default=Path("data/template_graphs"))
    parser.add_argument("--neural-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-file", type=Path, default=Path("val.txt"))
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_attention_v4_structured_req"))
    parser.add_argument("--logit-branch", choices=["final", "neural", "kg"], default="final")
    parser.add_argument("--num-classes", type=int, default=27)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-neural-epochs", type=int, default=0)
    parser.add_argument("--freeze-neural-for-kg", action="store_true")
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
    parser.add_argument("--directed-temporal-edges", action="store_true")
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--save-best-test", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-metric", choices=["accuracy", "macro_f1"], default="accuracy")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
