import argparse
import csv
import json
import time
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.models.cross_attention_v2 import (
    NeuralBaselineAdapter,
    TemplateQueryNeuroSymbolicModel,
    load_checkpoint_compat,
)
from src_neurosymbolic.models.cross_attention_v3 import DualCrossAttentionNeuroSymbolicModel
from src_neurosymbolic.models.cross_attention_v4 import DualCrossAttentionV4NeuroSymbolicModel
from src_neurosymbolic.models.cross_attention_v5 import DualCrossAttentionV5NeuroSymbolicModel
from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset
from src_neurosymbolic.train_cross_attention_v2 import collate_batch


BRANCHES = ("final", "neural", "kg")


def collate_optional_graph_batch(items: list[dict]) -> dict:
    return collate_batch(
        [
            {
                **item,
                "instance_graph_json": item.get("instance_graph_json", '{"nodes": [], "edges": []}'),
            }
            for item in items
        ]
    )


def outputs_for_model(model: torch.nn.Module, keypoints: torch.Tensor, instance_graph_json: list[str]) -> dict:
    if isinstance(model, NeuralBaselineAdapter):
        neural = model(keypoints)
        logits = neural["logits"]
        return {
            "logits": logits,
            "neural_logits": logits,
            "kg_logits": logits.detach(),
            "gate": torch.ones_like(logits),
        }
    return model(keypoints, instance_graph_json)


def label_name(label_names: dict, label: int) -> str:
    return str(label_names.get(label, label_names.get(str(label), label)))


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_confusion(labels: list[int], predictions: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true, predicted in zip(labels, predictions):
        matrix[true][predicted] += 1
    return matrix


def branch_report(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
    label_names: dict,
) -> tuple[dict, list[dict], list[list[int]]]:
    matrix = build_confusion(labels, predictions, num_classes)
    per_class = []
    for class_id in range(num_classes):
        true_positive = matrix[class_id][class_id]
        support = sum(matrix[class_id])
        predicted_count = sum(matrix[row][class_id] for row in range(num_classes))
        precision = safe_div(true_positive, predicted_count)
        recall = safe_div(true_positive, support)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": label_name(label_names, class_id),
                "support": support,
                "correct": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    correct = sum(matrix[index][index] for index in range(num_classes))
    summary = {
        "accuracy": safe_div(correct, len(labels)),
        "correct": correct,
        "total": len(labels),
        "macro_precision": safe_div(sum(row["precision"] for row in per_class), num_classes),
        "macro_recall": safe_div(sum(row["recall"] for row in per_class), num_classes),
        "macro_f1": safe_div(sum(row["f1"] for row in per_class), num_classes),
    }
    return summary, per_class, matrix


def classify_neural_vs_kg(true: int, neural: int, kg: int) -> str:
    neural_ok = neural == true
    kg_ok = kg == true
    if not neural_ok and kg_ok:
        return "kg_fix"
    if neural_ok and not kg_ok:
        return "kg_harm"
    if neural_ok and kg_ok:
        return "both_correct"
    return "both_wrong"


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint_compat(args.checkpoint)
    saved = checkpoint["args"]
    label_names = checkpoint.get("label_names", {})
    num_classes = int(saved["num_classes"])
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    common_model_args = {
        "template_dir": args.template_dir,
        "num_classes": num_classes,
        "num_frames": int(saved["num_frames"]),
        "d_model": int(saved["d_model"]),
        "layers": int(saved["layers"]),
        "heads": int(saved["heads"]),
        "graph_layers": int(saved["graph_layers"]),
        "decoder_layers": int(saved["decoder_layers"]),
        "instance_tokens": int(saved["instance_tokens"]),
        "dropout": float(saved["dropout"]),
    }
    directed_temporal_edges = str(saved.get("directed_temporal_edges", "False")).lower() == "true"
    if checkpoint.get("model_version") == "v4_neural_logits_only":
        model = NeuralBaselineAdapter(
            num_classes=num_classes,
            num_frames=int(saved["num_frames"]),
            d_model=int(saved["d_model"]),
            layers=int(saved["layers"]),
            heads=int(saved["heads"]),
            dropout=float(saved["dropout"]),
        )
        fusion_mode = "v4_neural_logits_only"
    elif checkpoint.get("model_version") == "v5_template_tuned_dual_cross_attention":
        model = DualCrossAttentionV5NeuroSymbolicModel(
            **common_model_args,
            directed_temporal_edges=directed_temporal_edges,
        )
        fusion_mode = "v5_template_tuned_dual_cross_attention"
    elif checkpoint.get("model_version") == "v4_structured_requirement_dual_cross_attention":
        model = DualCrossAttentionV4NeuroSymbolicModel(
            **common_model_args,
            directed_temporal_edges=directed_temporal_edges,
        )
        fusion_mode = "v4_structured_requirement_dual_cross_attention"
    elif checkpoint.get("model_version") == "v3_dual_cross_attention":
        model = DualCrossAttentionNeuroSymbolicModel(
            **common_model_args,
            directed_temporal_edges=directed_temporal_edges,
        )
        fusion_mode = "dual_cross_attention"
    else:
        model = TemplateQueryNeuroSymbolicModel(
            **common_model_args,
            fusion_mode=saved["fusion_mode"],
        )
        fusion_mode = saved["fusion_mode"]
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    dataset = KeypointSequenceDataset(
        args.test_file,
        args.keypoint_root,
        args.split,
        int(saved["num_frames"]),
        args.limit,
        args.cache_root,
        None if checkpoint.get("model_version") == "v4_neural_logits_only" else args.instance_graph_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_optional_graph_batch,
    )

    labels = []
    predictions = {branch: [] for branch in BRANCHES}
    prediction_rows = []
    category_counts = Counter()
    fixes_by_class = Counter()
    harms_by_class = Counter()
    fixed_confusions = Counter()
    harmful_confusions = Counter()
    loss_sum = 0.0
    gate_sum = 0.0
    gate_count = 0
    inference_seconds = 0.0
    timed_samples = 0

    for batch_index, batch in enumerate(loader):
        keypoints = batch["keypoints"].to(device)
        batch_labels = batch["label"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.timing_repeats):
            outputs = outputs_for_model(model, keypoints, batch["instance_graph_json"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if batch_index >= args.timing_warmup_batches:
            inference_seconds += elapsed
            timed_samples += batch_labels.numel() * args.timing_repeats
        logits = {
            "final": outputs["logits"],
            "neural": outputs["neural_logits"],
            "kg": outputs["kg_logits"],
        }
        batch_predictions = {branch: value.argmax(dim=1) for branch, value in logits.items()}
        confidences = {
            branch: torch.softmax(value, dim=1).max(dim=1).values
            for branch, value in logits.items()
        }
        loss_sum += float(torch.nn.functional.cross_entropy(logits[args.eval_branch], batch_labels)) * batch_labels.numel()
        gate = outputs["gate"]
        gate_sum += float(gate.sum())
        gate_count += gate.numel()

        true_values = batch_labels.tolist()
        labels.extend(true_values)
        for branch in BRANCHES:
            predictions[branch].extend(batch_predictions[branch].tolist())

        for index, path in enumerate(batch["path"]):
            true = true_values[index]
            neural = int(batch_predictions["neural"][index])
            kg = int(batch_predictions["kg"][index])
            final = int(batch_predictions["final"][index])
            category = classify_neural_vs_kg(true, neural, kg)
            category_counts[category] += 1
            if category == "kg_fix":
                fixes_by_class[true] += 1
                fixed_confusions[(true, neural)] += 1
            elif category == "kg_harm":
                harms_by_class[true] += 1
                harmful_confusions[(true, kg)] += 1
            prediction_rows.append(
                {
                    "path": path,
                    "true_id": true,
                    "true_name": label_name(label_names, true),
                    "final_pred_id": final,
                    "final_pred_name": label_name(label_names, final),
                    "final_correct": int(final == true),
                    "final_confidence": float(confidences["final"][index]),
                    "neural_pred_id": neural,
                    "neural_pred_name": label_name(label_names, neural),
                    "neural_correct": int(neural == true),
                    "neural_confidence": float(confidences["neural"][index]),
                    "kg_pred_id": kg,
                    "kg_pred_name": label_name(label_names, kg),
                    "kg_correct": int(kg == true),
                    "kg_confidence": float(confidences["kg"][index]),
                    "neural_vs_kg": category,
                    "gate_mean": float(gate[index].mean()),
                }
            )

    if not labels:
        raise RuntimeError("Test dataset is empty.")

    summaries = {}
    per_class_by_branch = {}
    confusion_matrices = {}
    for branch in BRANCHES:
        summary, per_class, matrix = branch_report(
            labels,
            predictions[branch],
            num_classes,
            label_names,
        )
        summaries[branch] = summary
        per_class_by_branch[branch] = per_class
        confusion_matrices[branch] = matrix

    per_class_rows = []
    for class_id in range(num_classes):
        neural_row = per_class_by_branch["neural"][class_id]
        kg_row = per_class_by_branch["kg"][class_id]
        final_row = per_class_by_branch["final"][class_id]
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": label_name(label_names, class_id),
                "support": neural_row["support"],
                "neural_correct": neural_row["correct"],
                "neural_recall": neural_row["recall"],
                "neural_f1": neural_row["f1"],
                "kg_correct": kg_row["correct"],
                "kg_recall": kg_row["recall"],
                "kg_f1": kg_row["f1"],
                "final_correct": final_row["correct"],
                "final_recall": final_row["recall"],
                "final_f1": final_row["f1"],
                "kg_fixes": fixes_by_class[class_id],
                "kg_harms": harms_by_class[class_id],
                "net_kg_gain": fixes_by_class[class_id] - harms_by_class[class_id],
            }
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation": checkpoint.get("val"),
        "fusion_mode": fusion_mode,
        "eval_branch": args.eval_branch,
        "test_loss": loss_sum / len(labels),
        "accuracy": summaries[args.eval_branch]["accuracy"],
        "branches": summaries,
        "gate_mean": safe_div(gate_sum, gate_count),
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
        "neural_vs_kg": {
            "counts": dict(category_counts),
            "net_kg_gain": category_counts["kg_fix"] - category_counts["kg_harm"],
            "top_classes_fixed_by_kg": [
                {
                    "class_id": class_id,
                    "class_name": label_name(label_names, class_id),
                    "count": count,
                }
                for class_id, count in fixes_by_class.most_common()
            ],
            "top_classes_harmed_by_kg": [
                {
                    "class_id": class_id,
                    "class_name": label_name(label_names, class_id),
                    "count": count,
                }
                for class_id, count in harms_by_class.most_common()
            ],
            "top_neural_confusions_fixed_by_kg": [
                {
                    "true_id": true,
                    "true_name": label_name(label_names, true),
                    "neural_wrong_id": wrong,
                    "neural_wrong_name": label_name(label_names, wrong),
                    "count": count,
                }
                for (true, wrong), count in fixed_confusions.most_common(30)
            ],
            "top_kg_confusions_harming_neural": [
                {
                    "true_id": true,
                    "true_name": label_name(label_names, true),
                    "kg_wrong_id": wrong,
                    "kg_wrong_name": label_name(label_names, wrong),
                    "count": count,
                }
                for (true, wrong), count in harmful_confusions.most_common(30)
            ],
        },
        "weakest_neural_classes": sorted(
            [row for row in per_class_rows if row["support"] > 0],
            key=lambda row: (row["neural_recall"], row["support"]),
        )[:10],
    }

    output_path = args.output or args.checkpoint.with_name("test_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path.with_name("test_predictions.csv")
    per_class_path = output_path.with_name("test_per_class.csv")
    confusion_path = output_path.with_name("test_confusion_matrices.json")

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    confusion_path.write_text(
        json.dumps(confusion_matrices, indent=2),
        encoding="utf-8",
    )
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)
    with per_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_class_rows)

    print(json.dumps({
        "test_loss": report["test_loss"],
        "eval_branch": report["eval_branch"],
        "accuracy": report["accuracy"],
        "branches": summaries,
        "gate_mean": report["gate_mean"],
        "neural_vs_kg": report["neural_vs_kg"]["counts"],
        "net_kg_gain": report["neural_vs_kg"]["net_kg_gain"],
        "weakest_neural_classes": report["weakest_neural_classes"],
        "saved": {
            "metrics": str(output_path),
            "predictions": str(predictions_path),
            "per_class": str(per_class_path),
            "confusion_matrices": str(confusion_path),
        },
    }, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate template-query cross-attention V2.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--instance-graph-root", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, default=Path("data/template_graphs"))
    parser.add_argument("--test-file", type=Path, default=Path("test.txt"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-warmup-batches", type=int, default=2)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eval-branch", choices=BRANCHES, default="final")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
