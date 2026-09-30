import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.models.cross_attention_v2 import (
    TemplateQueryNeuroSymbolicModel,
    load_checkpoint_compat,
)
from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset
from src_neurosymbolic.train_cross_attention_v2 import collate_batch


def label_name(label_names: dict, label: int) -> str:
    return str(label_names.get(label, label_names.get(str(label), label)))


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint_compat(args.checkpoint)
    saved = checkpoint["args"]
    label_names = checkpoint.get("label_names", {})
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = TemplateQueryNeuroSymbolicModel(
        template_dir=args.template_dir,
        num_classes=int(saved["num_classes"]),
        num_frames=int(saved["num_frames"]),
        d_model=int(saved["d_model"]),
        layers=int(saved["layers"]),
        heads=int(saved["heads"]),
        graph_layers=int(saved["graph_layers"]),
        decoder_layers=int(saved["decoder_layers"]),
        instance_tokens=int(saved["instance_tokens"]),
        dropout=float(saved["dropout"]),
        fusion_mode=saved["fusion_mode"],
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    dataset = KeypointSequenceDataset(
        args.test_file,
        args.keypoint_root,
        args.split,
        int(saved["num_frames"]),
        None,
        args.cache_root,
        args.instance_graph_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    rows = []
    category_counts = Counter()
    fixes_by_true = Counter()
    harms_by_true = Counter()
    fix_pairs = Counter()
    harm_pairs = Counter()
    for batch in loader:
        labels = batch["label"].to(device)
        outputs = model(batch["keypoints"].to(device), batch["instance_graph_json"])
        neural_pred = outputs["neural_logits"].argmax(dim=1)
        kg_pred = outputs["kg_logits"].argmax(dim=1)
        for path, true, neural, kg in zip(
            batch["path"],
            labels.tolist(),
            neural_pred.tolist(),
            kg_pred.tolist(),
        ):
            neural_ok = neural == true
            kg_ok = kg == true
            if not neural_ok and kg_ok:
                category = "kg_fix"
                fixes_by_true[true] += 1
                fix_pairs[(true, neural)] += 1
            elif neural_ok and not kg_ok:
                category = "kg_harm"
                harms_by_true[true] += 1
                harm_pairs[(true, kg)] += 1
            elif neural_ok and kg_ok:
                category = "both_correct"
            else:
                category = "both_wrong"
            category_counts[category] += 1
            rows.append(
                {
                    "category": category,
                    "path": path,
                    "true_id": true,
                    "true_name": label_name(label_names, true),
                    "neural_pred_id": neural,
                    "neural_pred_name": label_name(label_names, neural),
                    "kg_pred_id": kg,
                    "kg_pred_name": label_name(label_names, kg),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "kg_fix_harm_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "checkpoint": str(args.checkpoint),
        "total": len(rows),
        "counts": dict(category_counts),
        "net_kg_gain": category_counts["kg_fix"] - category_counts["kg_harm"],
        "top_fixes_by_true_class": [
            {"class_id": class_id, "class_name": label_name(label_names, class_id), "count": count}
            for class_id, count in fixes_by_true.most_common()
        ],
        "top_harms_by_true_class": [
            {"class_id": class_id, "class_name": label_name(label_names, class_id), "count": count}
            for class_id, count in harms_by_true.most_common()
        ],
        "top_neural_confusions_fixed_by_kg": [
            {
                "true_id": true,
                "true_name": label_name(label_names, true),
                "neural_wrong_id": wrong,
                "neural_wrong_name": label_name(label_names, wrong),
                "count": count,
            }
            for (true, wrong), count in fix_pairs.most_common(30)
        ],
        "top_kg_confusions_harming_neural": [
            {
                "true_id": true,
                "true_name": label_name(label_names, true),
                "kg_wrong_id": wrong,
                "kg_wrong_name": label_name(label_names, wrong),
                "count": count,
            }
            for (true, wrong), count in harm_pairs.most_common(30)
        ],
    }
    (args.output_dir / "kg_fix_harm_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze samples fixed or harmed by the KG branch.")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
