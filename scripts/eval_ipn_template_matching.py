"""Chấm nhánh symbolic thuần luật (graph_matching.score_template) trên một split IPN.

Không train gì: mỗi instance graph được so với 13 template, lớp có điểm cao nhất là dự đoán.
Đồng thời thống kê tỉ lệ clip của mỗi lớp có chứa từng concept (predicate/event), để biết
template nào dựa trên tín hiệu thực sự xuất hiện trong dữ liệu.

Ví dụ:
    python scripts/eval_ipn_template_matching.py --split val
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "final" / "code"))

from src_neurosymbolic.kg.graph_matching import instance_facts, score_template  # noqa: E402

IPN_LABELS = ["B0A", "B0B", "G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11"]


def load_templates(template_dir: Path) -> list[dict]:
    graphs = []
    for path in sorted(template_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        graphs.append(json.loads(path.read_text(encoding="utf-8")))
    graphs.sort(key=lambda graph: int(graph["label_id"]))
    return graphs


def instance_graph_path(source: str, keypoint_root: Path, graph_root: Path) -> Path:
    rel = Path(source).relative_to(keypoint_root)
    return graph_root / rel.with_suffix(".json")


def f1_table(matrix: list[list[int]]) -> list[dict]:
    rows = []
    n = len(matrix)
    for c in range(n):
        tp = matrix[c][c]
        predicted = sum(matrix[r][c] for r in range(n))
        support = sum(matrix[c])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": IPN_LABELS[c], "support": support, "precision": precision, "recall": recall, "f1": f1})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external_data/ipn_processed"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_root = args.data_root
    keypoint_root = data_root / "keypoints"
    templates = load_templates(data_root / "template_graphs")
    assert [t["label_name"] for t in templates] == IPN_LABELS, [t["label_name"] for t in templates]

    n = len(IPN_LABELS)
    matrix = [[0] * n for _ in range(n)]
    concept_counts = defaultdict(Counter)
    class_counts = Counter()
    missing = 0
    for line in (data_root / f"{args.split}.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source, _, label_text = line.rsplit(maxsplit=2)
        label = int(label_text)
        path = instance_graph_path(source, keypoint_root, data_root / "instance_graphs")
        if not path.exists():
            missing += 1
            continue
        graph = json.loads(path.read_text(encoding="utf-8"))
        scores = [score_template(graph, template)["score"] for template in templates]
        prediction = max(range(n), key=lambda c: scores[c])
        matrix[label][prediction] += 1
        class_counts[label] += 1
        facts, _ = instance_facts(graph)
        for concept in {concept for _, _, concept in facts}:
            concept_counts[label][concept] += 1

    total = sum(class_counts.values())
    correct = sum(matrix[c][c] for c in range(n))
    rows = f1_table(matrix)
    present = [row for row in rows if row["support"] > 0]
    macro_f1 = sum(row["f1"] for row in present) / max(len(present), 1)
    print(f"split={args.split} clips={total} missing_graphs={missing}")
    print(f"accuracy={correct / max(total, 1):.4f} macro_f1={macro_f1:.4f}")
    print("label  support  precision  recall  f1")
    for row in rows:
        print(f"{row['label']:5}  {row['support']:7}  {row['precision']:9.3f}  {row['recall']:6.3f}  {row['f1']:.3f}")

    confusions = sorted(
        ((matrix[r][c], IPN_LABELS[r], IPN_LABELS[c]) for r in range(n) for c in range(n) if r != c and matrix[r][c]),
        reverse=True,
    )
    print("top confusions (true -> pred):")
    for count, truth, pred in confusions[:10]:
        print(f"  {truth} -> {pred}: {count}")

    presence = {
        IPN_LABELS[c]: {
            concept: round(count / class_counts[c], 3)
            for concept, count in concept_counts[c].most_common()
        }
        for c in range(n)
        if class_counts[c]
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "clips": total,
                    "accuracy": correct / max(total, 1),
                    "macro_f1": macro_f1,
                    "per_class": rows,
                    "confusion_matrix": matrix,
                    "concept_presence_by_class": presence,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
