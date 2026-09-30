import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "confusion_matrices"


def labels_from_predictions(path: Path) -> list[str]:
    names: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "true_id" in row:
                true_id = int(row["true_id"])
                true_name = row.get("true_label") or row.get("true_name") or str(true_id)
                names[true_id] = true_name
            if "pred_id" in row:
                pred_id = int(row["pred_id"])
                pred_name = row.get("pred_label") or str(pred_id)
                names.setdefault(pred_id, pred_name)
            for prefix in ("final", "neural", "kg"):
                key = f"{prefix}_pred_id"
                if key in row:
                    pred_id = int(row[key])
                    pred_name = row.get(f"{prefix}_pred_name") or str(pred_id)
                    names.setdefault(pred_id, pred_name)
    return [names[index] for index in sorted(names)]


def matrix_from_neural_predictions(path: Path, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            matrix[int(row["true_id"]), int(row["pred_id"])] += 1
    return matrix


def matrix_from_json(path: Path, branch: str) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data[branch], dtype=int)


def plot_matrix(matrix: np.ndarray, labels: list[str], title: str, output_stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = {
        "counts": matrix.astype(float),
        "normalized": matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1),
    }

    for suffix, values in variants.items():
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(values, cmap="Blues", aspect="auto")
        ax.set_title(title if suffix == "counts" else f"{title} (normalized)")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

        if suffix == "counts":
            max_value = values.max() if values.size else 0
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    value = int(values[i, j])
                    if value:
                        ax.text(
                            j,
                            i,
                            str(value),
                            ha="center",
                            va="center",
                            fontsize=5,
                            color="white" if value > max_value * 0.55 else "black",
                        )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{output_stem}_{suffix}.png", dpi=300)
        fig.savefig(OUT_DIR / f"{output_stem}_{suffix}.pdf")
        plt.close(fig)


def main() -> None:
    neural_csv = ROOT / "results" / "neural_only_transformer_30epoch" / "test_last_pt_predictions.csv"
    kg_csv = ROOT / "results" / "kg_only_scratch_same_final_code" / "test_last_pt_predictions.csv"
    kg_json = ROOT / "results" / "kg_only_scratch_same_final_code" / "test_last_pt_confusion_matrices.json"
    final_csv = ROOT / "results" / "final_structured_req_epoch19" / "test_last_pt_predictions.csv"
    final_json = ROOT / "results" / "final_structured_req_epoch19" / "test_last_pt_confusion_matrices.json"

    labels = labels_from_predictions(final_csv)
    num_classes = len(labels)

    plot_matrix(
        matrix_from_neural_predictions(neural_csv, num_classes),
        labels,
        "Neural-only last checkpoint confusion matrix",
        "neural_only_last",
    )
    plot_matrix(
        matrix_from_json(kg_json, "kg"),
        labels,
        "KG-only last checkpoint confusion matrix",
        "kg_only_last",
    )
    plot_matrix(
        matrix_from_json(final_json, "final"),
        labels,
        "Final fusion last checkpoint confusion matrix",
        "final_fusion_last",
    )

    print(f"Saved confusion matrix figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
