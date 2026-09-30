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
        for row in csv.DictReader(f):
            true_id = int(row["true_id"])
            names[true_id] = row.get("true_label") or row.get("true_name") or str(true_id)
            for key, name_key in (
                ("pred_id", "pred_label"),
                ("final_pred_id", "final_pred_name"),
                ("kg_pred_id", "kg_pred_name"),
                ("neural_pred_id", "neural_pred_name"),
            ):
                if key in row and row[key] != "":
                    pred_id = int(row[key])
                    names.setdefault(pred_id, row.get(name_key) or str(pred_id))
    return [names[index] for index in sorted(names)]


def matrix_from_neural_predictions(path: Path, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            matrix[int(row["true_id"]), int(row["pred_id"])] += 1
    return matrix


def matrix_from_json(path: Path, branch: str) -> np.ndarray:
    return np.asarray(json.loads(path.read_text(encoding="utf-8"))[branch], dtype=int)


def plot_count_matrix(matrix: np.ndarray, labels: list[str], title: str, output_stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    max_value = matrix.max() if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
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
    fig.savefig(OUT_DIR / f"{output_stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{output_stem}.pdf")
    plt.close(fig)


def main() -> None:
    neural_csv = ROOT / "results" / "neural_only_transformer_30epoch" / "test_last_pt_predictions.csv"
    kg_json = ROOT / "results" / "kg_only_scratch_same_final_code" / "test_last_pt_confusion_matrices.json"
    final_json = ROOT / "results" / "final_structured_req_epoch19" / "test_last_pt_confusion_matrices.json"
    final_csv = ROOT / "results" / "final_structured_req_epoch19" / "test_last_pt_predictions.csv"

    labels = labels_from_predictions(final_csv)
    num_classes = len(labels)

    plot_count_matrix(
        matrix_from_neural_predictions(neural_csv, num_classes),
        labels,
        "Method 1",
        "method_1_neural_only_counts",
    )
    plot_count_matrix(
        matrix_from_json(kg_json, "kg"),
        labels,
        "Method 2",
        "method_2_kg_only_counts",
    )
    plot_count_matrix(
        matrix_from_json(final_json, "final"),
        labels,
        "Method 3",
        "method_3_final_fusion_counts",
    )

    print(f"Saved method count confusion matrices to {OUT_DIR}")


if __name__ == "__main__":
    main()
