#!/usr/bin/env python3
"""Summarize completed neural validation ablations (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

DEFAULT_EXPERIMENTS = ["A_attention", "B_masked_mean", "C_masked_mean_light_dropout"]
FOCUSED_PAIRS = [("G01", "G08"), ("G02", "G09"), ("G10", "G11")]
CANONICAL_LABELS = ["B0A", "B0B"] + [f"G{i:02d}" for i in range(1, 12)]
PER_CLASS_FIELDS = ["class_id", "label_name", "support", "correct", "predicted", "accuracy", "precision", "recall", "f1"]
TOLERANCE = 1e-6


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def close(value: object, expected: float) -> bool:
    return finite(value) and math.isclose(float(value), expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def load_experiment(root: Path, name: str, seed: int, num_classes: int) -> dict | None:
    output = root / f"neural_ablation_{name}" / f"transformer_seed{seed}"
    metrics_path = output / "val_eval" / "val_metrics.json"
    marker_path = output / "ABLATION_COMPLETE.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: {name}: missing/unreadable completion or metrics ({exc})", file=sys.stderr)
        return None
    expected_output = output.resolve()
    expected_metrics = metrics_path.resolve()
    marker_metrics = marker.get("validation_metrics") if isinstance(marker, dict) else None
    marker_output = marker.get("output_dir") if isinstance(marker, dict) else None
    try:
        marker_metrics_path = Path(marker_metrics).resolve() if isinstance(marker_metrics, str) else None
        marker_output_path = Path(marker_output).resolve() if isinstance(marker_output, str) else None
    except (OSError, RuntimeError, ValueError):
        marker_metrics_path = marker_output_path = None
    if (
        not isinstance(marker, dict)
        or marker.get("status") != "complete"
        or marker.get("name") != name
        or marker.get("seed") != seed
        or marker_output_path != expected_output
        or marker_metrics_path != expected_metrics
    ):
        print(f"warning: {name}: invalid/incomplete completion marker; skipped", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        print(f"warning: {name}: metrics is not a dict; skipped", file=sys.stderr)
        return None
    if num_classes != len(CANONICAL_LABELS):
        print(f"warning: {name}: num_classes must be exactly {len(CANONICAL_LABELS)}; skipped", file=sys.stderr)
        return None
    required_metrics = {"accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class", "confusion_matrix"}
    if any(not finite(payload.get(key)) or not 0 <= float(payload[key]) <= 1 for key in required_metrics - {"per_class", "confusion_matrix"}):
        print(f"warning: {name}: missing/non-numeric summary metrics; skipped", file=sys.stderr)
        return None
    rows = payload.get("per_class")
    required_row = {"class_id", "label_name", "support", "correct", "predicted", "accuracy", "precision", "recall", "f1"}
    if not isinstance(rows, list) or len(rows) != num_classes:
        print(f"warning: {name}: per_class must contain exactly {num_classes} rows; skipped", file=sys.stderr)
        return None
    for row in rows:
        if not isinstance(row, dict) or not required_row.issubset(row):
            print(f"warning: {name}: per_class row fields are invalid; skipped", file=sys.stderr)
            return None
        if not isinstance(row["class_id"], int) or isinstance(row["class_id"], bool) or not isinstance(row["label_name"], str):
            print(f"warning: {name}: invalid class ID/name; skipped", file=sys.stderr)
            return None
        if any(not nonnegative_int(row[key]) for key in ("support", "correct", "predicted")):
            print(f"warning: {name}: non-numeric per-class field; skipped", file=sys.stderr)
            return None
        if any(not finite(row[key]) or not 0 <= float(row[key]) <= 1 for key in ("accuracy", "precision", "recall", "f1")):
            print(f"warning: {name}: per-class metric outside [0, 1]; skipped", file=sys.stderr)
            return None
    if sorted(row["class_id"] for row in rows) != list(range(num_classes)) or len({row["class_id"] for row in rows}) != num_classes:
        print(f"warning: {name}: per_class IDs are not exactly 0..{num_classes - 1}; skipped", file=sys.stderr)
        return None
    if [row["label_name"] for row in sorted(rows, key=lambda r: r["class_id"])] != CANONICAL_LABELS or len({row["label_name"] for row in rows}) != num_classes:
        print(f"warning: {name}: per_class labels do not match canonical contract; skipped", file=sys.stderr)
        return None
    matrix = payload["confusion_matrix"]
    if not isinstance(matrix, list) or len(matrix) != num_classes or any(not isinstance(r, list) or len(r) != num_classes for r in matrix):
        print(f"warning: {name}: confusion_matrix has wrong dimensions; skipped", file=sys.stderr)
        return None
    if any(not nonnegative_int(v) for r in matrix for v in r):
        print(f"warning: {name}: confusion_matrix contains non-numeric values; skipped", file=sys.stderr)
        return None
    ordered = sorted(rows, key=lambda r: r["class_id"])
    row_supports = [sum(row) for row in matrix]
    col_predicted = [sum(matrix[i][j] for i in range(num_classes)) for j in range(num_classes)]
    total = sum(row_supports)
    if total == 0 or any(row_supports[i] != row["support"] or col_predicted[i] != row["predicted"] or matrix[i][i] != row["correct"] for i, row in enumerate(ordered)):
        print(f"warning: {name}: per_class counts do not agree with confusion_matrix; skipped", file=sys.stderr)
        return None
    for i, row in enumerate(ordered):
        accuracy = row["correct"] / row["support"] if row["support"] else 0.0
        precision = row["correct"] / row["predicted"] if row["predicted"] else 0.0
        recall = row["correct"] / row["support"] if row["support"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if not all(close(row[key], expected) for key, expected in (("accuracy", accuracy), ("precision", precision), ("recall", recall), ("f1", f1))):
            print(f"warning: {name}: per_class metrics are inconsistent with counts; skipped", file=sys.stderr)
            return None
    overall = sum(matrix[i][i] for i in range(num_classes)) / total
    macro = {key: sum(row[key] for row in ordered) / num_classes for key in ("precision", "recall", "f1")}
    if not close(payload["accuracy"], overall) or not close(payload["macro_precision"], macro["precision"]) or not close(payload["macro_recall"], macro["recall"]) or not close(payload["macro_f1"], macro["f1"]):
        print(f"warning: {name}: summary metrics are inconsistent with per_class counts; skipped", file=sys.stderr)
        return None
    names = {row["label_name"]: row["class_id"] for row in ordered}
    return {"name": name, "payload": payload, "rows": ordered, "matrix": matrix, "names": names}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--num-classes", type=int, default=13)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="report valid experiments even when requested runs are incomplete")
    args = parser.parse_args(argv)
    report_dir = args.report_dir or args.output_root / f"ablation_comparison_seed{args.seed}"
    complete = [item for name in args.experiments if (item := load_experiment(args.output_root, name, args.seed, args.num_classes))]
    incomplete = [name for name in args.experiments if name not in {item["name"] for item in complete}]
    if incomplete and not args.allow_partial:
        print(f"error: requested experiments are incomplete or invalid: {', '.join(incomplete)}", file=sys.stderr)
        return 1
    if not complete:
        print("error: no complete valid ablation experiments found", file=sys.stderr)
        return 1
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, class_rows, pair_rows = [], [], []
    for exp in complete:
        p, rows = exp["payload"], exp["rows"]
        weakest = min(rows, key=lambda r: (r["f1"], r["class_id"]))
        summary_rows.append({"experiment": exp["name"], "accuracy": p["accuracy"], "macro_precision": p["macro_precision"], "macro_recall": p["macro_recall"], "macro_f1": p["macro_f1"], "min_per_class_f1": weakest["f1"], "weakest_class": weakest["label_name"], "gate_pass": all(r["f1"] >= 0.65 for r in rows)})
        for row in rows:
            class_rows.append({"experiment": exp["name"], **{field: row[field] for field in PER_CLASS_FIELDS}})
        labels = {r["label_name"]: r["class_id"] for r in rows}
        names_by_id = {r["class_id"]: r["label_name"] for r in rows}
        for a, b in FOCUSED_PAIRS:
            for true_name, pred_name in ((a, b), (b, a)):
                pair_rows.append({"experiment": exp["name"], "pair": f"{a}/{b}", "true_label": true_name, "predicted_label": pred_name, "count": exp["matrix"][labels[true_name]][labels[pred_name]]})
        # G07 is reported as its diagonal and its largest off-diagonal confusion.
        gid = labels["G07"]
        pair_rows.append({"experiment": exp["name"], "pair": "G07", "true_label": "G07", "predicted_label": "G07", "count": exp["matrix"][gid][gid]})
        wrong = max((j for j in range(args.num_classes) if j != gid), key=lambda j: exp["matrix"][gid][j], default=gid)
        pair_rows.append({"experiment": exp["name"], "pair": "G07", "true_label": "G07", "predicted_label": names_by_id.get(wrong, str(wrong)), "count": exp["matrix"][gid][wrong]})
    summary_rows.sort(key=lambda r: (-r["macro_f1"], -r["min_per_class_f1"], r["experiment"]))
    write_csv(report_dir / "ablation_summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(report_dir / "ablation_per_class.csv", list(class_rows[0]), class_rows)
    pair_fields = ["experiment", "pair", "true_label", "predicted_label", "count"]
    write_csv(report_dir / "ablation_pair_confusions.csv", pair_fields, pair_rows)
    lines = ["# Neural ablation comparison", "", "Validation-only artifact summary (not a test-set claim).", "", "## Ranking", "", "| Experiment | Accuracy | Macro F1 | Min per-class F1 | Weakest class | Gate (all F1 >= 0.65) |", "|---|---:|---:|---:|---|---|"]
    for row in summary_rows:
        lines.append(f"| {row['experiment']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['min_per_class_f1']:.4f} | {row['weakest_class']} | {'PASS' if row['gate_pass'] else 'FAIL'} |")
    lines += ["", "## Focused confusions", "", "| Experiment | Pair | True | Predicted | Count |", "|---|---|---|---|---:|"]
    for row in pair_rows:
        lines.append(f"| {row['experiment']} | {row['pair']} | {row['true_label']} | {row['predicted_label']} | {row['count']} |")
    lines += ["", "## Scope and gate", "", "All values above come from the validated `val_eval/val_metrics.json` artifacts. Experiments are ranked by macro-F1, then minimum per-class F1. The gate passes only when every per-class F1 is at least 0.65. This report is validation-only.", ""]
    if incomplete:
        lines.insert(2, f"**INCOMPLETE:** omitted requested experiments: {', '.join(incomplete)}. This is a partial report.")
    report_path = report_dir / "ablation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Saved comparison artifacts to: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
