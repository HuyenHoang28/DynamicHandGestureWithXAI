import csv
import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("summarize_neural_ablations", Path(__file__).parents[1] / "scripts" / "summarize_neural_ablations.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_run(root: Path, name: str, *, complete: bool = True, matrix=None):
    run = root / f"neural_ablation_{name}" / "transformer_seed42"
    metrics = run / "val_eval" / "val_metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    if matrix is None:
        matrix = [[1 if i == j else 0 for j in range(13)] for i in range(13)]
        matrix[8][2] = 3  # G07 is most often confused with G01 in this fixture.
    rows = []
    for i in range(13):
        support = sum(matrix[i])
        predicted = sum(matrix[j][i] for j in range(13))
        correct = matrix[i][i]
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class_id": i, "label_name": "B0A" if i == 0 else "B0B" if i == 1 else f"G{i - 1:02d}", "support": support, "correct": correct, "predicted": predicted, "accuracy": recall, "precision": precision, "recall": recall, "f1": f1})
    total = sum(map(sum, matrix))
    payload = {"accuracy": sum(matrix[i][i] for i in range(13)) / total, "macro_precision": sum(r["precision"] for r in rows) / 13, "macro_recall": sum(r["recall"] for r in rows) / 13, "macro_f1": sum(r["f1"] for r in rows) / 13, "per_class": rows, "confusion_matrix": matrix}
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    if complete:
        (run / "ABLATION_COMPLETE.json").write_text(json.dumps({"status": "complete", "name": name, "seed": 42, "output_dir": str(run), "validation_metrics": str(metrics)}), encoding="utf-8")
    return run, metrics


def test_partial_requires_explicit_flag_and_is_marked_incomplete(tmp_path, capsys):
    _write_run(tmp_path, "A_attention")
    _write_run(tmp_path, "B_masked_mean", complete=False)
    args = ["--output-root", str(tmp_path), "--experiments", "A_attention", "B_masked_mean"]
    assert MODULE.main(args) == 1
    assert MODULE.main([*args, "--allow-partial"]) == 0
    report = tmp_path / "ablation_comparison_seed42"
    assert (report / "ablation_summary.csv").exists()
    report_text = (report / "ablation_report.md").read_text(encoding="utf-8")
    assert "validation-only" in report_text
    assert "INCOMPLETE" in report_text
    with (report / "ablation_pair_confusions.csv").open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    g07 = [row for row in pairs if row["pair"] == "G07"]
    assert [(row["predicted_label"], row["count"]) for row in g07] == [("G07", "1"), ("G01", "3")]
    assert "warning:" in capsys.readouterr().err


def test_returns_failure_when_no_complete_runs(tmp_path):
    assert MODULE.main(["--output-root", str(tmp_path), "--experiments", "missing"]) == 1


def test_rejects_inconsistent_matrix(tmp_path):
    _, metrics = _write_run(tmp_path, "A_attention")
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload["confusion_matrix"][0][0] += 1
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    assert MODULE.main(["--output-root", str(tmp_path), "--experiments", "A_attention"]) == 1


def test_rejects_duplicate_label_and_wrong_marker_provenance(tmp_path):
    run, metrics = _write_run(tmp_path, "A_attention")
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload["per_class"][1]["label_name"] = "B0A"
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    assert MODULE.main(["--output-root", str(tmp_path), "--experiments", "A_attention"]) == 1

    _write_run(tmp_path, "A_attention")
    marker_path = run / "ABLATION_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["seed"] = 123
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert MODULE.main(["--output-root", str(tmp_path), "--experiments", "A_attention"]) == 1


def test_ranking_and_gate_include_exact_point_65(tmp_path):
    def cyclic_matrix(correct: int):
        matrix = [[0 for _ in range(13)] for _ in range(13)]
        for i in range(13):
            matrix[i][i] = correct
            matrix[i][(i + 1) % 13] = 20 - correct
        return matrix

    _write_run(tmp_path, "A_attention", matrix=cyclic_matrix(13))
    _write_run(tmp_path, "B_masked_mean", matrix=cyclic_matrix(14))
    assert MODULE.main(["--output-root", str(tmp_path), "--experiments", "A_attention", "B_masked_mean"]) == 0
    with (tmp_path / "ablation_comparison_seed42" / "ablation_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment"] for row in rows] == ["B_masked_mean", "A_attention"]
    assert rows[1]["min_per_class_f1"] == "0.65"
    assert rows[1]["gate_pass"] == "True"
