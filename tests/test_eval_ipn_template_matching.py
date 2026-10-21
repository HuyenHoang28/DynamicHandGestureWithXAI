"""Focused tests for optional symbolic evaluator abstention diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "eval_ipn_template_matching.py"
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ipn_template_matching import IPN_LABELS, diagnostic_record  # noqa: E402


@pytest.mark.parametrize("flag", ["--min-best-score", "--min-top2-margin"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.1"])
def test_cli_rejects_non_finite_or_negative_abstention_thresholds(
    flag: str,
    value: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(EVALUATOR), f"{flag}={value}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "finite and non-negative" in completed.stderr


def _write_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    data_root = root / "ipn_processed"
    keypoint_root = data_root / "keypoints"
    graph_root = data_root / "instance_graphs"
    template_root = data_root / "template_graphs"
    keypoint_root.mkdir(parents=True)
    graph_root.mkdir(parents=True)
    template_root.mkdir(parents=True)

    predicate_config = {"version": "ipn_predicate_v2"}
    build_config = {"schema_version": 2, "predicate_config": predicate_config}
    split_lines = []
    for label_id, label_name in enumerate(IPN_LABELS):
        relative = Path("val") / label_name / "SubjectVal" / f"clip_{label_id}.json"
        source = (keypoint_root / relative).as_posix()
        split_lines.append(f"{source} 0 {label_id}")
        graph_path = graph_root / relative
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(
            json.dumps(
                {
                    "graph_type": "instance_graph",
                    "schema_version": 2,
                    "video_id": f"clip_{label_id}",
                    "predicate_config": predicate_config,
                    "provenance": {
                        "predicate_config": predicate_config,
                        "build_config": build_config,
                        "source_fingerprint": {"clip": label_id},
                    },
                    "nodes": [{"id": f"video:clip_{label_id}"}],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )

        # Empty templates deliberately produce equal zero scores, making both
        # the minimum-score and minimum-margin abstention paths deterministic.
        (template_root / f"{label_name}.json").write_text(
            json.dumps(
                {
                    "graph_type": "template_graph",
                    "label_id": label_id,
                    "label_name": label_name,
                    "nodes": [],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )

    split_file = data_root / "val.txt"
    split_file.write_text("\n".join(split_lines) + "\n", encoding="utf-8")
    return data_root, keypoint_root, graph_root, template_root


def test_diagnostic_record_preserves_event_matches_and_abstain_shape() -> None:
    scored = [
        {
            "label_id": label_id,
            "label_name": label_name,
            "score": 0.2 if label_id == 0 else 0.1,
            "event_matches": [{"template_event": "event:0"}],
        }
        for label_id, label_name in enumerate(IPN_LABELS)
    ]
    record = diagnostic_record(
        source="clip.json",
        true_label_id=1,
        scored=scored,
        ordered_indices=list(range(len(scored))),
        prediction=None,
        abstained=True,
        abstention_reasons=["top2_margin_below_threshold"],
        tie=False,
        best_score=0.2,
        margin=0.1,
        min_best_score=None,
        min_top2_margin=0.2,
    )

    assert record["source"] == "clip.json"
    assert record["true_label"] == IPN_LABELS[1]
    assert record["predicted_label"] == "abstain"
    assert record["predicted_label_id"] is None
    assert record["abstain"] is True
    assert record["scores"][0]["score"] == 0.2
    assert record["top_labels"][0] == IPN_LABELS[0]
    assert record["score_template"][0]["event_matches"] == [{"template_event": "event:0"}]
    assert record["score_template_event_diagnostics"][0]["event_matches"]


def test_cli_reports_threshold_abstentions_and_jsonl_diagnostics() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        data_root, keypoint_root, graph_root, template_root = _write_fixture(Path(temporary))
        output = data_root / "evaluation.json"
        diagnostics = data_root / "diagnostics.jsonl"
        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--data-root",
                str(data_root),
                "--split",
                "val",
                "--keypoint-root",
                str(keypoint_root),
                "--template-dir",
                str(template_root),
                "--instance-graph-root",
                str(graph_root),
                "--min-best-score",
                "0.1",
                "--min-top2-margin",
                "0.1",
                "--diagnostics-jsonl",
                str(diagnostics),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["clips"] == len(IPN_LABELS)
        assert payload["abstained_count"] == len(IPN_LABELS)
        assert payload["abstention_rate"] == 1.0
        assert payload["accepted_count"] == 0
        assert payload["coverage"] == 0.0
        assert payload["selective_accuracy"] == 0.0
        assert payload["abstentions_by_class"] == {label: 1 for label in IPN_LABELS}
        assert "abstained=13/13" in completed.stdout

        records = [json.loads(line) for line in diagnostics.read_text(encoding="utf-8").splitlines()]
        assert len(records) == len(IPN_LABELS)
        assert records[0]["predicted_label"] == "abstain"
        assert records[0]["abstention_reasons"] == [
            "best_score_below_threshold",
            "top2_margin_below_threshold",
        ]
        assert len(records[0]["scores"]) == len(IPN_LABELS)
        assert records[0]["score_template"][0]["event_matches"] == []


def test_diagnostics_temp_file_is_cleaned_when_evaluation_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        data_root, keypoint_root, graph_root, template_root = _write_fixture(Path(temporary))
        first_graph = next(graph_root.rglob("*.json"))
        first_graph.write_text("{malformed", encoding="utf-8")
        diagnostics = data_root / "diagnostics.jsonl"

        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--data-root",
                str(data_root),
                "--split",
                "val",
                "--keypoint-root",
                str(keypoint_root),
                "--template-dir",
                str(template_root),
                "--instance-graph-root",
                str(graph_root),
                "--diagnostics-jsonl",
                str(diagnostics),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode != 0
        assert not diagnostics.exists()
        assert list(data_root.glob(".diagnostics.jsonl.*.tmp")) == []
