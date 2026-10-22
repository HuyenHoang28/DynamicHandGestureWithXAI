import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ipn_pair_arbitration import (  # noqa: E402
    IPN_LABELS,
    canonical_source,
    evaluate,
    load_neural_predictions,
    symbolic_pair_decision,
)


def _score(label, score=0.0, occurrences=0, *, active=False):
    return {
        "label_id": IPN_LABELS.index(label),
        "label_name": label,
        "score": score,
        "missing_required_count": 0,
        "temporal_score": 1.0,
        "event_matches": [{"observed_occurrences": occurrences}],
        "active_intervals": ([{"start_frame": 1, "end_frame": 3}] if active else []),
    }


def _symbolic(truth="G01", winner="G08", *, abstain=False, tie=False):
    scores = [_score(label) for label in IPN_LABELS]
    values = {"G01": (0.4, 1), "G08": (0.7, 2), "G10": (0.7, 1), "G11": (0.1, 1)}
    for label, (score, occurrences) in values.items():
        scores[IPN_LABELS.index(label)] = _score(
            label, score, occurrences, active=label == winner and label in {"G10", "G11"}
        )
    return {
        "source": "/content/drive/data/keypoints/val/sample.json",
        "source_key": "keypoints/val/sample.json",
        "true_label": truth,
        "true_label_id": IPN_LABELS.index(truth),
        "abstain": abstain,
        "tie": tie,
        "score_template": scores,
    }


def _neural(truth="G01", pred="G01"):
    other = "G08" if pred == "G01" else "G01"
    return {
        "source_key": "keypoints/val/sample.json",
        "true_label": truth,
        "pred_label": pred,
        "confidence": 0.55,
        "top_labels": [pred, other, "B0A"],
        "top_probs": [0.55, 0.45, 0.0],
    }


def test_canonical_source_joins_colab_and_local_paths():
    assert canonical_source("/content/x/keypoints/val/A.json") == canonical_source(
        r"external_data\x\keypoints\val\a.json"
    )
    with pytest.raises(ValueError, match="keypoints"):
        canonical_source("/content/no-marker/a.json")


def test_pair_decision_is_restricted_and_gated():
    accepted = symbolic_pair_decision(
        _symbolic(), ("G01", "G08"), min_score=0.15, min_margin=0.02, min_temporal_score=0.5
    )
    assert accepted["winner"] == "G08"
    # A global 13-way abstention may come from unrelated labels. Pair scoring
    # therefore uses only the two supplied candidates and its own gates.
    pair_only = symbolic_pair_decision(
        _symbolic(abstain=True, tie=True), ("G01", "G08"), min_score=0.15,
        min_margin=0.02, min_temporal_score=0.5,
    )
    assert pair_only["winner"] == "G08"

    tied = _symbolic()
    tied["score_template"][IPN_LABELS.index("G01")]["score"] = 0.7
    held = symbolic_pair_decision(
        tied, ("G01", "G08"), min_score=0.15, min_margin=0.02, min_temporal_score=0.5
    )
    assert held["winner"] is None
    assert held["reason"] == "pair_margin_below_threshold"

    held_at_zero_threshold = symbolic_pair_decision(
        tied, ("G01", "G08"), min_score=0.15, min_margin=0.0, min_temporal_score=0.5
    )
    assert held_at_zero_threshold["winner"] is None


def test_zoom_pair_rejects_opposite_active_direction():
    row = _symbolic(truth="G10", winner="G10")
    row["score_template"][IPN_LABELS.index("G11")]["active_intervals"] = [
        {"start_frame": 8, "end_frame": 10}
    ]
    result = symbolic_pair_decision(
        row, ("G10", "G11"), min_score=0.15, min_margin=0.02, min_temporal_score=0.5
    )
    assert result["winner"] is None
    assert result["reason"] == "opposite_active_direction_present"


def test_audit_mode_never_changes_prediction_and_override_is_explicit():
    neural = {"keypoints/val/sample.json": _neural()}
    symbolic = {"keypoints/val/sample.json": _symbolic()}
    audit_report, audit_rows = evaluate(
        neural, symbolic, enable_override=False, max_neural_confidence=1.0,
        max_neural_margin=0.2, min_symbolic_score=0.15,
        min_symbolic_margin=0.02, min_temporal_score=0.5,
    )
    assert audit_report["override_count"] == 0
    assert audit_report["proposed_override_count"] == 1
    assert audit_report["proposed_wrong_overrides"] == 1
    assert audit_rows[0]["final_prediction"] == "G01"
    override_report, override_rows = evaluate(
        neural, symbolic, enable_override=True, max_neural_confidence=1.0,
        max_neural_margin=0.2, min_symbolic_score=0.15,
        min_symbolic_margin=0.02, min_temporal_score=0.5,
    )
    assert override_report["override_count"] == 1
    assert override_rows[0]["final_prediction"] == "G08"


def test_non_pair_top2_is_never_overridden():
    neural_row = _neural()
    neural_row["top_labels"][:2] = ["G01", "G02"]
    report, rows = evaluate(
        {neural_row["source_key"]: neural_row},
        {"keypoints/val/sample.json": _symbolic()},
        enable_override=True, max_neural_confidence=1.0, max_neural_margin=1.0,
        min_symbolic_score=0.0, min_symbolic_margin=0.0, min_temporal_score=0.0,
    )
    assert report["override_count"] == 0
    assert rows[0]["reason"] == "unsupported_neural_top2"


def test_wrong_to_different_wrong_is_reported_as_wrong_but_accuracy_neutral():
    neural_row = _neural(truth="G02", pred="G01")
    report, _ = evaluate(
        {neural_row["source_key"]: neural_row},
        {"keypoints/val/sample.json": _symbolic(truth="G02")},
        enable_override=True, max_neural_confidence=1.0, max_neural_margin=1.0,
        min_symbolic_score=0.0, min_symbolic_margin=0.0, min_temporal_score=0.0,
    )
    assert report["override_count"] == 1
    assert report["wrong_overrides"] == 1
    assert report["proposed_wrong_overrides"] == 1
    assert report["proposed_neutral_overrides"] == 1
    assert report["proposed_net_correct_change"] == 0


def test_cli_requires_explicit_neural_margin_for_override(tmp_path):
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "eval_ipn_pair_arbitration.py"),
            "--neural-predictions", str(tmp_path / "n.csv"),
            "--symbolic-diagnostics", str(tmp_path / "s.jsonl"),
            "--output", str(tmp_path / "out.json"),
            "--enable-override",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2
    assert "requires an explicit --max-neural-margin" in completed.stderr


@pytest.mark.parametrize(
    "confidence,top_probs,error",
    [
        ("0.9", "0.8 0.2", "confidence does not match"),
        ("0.6", "0.6 0.7", "not descending"),
        ("1.2", "1.2 0.1", "outside \\[0, 1\\]"),
    ],
)
def test_neural_csv_rejects_inconsistent_probabilities(tmp_path, confidence, top_probs, error):
    path = tmp_path / "predictions.csv"
    path.write_text(
        "path,true_id,true_label,pred_id,pred_label,confidence,top_labels,top_probs\n"
        f"/x/keypoints/val/a.json,2,G01,2,G01,{confidence},G01 G08,{top_probs}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=error):
        load_neural_predictions(path)
