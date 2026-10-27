"""Unit coverage for train/validation-only temporal-factor calibration."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_ipn_temporal_factors.py"
spec = importlib.util.spec_from_file_location("calibrate_ipn_temporal_factors", SCRIPT)
calibrator = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = calibrator
spec.loader.exec_module(calibrator)


def sample(truth: str, prediction: str) -> dict:
    return {"true_label": truth, "prediction": prediction, "subject": "fixture"}


def test_aggregate_counts_abstention_as_wrong_and_reports_unknown() -> None:
    rows = [sample("G01", "G01"), sample("G01", "unknown"), sample("G08", "G01")]
    result = calibrator.aggregate(rows, ("G01", "G08"))
    assert result["support"] == 3
    assert result["covered"] == 2
    assert result["correct"] == 1
    assert result["accuracy_on_covered"] == .5
    assert result["overall_accuracy_abstain_wrong"] == pytest.approx(1 / 3)
    assert result["confusion_unknown_included"]["G01"]["unknown"] == 1


def test_config_selection_uses_validation_scores_not_train_scores() -> None:
    train_scores = [
        {"config": {"id": "train-favorite"}, "objective": .99, "coverage": .9},
    ]
    validation_scores = [
        {"config": {"id": "validation-winner"}, "objective": .8, "coverage": .6},
        {"config": {"id": "train-favorite"}, "objective": .4, "coverage": .95},
    ]
    selected = calibrator.select_config(validation_scores)
    assert selected["config"]["id"] == "validation-winner"
    assert train_scores[0]["config"]["id"] != selected["config"]["id"]


def test_min_coverage_is_a_per_pair_feasibility_gate() -> None:
    rows = [sample("G01", "G01"), sample("G08", "unknown"), sample("G02", "G02"), sample("G09", "G09")]
    score = calibrator.score_config(rows, {"x": 1}, min_coverage=.75)
    assert score["per_pair"]["G01/G08"]["coverage"] == .5
    assert score["feasible"] is False
    assert score["objective"] == -1.0


def test_cli_contract_has_no_test_split_or_test_manifest_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        calibrator.parse_args(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--limit" in help_text
    assert "--keypoint-root" in help_text
    assert "--test" not in help_text
    assert "test_manifest" not in help_text


def test_pair_decision_counts_only_the_dominant_hand() -> None:
    predicates = [
        {"name": "index_click_cycle", "subject": "left_hand", "confidence": .9},
        {"name": "index_click_cycle", "subject": "left_hand", "confidence": .8},
        {"name": "index_click_cycle", "subject": "right_hand", "confidence": .2},
    ]
    prediction, evidence = calibrator.pair_decision("G08", predicates, {})
    assert prediction == "G08"
    assert evidence["subject"] == "left_hand"
    assert evidence["count"] == 2


def test_zoom_decision_abstains_when_opposite_hands_are_not_decisive() -> None:
    predicates = [
        {"name": "pinch_opening", "subject": "left_hand", "confidence": .8},
        {"name": "pinch_closing", "subject": "right_hand", "confidence": .7},
    ]
    prediction, _ = calibrator.pair_decision("G10", predicates, {"pinch_dominance_ratio": 1.2})
    assert prediction == "unknown"


@pytest.mark.parametrize("label,factor", [("G08", "index_click_cycle"), ("G09", "two_finger_click_cycle")])
def test_pair_decision_abstains_above_exact_double_count(label: str, factor: str) -> None:
    predicates = [
        {"name": factor, "subject": "left_hand", "confidence": .9},
        {"name": factor, "subject": "left_hand", "confidence": .8},
        {"name": factor, "subject": "left_hand", "confidence": .7},
    ]
    prediction, evidence = calibrator.pair_decision(label, predicates, {})
    assert evidence["count"] == 3
    assert prediction == "unknown"


def test_project_dataset_resolves_ipn_split_label_subject_cache_layout(tmp_path: Path) -> None:
    source = "external_data/ipn_processed/keypoints/train/G01/1CM1_4/example.json"
    split_file = tmp_path / "train.txt"
    split_file.write_text(f"{source} 2 2\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    cache_path = cache_root / "T64" / "train" / "G01" / "1CM1_4" / "example.pt"
    cache_path.parent.mkdir(parents=True)
    expected = torch.zeros(64, len(calibrator.KEYPOINT_NAMES), 3)
    torch.save(expected, cache_path)

    dataset = calibrator.KeypointSequenceDataset(
        split_file,
        tmp_path / "raw-keypoints-not-needed",
        "train",
        cache_root=cache_root,
    )
    item = dataset[0]
    assert torch.equal(item["keypoints"], expected)


@pytest.mark.parametrize("source", [
    "external_data/ipn_processed/keypoints/test/G01/1CM1_4/example.json",
    "external_data/ipn_processed/keypoints/G01/1CM1_4/example.json",
])
def test_train_calibration_rejects_cross_split_or_unscoped_source(source: str) -> None:
    with pytest.raises(ValueError, match="missing/cross-split"):
        calibrator.validate_source_split(source, "train")


def test_validation_calibration_accepts_val_alias_only() -> None:
    calibrator.validate_source_split("root/validation/G01/subject/example.json", "val")
    with pytest.raises(ValueError, match="missing/cross-split"):
        calibrator.validate_source_split("root/train/G01/subject/example.json", "val")

