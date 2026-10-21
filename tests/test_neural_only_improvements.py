import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NEURAL_ONLY_DIR = ROOT / "final" / "code" / "neural_only_experiments"
NEURAL_COLAB_NOTEBOOK = ROOT / "notebooks" / "Neural_Only_Improved_Colab.ipynb"
sys.path.insert(0, str(NEURAL_ONLY_DIR))

from dataset import cache_path_for_source, label_names_from_split  # noqa: E402
from models import NeuralOnlyClassifier  # noqa: E402
from test_neural_only import (  # noqa: E402
    classification_metrics,
    parse_args,
    validate_evaluation_label_contract,
)
from train_neural_only import (  # noqa: E402
    augment_keypoints,
    class_weights,
    macro_f1,
    validate_no_split_overlap,
)


@pytest.mark.parametrize("backbone", ["transformer", "tcn"])
def test_motion_feature_models_forward_with_expected_shape(backbone: str) -> None:
    model = NeuralOnlyClassifier(
        backbone=backbone,
        num_classes=4,
        num_keypoints=3,
        input_dims=3,
        num_frames=5,
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        use_motion_features=True,
        pooling="mean",
    ).eval()
    keypoints = torch.randn(2, 5, 3, 3)
    keypoints[..., 2] = 1.0

    with torch.no_grad():
        logits = model(keypoints)

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()


def test_attention_pooling_all_zero_input_is_finite() -> None:
    model = NeuralOnlyClassifier(
        backbone="transformer",
        num_classes=3,
        num_keypoints=2,
        input_dims=3,
        num_frames=4,
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        use_motion_features=True,
        pooling="attention",
    ).eval()

    with torch.no_grad():
        logits = model(torch.zeros(2, 4, 2, 3))

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_mean_pooling_exactly_matches_legacy_forward() -> None:
    model = NeuralOnlyClassifier(
        backbone="transformer",
        num_classes=3,
        num_keypoints=2,
        input_dims=3,
        num_frames=4,
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        use_motion_features=False,
        pooling="mean",
    ).eval()
    keypoints = torch.randn(2, 4, 2, 3)
    keypoints[:, -1] = 0.0

    with torch.no_grad():
        actual = model(keypoints)
        projected = model.input_proj(keypoints.reshape(2, 4, -1))
        expected = model.classifier(model.encoder(projected).mean(dim=1))

    assert torch.equal(actual, expected)


def test_augmentation_is_seeded_and_preserves_missing_joint_coordinates() -> None:
    keypoints = torch.tensor(
        [
            [
                [[0.2, 0.3, 1.0], [0.0, 0.0, 0.0]],
                [[0.4, 0.5, 1.0], [0.0, 0.0, 0.0]],
                [[0.6, 0.7, 1.0], [0.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(1234)
    augmented = augment_keypoints(keypoints, 0.1, 0.0, 0.0)
    torch.manual_seed(1234)
    repeated = augment_keypoints(keypoints, 0.1, 0.0, 0.0)

    assert torch.equal(augmented, repeated)
    assert torch.equal(augmented[..., 2], keypoints[..., 2])
    assert torch.equal(augmented[:, :, 1, :2], keypoints[:, :, 1, :2])
    assert not torch.equal(augmented[:, :, 0, :2], keypoints[:, :, 0, :2])


def test_class_weights_are_normalized_and_zero_power_disables_weighting() -> None:
    samples = [("a", 0), ("b", 0), ("c", 1), ("d", 2)]

    assert class_weights(samples, num_classes=3, power=0.0) is None
    result = class_weights(samples, num_classes=3, power=0.5)
    counts = torch.tensor([2.0, 1.0, 1.0])
    expected = (counts.mean() / counts).pow(0.5)
    expected = expected / expected.mean()
    assert result is not None
    assert torch.allclose(result, expected)
    assert torch.isclose(result.mean(), torch.tensor(1.0))


@pytest.mark.parametrize("samples", [[("a", -1)], [("a", 3)]])
def test_class_weights_reject_invalid_labels(samples: list[tuple[str, int]]) -> None:
    with pytest.raises(ValueError, match="Labels outside"):
        class_weights(samples, num_classes=3, power=0.5)


def test_class_weights_reject_missing_training_classes() -> None:
    with pytest.raises(ValueError, match="no samples for class IDs"):
        class_weights([("a", 0), ("b", 2)], num_classes=3, power=0.5)


def test_cache_lookup_does_not_cross_split_or_class(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    (cache_root / "T4" / "val" / "G01").mkdir(parents=True)
    (cache_root / "T4" / "train" / "G02").mkdir(parents=True)
    (cache_root / "T4" / "val" / "G01" / "shared.pt").touch()
    (cache_root / "T4" / "train" / "G02" / "shared.pt").touch()

    source = "dataset/train/G01/shared.json"
    assert cache_path_for_source(source, cache_root, num_frames=4, fallback_split="train") is None


def test_cache_lookup_rejects_ambiguous_same_class_split(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    first = cache_root / "T4" / "train" / "G01" / "subject_a" / "shared.pt"
    second = cache_root / "T4" / "train" / "G01" / "subject_b" / "shared.pt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()

    with pytest.raises(RuntimeError, match="Ambiguous cache tensor"):
        cache_path_for_source("dataset/train/G01/shared.json", cache_root, 4, "train")


def test_cache_lookup_prefers_split_scoped_over_legacy(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    scoped = cache_root / "T4" / "test" / "G01" / "subject_a" / "shared.pt"
    legacy = cache_root / "T4" / "G01" / "shared.pt"
    scoped.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    scoped.touch()
    legacy.touch()

    result = cache_path_for_source("dataset/test/G01/shared.json", cache_root, 4, "test")

    assert result == scoped


def test_macro_f1_penalizes_predictions_into_absent_class() -> None:
    predictions = [0, 2]
    labels = [0, 1]

    assert macro_f1(predictions, labels, 3) == pytest.approx(1 / 3)
    assert classification_metrics(predictions, labels, 3)["macro_f1"] == pytest.approx(1 / 3)


def test_classification_metrics_reports_exact_class_metrics_and_labels() -> None:
    result = classification_metrics(
        predictions=[0, 1, 1, 2, 2],
        labels=[0, 0, 1, 2, 2],
        num_classes=3,
        label_names={0: "wave", 1: "fist", 2: "open"},
    )

    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 2]]
    assert result["macro_precision"] == pytest.approx((1.0 + 1 / 2 + 1.0) / 3)
    assert result["macro_recall"] == pytest.approx((1 / 2 + 1.0 + 1.0) / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1.0) / 3)
    assert result["balanced_accuracy"] == result["macro_recall"]
    assert result["per_class"] == [
        {
            "class_id": 0,
            "label_name": "wave",
            "support": 2,
            "correct": 1,
            "predicted": 1,
            "accuracy": 0.5,
            "precision": 1.0,
            "recall": 0.5,
            "f1": 2 / 3,
        },
        {
            "class_id": 1,
            "label_name": "fist",
            "support": 1,
            "correct": 1,
            "predicted": 2,
            "accuracy": 1.0,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 2 / 3,
        },
        {
            "class_id": 2,
            "label_name": "open",
            "support": 2,
            "correct": 2,
            "predicted": 2,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
    ]


def test_classification_metrics_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        classification_metrics([0], [0, 1], 2)
    with pytest.raises(ValueError, match="outside"):
        classification_metrics([-1], [0], 2)
    with pytest.raises(ValueError, match="outside"):
        classification_metrics([0], [2], 2)


def test_balanced_accuracy_averages_only_classes_with_support() -> None:
    result = classification_metrics([0, 0], [0, 1], num_classes=3)
    assert result["macro_recall"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert result["balanced_accuracy"] == pytest.approx((1.0 + 0.0) / 2)


def test_evaluation_rejects_labels_outside_checkpoint_output_range() -> None:
    with pytest.raises(ValueError, match="outside checkpoint output range"):
        validate_evaluation_label_contract(
            [("dataset/test/class_13/clip.json", 13)],
            num_classes=13,
            checkpoint_label_names={0: "B0A"},
            evaluation_label_names={13: "OLD_13"},
        )


def test_evaluation_parser_supports_validation_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_neural_only.py",
            "--checkpoint",
            "best.pt",
            "--cache-root",
            "cache",
            "--split",
            "val",
            "--test-file",
            "val.txt",
        ],
    )
    args = parse_args()
    assert args.split == "val"


def test_evaluation_rejects_same_id_with_different_label_semantics() -> None:
    with pytest.raises(ValueError, match="label mapping mismatch"):
        validate_evaluation_label_contract(
            [("dataset/test/G01/clip.json", 0)],
            num_classes=13,
            checkpoint_label_names={0: "B0A"},
            evaluation_label_names={0: "G01"},
        )


def test_evaluation_rejects_incomplete_label_metadata() -> None:
    with pytest.raises(ValueError, match="Incomplete label mapping"):
        validate_evaluation_label_contract(
            [("dataset/test/B0A/clip.json", 0)],
            num_classes=13,
            checkpoint_label_names={},
            evaluation_label_names={0: "B0A"},
        )


def test_evaluation_accepts_matching_label_contract() -> None:
    validate_evaluation_label_contract(
        [("dataset/test/B0A/clip.json", 0), ("dataset/test/G01/clip.json", 2)],
        num_classes=13,
        checkpoint_label_names={0: "B0A", 2: "G01"},
        evaluation_label_names={0: "B0A", 2: "G01"},
    )


def test_label_names_reject_conflicting_id_or_name_mapping(tmp_path: Path) -> None:
    conflicting_id = tmp_path / "conflicting_id.txt"
    conflicting_id.write_text(
        "dataset/train/B0A/a.json unused 0\n"
        "dataset/train/G01/b.json unused 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Conflicting label names"):
        label_names_from_split(conflicting_id)

    conflicting_name = tmp_path / "conflicting_name.txt"
    conflicting_name.write_text(
        "dataset/train/B0A/a.json unused 0\n"
        "dataset/train/B0A/b.json unused 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="maps to multiple IDs"):
        label_names_from_split(conflicting_name)


def test_split_overlap_is_rejected_after_path_normalization() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_no_split_overlap([("Data\\Clip.JSON", 0)], [("data/clip.json", 0)])


def test_neural_colab_notebook_keeps_locked_test_off_and_writes_artifacts() -> None:
    notebook = json.loads(NEURAL_COLAB_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    code = "\n".join(code_cells)
    locked_code = next(cell for cell in code_cells if "FINAL_TEST_SEEDS" in cell)
    assert "RUN_LOCKED_TEST = False" in code
    assert "FINAL_TEST_SEEDS = SEEDS if RUN_ALL_SEEDS else SEEDS[:1]" in code
    assert "ALLOW_LOCKED_TEST_OVERWRITE = False" in code
    assert "locked_test" in code
    assert "LOCKED_TEST_COMPLETE.json" in code
    assert "provenance.json" in code
    assert "checkpoint_sha256" in code
    assert "test_file_sha256" in code
    assert "completed_at_utc" in code
    assert "trained_dirs" not in locked_code
    assert locked_code.index("report_path.write_text") < locked_code.index("provenance_path.write_text")
    assert locked_code.index("provenance_path.write_text") < locked_code.index("completion_marker.write_text")
    for artifact_name in (
        "test_per_class_metrics.csv",
        "test_confusion_matrix.csv",
        "test_report.md",
    ):
        assert artifact_name in code
    assert "predictions.csv" in code
    assert "Accuracy/Recall" in code
