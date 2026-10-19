import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NEURAL_ONLY_DIR = ROOT / "final" / "code" / "neural_only_experiments"
sys.path.insert(0, str(NEURAL_ONLY_DIR))

from dataset import cache_path_for_source  # noqa: E402
from models import NeuralOnlyClassifier  # noqa: E402
from test_neural_only import classification_metrics  # noqa: E402
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


def test_split_overlap_is_rejected_after_path_normalization() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_no_split_overlap([("Data\\Clip.JSON", 0)], [("data/clip.json", 0)])
