"""Tests for the opt-in IPN accuracy improvements (hand geometry, temporal/geometric augmentation)."""

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NEURAL_ONLY_DIR = ROOT / "final" / "code" / "neural_only_experiments"
sys.path.insert(0, str(NEURAL_ONLY_DIR))

from hand_geometry import (  # noqa: E402
    FEATURES_PER_HAND,
    HAND_GEOMETRY_DIMS,
    LEFT_HAND,
    RIGHT_HAND,
    hand_geometry_features,
    hand_geometry_inputs,
)
from models import NeuralOnlyClassifier  # noqa: E402
from train_neural_only import augment_keypoints, temporal_crop_resample  # noqa: E402


def _hand_clip(frames: int = 4) -> torch.Tensor:
    """Right hand with hand size 1.0: wrist (0,0), middle knuckle (0,-1)."""
    clip = torch.zeros(1, frames, 52, 3)
    for idx in RIGHT_HAND.values():
        clip[0, :, idx, 2] = 1.0
    clip[0, :, RIGHT_HAND["root"], :2] = torch.tensor([0.0, 0.0])
    clip[0, :, RIGHT_HAND["mid1"], :2] = torch.tensor([0.0, -1.0])
    clip[0, :, RIGHT_HAND["fore1"], :2] = torch.tensor([0.2, -1.0])
    clip[0, :, RIGHT_HAND["fore4"], :2] = torch.tensor([0.2, -2.0])
    clip[0, :, RIGHT_HAND["mid4"], :2] = torch.tensor([0.0, -2.0])
    clip[0, :, RIGHT_HAND["thumb4"], :2] = torch.tensor([-0.5, -1.0])
    clip[0, :, RIGHT_HAND["ring4"], :2] = torch.tensor([-0.2, -1.8])
    clip[0, :, RIGHT_HAND["pinky4"], :2] = torch.tensor([-0.4, -1.5])
    return clip


def test_geometry_dimensions_match_documented_layout() -> None:
    assert FEATURES_PER_HAND == 10
    assert HAND_GEOMETRY_DIMS == 20
    features, valid = hand_geometry_features(_hand_clip())
    assert features.shape == (1, 4, 20)
    assert valid.shape == (1, 4, 20)
    assert hand_geometry_inputs(_hand_clip()).shape == (1, 4, 40)


def test_finger_flexion_signal_is_scale_invariant_and_counts_clicks() -> None:
    clip = _hand_clip(6)
    # Index tip bends toward its knuckle on frames 2 and 4: two "clicks".
    for frame in (2, 4):
        clip[0, frame, RIGHT_HAND["fore4"], :2] = torch.tensor([0.2, -1.2])
    features, valid = hand_geometry_features(clip)
    right_index_flex = features[0, :, FEATURES_PER_HAND + 5]  # right hand, index tip-to-knuckle
    assert valid[0, :, FEATURES_PER_HAND + 5].all()
    assert right_index_flex[0].item() == pytest.approx(1.0, abs=1e-5)
    assert right_index_flex[2].item() == pytest.approx(0.2, abs=1e-5)
    dips = ((right_index_flex[1:-1] < right_index_flex[:-2]) & (right_index_flex[1:-1] < right_index_flex[2:])).sum()
    assert int(dips) == 2  # one dip per click, so a click count is readable from this signal

    # Doubling the whole hand (a closer camera) leaves every ratio unchanged.
    scaled = clip.clone()
    scaled[..., :2] *= 2.0
    scaled_features, _ = hand_geometry_features(scaled)
    assert torch.allclose(features, scaled_features, atol=1e-5)


def test_pinch_distance_feature() -> None:
    clip = _hand_clip()
    clip[0, :, RIGHT_HAND["thumb4"], :2] = torch.tensor([0.2, -2.0])  # touches index tip
    features, _ = hand_geometry_features(clip)
    assert features[0, 0, FEATURES_PER_HAND + 7].item() == pytest.approx(0.0, abs=1e-5)
    clip[0, :, RIGHT_HAND["thumb4"], :2] = torch.tensor([1.2, -2.0])
    features, _ = hand_geometry_features(clip)
    assert features[0, 0, FEATURES_PER_HAND + 7].item() == pytest.approx(1.0, abs=1e-5)


def test_missing_joint_gives_zero_feature_and_zero_change_not_a_fake_value() -> None:
    clip = _hand_clip(4)
    clip[0, 2, RIGHT_HAND["fore4"], 2] = 0.0  # index tip lost on frame 2
    features, valid = hand_geometry_features(clip)
    column = FEATURES_PER_HAND + 5
    assert not valid[0, 2, column]
    assert features[0, 2, column].item() == 0.0
    inputs = hand_geometry_inputs(clip)
    change = inputs[0, :, HAND_GEOMETRY_DIMS + column]
    assert change[2].item() == 0.0  # no jump into the gap
    assert change[3].item() == 0.0  # no jump out of the gap


def test_all_zero_frame_is_invalid_without_nan() -> None:
    clip = torch.zeros(2, 5, 52, 3)
    clip[..., 2] = 1.0  # confident but every coordinate is zero (failed normalisation)
    features, valid = hand_geometry_features(clip)
    assert torch.isfinite(features).all()
    assert not valid.any()


def test_left_hand_uses_its_own_joints() -> None:
    clip = torch.zeros(1, 3, 52, 3)
    for idx in LEFT_HAND.values():
        clip[0, :, idx, 2] = 1.0
    clip[0, :, LEFT_HAND["mid1"], :2] = torch.tensor([0.0, -1.0])
    clip[0, :, LEFT_HAND["mid4"], :2] = torch.tensor([0.0, -3.0])
    features, valid = hand_geometry_features(clip)
    assert valid[0, 0, :FEATURES_PER_HAND].any()
    assert not valid[0, 0, FEATURES_PER_HAND:].any()
    assert features[0, 0, 2].item() == pytest.approx(3.0, abs=1e-5)  # middle tip-to-wrist / hand size


def test_model_with_hand_features_forward_and_default_is_unchanged() -> None:
    kwargs = dict(
        backbone="transformer", num_classes=13, num_keypoints=52, input_dims=3, num_frames=8,
        d_model=16, layers=1, heads=2, dropout=0.0, use_motion_features=True, pooling="masked_mean",
    )
    plain = NeuralOnlyClassifier(**kwargs)
    with_geometry = NeuralOnlyClassifier(**kwargs, hand_features=True)
    assert plain.input_proj[0].in_features == 52 * 5  # old checkpoints keep this shape
    assert with_geometry.input_proj[0].in_features == 52 * 5 + 40
    clip = _hand_clip(8)
    assert with_geometry.eval()(clip).shape == (1, 13)
    assert torch.isfinite(with_geometry.eval()(clip)).all()
    plain_state = set(plain.state_dict())
    assert plain_state == set(with_geometry.state_dict())  # only the projection width differs


def test_hand_features_reject_other_keypoint_layouts() -> None:
    with pytest.raises(ValueError):
        NeuralOnlyClassifier(num_keypoints=3, num_frames=4, d_model=8, layers=1, heads=2, hand_features=True)


def test_temporal_crop_copies_real_frames_and_keeps_shape() -> None:
    torch.manual_seed(0)
    clip = torch.rand(16, 20, 4, 3) + 0.5
    cropped = temporal_crop_resample(clip, 0.5)
    assert cropped.shape == clip.shape
    for b in range(clip.shape[0]):
        for t in range(clip.shape[1]):
            matches = (clip[b] == cropped[b, t]).flatten(1).all(dim=1)
            assert matches.any(), "every output frame must be an untouched input frame"


def test_temporal_crop_changes_clips_and_off_by_default() -> None:
    torch.manual_seed(1)
    clip = torch.arange(20).float().view(1, 20, 1, 1).expand(4, 20, 3, 3).clone()
    clip[..., 2] = 1.0
    assert torch.equal(augment_keypoints(clip, 0.0, 0.0, 0.0), clip)
    changed = temporal_crop_resample(clip, 0.4)
    assert not torch.equal(changed, clip)


def test_scale_and_shift_only_move_valid_points() -> None:
    torch.manual_seed(2)
    clip = torch.randn(3, 6, 5, 3)
    clip[..., 2] = 1.0
    clip[:, :, 0, 2] = 0.0
    clip[:, :, 0, :2] = 0.0  # a missing joint must stay at zero
    out = augment_keypoints(clip, 0.0, 0.0, 0.0, scale_jitter=0.2, shift_std=0.05)
    assert torch.equal(out[:, :, 0], clip[:, :, 0])
    assert not torch.equal(out[:, :, 1:, :2], clip[:, :, 1:, :2])
    assert torch.equal(out[..., 2], clip[..., 2])
    ratio = (out[:, :, 1:, :2].abs().mean() / clip[:, :, 1:, :2].abs().mean()).item()
    assert 0.5 < ratio < 1.5


def test_existing_augmentation_defaults_are_bit_identical() -> None:
    clip = torch.randn(2, 8, 4, 3)
    clip[..., 2] = 1.0
    torch.manual_seed(3)
    old_style = augment_keypoints(clip, 0.005, 0.05, 0.03)
    torch.manual_seed(3)
    explicit = augment_keypoints(clip, 0.005, 0.05, 0.03, time_crop_min=1.0, scale_jitter=0.0, shift_std=0.0)
    assert torch.equal(old_style, explicit)


# --- V4 neural adapter parity -------------------------------------------------

V4_CODE_DIR = ROOT / "final" / "code"
if str(V4_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V4_CODE_DIR))

from src_neurosymbolic.models.cross_attention_v2 import NeuralBaselineAdapter  # noqa: E402
from src_neurosymbolic.models import hand_geometry as v4_hand_geometry  # noqa: E402


@pytest.mark.parametrize("motion", [False, True])
@pytest.mark.parametrize("pooling", ["mean", "masked_mean", "attention"])
@pytest.mark.parametrize("hand", [False, True])
def test_v4_adapter_matches_neural_only_model_exactly(motion: bool, pooling: str, hand: bool) -> None:
    """A neural-only checkpoint must load strictly into V4's adapter and give the same logits."""
    torch.manual_seed(0)
    shape = dict(num_classes=13, num_frames=10, d_model=16, layers=2, heads=2, dropout=0.0)
    neural_only = NeuralOnlyClassifier(
        backbone="transformer", use_motion_features=motion, pooling=pooling, hand_features=hand, **shape
    ).eval()
    adapter = NeuralBaselineAdapter(use_motion_features=motion, pooling=pooling, hand_features=hand, **shape).eval()
    adapter.load_state_dict(neural_only.state_dict(), strict=True)

    clip = torch.randn(3, 10, 52, 3)
    clip[..., 2] = torch.rand(3, 10, 52) * (torch.rand(3, 10, 52) > 0.2)  # some missing joints
    clip[1, 4:7] = 0.0  # a run of empty frames
    with torch.no_grad():
        assert torch.allclose(adapter(clip)["logits"], neural_only(clip), atol=1e-5)


def test_v4_adapter_defaults_keep_old_checkpoint_layout() -> None:
    adapter = NeuralBaselineAdapter(num_classes=13, num_frames=8, d_model=16, layers=1, heads=2)
    assert adapter.input_proj[0].in_features == 52 * 3
    assert not hasattr(adapter, "pool_score")
    clip = torch.randn(2, 8, 52, 3)
    assert adapter(clip)["logits"].shape == (2, 13)


def test_v4_adapter_rejects_mismatched_checkpoint_instead_of_loading_silently(tmp_path) -> None:
    with_velocity = NeuralOnlyClassifier(num_classes=13, num_frames=8, d_model=16, layers=1, heads=2, pooling="masked_mean")
    path = tmp_path / "ckpt.pt"
    torch.save({"model": with_velocity.state_dict()}, path)
    plain = NeuralBaselineAdapter(num_classes=13, num_frames=8, d_model=16, layers=1, heads=2)
    with pytest.raises(RuntimeError):
        plain.load_baseline_checkpoint(path)
    matching = NeuralBaselineAdapter(
        num_classes=13, num_frames=8, d_model=16, layers=1, heads=2, use_motion_features=True, pooling="masked_mean"
    )
    matching.load_baseline_checkpoint(path)


def test_hand_geometry_copies_stay_identical() -> None:
    torch.manual_seed(4)
    clip = torch.randn(2, 6, 52, 3)
    clip[..., 2] = torch.rand(2, 6, 52)
    a, av = hand_geometry_features(clip)
    b, bv = v4_hand_geometry.hand_geometry_features(clip)
    assert torch.equal(a, b) and torch.equal(av, bv)
    assert torch.equal(hand_geometry_inputs(clip), v4_hand_geometry.hand_geometry_inputs(clip))


# --- ensembling ----------------------------------------------------------------

from ensemble_neural_only import average_probabilities, build_model  # noqa: E402


def test_average_probabilities_is_the_mean_of_softmax_not_of_logits() -> None:
    a = torch.tensor([[10.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 3.0, 0.0]])
    mean = average_probabilities([a, b])
    expected = (a.softmax(-1) + b.softmax(-1)) / 2
    assert torch.allclose(mean, expected)
    assert torch.allclose(mean.sum(dim=-1), torch.ones(1))
    # A very confident member must not drown out the others the way raw-logit averaging would.
    assert mean[0, 1].item() > 0.4


def test_average_probabilities_weights_and_validation() -> None:
    a = torch.tensor([[2.0, 0.0]])
    b = torch.tensor([[0.0, 2.0]])
    weighted = average_probabilities([a, b], [3.0, 1.0])
    assert weighted[0, 0].item() > 0.5
    assert torch.allclose(average_probabilities([a, b], [1.0, 0.0]), a.softmax(-1))
    with pytest.raises(ValueError):
        average_probabilities([])
    with pytest.raises(ValueError):
        average_probabilities([a, b], [1.0])
    with pytest.raises(ValueError):
        average_probabilities([a, b], [-1.0, 2.0])
    with pytest.raises(ValueError):
        average_probabilities([a, torch.zeros(1, 3)])


def test_ensemble_rebuilds_models_from_checkpoint_args_including_new_flags() -> None:
    import argparse

    args = argparse.Namespace(
        backbone="transformer", num_classes=13, num_frames=8, d_model=16, layers=1, heads=2, dropout=0.0,
        disable_motion_features=False, pooling="masked_mean", hand_features=True,
    )
    model = build_model(args)
    assert model.hand_features and model.use_motion_features and model.pooling == "masked_mean"
    legacy = argparse.Namespace(
        backbone="transformer", num_classes=13, num_frames=8, d_model=16, layers=1, heads=2, dropout=0.0,
    )
    legacy_model = build_model(legacy)  # checkpoints from before the options existed
    assert not legacy_model.hand_features and not legacy_model.use_motion_features and legacy_model.pooling == "mean"
