import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
NEURAL_ROOT = ROOT / "final" / "code" / "neural_only_experiments"
CODE_ROOT = ROOT / "final" / "code"
sys.path.insert(0, str(NEURAL_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from models import NeuralOnlyClassifier  # noqa: E402
from src_neurosymbolic.models.cross_attention_v2 import NeuralBaselineAdapter  # noqa: E402


def sample(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    keypoints = torch.randn(1, length, 52, 3)
    keypoints[..., 2] = 1.0
    velocity = torch.zeros(1, length, 52, 2)
    velocity[:, 1:] = keypoints[:, 1:, :, :2] - keypoints[:, :-1, :, :2]
    return keypoints, velocity


class VariableLengthModelTests(unittest.TestCase):
    def model(self, pooling="masked_mean"):
        return NeuralOnlyClassifier(
            backbone="transformer",
            num_classes=3,
            num_frames=8,
            d_model=16,
            layers=1,
            heads=2,
            dropout=0.0,
            use_motion_features=True,
            pooling=pooling,
        ).eval()

    def test_padding_values_do_not_change_transformer_logits(self):
        model = self.model()
        keypoints, velocity = sample(5)
        real_mask = torch.zeros((1, 5), dtype=torch.bool)
        with torch.no_grad():
            expected = model(keypoints, padding_mask=real_mask, velocity=velocity)

        padded_keypoints = torch.zeros((1, 8, 52, 3))
        padded_velocity = torch.zeros((1, 8, 52, 2))
        padded_keypoints[:, :5] = keypoints
        padded_velocity[:, :5] = velocity
        padded_keypoints[:, 5:] = 999.0
        padded_velocity[:, 5:] = 999.0
        padding_mask = torch.tensor([[False] * 5 + [True] * 3])
        with torch.no_grad():
            actual = model(
                padded_keypoints,
                padding_mask=padding_mask,
                velocity=padded_velocity,
            )
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))

    def test_legacy_motion_path_matches_explicit_velocity(self):
        model = self.model()
        keypoints, velocity = sample(5)
        with torch.no_grad():
            legacy = model(keypoints)
            explicit = model(keypoints, velocity=velocity)
        self.assertTrue(torch.equal(legacy, explicit))

    def test_explicit_mask_controls_mean_pooling_and_validates_inputs(self):
        model = self.model(pooling="mean")
        keypoints, velocity = sample(5)
        mask = torch.zeros((1, 5), dtype=torch.bool)
        with torch.no_grad():
            logits = model(keypoints, padding_mask=mask, velocity=velocity)
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaisesRegex(ValueError, "bool"):
            model(keypoints, padding_mask=mask.float(), velocity=velocity)
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(keypoints, padding_mask=torch.ones_like(mask), velocity=velocity)

    def test_positional_capacity_and_non_transformer_mask_fail_clearly(self):
        keypoints, velocity = sample(9)
        mask = torch.zeros((1, 9), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "positional capacity"):
            self.model()(keypoints, padding_mask=mask, velocity=velocity)
        tcn = NeuralOnlyClassifier(
            backbone="tcn", num_classes=3, num_frames=8, d_model=8,
            layers=1, heads=2, dropout=0.0,
        )
        short, short_velocity = sample(5)
        with self.assertRaisesRegex(ValueError, "only by transformer"):
            tcn(short, padding_mask=torch.zeros((1, 5), dtype=torch.bool), velocity=short_velocity)

    def test_neurosymbolic_adapter_propagates_padding_mask(self):
        adapter = NeuralBaselineAdapter(
            num_classes=3,
            num_frames=8,
            d_model=16,
            layers=1,
            heads=2,
            dropout=0.0,
            use_motion_features=True,
            pooling="masked_mean",
        ).eval()
        keypoints, velocity = sample(5)
        mask = torch.zeros((1, 5), dtype=torch.bool)
        with torch.no_grad():
            output = adapter(keypoints, padding_mask=mask, velocity=velocity)
        self.assertTrue(torch.equal(output["padding_mask"], mask))
        self.assertEqual(output["tokens"].shape, (1, 5, 16))
        keypoints[:, 2, :, 2] = 0.0
        with torch.no_grad():
            legacy = adapter(keypoints, velocity=velocity)
        self.assertFalse(legacy["padding_mask"].any())


if __name__ == "__main__":
    unittest.main()
