import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code" / "neural_only_experiments"))

import train_neural_only as trainer  # noqa: E402
from models import NeuralOnlyClassifier  # noqa: E402


def build(aux):
    return NeuralOnlyClassifier(backbone="transformer", num_classes=13, num_frames=8, d_model=16, layers=1,
                                heads=2, dropout=0.0, pooling="masked_mean", aux_heads=aux)


class AuxHeadTests(unittest.TestCase):
    def test_targets_follow_finger_x_count_factorisation(self):
        labels = torch.tensor([0, 1, 2, 3, 4, 9, 10, 11, 12])
        count, fingers = trainer.auxiliary_targets(labels)
        ignore = trainer.IGNORE_AUX
        self.assertEqual(count.tolist(), [ignore, ignore, 0, 0, ignore, 1, 1, ignore, ignore])
        self.assertEqual(fingers.tolist(), [ignore, ignore, 0, 1, ignore, 0, 1, ignore, ignore])

    def test_loss_is_zero_without_click_clips_and_positive_with_them(self):
        aux = {"count": torch.randn(3, 2, requires_grad=True), "fingers": torch.randn(3, 2, requires_grad=True)}
        self.assertEqual(float(trainer.auxiliary_loss(aux, torch.tensor([0, 1, 4]))), 0.0)
        loss = trainer.auxiliary_loss(aux, torch.tensor([2, 9, 4]))
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(aux["count"].grad).all())

    def test_default_model_is_unchanged_and_aux_returns_two_heads(self):
        x = torch.randn(2, 8, 52, 3)
        plain = build(False)
        self.assertFalse(any(k.startswith(("count_head", "finger_head")) for k in plain.state_dict()))
        self.assertEqual(tuple(plain(x).shape), (2, 13))
        with self.assertRaises(ValueError):
            plain(x, return_aux=True)
        aux_model = build(True)
        logits, aux = aux_model(x, return_aux=True)
        self.assertEqual(tuple(logits.shape), (2, 13))
        self.assertEqual(tuple(aux["count"].shape), (2, 2))
        self.assertEqual(tuple(aux["fingers"].shape), (2, 2))

    def test_filtered_state_dict_loads_into_plain_model(self):
        trained = build(True)
        state = {k: v for k, v in trained.state_dict().items() if not k.startswith(("count_head.", "finger_head."))}
        build(False).load_state_dict(state)

    def test_cli_flags(self):
        old = sys.argv
        sys.argv = ["x", "--cache-root", "c", "--output-dir", "o", "--aux-heads", "--aux-weight", "0.5"]
        try:
            args = trainer.parse_args()
        finally:
            sys.argv = old
        self.assertTrue(args.aux_heads)
        self.assertEqual(args.aux_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
