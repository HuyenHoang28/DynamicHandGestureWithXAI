import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code" / "neural_only_experiments"))

from models import NeuralOnlyClassifier  # noqa: E402


class LSTMBackboneTests(unittest.TestCase):
    def test_forward_shape_and_gradient(self):
        model = NeuralOnlyClassifier(backbone="lstm", num_classes=13, num_frames=64, d_model=32, layers=3,
                                     heads=2, dropout=0.3, pooling="masked_mean")
        logits = model(torch.randn(4, 64, 52, 3))
        self.assertEqual(tuple(logits.shape), (4, 13))
        logits.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.encoder.parameters()))

    def test_state_dict_roundtrip(self):
        kwargs = dict(backbone="lstm", num_classes=13, num_frames=64, d_model=32, layers=2, heads=2,
                      dropout=0.0, pooling="masked_mean")
        a, b = NeuralOnlyClassifier(**kwargs), NeuralOnlyClassifier(**kwargs)
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, 64, 52, 3)
        a.eval(); b.eval()
        self.assertTrue(torch.allclose(a(x), b(x)))


if __name__ == "__main__":
    unittest.main()
