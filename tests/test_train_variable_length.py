import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))
sys.path.insert(0, str(ROOT / "final" / "code" / "neural_only_experiments"))

import train_variable_length as tvl  # noqa: E402
from neural_only_experiments.variable_dataset import variable_cache_path  # noqa: E402
from src_neurosymbolic.variable_keypoint_cache import build_variable_cache, save_variable_cache  # noqa: E402


class WeightedLossTests(unittest.TestCase):
    def test_unit_weights_match_class_weighted_cross_entropy(self):
        logits = torch.randn(6, 3)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        weight = torch.tensor([0.5, 1.0, 2.0])
        reference = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)(logits, labels)
        per = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05, reduction="none")(logits, labels)
        got = tvl.weighted_window_loss(per, torch.ones(6), weight, labels)
        self.assertAlmostEqual(float(got), float(reference), places=5)

    def test_split_clip_counts_once(self):
        logits = torch.randn(3, 2)
        labels = torch.tensor([0, 1, 1])
        per = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)
        # clip B (label 1) has two windows of weight 0.5 each
        got = tvl.weighted_window_loss(per, torch.tensor([1.0, 0.5, 0.5]), None, labels)
        expected = (per[0] + 0.5 * per[1] + 0.5 * per[2]) / 2.0
        self.assertAlmostEqual(float(got), float(expected), places=6)

    def test_length_groups(self):
        groups = tvl.length_group_metrics([0, 1, 1], [0, 1, 0], [10, 100, 300])
        self.assertEqual(groups["lt64"]["accuracy"], 1.0)
        self.assertEqual(groups["ge256"]["accuracy"], 0.0)
        self.assertEqual(groups["128_255"]["clips"], 0)


class EndToEndTests(unittest.TestCase):
    def test_train_one_epoch_then_evaluate_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i, (n, label) in enumerate([(10, 0), (300, 1), (40, 0), (270, 1)]):
                source = f"train/C{label}/clip{i}.avi"
                frames = [{"keypoints": {"nose": {"xy": [t, t + 1], "score": 1.0}}} for t in range(n)]
                save_variable_cache(build_variable_cache({"frames": frames}), variable_cache_path(source, root))
                rows.append(f"{source} {n} {label}")
            manifest = root / "train.txt"
            manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
            argv = ["x", "--cache-root", str(root), "--train-file", str(manifest), "--val-file", str(manifest),
                    "--output-dir", str(root / "out"), "--num-classes", "2", "--epochs", "1", "--d-model", "16",
                    "--layers", "1", "--heads", "2", "--batch-size", "4", "--device", "cpu", "--log-every", "0"]
            old = sys.argv
            sys.argv = argv
            try:
                args = tvl.parse_args()
            finally:
                sys.argv = old
            # validation overlaps train on purpose here, so skip the overlap guard
            original = tvl.validate_no_split_overlap
            tvl.validate_no_split_overlap = lambda *a: None
            try:
                tvl.train(args)
            finally:
                tvl.validate_no_split_overlap = original
            self.assertTrue((root / "out" / "best.pt").is_file())
            evaluate_args = tvl.argparse.Namespace(
                checkpoint=root / "out" / "best.pt", eval_file=manifest, split="train", output=root / "ev" / "m.json",
                device="cpu", eval_batch_size=4, cache_root=root, num_workers=0)
            tvl.evaluate_checkpoint(evaluate_args)
            import json
            metrics = json.loads((root / "ev" / "m.json").read_text())
            self.assertEqual(metrics["total"], 4)
            self.assertEqual(metrics["windows"], 6)
            self.assertEqual(metrics["length_groups"]["ge256"]["clips"], 2)
            saved = torch.load(root / "ev" / "clip_logits.pt")
            self.assertEqual(tuple(saved["logits"].shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
