import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from neural_only_experiments.dataset import NeuralOnlyKeypointDataset, collate_batch  # noqa: E402
from neural_only_experiments.variable_dataset import (  # noqa: E402
    VariableKeypointDataset,
    collate_variable_batch,
    variable_cache_path,
)
from src_neurosymbolic.variable_keypoint_cache import build_variable_cache, save_variable_cache  # noqa: E402


class VariableLengthDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "train.txt"

    def tearDown(self):
        self.temp.cleanup()

    def write_cache(self, source, n):
        frames = [{"keypoints": {"nose": {"xy": [i, i + 1], "score": 1.0}}} for i in range(n)]
        target = variable_cache_path(source, self.root)
        save_variable_cache(build_variable_cache({"frames": frames}), target)
        return target

    def dataset(self, rows, max_frames=256):
        self.manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return VariableKeypointDataset(self.manifest, self.root, "train", max_frames=max_frames)

    def test_mixed_lengths_pad_with_zero_and_mask_only_padding(self):
        a, b = "train/wave/a.json", "train/wave/b.json"
        self.write_cache(a, 5)
        self.write_cache(b, 8)
        batch = collate_variable_batch([self.dataset([f"{a} 0 2", f"{b} 0 3"])[0],
                                        self.dataset([f"{a} 0 2", f"{b} 0 3"])[1]])
        self.assertEqual(batch["keypoints"].shape, (2, 8, 52, 3))
        self.assertEqual(batch["velocity"].shape, (2, 8, 52, 2))
        self.assertEqual(batch["padding_mask"].tolist(), [[False] * 5 + [True] * 3, [False] * 8])
        self.assertTrue(torch.equal(batch["keypoints"][0, :5], self.dataset([f"{a} 0 2"])[0]["keypoints"]))
        self.assertTrue(torch.equal(batch["velocity"][0, :5], self.dataset([f"{a} 0 2"])[0]["velocity"]))
        self.assertEqual(torch.count_nonzero(batch["keypoints"][0, 5:]).item(), 0)
        self.assertEqual(torch.count_nonzero(batch["velocity"][0, 5:]).item(), 0)
        self.assertEqual(batch["label"].tolist(), [2, 3])

    def test_same_length_and_no_repeat_last(self):
        source = "train/wave/a.json"
        self.write_cache(source, 5)
        item = self.dataset([f"{source} 0 1"])[0]
        batch = collate_variable_batch([item, item])
        self.assertEqual(batch["keypoints"].shape, (2, 5, 52, 3))
        self.assertFalse(batch["padding_mask"].any())
        self.assertTrue(torch.equal(batch["keypoints"][0], batch["keypoints"][1]))

    def test_long_and_empty_clips_rejected(self):
        long_source, empty_source = "train/wave/long.json", "train/wave/empty.json"
        self.write_cache(long_source, 257)
        self.write_cache(empty_source, 0)
        with self.assertRaisesRegex(ValueError, "Step 5 windowing"):
            self.dataset([f"{long_source} 0 1"])[0]
        with self.assertRaisesRegex(ValueError, "Empty cache"):
            self.dataset([f"{empty_source} 0 1"])[0]

    def test_absolute_drive_and_relative_cache_resolution(self):
        self.assertEqual(variable_cache_path(r"D:\\dataset\\train\\wave\\clip.json", self.root),
                         self.root / "variable_v1" / "train" / "wave" / "clip.pt")
        self.assertEqual(variable_cache_path("val/wave/clip.json", self.root),
                         self.root / "variable_v1" / "val" / "wave" / "clip.pt")

    def test_empty_batch_rejected_and_legacy_api_stays_separate(self):
        with self.assertRaisesRegex(ValueError, "empty batch"):
            collate_variable_batch([])
        self.assertTrue(callable(collate_batch))
        self.assertTrue(NeuralOnlyKeypointDataset)


if __name__ == "__main__":
    unittest.main()
