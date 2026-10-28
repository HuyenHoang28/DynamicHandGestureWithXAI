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
    WindowedVariableKeypointDataset,
    collate_variable_batch,
    aggregate_window_logits,
    variable_cache_path,
    window_spans,
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

    def test_window_spans_cover_edges_and_final_window(self):
        self.assertEqual(window_spans(256), [(0, 256)])
        self.assertEqual(window_spans(257), [(0, 256), (1, 257)])
        self.assertEqual(window_spans(525), [(0, 256), (128, 384), (256, 512), (269, 525)])
        for length in (256, 257, 525):
            covered = set(i for start, end in window_spans(length) for i in range(start, end))
            self.assertEqual(covered, set(range(length)))

    def test_windowed_cache_slices_boundary_velocity_and_weights(self):
        source = "train/wave/long.json"
        self.write_cache(source, 525)
        original = self.dataset([f"{source} 0 4"], max_frames=525)[0]
        ds = WindowedVariableKeypointDataset(self.manifest, self.root, "train")
        # Manifest is populated by dataset helper above.
        self.assertEqual(len(ds), 4)
        weights = 0.0
        for i, (start, end) in enumerate(window_spans(525)):
            item = ds[i]
            self.assertTrue(torch.equal(item["keypoints"], original["keypoints"][start:end]))
            expected_velocity = original["velocity"][start:end].clone()
            expected_velocity[0].zero_()
            self.assertTrue(torch.equal(item["velocity"], expected_velocity))
            self.assertEqual((item["start"], item["end"], item["window_index"], item["window_count"]),
                             (start, end, i, 4))
            weights += item["sample_weight"]
        self.assertAlmostEqual(weights, 1.0)
        batch = collate_variable_batch([ds[0], ds[3]])
        self.assertEqual(batch["keypoints"].shape, (2, 256, 52, 3))
        self.assertEqual(batch["sample_weight"].tolist(), [0.25, 0.25])
        self.assertEqual(batch["window_index"], [0, 3])

    def test_aggregate_shuffled_windows_across_batches_and_reject_bad_groups(self):
        metadata = {"label": [1, 1, 2], "source": ["a", "a", "b"],
                    "split": ["train"] * 3, "clip_index": [0, 0, 1],
                    "window_index": [1, 0, 0], "window_count": [2, 2, 1]}
        logits = torch.tensor([[3., 5.], [1., 3.], [9., 7.]])
        result = aggregate_window_logits(logits, metadata)
        self.assertEqual(result["paths"], ["a", "b"])
        self.assertEqual(result["labels"].tolist(), [1, 2])
        self.assertTrue(torch.equal(result["logits"], torch.tensor([[2., 4.], [9., 7.]])))
        partial = {k: [v[0]] for k, v in metadata.items()}
        with self.assertRaisesRegex(ValueError, "Missing windows"):
            aggregate_window_logits(logits[:1], partial)
        duplicate = dict(metadata, window_index=[0, 0, 0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_window_logits(logits, duplicate)
        inconsistent = dict(metadata, label=[1, 3, 2])
        with self.assertRaisesRegex(ValueError, "Inconsistent labels"):
            aggregate_window_logits(logits, inconsistent)


if __name__ == "__main__":
    unittest.main()
