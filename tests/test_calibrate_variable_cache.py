import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))
spec = importlib.util.spec_from_file_location("calibrator", ROOT / "scripts" / "calibrate_ipn_temporal_factors.py")
calibrator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrator)

from neural_only_experiments.variable_dataset import variable_cache_path  # noqa: E402
from src_neurosymbolic.variable_keypoint_cache import save_variable_cache  # noqa: E402
from src_neurosymbolic.variable_keypoint_cache import build_variable_cache  # noqa: E402


class VariableCacheCalibrationTests(unittest.TestCase):
    def test_full_rate_grid_has_consistent_keys(self):
        grid = calibrator.fullrate_grid()
        self.assertEqual(len(grid), 12)
        self.assertTrue(all(set(g) == set(calibrator.DEFAULT_GRID[0]) for g in grid))

    def test_run_split_reads_full_length_variable_cache_without_raw_keypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "keypoints/train/G01/1CM1_4/clip.json"
            frames = [{"keypoints": {"nose": {"xy": [t, t + 1], "score": 1.0}}} for t in range(300)]
            save_variable_cache(build_variable_cache({"frames": frames}), variable_cache_path(source, root))
            split = root / "train.txt"
            split.write_text(f"{source} 300 2\n", encoding="utf-8")
            records, skipped, inputs = calibrator.run_split(
                "train", split, root / "missing_manifest.csv", root / "no_raw", None,
                calibrator.fullrate_grid()[:2], None, variable_cache_root=root)
            self.assertEqual(len(records), 2)
            self.assertEqual(skipped, {})
            self.assertIn("variable_v1", inputs[0]["resolved_path"])
            self.assertEqual(records[0]["true_label"], "G01")


if __name__ == "__main__":
    unittest.main()
