import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.models.keypoint_dataset import KEYPOINT_NAMES, keypoints_to_tensor  # noqa: E402
from src_neurosymbolic.variable_keypoint_cache import (  # noqa: E402
    build_variable_cache,
    cache_matches_source,
    load_variable_cache,
    precompute_manifest,
    save_variable_cache,
    source_fingerprint,
    validate_variable_cache,
)


def normalized(count):
    return {"frames": [
        {"frame_id": i, "keypoints": {
            "nose": {"xy": [float(i), 1.0], "score": 1.0 if i != 2 else 0.0},
            "left_eye": {"xy": [float(i * 2), 3.0], "score": 0.5},
        }} for i in range(count)
    ]}


class VariableKeypointCacheTests(unittest.TestCase):
    def test_lengths_are_preserved_without_windowing(self):
        for count in (5, 64, 300):
            with self.subTest(count=count):
                cache = build_variable_cache(normalized(count))
                self.assertEqual(cache["keypoints"].shape, (count, 52, 3))
                self.assertEqual(cache["velocity"].shape, (count, 52, 2))
                self.assertEqual(cache["length"], count)
                self.assertEqual(cache["source_frames"], count)
                self.assertEqual(cache["exceeds_max_frames"], count > 256)

    def test_velocity_confidence_and_first_frame(self):
        cache = build_variable_cache(normalized(4))
        velocity = cache["velocity"]
        nose = KEYPOINT_NAMES.index("nose")
        eye = KEYPOINT_NAMES.index("left_eye")
        self.assertTrue(torch.equal(velocity[0], torch.zeros_like(velocity[0])))
        self.assertEqual(velocity[1, nose, 0].item(), 1.0)
        self.assertEqual(velocity[2, nose, 0].item(), 0.0)  # current confidence is zero
        self.assertEqual(velocity[3, nose, 0].item(), 0.0)  # previous confidence is zero
        self.assertEqual(velocity[3, eye, 0].item(), 2.0)

    def test_invalid_normalized_frame_zeros_confidence_and_velocity_pairs(self):
        data = normalized(3)
        data["frames"][1]["valid"] = False
        cache = build_variable_cache(data)
        nose = KEYPOINT_NAMES.index("nose")
        self.assertEqual(cache["keypoints"][1, nose, 2].item(), 0.0)
        self.assertEqual(cache["velocity"][1, nose, 0].item(), 0.0)
        self.assertEqual(cache["velocity"][2, nose, 0].item(), 0.0)

    def test_empty_cache_is_explicit_and_round_trips(self):
        cache = build_variable_cache({"frames": []})
        self.assertEqual(cache["keypoints"].shape, (0, 52, 3))
        self.assertEqual(cache["velocity"].shape, (0, 52, 2))
        self.assertEqual((cache["length"], cache["source_frames"]), (0, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.pt"
            save_variable_cache(cache, path)
            self.assertEqual(load_variable_cache(path)["length"], 0)

    def test_source_fingerprint_controls_cache_freshness(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text("first", encoding="utf-8")
            fingerprint = source_fingerprint(source)
            cache = build_variable_cache(
                normalized(1), **fingerprint
            )
            self.assertTrue(cache_matches_source(cache, source))
            source.write_text("other", encoding="utf-8")  # same size; SHA-256 must detect it
            self.assertFalse(cache_matches_source(cache, source))

    def test_precompute_accepts_absolute_keypoint_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = (base / "keypoints").resolve()
            source = root / "train" / "G01" / "subject" / "clip.json"
            source.parent.mkdir(parents=True)
            raw_points = {
                "left_shoulder": {"xy": [0.0, 0.0], "score": 1.0},
                "right_shoulder": {"xy": [2.0, 0.0], "score": 1.0},
                "nose": {"xy": [1.0, 1.0], "score": 1.0},
            }
            raw_frame = {
                "frame_id": 0,
                "instances": [{"keypoints_by_name": raw_points}],
            }
            source.write_text(json.dumps([raw_frame, raw_frame, raw_frame]), encoding="utf-8")
            manifest = base / "train.txt"
            manifest.write_text("train/G01/subject/clip.json ignored 2\n", encoding="utf-8")
            summary = precompute_manifest(manifest, root, base / "cache", "train", max_frames=2)
            self.assertEqual(summary, {"split": "train", "written": 1, "skipped": 0, "failed": 0})
            cached = base / "cache" / "variable_v1" / "train" / "G01" / "subject" / "clip.pt"
            self.assertTrue(load_variable_cache(cached)["exceeds_max_frames"])
            rebuilt = precompute_manifest(manifest, root, base / "cache", "train", max_frames=4)
            self.assertEqual(rebuilt, {"split": "train", "written": 1, "skipped": 0, "failed": 0})
            self.assertFalse(load_variable_cache(cached)["exceeds_max_frames"])

    def test_malformed_cache_rejected(self):
        cache = build_variable_cache(normalized(3))
        cache["length"] = 2
        with self.assertRaises(ValueError):
            validate_variable_cache(cache)
        with self.assertRaisesRegex(ValueError, "positive"):
            build_variable_cache(normalized(3), max_frames=0)
        cache = build_variable_cache(normalized(3))
        cache["velocity"] = torch.zeros((2, 52, 2))
        with self.assertRaises(ValueError):
            validate_variable_cache(cache)

    def test_legacy_tensor_behavior_is_unchanged(self):
        repeated = keypoints_to_tensor(normalized(2), 5)
        self.assertTrue(torch.equal(repeated[1], repeated[4]))
        sampled = keypoints_to_tensor(normalized(6), 3)
        self.assertEqual(sampled.shape, (3, 52, 3))
        self.assertEqual(sampled[0, 0, 0].item(), 0.0)
        self.assertEqual(sampled[-1, 0, 0].item(), 5.0)


if __name__ == "__main__":
    unittest.main()
