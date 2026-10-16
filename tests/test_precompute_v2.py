import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from src_neurosymbolic.precompute_instance_graphs_phase_v2 import (  # noqa: E402
    output_path_for_source,
    precompute_split,
)
from eval_ipn_template_matching import instance_graph_path  # noqa: E402


class PrecomputePathSafetyTests(unittest.TestCase):
    def test_colab_absolute_source_maps_below_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "graphs"
            result = output_path_for_source(
                "/content/drive/data/keypoints/val/G01/subject/clip.json",
                root / "keypoints",
                output,
                "val",
            )
            result.relative_to(output.resolve())
            self.assertEqual(result.relative_to(output.resolve()).as_posix(), "val/G01/subject/clip.json")

    def test_foreign_absolute_source_without_keypoint_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            foreign = Path("C:/foreign/clip.json") if sys.platform == "win32" else Path("/foreign/clip.json")
            with self.assertRaises(ValueError):
                output_path_for_source(str(foreign), root / "keypoints", root / "graphs", "val")

    def test_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                output_path_for_source("../clip.json", root / "keypoints", root / "graphs", "val")

    def test_parent_traversal_before_keypoints_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "../foreign/keypoints/val/G01/clip.json"
            with self.assertRaises(ValueError):
                output_path_for_source(source, root / "keypoints", root / "graphs", "val")
            with self.assertRaises(ValueError):
                instance_graph_path(source, root / "keypoints", root / "graphs")

    def test_subject_style_output_matches_evaluator_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keypoint_root = root / "keypoints"
            output_root = root / "graphs"
            source = keypoint_root / "val" / "G01" / "G01_Subject11_HoaiLinh_clip_01.json"

            produced = output_path_for_source(str(source), keypoint_root, output_root, "val")
            looked_up = instance_graph_path(str(source), keypoint_root, output_root)

            self.assertEqual(produced, looked_up)
            self.assertEqual(
                produced.relative_to(output_root),
                Path("G01") / "Subject11_HoaiLinh" / "clip_01.json",
            )

    def test_stale_graph_fails_immediately_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keypoint_root = root / "keypoints"
            source = keypoint_root / "val" / "G01" / "subject" / "clip.json"
            source.parent.mkdir(parents=True)
            source.write_text("{}", encoding="utf-8")
            split_file = root / "val.txt"
            split_file.write_text(f"{source} 0 2\n", encoding="utf-8")
            output_root = root / "graphs"
            output = output_path_for_source(str(source), keypoint_root, output_root, "val")
            output.parent.mkdir(parents=True)
            output.write_text('{"graph_type":"instance_graph","provenance":{}}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale instance graph"):
                precompute_split(
                    split_file,
                    keypoint_root,
                    output_root,
                    "val",
                    None,
                    64,
                    None,
                    False,
                    16,
                    4,
                    0.4,
                    2,
                    True,
                )


if __name__ == "__main__":
    unittest.main()
