"""Unit tests for the standard-library symbolic input validator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_symbolic_inputs.py"
EVALUATOR_SCRIPT = ROOT / "scripts" / "eval_ipn_template_matching.py"
sys.path.insert(0, str(ROOT / "scripts"))

from validate_symbolic_inputs import LABELS, validate_symbolic_inputs  # noqa: E402


class ValidateSymbolicInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "ipn_processed"
        self.keypoints = self.root / "keypoints"
        self.graphs = self.root / "instance_graphs"
        self.templates = self.root / "template_graphs"
        self.keypoints.mkdir(parents=True)
        self.graphs.mkdir(parents=True)
        self.templates.mkdir(parents=True)
        self._write_fixture()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_fixture(self) -> None:
        # Every ID occurs once across the three splits, while each split has a
        # distinct subject.  This keeps the fixture small but exercises all
        # label and disjointness checks.
        assignments = {
            "train": list(range(5)),
            "val": list(range(5, 9)),
            "test": list(range(9, 13)),
        }
        split_lines: dict[str, list[str]] = {}
        predicate_config = {
            "version": "ipn_predicate_v2",
            "thresholds": {"hand_motion_threshold": 0.08},
        }
        build_config = {
            "schema_version": 2,
            "predicate_config": predicate_config,
            "event_config": {"builder": "phase_v2"},
        }
        for split, ids in assignments.items():
            subject = {"train": "SubjectTrain", "val": "SubjectVal", "test": "SubjectTest"}[split]
            lines: list[str] = []
            for index, label_id in enumerate(ids):
                label = LABELS[label_id]
                relative = Path(split) / label / subject / f"clip_{label_id}.json"
                source = (self.keypoints / relative).as_posix()
                lines.append(f"{source} 0 {label_id}")
                graph_path = self.graphs / relative
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                graph_path.write_text(
                    json.dumps(
                        {
                            "graph_type": "instance_graph",
                            "schema_version": 2,
                            "video_id": f"clip_{label_id}",
                            "predicate_config": predicate_config,
                            "provenance": {
                                "predicate_config": predicate_config,
                                "build_config": build_config,
                                # This is intentionally different per clip;
                                # the validator must not mistake it for a
                                # mixed build recipe.
                                "source_fingerprint": {"clip": label_id},
                            },
                            "nodes": [{"id": f"video:clip_{label_id}"}],
                            "edges": [],
                        }
                    ),
                    encoding="utf-8",
                )
            split_path = self.root / f"{split}.txt"
            split_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            split_lines[split] = lines

        # Template graphs and their index use the production schema's ID
        # fields.  The validator only relies on those fields, so no model
        # dependencies are needed in this fixture.
        template_index = []
        for label_id, label in enumerate(LABELS):
            graph = {
                "graph_type": "template_graph",
                "label_id": label_id,
                "label_name": label,
                "nodes": [{"id": f"gesture:{label}"}],
                "edges": [],
            }
            (self.templates / f"{label}.json").write_text(json.dumps(graph), encoding="utf-8")
            template_index.append({"label_id": label_id, "label_name": label})
        (self.templates / "index.json").write_text(
            json.dumps({"graph_type": "template_graph_index", "template_count": 13, "templates": template_index}),
            encoding="utf-8",
        )

        hashes = {
            f"{split}.txt": hashlib.sha256((self.root / f"{split}.txt").read_bytes()).hexdigest()
            for split in assignments
        }
        (self.root / "split_subjects.json").write_text(
            json.dumps(
                {
                    "clip_counts": {split: len(ids) for split, ids in assignments.items()},
                    "hashes": hashes,
                    "excluded_subjects": ["1CM42_3"],
                }
            ),
            encoding="utf-8",
        )

    def _validate(self, **kwargs):
        return validate_symbolic_inputs(
            self.root,
            keypoint_root=self.keypoints,
            template_dir=self.templates,
            instance_graph_root=self.graphs,
            **kwargs,
        )

    def test_valid_fixture_checks_all_inputs(self) -> None:
        report = self._validate()
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.stats["split_counts"], {"train": 5, "val": 4, "test": 4})
        self.assertEqual(report.stats["template_ids"], list(range(13)))
        self.assertEqual(report.stats["instance_graph_found"], 13)
        self.assertEqual(report.stats["instance_graph_coverage"], 1.0)

    def test_hash_mismatch_fails(self) -> None:
        metadata_path = self.root / "split_subjects.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["hashes"]["val.txt"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        report = self._validate()
        self.assertFalse(report.ok)
        self.assertTrue(any("SHA-256 mismatch" in error for error in report.errors))

    def test_flat_input_hashes_layout_is_supported(self) -> None:
        metadata_path = self.root / "split_subjects.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_path.write_text(json.dumps(metadata["hashes"]), encoding="utf-8")
        report = self._validate(
            require_hashes=True,
            expected_split_counts={"train": 5, "val": 4, "test": 4},
        )
        self.assertTrue(report.ok, report.errors)

    def test_invalid_label_id_fails(self) -> None:
        path = self.root / "test.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[-1] = lines[-1].rsplit(" ", 1)[0] + " 13"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = self._validate()
        self.assertFalse(report.ok)
        self.assertTrue(any("outside 0..12" in error for error in report.errors))

    def test_subject_overlap_fails(self) -> None:
        path = self.root / "val.txt"
        line = path.read_text(encoding="utf-8").splitlines()[0]
        line = line.replace("SubjectVal", "SubjectTrain")
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = line
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = self._validate()
        self.assertFalse(report.ok)
        self.assertTrue(any("Subject overlap" in error for error in report.errors))

    def test_missing_or_malformed_instance_graph_fails_coverage(self) -> None:
        graph_files = sorted(self.graphs.rglob("*.json"))
        graph_files[0].unlink()
        malformed = graph_files[1]
        malformed.write_text("{not-json", encoding="utf-8")
        report = self._validate()
        self.assertFalse(report.ok)
        self.assertTrue(any("coverage is not 100%" in error for error in report.errors))
        self.assertTrue(any("Invalid/nonempty instance graph JSON" in error for error in report.errors))

    def test_wrong_instance_graph_schema_fails(self) -> None:
        graph_path = sorted(self.graphs.rglob("*.json"))[0]
        graph_path.write_text(json.dumps({"label_id": 0}), encoding="utf-8")
        report = self._validate()
        self.assertFalse(report.ok)
        self.assertTrue(any("graph_type=instance_graph" in error for error in report.errors))

    def test_template_label_name_must_match_canonical_label_id(self) -> None:
        path = self.templates / "B0A.json"
        graph = json.loads(path.read_text(encoding="utf-8"))
        graph["label_name"] = "B0B"
        path.write_text(json.dumps(graph), encoding="utf-8")
        report = self._validate()
        self.assertFalse(report.ok)
        self.assertTrue(any("non-canonical label_name" in error for error in report.errors))

    def test_legacy_graphs_require_explicit_opt_in(self) -> None:
        path = sorted(self.graphs.rglob("*.json"))[0]
        graph = json.loads(path.read_text(encoding="utf-8"))
        graph.pop("predicate_config", None)
        graph.pop("provenance", None)
        path.write_text(json.dumps(graph), encoding="utf-8")

        strict_report = self._validate()
        self.assertFalse(strict_report.ok)
        self.assertTrue(any("predicate_config" in error for error in strict_report.errors))

        legacy_report = self._validate(allow_legacy_graphs=True)
        self.assertTrue(legacy_report.ok, legacy_report.errors)

    def test_mixed_v2_predicate_config_fails(self) -> None:
        path = sorted(self.graphs.rglob("*.json"))[0]
        graph = json.loads(path.read_text(encoding="utf-8"))
        graph["predicate_config"]["thresholds"]["hand_motion_threshold"] = 0.12
        graph["provenance"]["predicate_config"] = graph["predicate_config"]
        graph["provenance"]["build_config"]["predicate_config"] = graph["predicate_config"]
        path.write_text(json.dumps(graph), encoding="utf-8")
        report = self._validate()
        self.assertFalse(report.ok)
        self.assertTrue(any("Mixed predicate_config" in error for error in report.errors))

    def test_evaluator_requires_full_support_and_records_build_config(self) -> None:
        combined = self.root / "all.txt"
        combined.write_text(
            "\n".join(
                (self.root / f"{split}.txt").read_text(encoding="utf-8").strip()
                for split in ("train", "val", "test")
            )
            + "\n",
            encoding="utf-8",
        )
        output = self.root / "evaluation.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR_SCRIPT),
                "--data-root",
                str(self.root),
                "--split",
                "val",
                "--split-file",
                str(combined),
                "--keypoint-root",
                str(self.keypoints),
                "--template-dir",
                str(self.templates),
                "--instance-graph-root",
                str(self.graphs),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["clips"], 13)
        self.assertEqual(payload["build_config"], {
            "schema_version": 2,
            "predicate_config": {
                "version": "ipn_predicate_v2",
                "thresholds": {"hand_motion_threshold": 0.08},
            },
            "event_config": {"builder": "phase_v2"},
        })
        self.assertEqual(len(payload["build_config_sha256"]), 64)

    def test_evaluator_partial_support_requires_explicit_flag(self) -> None:
        strict = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR_SCRIPT),
                "--data-root",
                str(self.root),
                "--split",
                "val",
                "--keypoint-root",
                str(self.keypoints),
                "--template-dir",
                str(self.templates),
                "--instance-graph-root",
                str(self.graphs),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(strict.returncode, 0)
        self.assertIn("allow-partial-labels", strict.stderr)

        allowed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR_SCRIPT),
                "--data-root",
                str(self.root),
                "--split",
                "val",
                "--keypoint-root",
                str(self.keypoints),
                "--template-dir",
                str(self.templates),
                "--instance-graph-root",
                str(self.graphs),
                "--allow-partial-labels",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_cli_returns_nonzero_on_mismatch(self) -> None:
        path = self.root / "train.txt"
        path.write_text(path.read_text(encoding="utf-8") + "bad source 0 13\n", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--data-root",
                str(self.root),
                "--keypoint-root",
                str(self.keypoints),
                "--template-dir",
                str(self.templates),
                "--instance-graph-root",
                str(self.graphs),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Symbolic input validation: FAIL", completed.stderr)


if __name__ == "__main__":
    unittest.main()
