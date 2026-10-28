import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_ipn_sequence_lengths", ROOT / "scripts/audit_ipn_sequence_lengths.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summarize_reports_boundaries_and_percentiles():
    rows = [{"frames": value} for value in (20, 63, 64, 127, 128, 255, 256, 400)]
    result = MODULE.summarize(rows)
    assert result["clips"] == 8
    assert result["min"] == 20
    assert result["median"] == pytest.approx(127.5)
    assert result["p90"] == pytest.approx(299.2)
    assert result["p95"] == pytest.approx(349.6)
    assert result["p99"] == pytest.approx(389.92)
    assert result["max"] == 400
    assert result["buckets"] == {"lt64": 2, "64_127": 2, "128_255": 2, "ge256": 2}
    assert result["coverage_at_max_frames"]["64"] == {"clips": 3, "fraction": 3 / 8}
    assert result["coverage_at_max_frames"]["128"] == {"clips": 5, "fraction": 5 / 8}
    assert result["coverage_at_max_frames"]["256"] == {"clips": 7, "fraction": 7 / 8}
    assert result["coverage_at_max_frames"]["384"] == {"clips": 7, "fraction": 7 / 8}


def test_read_manifests_rejects_duplicate_clip_ids(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"manifest_{index}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("clip_id", "split", "label", "frames"))
            writer.writeheader()
            writer.writerow({"clip_id": "same", "split": "train", "label": "G01", "frames": 50})
        paths.append(path)
    with pytest.raises(ValueError, match="Duplicate clip_id"):
        MODULE.read_manifests(paths)


def test_build_report_does_not_claim_click_positions_are_ground_truth(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"clip_id": "a", "split": "train", "label": "G01", "frames": 50},
        {"clip_id": "b", "split": "val", "label": "G08", "frames": 90},
    ]
    report = MODULE.build_report(rows, [manifest])
    assert report["click_labels"]["G01"]["median"] == 50
    assert report["click_labels"]["G08"]["median"] == 90
    assert report["click_position_ground_truth"]["available"] is False
