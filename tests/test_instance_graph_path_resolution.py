import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset  # noqa: E402


def make_dataset(tmp_path: Path, split: str = "val") -> KeypointSequenceDataset:
    dataset = object.__new__(KeypointSequenceDataset)
    dataset.keypoint_root = tmp_path / "data" / "keypoints"
    dataset.instance_graph_root = tmp_path / "data" / "graphs"
    dataset.split = split
    return dataset


def test_repo_relative_list_path_finds_graph_under_absolute_roots(tmp_path):
    """Split lists hold repo-relative paths while --keypoint-root is absolute (regression: graphs were silently empty)."""

    graph = tmp_path / "data" / "graphs" / "val" / "B0A" / "subj1" / "clip_a.json"
    graph.parent.mkdir(parents=True)
    graph.write_text('{"nodes": [{"id": "x"}], "edges": []}', encoding="utf-8")
    source = "external_data/run/keypoints/val/B0A/subj1/clip_a.json"
    assert make_dataset(tmp_path).instance_graph_path_for_source(source) == graph


def test_root_relative_path_still_wins_when_it_exists(tmp_path):
    graph = tmp_path / "data" / "graphs" / "val" / "G01" / "subj2" / "clip_b.json"
    graph.parent.mkdir(parents=True)
    graph.write_text("{}", encoding="utf-8")
    source = str(tmp_path / "data" / "keypoints" / "val" / "G01" / "subj2" / "clip_b.json")
    assert make_dataset(tmp_path).instance_graph_path_for_source(source) == graph


def test_missing_graph_still_returns_a_nonexistent_candidate(tmp_path):
    path = make_dataset(tmp_path).instance_graph_path_for_source("a/val/B0A/s/missing.json")
    assert path is not None and not path.exists()
