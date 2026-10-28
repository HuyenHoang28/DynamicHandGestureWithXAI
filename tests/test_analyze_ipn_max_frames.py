import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_ipn_max_frames", ROOT / "scripts/analyze_ipn_max_frames.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("frames,expected", [
    (1, [(0, 1)]),
    (64, [(0, 64)]),
    (65, [(0, 64), (1, 65)]),
    (96, [(0, 64), (32, 96)]),
    (97, [(0, 64), (32, 96), (33, 97)]),
])
def test_window_spans_cover_exact_tail(frames, expected):
    assert MODULE.window_spans(frames, 64) == expected


def test_candidate_metrics_include_click_groups_windows_and_relative_cost():
    rows = [
        {"clip_id": "a", "split": "train", "label": "G01", "frames": 64},
        {"clip_id": "b", "split": "train", "label": "G02", "frames": 65},
        {"clip_id": "c", "split": "val", "label": "G09", "frames": 32},
    ]
    result = MODULE.build_report(rows, [Path("train.csv")])["candidates"]
    at_64 = result["64"]
    assert at_64["all"]["covered_clips"] == 2
    assert at_64["all"]["coverage_fraction"] == pytest.approx(2 / 3)
    assert at_64["all"]["clips_needing_sliding_windows"] == 1
    assert at_64["all"]["sliding_window_fraction"] == pytest.approx(1 / 3)
    assert at_64["all"]["sliding_window_count"] == 2
    assert at_64["all"]["total_windows"] == 4
    assert at_64["all"]["token_count_proxy"] == 224
    assert at_64["all"]["attention_pair_proxy"] == 13312
    assert at_64["all"]["max_window_tokens"] == 64
    assert at_64["all"]["peak_attention_pair_proxy"] == 4096
    assert at_64["click_labels"]["G02"]["total_windows"] == 2
    assert at_64["relative_attention_proxy_vs_64"] == 1
    assert at_64["relative_peak_attention_proxy_vs_64"] == 1
    assert result["128"]["all"]["coverage_fraction"] == 1
    assert result["128"]["relative_attention_proxy_vs_64"] < 1
    assert result["128"]["all"]["max_window_tokens"] == 65
    assert result["128"]["relative_peak_attention_proxy_vs_64"] == pytest.approx(65 ** 2 / 64 ** 2)


def test_rejects_nonpositive_frames_and_duplicate_ids():
    with pytest.raises(ValueError, match="positive"):
        MODULE.analyze_candidate([{"clip_id": "a", "label": "G01", "frames": 0}], 64)
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.analyze_candidate([
            {"clip_id": "a", "label": "G01", "frames": 10},
            {"clip_id": "a", "label": "G02", "frames": 20},
        ], 64)


def test_rejects_limit_one_and_test_split_for_architecture_selection():
    with pytest.raises(ValueError, match="at least 2"):
        MODULE.window_spans(2, 1)
    with pytest.raises(ValueError, match="train/val"):
        MODULE.build_report(
            [{"clip_id": "a", "split": "test", "label": "G01", "frames": 32}],
            [Path("test.csv")],
        )
