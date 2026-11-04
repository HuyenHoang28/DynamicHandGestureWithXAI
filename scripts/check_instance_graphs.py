"""Fail when a split list does not resolve to non-empty instance graphs.

``KeypointSequenceDataset`` silently returns an empty graph when a graph file is not found, which makes V4 train
without any symbolic evidence.  Run this before training or evaluating V4; the exit code is 1 if any sampled clip
resolves to an empty graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset  # noqa: E402


def non_empty_ratio(dataset: KeypointSequenceDataset, samples: int) -> tuple[int, int]:
    step = max(1, len(dataset) // max(1, samples))
    indices = list(range(0, len(dataset), step))
    found = sum(1 for index in indices if json.loads(dataset[index]["instance_graph_json"]).get("nodes"))
    return found, len(indices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--instance-graph-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--samples", type=int, default=40)
    args = parser.parse_args(argv)
    dataset = KeypointSequenceDataset(
        args.list_file, args.keypoint_root, args.split, args.num_frames, None, args.cache_root, args.instance_graph_root
    )
    found, total = non_empty_ratio(dataset, args.samples)
    print(f"non-empty instance graphs: {found}/{total} ({args.split})")
    return 0 if found == total and total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
