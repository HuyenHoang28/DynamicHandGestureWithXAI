"""Audit original IPN clip lengths before fixed-length tensor conversion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CLICK_LABELS = ("G01", "G02", "G08", "G09")
BUCKETS = (
    ("lt64", 0, 63),
    ("64_127", 64, 127),
    ("128_255", 128, 255),
    ("ge256", 256, None),
)


def percentile(values: list[int], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bucket_name(frames: int) -> str:
    for name, lower, upper in BUCKETS:
        if frames >= lower and (upper is None or frames <= upper):
            return name
    raise ValueError(f"Invalid frame count: {frames}")


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No manifest rows were provided")
    values = [int(row["frames"]) for row in rows]
    if any(value <= 0 for value in values):
        raise ValueError("Every clip must have a positive frame count")
    buckets = Counter(bucket_name(value) for value in values)
    coverage = {}
    for limit in (64, 128, 256, 384):
        covered = sum(value <= limit for value in values)
        coverage[str(limit)] = {
            "clips": covered,
            "fraction": covered / len(values),
        }
    return {
        "clips": len(values),
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "buckets": {name: buckets[name] for name, _, _ in BUCKETS},
        "coverage_at_max_frames": coverage,
    }


def read_manifests(paths: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    required = {"clip_id", "split", "label", "frames"}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                clip_id = row["clip_id"]
                if clip_id in seen:
                    raise ValueError(f"Duplicate clip_id {clip_id!r} at {path}:{line_number}")
                try:
                    frames = int(row["frames"])
                except ValueError as exc:
                    raise ValueError(f"Invalid frames at {path}:{line_number}") from exc
                if frames <= 0:
                    raise ValueError(f"Non-positive frames at {path}:{line_number}")
                seen.add(clip_id)
                rows.append({**row, "frames": frames})
    return rows


def build_report(rows: list[dict], manifest_paths: list[Path]) -> dict:
    by_split = defaultdict(list)
    by_label = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
        by_label[row["label"]].append(row)
    return {
        "schema_version": 1,
        "source": "manifest frames column (original annotated clip length)",
        "manifests": [str(path.resolve()) for path in manifest_paths],
        "all": summarize(rows),
        "by_split": {name: summarize(group) for name, group in sorted(by_split.items())},
        "click_labels": {
            label: summarize(by_label[label]) for label in CLICK_LABELS if by_label[label]
        },
        "click_position_ground_truth": {
            "available": False,
            "reason": (
                "The dataset labels clip boundaries and gesture class, not per-click occurrence frames. "
                "Detector-derived peaks must not be reported as ground truth."
            ),
        },
    }


def write_outputs(report: dict, rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sequence_length_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "sequence_lengths.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("clip_id", "split", "label", "frames", "length_bucket")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "clip_id": row["clip_id"],
                "split": row["split"],
                "label": row["label"],
                "frames": row["frames"],
                "length_bucket": bucket_name(row["frames"]),
            })


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="+",
        default=[
            root / "external_data/ipn_processed/train_manifest.csv",
            root / "external_data/ipn_processed/val_manifest.csv",
            root / "external_data/ipn_processed/test_manifest.csv",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "external_data/ipn_sequence_length_audit",
    )
    args = parser.parse_args()
    rows = read_manifests(args.manifests)
    report = build_report(rows, args.manifests)
    write_outputs(report, rows, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
