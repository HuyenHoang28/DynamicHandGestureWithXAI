"""Compare fixed frame caps and sliding-window costs for IPN manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from audit_ipn_sequence_lengths import CLICK_LABELS, read_manifests
except ImportError:  # import when loaded as a package/module in tests
    from scripts.audit_ipn_sequence_lengths import CLICK_LABELS, read_manifests


CANDIDATES = (64, 128, 256, 384)


def window_spans(frames: int, limit: int) -> list[tuple[int, int]]:
    """Return half-open spans; the final full window ends exactly at frames."""
    if frames <= 0 or limit < 2:
        raise ValueError("frames must be positive and limit must be at least 2")
    if frames <= limit:
        return [(0, frames)]
    stride = limit // 2
    starts = list(range(0, frames - limit + 1, stride))
    last = frames - limit
    if starts[-1] != last:
        starts.append(last)
    return [(start, start + limit) for start in starts]


def analyze_candidate(rows: list[dict], limit: int) -> dict:
    if limit < 2:
        raise ValueError("limit must be at least 2")
    if not rows:
        raise ValueError("No manifest rows were provided")
    frames = [int(row["frames"]) for row in rows]
    if any(value <= 0 for value in frames):
        raise ValueError("Every clip must have a positive frame count")
    ids = [row["clip_id"] for row in rows if "clip_id" in row]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip_id values")

    def metrics(group: list[dict]) -> dict:
        count = len(group)
        covered = sum(int(row["frames"]) <= limit for row in group)
        windows = [window_spans(int(row["frames"]), limit) for row in group]
        window_lengths = [end - start for spans in windows for start, end in spans]
        return {
            "clips": count,
            "covered_clips": covered,
            "coverage_fraction": covered / count if count else 0.0,
            "clips_needing_sliding_windows": count - covered,
            "sliding_window_fraction": (count - covered) / count if count else 0.0,
            "sliding_window_count": sum(
                len(spans) for row, spans in zip(group, windows) if int(row["frames"]) > limit
            ),
            "total_windows": sum(len(spans) for spans in windows),
            "token_count_proxy": sum(window_lengths),
            "attention_pair_proxy": sum(length ** 2 for length in window_lengths),
            "max_window_tokens": max(window_lengths, default=0),
            "peak_attention_pair_proxy": max((length ** 2 for length in window_lengths), default=0),
        }

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)
    return {
        "limit": limit,
        "all": metrics(rows),
        "click_labels": {label: metrics(grouped[label]) for label in CLICK_LABELS if grouped[label]},
    }


def build_report(rows: list[dict], manifest_paths: list[Path]) -> dict:
    if not rows:
        raise ValueError("No manifest rows were provided")
    invalid_splits = sorted({str(row.get("split", "")) for row in rows} - {"train", "val"})
    if invalid_splits:
        raise ValueError(f"Step-2 architecture selection only accepts train/val splits: {invalid_splits}")
    candidates = {str(limit): analyze_candidate(rows, limit) for limit in CANDIDATES}
    base = candidates["64"]["all"]["attention_pair_proxy"]
    peak_base = candidates["64"]["all"]["peak_attention_pair_proxy"]
    for result in candidates.values():
        result["relative_attention_proxy_vs_64"] = (
            result["all"]["attention_pair_proxy"] / base if base else 1.0
        )
        result["relative_peak_attention_proxy_vs_64"] = (
            result["all"]["peak_attention_pair_proxy"] / peak_base if peak_base else 1.0
        )
    return {
        "schema_version": 1,
        "source": "train and validation manifests; original annotated clip frame counts",
        "manifests": [str(path.resolve()) for path in manifest_paths],
        "window_policy": "stride=limit//2; regular starts plus final start N-limit to cover last frame",
        "candidates": candidates,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", default=[
        root / "external_data/ipn_processed/train_manifest.csv",
        root / "external_data/ipn_processed/val_manifest.csv",
    ])
    parser.add_argument("--output", type=Path, default=(
        root / "external_data/ipn_sequence_length_audit/max_frames_analysis.json"
    ))
    args = parser.parse_args()
    rows = read_manifests(args.manifests)
    report = build_report(rows, args.manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
