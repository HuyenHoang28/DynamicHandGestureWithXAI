"""Calibrate experimental V3 temporal factor thresholds on IPN train/validation.

The emitted pair decisions are diagnostics only. This command deliberately has
no test split option and never opens a test manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.models.keypoint_dataset import (  # noqa: E402
    KEYPOINT_NAMES,
    KeypointSequenceDataset,
    resolve_keypoint_path,
)
from src_neurosymbolic.predicates.extract_predicates import (  # noqa: E402
    extract_temporal_factor_predicates,
)
from src_neurosymbolic.models.keypoint_dataset import read_split_file  # noqa: E402
from src_neurosymbolic.variable_keypoint_cache import load_variable_cache  # noqa: E402
from neural_only_experiments.variable_dataset import variable_cache_path  # noqa: E402

TARGET_PAIRS = (("G01", "G08"), ("G02", "G09"), ("G10", "G11"))
PAIR_TO_FAMILY = {("G01", "G08"): "index", ("G02", "G09"): "two_finger", ("G10", "G11"): "pinch"}
LABEL_IDS = {"B0A": 0, "B0B": 1, **{f"G{i:02}": i + 1 for i in range(1, 12)}}
DEFAULT_GRID = [
    {"click_active_threshold": .50, "click_inactive_threshold": .42, "click_min_confidence": .30, "two_finger_start_delta": 3, "two_finger_overlap": .5, "pinch_min_excursion": .08, "pinch_min_valid_ratio": .5, "temporal_max_gap": 2, "pinch_dominance_ratio": 1.1},
    {"click_active_threshold": .56, "click_inactive_threshold": .46, "click_min_confidence": .35, "two_finger_start_delta": 4, "two_finger_overlap": .4, "pinch_min_excursion": .12, "pinch_min_valid_ratio": .6, "temporal_max_gap": 3, "pinch_dominance_ratio": 1.2},
    {"click_active_threshold": .62, "click_inactive_threshold": .50, "click_min_confidence": .45, "two_finger_start_delta": 5, "two_finger_overlap": .35, "pinch_min_excursion": .16, "pinch_min_valid_ratio": .7, "temporal_max_gap": 4, "pinch_dominance_ratio": 1.35},
]


def fullrate_grid() -> list[dict[str, Any]]:
    """Grid for caches that keep the original frame rate (gaps/deltas are in real frames)."""
    grid = []
    for active, inactive in ((.50, .42), (.56, .46), (.62, .50)):
        for gap, delta in ((3, 4), (5, 6)):
            for min_confidence in (.30, .40):
                grid.append({
                    "click_active_threshold": active, "click_inactive_threshold": inactive,
                    "click_min_confidence": min_confidence, "two_finger_start_delta": delta,
                    "two_finger_overlap": .4, "pinch_min_excursion": .12, "pinch_min_valid_ratio": .6,
                    "temporal_max_gap": gap, "pinch_dominance_ratio": 1.2,
                })
    return grid


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_labels(path: Path) -> dict[str, tuple[str, str | None]]:
    """Map source/keypoint path variants to authoritative label and subject metadata."""
    result: dict[str, tuple[str, str | None]] = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = str(row.get("label", "")).strip().upper()
            if not label:
                continue
            for field in ("keypoint_path", "source", "path"):
                value = str(row.get(field, "")).strip()
                if value:
                    result[canonical_source(value)] = (label, row.get("subject") or None)
    return result


def canonical_source(source: str) -> str:
    return source.replace("\\", "/").lower().lstrip("./")


def label_from_source(source: str, split: str, metadata: dict[str, tuple[str, str | None]], label_id: int) -> tuple[str, str | None]:
    parts = source.replace("\\", "/").split("/")
    for i, part in enumerate(parts[:-1]):
        if part.lower() == split.lower() and parts[i + 1].upper() in LABEL_IDS:
            label = parts[i + 1].upper()
            return label, metadata.get(canonical_source(source), (label, None))[1]
    meta = metadata.get(canonical_source(source))
    if meta:
        return meta
    # Match metadata paths by basename when manifests store a different root.
    basename = canonical_source(source).rsplit("/", 1)[-1]
    for key, value in metadata.items():
        if key.rsplit("/", 1)[-1] == basename:
            return value
    label = next((name for name, index in LABEL_IDS.items() if index == label_id), "")
    if not label:
        raise ValueError(f"Cannot determine label for {source!r} (label id {label_id})")
    return label, None


def subject_from(source: str, metadata_subject: str | None) -> str:
    if metadata_subject:
        return str(metadata_subject)
    normalized = source.replace("\\", "/")
    match = re.search(r"(Subject\d+_[A-Za-z0-9]+)", normalized, re.IGNORECASE)
    if match:
        return match.group(1)
    parts = normalized.split("/")
    for i, part in enumerate(parts[:-1]):
        if part.lower() in {"train", "val", "validation"} and i + 2 < len(parts):
            return parts[i + 2]
    stem = Path(parts[-1]).stem
    match = re.match(r"(.+?_[0-9]+)_[RL]_#?\d+", stem)
    return match.group(1) if match else "unknown"


def validate_source_split(source: str, expected_split: str) -> None:
    """Reject missing or cross-split paths before any tensor/raw input is resolved."""
    aliases = {"val", "validation"} if expected_split == "val" else {expected_split}
    declared = [
        part.lower()
        for part in source.replace("\\", "/").split("/")
        if part.lower() in {"train", "val", "validation", "test"}
    ]
    if not declared or any(part not in aliases for part in declared):
        raise ValueError(
            f"{expected_split} manifest contains a missing/cross-split source path: {source!r} "
            f"(declared split tokens: {declared or ['none']})"
        )


def tensor_to_frames(tensor: Any) -> list[dict]:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    frames = []
    for frame_id, row in enumerate(tensor):
        points = {}
        for name, value in zip(KEYPOINT_NAMES, row):
            points[name] = {"xy": [float(value[0]), float(value[1])], "score": float(value[2])}
        frames.append({"frame_id": frame_id, "valid": True, "keypoints": points})
    return frames


def pair_decision(label: str, predicates: list[dict], config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    pair = next((p for p in TARGET_PAIRS if label in p), None)
    if pair is None:
        return "", {"reason": "outside_target_pairs"}
    family = PAIR_TO_FAMILY[pair]
    selected = [p for p in predicates if p.get("subject", "").endswith("_hand")]
    by_subject: dict[str, list[dict]] = {}
    for predicate in selected:
        by_subject.setdefault(predicate["subject"], []).append(predicate)
    if family == "index":
        factor = "index_click_cycle"
        subject, subject_rows = max(
            by_subject.items(),
            default=("unknown", []),
            key=lambda item: (
                sum(float(row.get("confidence", 0.0)) for row in item[1] if row["name"] == factor),
                item[0],
            ),
        )
        count = sum(p["name"] == factor for p in subject_rows)
        guess = pair[0] if count == 1 else pair[1] if count == 2 else "unknown"
        evidence = {"factor": factor, "count": count, "subject": subject}
    elif family == "two_finger":
        factor = "two_finger_click_cycle"
        subject, subject_rows = max(
            by_subject.items(),
            default=("unknown", []),
            key=lambda item: (
                sum(float(row.get("confidence", 0.0)) for row in item[1] if row["name"] == factor),
                item[0],
            ),
        )
        count = sum(p["name"] == factor for p in subject_rows)
        guess = pair[0] if count == 1 else pair[1] if count == 2 else "unknown"
        evidence = {"factor": factor, "count": count, "subject": subject}
    else:
        transitions = sorted(
            (p for p in selected if p["name"] in {"pinch_opening", "pinch_closing"}),
            key=lambda row: (-float(row.get("confidence", 0.0)), row["subject"], row["name"]),
        )
        best = transitions[0] if transitions else None
        opposite = next((row for row in transitions[1:] if row["name"] != best["name"]), None) if best else None
        dominance = float(config.get("pinch_dominance_ratio", 1.1))
        decisive = bool(
            best
            and (
                opposite is None
                or float(best.get("confidence", 0.0))
                >= dominance * max(float(opposite.get("confidence", 0.0)), 1e-12)
            )
        )
        guess = pair[0] if decisive and best["name"] == "pinch_opening" else pair[1] if decisive else "unknown"
        evidence = {
            "factor": "pinch_transition",
            "subject": best["subject"] if best else "unknown",
            "best": best["name"] if best else "unknown",
            "best_confidence": float(best.get("confidence", 0.0)) if best else 0.0,
            "opposite_confidence": float(opposite.get("confidence", 0.0)) if opposite else 0.0,
        }
    return guess, evidence


def aggregate(samples: Iterable[dict], pair: tuple[str, str]) -> dict[str, Any]:
    rows = [r for r in samples if r["true_label"] in pair]
    covered = [r for r in rows if r["prediction"] != "unknown"]
    correct = sum(r["prediction"] == r["true_label"] for r in covered)
    confusion: dict[str, dict[str, int]] = {truth: {pred: 0 for pred in (*pair, "unknown")} for truth in pair}
    for r in rows:
        confusion[r["true_label"]][r["prediction"] if r["prediction"] in (*pair, "unknown") else "unknown"] += 1
    return {
        "pair": list(pair), "support": len(rows), "coverage": len(covered) / len(rows) if rows else 0.0,
        "covered": len(covered), "correct": correct,
        "accuracy_on_covered": correct / len(covered) if covered else 0.0,
        "overall_correct": correct, "overall_accuracy_abstain_wrong": correct / len(rows) if rows else 0.0,
        "confusion_unknown_included": confusion,
    }


def score_config(samples: list[dict], config: dict[str, Any], min_coverage: float) -> dict[str, Any]:
    per_pair = {"/".join(pair): aggregate(samples, pair) for pair in TARGET_PAIRS}
    per_subject = {}
    for subject in sorted({row["subject"] for row in samples}):
        subject_rows = [row for row in samples if row["subject"] == subject]
        per_subject[subject] = {
            "/".join(pair): aggregate(subject_rows, pair)
            for pair in TARGET_PAIRS
            if any(row["true_label"] in pair for row in subject_rows)
        }
    relevant = [v for v in per_pair.values() if v["support"]]
    total_support = sum(v["support"] for v in relevant)
    correct = sum(v["correct"] for v in relevant)
    coverage = sum(v["covered"] for v in relevant) / total_support if total_support else 0.0
    # Net exact pair accuracy is the primary objective; a coverage floor is a hard feasibility gate.
    feasible = bool(total_support) and all(v["coverage"] >= min_coverage for v in relevant)
    return {"config": config, "per_pair": per_pair, "per_subject": per_subject, "support": total_support, "coverage": coverage,
            "correct": correct, "overall_accuracy_abstain_wrong": correct / total_support if total_support else 0.0,
            "feasible": feasible, "objective": correct / total_support if feasible else -1.0}


def select_config(validation_scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not validation_scores:
        raise ValueError("No validation scores")
    # Stable lexicographic tie-break: objective, coverage, then canonical config JSON.
    return max(validation_scores, key=lambda row: (row["objective"], row["coverage"], json.dumps(row["config"], sort_keys=True)))


def resolve_sample_input(dataset: KeypointSequenceDataset, source: str) -> Path:
    """Return the tensor or raw JSON that ``dataset[source]`` will actually load."""
    source_cache = dataset.cache_path_for_source(source)
    if source_cache and source_cache.is_file():
        return source_cache
    raw_path = resolve_keypoint_path(source, dataset.keypoint_root, dataset.split)
    raw_cache = dataset.cache_path_for(raw_path)
    return raw_cache if raw_cache and raw_cache.is_file() else raw_path


def run_split(split: str, split_path: Path, manifest_path: Path, keypoint_root: Path,
              cache_root: Path | None, configs: list[dict[str, Any]], limit: int | None,
              variable_cache_root: Path | None = None) -> tuple[list[dict], dict[str, int], list[dict]]:
    metadata = manifest_labels(manifest_path)
    if variable_cache_root is None:
        dataset = KeypointSequenceDataset(split_path, keypoint_root, split, cache_root=cache_root)
        samples = dataset.samples
    else:
        dataset = None  # full-rate variable_v1 cache replaces the 64-frame tensors
        samples = read_split_file(split_path)
    records, skipped, inputs = [], Counter(), []
    relevant_seen = 0
    for i, (source, label_id) in enumerate(samples):
        validate_source_split(source, split)
        label, meta_subject = label_from_source(source, split, metadata, label_id)
        subject = subject_from(source, meta_subject)
        if label not in {item for pair in TARGET_PAIRS for item in pair}:
            skipped[label] += 1
            continue
        if limit is not None and relevant_seen >= limit:
            break
        relevant_seen += 1
        input_path = (
            variable_cache_path(source, variable_cache_root)
            if variable_cache_root is not None
            else resolve_sample_input(dataset, source)
        )
        if not input_path.is_file():
            raise FileNotFoundError(f"Cannot resolve input for {split} sample {source!r}: {input_path}")
        inputs.append({
            "split": split,
            "source": source,
            "resolved_path": str(input_path.resolve()),
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        })
        keypoints = (
            load_variable_cache(input_path)["keypoints"]
            if variable_cache_root is not None
            else dataset[i]["keypoints"]
        )
        frames = tensor_to_frames(keypoints)
        for config_index, config in enumerate(configs):
            predicates = extract_temporal_factor_predicates(frames, .3, **config)
            prediction, evidence = pair_decision(label, predicates, config)
            records.append({"split": split, "source": source, "subject": subject, "true_label": label,
                            "config_index": config_index, "prediction": prediction, "covered": prediction != "unknown",
                            "correct": prediction == label, "evidence": evidence})
    return records, dict(sorted(skipped.items())), inputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "external_data/ipn_processed", help="IPN processed root containing train.txt and val.txt.")
    parser.add_argument("--keypoint-root", type=Path, help="Raw keypoint JSON root; defaults to DATA_ROOT/keypoints.")
    parser.add_argument("--cache-root", type=Path, help="Optional normalized tensor cache, used by project loader.")
    parser.add_argument("--variable-cache-root", type=Path,
                        help="Use full-frame-rate variable_v1 caches under this root instead of 64-frame tensors; "
                             "defaults the grid to the full-rate grid.")
    parser.add_argument("--grid", type=Path, help="JSON list of temporal-factor threshold dictionaries. Defaults to a small built-in grid.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for calibration report and selected artifacts.")
    parser.add_argument("--min-coverage", type=float, default=.25, help="Minimum per-pair validation coverage for a config to be eligible.")
    parser.add_argument("--limit", type=int, help="Limit each split to its first N target-pair samples for smoke runs.")
    args = parser.parse_args(argv)
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be between 0 and 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root
    train_path, val_path = data_root / "train.txt", data_root / "val.txt"
    # Deliberately hard-coded train/val inputs: there is no test CLI or lookup.
    for path in (train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    configs = (
        json.loads(args.grid.read_text(encoding="utf-8")) if args.grid
        else fullrate_grid() if args.variable_cache_root else DEFAULT_GRID
    )
    if not isinstance(configs, list) or not configs or any(not isinstance(c, dict) for c in configs):
        raise ValueError("--grid must contain a non-empty JSON list of threshold objects")
    # Canonicalize configs to prevent accidental mutation and make output stable.
    configs = [json.loads(json.dumps(c, sort_keys=True)) for c in configs]
    keypoint_root = args.keypoint_root or data_root / "keypoints"
    train_records, train_skipped, train_inputs = run_split("train", train_path, data_root / "train_manifest.csv", keypoint_root, args.cache_root, configs, args.limit, args.variable_cache_root)
    val_records, val_skipped, val_inputs = run_split("val", val_path, data_root / "val_manifest.csv", keypoint_root, args.cache_root, configs, args.limit, args.variable_cache_root)
    train_scores = [score_config([r for r in train_records if r["config_index"] == i], c, args.min_coverage) for i, c in enumerate(configs)]
    val_scores = [score_config([r for r in val_records if r["config_index"] == i], c, args.min_coverage) for i, c in enumerate(configs)]
    selected = select_config(val_scores)
    selected_index = val_scores.index(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scope": "train_validation_only", "target_pairs": [list(p) for p in TARGET_PAIRS],
              "min_validation_coverage_per_pair": args.min_coverage, "train_diagnostic_scores": train_scores,
              "validation_scores": val_scores, "selected_config_index": selected_index,
              "selected_validation_score": selected, "skipped_labels": {"train": train_skipped, "val": val_skipped}}
    (args.output_dir / "calibration_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "selected_config.json").write_text(json.dumps(selected["config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["split", "source", "subject", "true_label", "config_index", "prediction", "covered", "correct", "evidence"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in train_records + val_records:
            writer.writerow({**row, "evidence": json.dumps(row["evidence"], sort_keys=True)})
    input_manifest_path = args.output_dir / "input_tensors.jsonl"
    with input_manifest_path.open("w", encoding="utf-8") as f:
        for row in train_inputs + val_inputs:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {"command": sys.argv, "scope": "train_validation_only", "data_root": str(data_root.resolve()),
                "inputs": {str(p.resolve()): sha256_file(p) for p in (train_path, val_path, data_root / "train_manifest.csv", data_root / "val_manifest.csv") if p.is_file()},
                "implementation_sha256": {
                    str(path.resolve()): sha256_file(path)
                    for path in (
                        Path(__file__),
                        ROOT / "final/code/src_neurosymbolic/models/keypoint_dataset.py",
                        ROOT / "final/code/src_neurosymbolic/pose/normalize_keypoints.py",
                        ROOT / "final/code/src_neurosymbolic/predicates/extract_predicates.py",
                        ROOT / "final/code/src_neurosymbolic/predicates/temporal_factors.py",
                    )
                },
                "grid_sha256": hashlib.sha256(json.dumps(configs, sort_keys=True).encode()).hexdigest(),
                "loaded_feature_manifest": {
                    "path": "input_tensors.jsonl",
                    "sha256": sha256_file(input_manifest_path),
                    "sample_count": len(train_inputs) + len(val_inputs),
                },
                "limit": args.limit,
                "variable_cache_root": str(args.variable_cache_root) if args.variable_cache_root else None,
                "outputs": ["calibration_report.json", "per_sample.csv", "selected_config.json", "input_tensors.jsonl"]}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Selected validation config {selected_index}: {selected['overall_accuracy_abstain_wrong']:.4f} exact pair accuracy; coverage {selected['coverage']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
