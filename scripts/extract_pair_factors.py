"""Compute label-free temporal-factor decisions for every clip of a train or validation manifest.

``calibrate_ipn_temporal_factors.py`` only scores clips of the six target classes
and picks the factor family from the *true* label. A deployed pair residual has
to decide from the clip alone, so this script evaluates all three families on
every clip (including B0A, B0B and the other gestures) with the selected config
and writes one row per clip for ``eval_pair_residual.py``.

There is deliberately no test split option: choose thresholds and residual
parameters on train/validation only.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

_spec = importlib.util.spec_from_file_location("calibrate_ipn_temporal_factors", ROOT / "scripts" / "calibrate_ipn_temporal_factors.py")
calibrator = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = calibrator
_spec.loader.exec_module(calibrator)

from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset  # noqa: E402
from src_neurosymbolic.predicates.extract_predicates import extract_temporal_factor_predicates  # noqa: E402

FAMILY_PAIRS = {"index": ("G01", "G08"), "two_finger": ("G02", "G09"), "pinch": ("G10", "G11")}
FIELDS = ["split", "source", "subject", "true_label"] + [
    f"{family}_{suffix}" for family in FAMILY_PAIRS for suffix in ("guess", "quality")
]


def family_decisions(predicates: list[dict], config: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """Return ``{family: (guess_label_or_unknown, quality)}`` without using any ground-truth label.

    The label argument of ``pair_decision`` only selects which family to read, so
    passing the pair's first class gives that family's decision for any clip.
    """

    result: dict[str, tuple[str, float]] = {}
    for family, pair in FAMILY_PAIRS.items():
        guess, evidence = calibrator.pair_decision(pair[0], predicates, config)
        if family == "pinch":
            quality = float(evidence.get("best_confidence", 0.0))
        else:
            subject = evidence.get("subject")
            factor = evidence.get("factor")
            confidences = [
                float(row.get("confidence", 0.0))
                for row in predicates
                if row.get("subject") == subject and row.get("name") == factor
            ]
            quality = sum(confidences) / len(confidences) if confidences else 0.0
        result[family] = (guess, max(0.0, min(1.0, quality)) if guess != "unknown" else 0.0)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--split-file", type=Path, required=True, help="train.txt or val.txt manifest.")
    parser.add_argument("--manifest-csv", type=Path, help="Optional train_manifest.csv/val_manifest.csv for subject metadata.")
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--config", type=Path, required=True, help="selected_config.json from the calibration run.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Smoke-test cap on the number of clips.")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("--config must contain a JSON object of thresholds")
    metadata = calibrator.manifest_labels(args.manifest_csv) if args.manifest_csv else {}
    dataset = KeypointSequenceDataset(args.split_file, args.keypoint_root, args.split, cache_root=args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, (source, label_id) in enumerate(dataset.samples):
            if args.limit is not None and written >= args.limit:
                break
            calibrator.validate_source_split(source, args.split)
            label, meta_subject = calibrator.label_from_source(source, args.split, metadata, label_id)
            frames = calibrator.tensor_to_frames(dataset[index]["keypoints"])
            predicates = extract_temporal_factor_predicates(frames, 0.3, **config)
            row = {"split": args.split, "source": source, "subject": calibrator.subject_from(source, meta_subject), "true_label": label}
            for family, (guess, quality) in family_decisions(predicates, config).items():
                row[f"{family}_guess"] = guess
                row[f"{family}_quality"] = f"{quality:.6f}"
            writer.writerow(row)
            written += 1
    print(f"wrote {written} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
