"""Rebuild and score symbolic V2 on the packaged 13-class IPN examples.

The packaged cohort contains one normalized train example and one normalized
validation example per class.  It is a smoke/regression cohort, not a valid
replacement for the full locked validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.events.build_events_phase_v2 import build_events_phase_v2  # noqa: E402
from src_neurosymbolic.kg.build_instance_graph_phase_v2 import build_instance_graph_phase_v2  # noqa: E402
from src_neurosymbolic.kg.build_template_graph import (  # noqa: E402
    DEFAULT_WEIGHTS,
    build_template_graph,
    parse_template_yaml,
)
from src_neurosymbolic.kg.graph_matching import score_template  # noqa: E402
from src_neurosymbolic.predicates.extract_predicates import (  # noqa: E402
    DEFAULT_MOTION_THRESHOLD,
    IPN_FINGER_MOTION_THRESHOLD,
    IPN_HAND_MOTION_THRESHOLD,
    IPN_PINCH_THRESHOLD,
    extract_predicates,
)


IPN_LABELS = ["B0A", "B0B", *[f"G{index:02d}" for index in range(1, 12)]]
IMPLEMENTATION_FILES = (
    "scripts/eval_ipn_symbolic_examples_v2.py",
    "final/code/src_neurosymbolic/predicates/extract_predicates.py",
    "final/code/src_neurosymbolic/events/build_events_phase_v2.py",
    "final/code/src_neurosymbolic/kg/build_instance_graph_phase_v2.py",
    "final/code/src_neurosymbolic/kg/build_template_graph.py",
    "final/code/src_neurosymbolic/kg/graph_matching.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def load_templates(path: Path) -> list[dict]:
    parsed = parse_template_yaml(path)
    weights = {
        level: float(parsed["template_policy"]["matching"][f"{level}_weight"])
        for level in DEFAULT_WEIGHTS
    }
    gestures = parsed["gestures"]
    missing = [label for label in IPN_LABELS if label not in gestures]
    extra = sorted(set(gestures) - set(IPN_LABELS))
    if missing or extra:
        raise ValueError(f"Template labels mismatch: missing={missing}, extra={extra}")
    return [
        build_template_graph(
            label,
            label_id,
            gestures[label],
            weights,
            schema_version=int(parsed["version"]),
        )
        for label_id, label in enumerate(IPN_LABELS)
    ]


def macro_f1(rows: list[dict]) -> float:
    values = []
    for label in IPN_LABELS:
        true_positive = sum(row["true_label"] == label and row["prediction"] == label for row in rows)
        predicted = sum(row["prediction"] == label for row in rows)
        support = sum(row["true_label"] == label for row in rows)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values)


def write_artifacts(root: Path, split: str, label: str, payloads: dict[str, dict]) -> None:
    output_dir = root / split / label
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output_dir / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=ROOT / "external_data/ipn_processed/person3_v4_20261008_165456/symbolic_examples",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=ROOT / "configs/ipn/gesture_templates.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--json-stdout", action="store_true", help="Print the complete JSON report instead of a concise summary.")
    parser.add_argument(
        "--min-val-macro-f1",
        type=float,
        help="Return exit code 2 when the validation smoke macro-F1 is below this gate.",
    )
    args = parser.parse_args()

    templates = load_templates(args.templates)
    rows = []
    cohort = []
    predicate_config = None
    for split in ("train", "val"):
        for label in IPN_LABELS:
            path = args.examples_root / split / label / "normalized.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing packaged symbolic example: {path}")
            normalized = json.loads(path.read_text(encoding="utf-8"))
            normalized_sha256 = sha256_bytes(path.read_bytes())
            predicates = extract_predicates(
                normalized,
                motion_threshold=DEFAULT_MOTION_THRESHOLD,
                hand_motion_threshold=IPN_HAND_MOTION_THRESHOLD,
                finger_motion_threshold=IPN_FINGER_MOTION_THRESHOLD,
                pinch_threshold=IPN_PINCH_THRESHOLD,
            )
            current_predicate_config = predicates.get("predicate_config", {})
            if predicate_config is None:
                predicate_config = current_predicate_config
            elif current_predicate_config != predicate_config:
                raise ValueError(f"Mixed predicate configuration in smoke cohort: {path}")
            events = build_events_phase_v2(predicates)
            instance_graph = build_instance_graph_phase_v2(events)
            scored = [score_template(instance_graph, template) for template in templates]
            ordered = sorted(
                scored,
                key=lambda item: (item["score"], item["required_hit"], item["positive_score"]),
                reverse=True,
            )
            best_score = ordered[0]["score"]
            true_index = next(index for index, item in enumerate(ordered) if item["label_name"] == label)
            true_result = ordered[true_index]
            ties = sum(abs(item["score"] - best_score) <= 1e-9 for item in ordered)
            margin = best_score - ordered[1]["score"]
            row = {
                "split": split,
                "true_label": label,
                "video_id": normalized.get("video_id", path.parent.name),
                "normalized_sha256": normalized_sha256,
                "prediction": ordered[0]["label_name"],
                "correct": ordered[0]["label_name"] == label,
                "top_score": best_score,
                "true_score": true_result["score"],
                "true_rank": true_index + 1,
                "top2_margin": margin,
                "tie_count": ties,
                "predicate_count": len(predicates["predicates"]),
                "event_count": len(events["events"]),
                "top3": [
                    {
                        "label": item["label_name"],
                        "score": item["score"],
                        "required": f"{item['required_hit']}/{item['required_total']}",
                        "temporal_score": item["temporal_score"],
                    }
                    for item in ordered[:3]
                ],
            }
            rows.append(row)
            cohort.append(
                {
                    "split": split,
                    "label": label,
                    "video_id": row["video_id"],
                    "normalized_sha256": normalized_sha256,
                }
            )
            if args.artifact_dir:
                write_artifacts(
                    args.artifact_dir,
                    split,
                    label,
                    {
                        "predicates_v2": predicates,
                        "events_v2": events,
                        "instance_graph_v2": instance_graph,
                        "scores_v2": {"true_label": label, "scores": ordered},
                    },
                )

    summaries = {}
    for split in ("train", "val"):
        split_rows = [row for row in rows if row["split"] == split]
        correct = sum(row["correct"] for row in split_rows)
        summaries[split] = {
            "samples": len(split_rows),
            "accuracy": correct / len(split_rows),
            "macro_f1": macro_f1(split_rows),
            "tie_rate": sum(row["tie_count"] > 1 for row in split_rows) / len(split_rows),
            "mean_top2_margin": sum(row["top2_margin"] for row in split_rows) / len(split_rows),
            "prediction_counts": dict(sorted(Counter(row["prediction"] for row in split_rows).items())),
        }

    report = {
        "scope": "Packaged IPN symbolic examples: one sample per class per split; smoke/regression only.",
        "templates": str(args.templates),
        "templates_sha256": sha256_bytes(args.templates.read_bytes()),
        "examples_root": str(args.examples_root),
        "cohort": cohort,
        "cohort_sha256": canonical_sha256(cohort),
        "predicate_config": predicate_config,
        "implementation_sha256": {
            relative_path: sha256_bytes((ROOT / relative_path).read_bytes())
            for relative_path in IMPLEMENTATION_FILES
        },
        "summaries": summaries,
        "rows": rows,
    }
    if args.json_stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for split in ("train", "val"):
            summary = summaries[split]
            print(
                f"{split}: samples={summary['samples']} accuracy={summary['accuracy']:.4f} "
                f"macro_f1={summary['macro_f1']:.4f} tie_rate={summary['tie_rate']:.4f}"
            )
        failures = [
            f"{row['split']}/{row['true_label']}->{row['prediction']} "
            f"(true_rank={row['true_rank']}, true_score={row['true_score']:.4f})"
            for row in rows
            if not row["correct"]
        ]
        if failures:
            print("misclassified: " + ", ".join(failures))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.min_val_macro_f1 is not None and summaries["val"]["macro_f1"] < args.min_val_macro_f1:
        print(
            f"validation gate failed: macro_f1={summaries['val']['macro_f1']:.4f} "
            f"< {args.min_val_macro_f1:.4f}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
