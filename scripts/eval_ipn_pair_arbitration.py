"""Audit or simulate conservative neural/symbolic pair arbitration on IPN.

The neural prediction remains authoritative by default.  Symbolic evidence may
only replace it with ``--enable-override`` and only when the neural top-2 is
exactly one of the configured confusion pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


IPN_LABELS = [
    "B0A", "B0B", "G01", "G02", "G03", "G04", "G05",
    "G06", "G07", "G08", "G09", "G10", "G11",
]
SUPPORTED_PAIRS = (("G01", "G08"), ("G02", "G09"), ("G10", "G11"))
PAIR_BY_LABEL = {
    label: pair for pair in SUPPORTED_PAIRS for label in pair
}


def canonical_source(value: str) -> str:
    """Return a portable key rooted at keypoints/, independent of Colab paths."""
    normalized = value.strip().replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    lowered = [part.casefold() for part in parts]
    try:
        index = len(lowered) - 1 - lowered[::-1].index("keypoints")
    except ValueError as exc:
        raise ValueError(f"Source path has no keypoints/ component: {value!r}") from exc
    key = "/".join(parts[index:]).casefold()
    if key == "keypoints" or key.endswith("/"):
        raise ValueError(f"Source path has no file below keypoints/: {value!r}")
    return key


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Invalid {field}: {value!r}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_label(label: str, label_id: Any, field: str) -> None:
    if label not in IPN_LABELS:
        raise ValueError(f"Unknown {field}: {label!r}")
    if int(label_id) != IPN_LABELS.index(label):
        raise ValueError(f"Mismatched {field}/id: {label!r}/{label_id!r}")


def load_neural_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "path", "true_id", "true_label", "pred_id", "pred_label",
            "confidence", "top_labels", "top_probs",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Neural CSV is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            _validate_label(row["true_label"], row["true_id"], "true_label")
            _validate_label(row["pred_label"], row["pred_id"], "pred_label")
            top_labels = row["top_labels"].split()
            top_probs = [
                _finite_float(value, f"top_probs line {line_number}")
                for value in row["top_probs"].split()
            ]
            if len(top_labels) < 2 or len(top_labels) != len(top_probs):
                raise ValueError(f"Invalid top-k fields at neural CSV line {line_number}")
            if len(set(top_labels)) != len(top_labels) or any(
                label not in IPN_LABELS for label in top_labels
            ):
                raise ValueError(f"Invalid top_labels at neural CSV line {line_number}")
            if top_labels[0] != row["pred_label"]:
                raise ValueError(f"pred_label is not top-1 at neural CSV line {line_number}")
            if any(probability < 0.0 or probability > 1.0 for probability in top_probs):
                raise ValueError(f"top_probs outside [0, 1] at neural CSV line {line_number}")
            if any(left < right for left, right in zip(top_probs, top_probs[1:])):
                raise ValueError(f"top_probs are not descending at neural CSV line {line_number}")
            confidence = _finite_float(row["confidence"], "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence outside [0, 1] at neural CSV line {line_number}")
            if not math.isclose(confidence, top_probs[0], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"confidence does not match top_probs[0] at neural CSV line {line_number}")
            key = canonical_source(row["path"])
            if key in rows:
                raise ValueError(f"Duplicate neural source: {key}")
            rows[key] = {
                **row,
                "source_key": key,
                "true_id": int(row["true_id"]),
                "pred_id": int(row["pred_id"]),
                "confidence": confidence,
                "top_labels": top_labels,
                "top_probs": top_probs,
            }
    return rows


def load_symbolic_diagnostics(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid symbolic JSONL line {line_number}: {exc}") from exc
            _validate_label(row.get("true_label"), row.get("true_label_id"), "true_label")
            score_rows = row.get("score_template")
            if not isinstance(score_rows, list) or len(score_rows) != len(IPN_LABELS):
                raise ValueError(f"Symbolic line {line_number} must contain all 13 score_template rows")
            labels = [item.get("label_name") for item in score_rows]
            if labels != IPN_LABELS:
                raise ValueError(f"Non-canonical symbolic label order at line {line_number}")
            for label_id, item in enumerate(score_rows):
                _validate_label(item.get("label_name"), item.get("label_id"), "score label")
                _finite_float(item.get("score"), f"symbolic score line {line_number}")
                if int(item["label_id"]) != label_id:
                    raise ValueError(f"Non-canonical symbolic label id at line {line_number}")
            key = canonical_source(str(row.get("source", "")))
            if key in rows:
                raise ValueError(f"Duplicate symbolic source: {key}")
            row["source_key"] = key
            rows[key] = row
    return rows


def _score_by_label(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {item["label_name"]: item for item in row["score_template"]}


def _occurrences(score: Mapping[str, Any]) -> list[int]:
    return [
        int(event.get("observed_occurrences", 0))
        for event in score.get("event_matches", [])
    ]


def symbolic_pair_decision(
    row: Mapping[str, Any],
    pair: tuple[str, str],
    *,
    min_score: float,
    min_margin: float,
    min_temporal_score: float,
) -> dict[str, Any]:
    """Return a conservative decision using only the supplied candidate pair."""
    scores = _score_by_label(row)
    left, right = pair
    left_score = _finite_float(scores[left]["score"], f"{left} score")
    right_score = _finite_float(scores[right]["score"], f"{right} score")
    ordered = sorted(((left_score, left), (right_score, right)), reverse=True)
    best_score, winner = ordered[0]
    margin = best_score - ordered[1][0]
    result = {
        "pair": list(pair),
        "scores": {left: left_score, right: right_score},
        "winner": None,
        "best_score": best_score,
        "margin": margin,
        "reason": "accepted",
    }
    # Deliberately ignore the 13-way ``abstain``/``tie`` flags here.  They may
    # be caused by an unrelated third template; this audit is conditioned on
    # the neural candidate pair and applies its own score/margin gates below.
    if best_score < min_score:
        result["reason"] = "pair_score_below_threshold"
        return result
    if margin <= 0.0 or margin < min_margin:
        result["reason"] = "pair_margin_below_threshold"
        return result
    winner_score = scores[winner]
    if int(winner_score.get("missing_required_count", 0)) != 0:
        result["reason"] = "missing_required_evidence"
        return result
    if _finite_float(winner_score.get("temporal_score", 0.0), "temporal_score") < min_temporal_score:
        result["reason"] = "temporal_score_below_threshold"
        return result
    observed = _occurrences(winner_score)
    expected_occurrences = 2 if winner in {"G08", "G09"} else 1
    if len(observed) != 1 or observed[0] != expected_occurrences:
        result["reason"] = "occurrence_gate_failed"
        return result
    if pair == ("G10", "G11"):
        loser = right if winner == left else left
        loser_score = scores[loser]
        loser_active = loser_score.get("active_intervals") or []
        if loser_active:
            result["reason"] = "opposite_active_direction_present"
            return result
        if not (winner_score.get("active_intervals") or []):
            result["reason"] = "missing_active_interval"
            return result
    result["winner"] = winner
    return result


def evaluate(
    neural_rows: Mapping[str, Mapping[str, Any]],
    symbolic_rows: Mapping[str, Mapping[str, Any]],
    *,
    enable_override: bool,
    max_neural_confidence: float,
    max_neural_margin: float,
    min_symbolic_score: float,
    min_symbolic_margin: float,
    min_temporal_score: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing_symbolic = sorted(set(neural_rows) - set(symbolic_rows))
    extra_symbolic = sorted(set(symbolic_rows) - set(neural_rows))
    if missing_symbolic or extra_symbolic:
        raise ValueError(
            "Neural/symbolic source mismatch: "
            f"missing_symbolic={len(missing_symbolic)}, extra_symbolic={len(extra_symbolic)}"
        )
    audit = []
    pair_stats: dict[str, Counter] = defaultdict(Counter)
    baseline_correct = final_correct = overrides = correct_overrides = wrong_overrides = 0
    proposed_overrides = proposed_correct = proposed_wrong = 0
    proposed_beneficial = proposed_harmful = proposed_neutral = 0
    for key in sorted(neural_rows):
        neural = neural_rows[key]
        symbolic = symbolic_rows[key]
        if neural["true_label"] != symbolic["true_label"]:
            raise ValueError(f"True-label mismatch for {key}")
        truth = neural["true_label"]
        baseline = neural["pred_label"]
        final = baseline
        reason = "unsupported_neural_top2"
        pair = PAIR_BY_LABEL.get(neural["top_labels"][0])
        neural_top2 = tuple(neural["top_labels"][:2])
        neural_margin = neural["top_probs"][0] - neural["top_probs"][1]
        decision = None
        if pair is not None and set(neural_top2) == set(pair):
            pair_name = "/".join(pair)
            pair_stats[pair_name]["eligible"] += 1
            decision = symbolic_pair_decision(
                symbolic,
                pair,
                min_score=min_symbolic_score,
                min_margin=min_symbolic_margin,
                min_temporal_score=min_temporal_score,
            )
            reason = decision["reason"]
            if decision["winner"] is not None:
                pair_stats[pair_name]["symbolic_decided"] += 1
            if neural["confidence"] > max_neural_confidence:
                reason = "neural_confidence_above_gate"
            elif neural_margin > max_neural_margin:
                reason = "neural_margin_above_gate"
            elif decision["winner"] is not None:
                pair_stats[pair_name]["override_eligible"] += 1
                pair_stats[pair_name][
                    "symbolic_correct" if decision["winner"] == truth else "symbolic_wrong"
                ] += 1
                if decision["winner"] != baseline:
                    proposed_overrides += 1
                    pair_stats[pair_name]["proposed_override"] += 1
                    if decision["winner"] == truth:
                        proposed_correct += 1
                        pair_stats[pair_name]["proposed_final_correct"] += 1
                    else:
                        proposed_wrong += 1
                        pair_stats[pair_name]["proposed_final_wrong"] += 1
                    if decision["winner"] == truth and baseline != truth:
                        proposed_beneficial += 1
                        pair_stats[pair_name]["proposed_beneficial"] += 1
                    elif baseline == truth and decision["winner"] != truth:
                        proposed_harmful += 1
                        pair_stats[pair_name]["proposed_harmful"] += 1
                    else:
                        proposed_neutral += 1
                        pair_stats[pair_name]["proposed_neutral"] += 1
                reason = "audit_accept"
                if enable_override:
                    final = decision["winner"]
                    reason = "override" if final != baseline else "symbolic_confirm"
            pair_stats[pair_name][f"reason:{reason}"] += 1
        baseline_ok = baseline == truth
        final_ok = final == truth
        baseline_correct += int(baseline_ok)
        final_correct += int(final_ok)
        if final != baseline:
            overrides += 1
            correct_overrides += int(final_ok)
            wrong_overrides += int(not final_ok)
        audit.append(
            {
                "source_key": key,
                "true_label": truth,
                "neural_prediction": baseline,
                "neural_top2": list(neural_top2),
                "neural_confidence": neural["confidence"],
                "neural_margin": neural_margin,
                "symbolic_pair": decision,
                "override_enabled": enable_override,
                "final_prediction": final,
                "reason": reason,
            }
        )
    total = len(audit)
    report = {
        "mode": "override_simulation" if enable_override else "audit_only",
        "clips": total,
        "configured_pairs": [list(pair) for pair in SUPPORTED_PAIRS],
        "thresholds": {
            "max_neural_confidence": max_neural_confidence,
            "max_neural_margin": max_neural_margin,
            "min_symbolic_score": min_symbolic_score,
            "min_symbolic_margin": min_symbolic_margin,
            "min_temporal_score": min_temporal_score,
        },
        "baseline_accuracy": baseline_correct / max(total, 1),
        "final_accuracy": final_correct / max(total, 1),
        "proposed_override_count": proposed_overrides,
        "proposed_correct_overrides": proposed_correct,
        "proposed_wrong_overrides": proposed_wrong,
        "proposed_beneficial_overrides": proposed_beneficial,
        "proposed_harmful_overrides": proposed_harmful,
        "proposed_neutral_overrides": proposed_neutral,
        "proposed_net_correct_change": proposed_beneficial - proposed_harmful,
        "override_count": overrides,
        "correct_overrides": correct_overrides,
        "wrong_overrides": wrong_overrides,
        "net_correct_change": final_correct - baseline_correct,
        "pair_stats": {name: dict(stats) for name, stats in sorted(pair_stats.items())},
    }
    return report, audit


def _nonnegative(value: str) -> float:
    parsed = _finite_float(value, "threshold")
    if parsed < 0:
        raise argparse.ArgumentTypeError("threshold must be non-negative")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-predictions", type=Path, required=True)
    parser.add_argument("--symbolic-diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument("--enable-override", action="store_true")
    parser.add_argument("--max-neural-confidence", type=_nonnegative, default=1.0)
    parser.add_argument("--max-neural-margin", type=_nonnegative)
    parser.add_argument("--min-symbolic-score", type=_nonnegative, default=0.15)
    parser.add_argument("--min-symbolic-margin", type=_nonnegative, default=0.02)
    parser.add_argument("--min-temporal-score", type=_nonnegative, default=0.5)
    args = parser.parse_args()
    if args.enable_override and args.max_neural_margin is None:
        parser.error("--enable-override requires an explicit --max-neural-margin tuned on validation")
    max_neural_margin = math.inf if args.max_neural_margin is None else args.max_neural_margin
    neural = load_neural_predictions(args.neural_predictions)
    symbolic = load_symbolic_diagnostics(args.symbolic_diagnostics)
    report, audit = evaluate(
        neural,
        symbolic,
        enable_override=args.enable_override,
        max_neural_confidence=args.max_neural_confidence,
        max_neural_margin=max_neural_margin,
        min_symbolic_score=args.min_symbolic_score,
        min_symbolic_margin=args.min_symbolic_margin,
        min_temporal_score=args.min_temporal_score,
    )
    report["inputs"] = {
        "neural_predictions": str(args.neural_predictions),
        "neural_predictions_sha256": sha256_file(args.neural_predictions),
        "symbolic_diagnostics": str(args.symbolic_diagnostics),
        "symbolic_diagnostics_sha256": sha256_file(args.symbolic_diagnostics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.audit_jsonl:
        args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.audit_jsonl.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in audit),
            encoding="utf-8",
        )
    print(
        f"mode={report['mode']} clips={report['clips']} overrides={report['override_count']} "
        f"baseline_accuracy={report['baseline_accuracy']:.4f} final_accuracy={report['final_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
