"""Chấm nhánh symbolic thuần luật (graph_matching.score_template) trên một split IPN.

Không train gì: mỗi instance graph được so với 13 template, lớp có điểm cao nhất là dự đoán.
Đồng thời thống kê tỉ lệ clip của mỗi lớp có chứa từng concept (predicate/event), để biết
template nào dựa trên tín hiệu thực sự xuất hiện trong dữ liệu.

Ví dụ:
    python scripts/eval_ipn_template_matching.py --split val
"""

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "final" / "code"))

from src_neurosymbolic.kg.graph_matching import instance_facts, score_template  # noqa: E402
from src_neurosymbolic.precompute_instance_graphs_phase_v2 import (  # noqa: E402
    output_path_for_source,
)

IPN_LABELS = ["B0A", "B0B", "G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11"]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_templates(template_dir: Path) -> list[dict]:
    graphs = []
    for path in sorted(template_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read template graph {path}: {exc}") from exc
        if not isinstance(graph, Mapping):
            raise ValueError(f"Template graph {path} must contain a JSON object")
        if graph.get("graph_type") != "template_graph":
            raise ValueError(f"Template graph {path} must have graph_type=template_graph")
        try:
            label_id = int(graph["label_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Template graph {path} has no integer label_id") from exc
        label_name = graph.get("label_name")
        if not isinstance(label_name, str):
            raise ValueError(f"Template graph {path} has no label_name")
        graphs.append(dict(graph))

    if len(graphs) != len(IPN_LABELS):
        raise ValueError(
            f"Template graph count mismatch: expected {len(IPN_LABELS)}, found {len(graphs)}"
        )
    ids = [int(graph["label_id"]) for graph in graphs]
    duplicates = sorted({label_id for label_id in ids if ids.count(label_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate template graph label IDs: {duplicates}")
    expected_ids = set(range(len(IPN_LABELS)))
    if set(ids) != expected_ids:
        raise ValueError(
            f"Template graph IDs mismatch: expected {sorted(expected_ids)}, found {sorted(set(ids))}"
        )
    for graph in graphs:
        label_id = int(graph["label_id"])
        expected_name = IPN_LABELS[label_id]
        if graph.get("label_name") != expected_name:
            raise ValueError(
                f"Template graph label_id {label_id} has non-canonical label_name "
                f"{graph.get('label_name')!r}; expected {expected_name!r}"
            )
    graphs.sort(key=lambda graph: int(graph["label_id"]))
    return graphs


def instance_graph_metadata(
    graph: Mapping[str, Any],
    path: Path,
    *,
    allow_legacy_graphs: bool,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Return predicate/build provenance and reject stale mixed V2 metadata."""

    predicate_config = graph.get("predicate_config")
    if predicate_config is None:
        if allow_legacy_graphs:
            return None, None
        raise ValueError(f"Instance graph {path} is missing predicate_config")
    if int(graph.get("schema_version") or 0) != 2:
        raise ValueError(f"Instance graph {path} must have schema_version=2")
    if not isinstance(predicate_config, Mapping) or not predicate_config:
        raise ValueError(f"Instance graph {path} has invalid predicate_config")

    provenance = graph.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError(f"Instance graph {path} is missing provenance")
    provenance_predicate = provenance.get("predicate_config")
    if not isinstance(provenance_predicate, Mapping) or canonical_json(provenance_predicate) != canonical_json(predicate_config):
        raise ValueError(f"Instance graph {path} has mixed predicate_config provenance")
    build_config = provenance.get("build_config")
    if not isinstance(build_config, Mapping) or not build_config:
        raise ValueError(f"Instance graph {path} is missing provenance.build_config")
    if int(build_config.get("schema_version") or 0) != 2:
        raise ValueError(f"Instance graph {path} build_config must have schema_version=2")
    source_fingerprint = provenance.get("source_fingerprint")
    if not isinstance(source_fingerprint, Mapping) or not source_fingerprint:
        raise ValueError(f"Instance graph {path} is missing provenance.source_fingerprint")
    build_predicate = build_config.get("predicate_config")
    if canonical_json(build_predicate) != canonical_json(predicate_config):
        raise ValueError(f"Instance graph {path} has mixed build_config predicate_config")
    return predicate_config, build_config


def instance_graph_path(source: str, keypoint_root: Path, graph_root: Path) -> Path:
    # Keep evaluator lookup byte-for-byte aligned with precompute output,
    # including the subject-style ``LABEL_Subject..._clip`` layout.
    return output_path_for_source(source, keypoint_root, graph_root, split="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.glob("*.json")):
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def f1_table(matrix: list[list[int]], support_counts: Mapping[int, int] | None = None) -> list[dict]:
    """Build per-class metrics, optionally counting abstentions in support.

    The evaluator historically derived support from the confusion matrix.  An
    abstained clip has no predicted column, so callers that enable abstention
    pass the true-label counts explicitly.  Keeping the default ``None``
    preserves the legacy behavior for callers that use this helper directly.
    """
    rows = []
    n = len(matrix)
    for c in range(n):
        tp = matrix[c][c]
        predicted = sum(matrix[r][c] for r in range(n))
        support = support_counts[c] if support_counts is not None else sum(matrix[c])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": IPN_LABELS[c], "support": support, "precision": precision, "recall": recall, "f1": f1})
    return rows


def diagnostic_record(
    *,
    source: str,
    true_label_id: int,
    scored: list[Mapping[str, Any]],
    ordered_indices: list[int],
    prediction: int | None,
    abstained: bool,
    abstention_reasons: list[str],
    tie: bool,
    best_score: float,
    margin: float,
    min_best_score: float | None,
    min_top2_margin: float | None,
) -> dict[str, Any]:
    """Return one JSONL-safe, per-clip score/debug record.

    ``score_template`` intentionally retains the complete result returned by
    ``graph_matching.score_template``.  In particular this keeps its
    ``event_matches`` evidence available when a clip is abstained, while the
    compact ``scores`` and ``top_labels`` fields stay convenient for consumers
    that only need ranking information.
    """

    compact_scores = [
        {
            "label_id": item.get("label_id", index),
            "label_name": item.get("label_name", IPN_LABELS[index]),
            "score": item.get("score"),
        }
        for index, item in enumerate(scored)
    ]
    top_labels = [
        scored[index].get("label_name", IPN_LABELS[index])
        for index in ordered_indices
    ]
    top_label_ids = [
        scored[index].get("label_id", index)
        for index in ordered_indices
    ]
    event_diagnostics = [
        {
            "label_id": item.get("label_id", index),
            "label_name": item.get("label_name", IPN_LABELS[index]),
            "score": item.get("score"),
            "temporal_score": item.get("temporal_score"),
            "matched_events": item.get("matched_events"),
            "required_events": item.get("required_events"),
            "event_matches": item.get("event_matches", []),
        }
        for index, item in enumerate(scored)
    ]
    predicted_label = "abstain" if abstained else IPN_LABELS[prediction]  # type: ignore[index]
    return {
        "source": source,
        "true_label": IPN_LABELS[true_label_id],
        "true_label_id": true_label_id,
        "predicted_label": predicted_label,
        "predicted_label_id": None if abstained else prediction,
        "abstain": abstained,
        "abstention_reasons": abstention_reasons,
        "best_score": best_score,
        "margin": margin,
        "top2_margin": margin,
        "tie": tie,
        "scores": compact_scores,
        "top_labels": top_labels,
        "top_label_ids": top_label_ids,
        "score_template": [dict(item) for item in scored],
        "score_template_event_diagnostics": event_diagnostics,
        "min_best_score": min_best_score,
        "min_top2_margin": min_top2_margin,
    }


def _run_main(cleanups: list[Callable[[], None]]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external_data/ipn_processed"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--split-file", type=Path, help="Override data-root/<split>.txt.")
    parser.add_argument("--keypoint-root", type=Path, help="Root used to relativize source paths.")
    parser.add_argument("--template-dir", type=Path, help="Override data-root/template_graphs.")
    parser.add_argument("--instance-graph-root", type=Path, help="Override data-root/instance_graphs.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true", help="Evaluate a partial graph cohort instead of failing.")
    parser.add_argument(
        "--allow-partial-labels",
        action="store_true",
        help="Do not require evaluated support for all 13 canonical labels.",
    )
    parser.add_argument(
        "--allow-legacy-graphs",
        action="store_true",
        help="Allow instance graphs without V2 predicate metadata.",
    )
    parser.add_argument("--tie-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--min-best-score",
        type=float,
        default=None,
        help="Abstain when the best template score is below this threshold.",
    )
    parser.add_argument(
        "--min-top2-margin",
        type=float,
        default=None,
        help="Abstain when the best-vs-second score margin is below this threshold.",
    )
    parser.add_argument(
        "--diagnostics-jsonl",
        type=Path,
        help="Optionally write one per-clip score/event diagnostic JSON object per line.",
    )
    args = parser.parse_args()

    if args.min_best_score is not None and (
        not math.isfinite(args.min_best_score) or args.min_best_score < 0
    ):
        parser.error("--min-best-score must be finite and non-negative")
    if args.min_top2_margin is not None and (
        not math.isfinite(args.min_top2_margin) or args.min_top2_margin < 0
    ):
        parser.error("--min-top2-margin must be finite and non-negative")

    data_root = args.data_root
    keypoint_root = args.keypoint_root or data_root / "keypoints"
    template_dir = args.template_dir or data_root / "template_graphs"
    instance_graph_root = args.instance_graph_root or data_root / "instance_graphs"
    split_file = args.split_file or data_root / f"{args.split}.txt"
    templates = load_templates(template_dir)
    if not split_file.is_file():
        raise ValueError(f"Missing split file: {split_file}")
    if not instance_graph_root.is_dir():
        raise ValueError(f"Missing instance graph directory: {instance_graph_root}")

    n = len(IPN_LABELS)
    matrix = [[0] * n for _ in range(n)]
    concept_counts = defaultdict(Counter)
    class_counts = Counter()
    missing = 0
    ties = 0
    tied_by_class = Counter()
    margins = []
    abstained_count = 0
    abstentions_by_class = Counter()
    cohort_predicate_config: Mapping[str, Any] | None = None
    cohort_build_config: Mapping[str, Any] | None = None
    predicate_config_signature: str | None = None
    build_config_signature: str | None = None
    split_lines = [
        line
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    diagnostics_handle = None
    diagnostics_temp_path = None
    if args.diagnostics_jsonl:
        args.diagnostics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.diagnostics_jsonl.parent,
            prefix=f".{args.diagnostics_jsonl.name}.",
            suffix=".tmp",
            delete=False,
        )
        diagnostics_temp_path = Path(diagnostics_handle.name)

        def cleanup_diagnostics() -> None:
            if diagnostics_handle is not None and not diagnostics_handle.closed:
                diagnostics_handle.close()
            if diagnostics_temp_path is not None:
                diagnostics_temp_path.unlink(missing_ok=True)

        cleanups.append(cleanup_diagnostics)
    for line in split_lines:
        try:
            source, _, label_text = line.rsplit(maxsplit=2)
            label = int(label_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed split line: {line!r}") from exc
        if not 0 <= label < n:
            raise ValueError(f"Label id outside 0..{n - 1}: {label} in {line}")
        path = instance_graph_path(source, keypoint_root, instance_graph_root)
        if not path.exists():
            missing += 1
            continue
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read instance graph {path}: {exc}") from exc
        if not isinstance(graph, Mapping) or graph.get("graph_type") != "instance_graph":
            raise ValueError(f"Instance graph {path} must have graph_type=instance_graph")
        if not isinstance(graph.get("nodes"), list) or not graph["nodes"]:
            raise ValueError(f"Instance graph {path} nodes must be a nonempty list")
        if not isinstance(graph.get("edges"), list):
            raise ValueError(f"Instance graph {path} edges must be a list")
        predicate_config, build_config = instance_graph_metadata(
            graph,
            path,
            allow_legacy_graphs=args.allow_legacy_graphs,
        )
        if predicate_config is not None:
            signature = canonical_json(predicate_config)
            if predicate_config_signature is None:
                predicate_config_signature = signature
                cohort_predicate_config = predicate_config
            elif signature != predicate_config_signature:
                raise ValueError(
                    "Mixed predicate_config across evaluated instance graphs; "
                    f"first={predicate_config_signature}, current={signature} ({path})"
                )
        if build_config is not None:
            signature = canonical_json(build_config)
            if build_config_signature is None:
                build_config_signature = signature
                cohort_build_config = build_config
            elif signature != build_config_signature:
                raise ValueError(
                    "Mixed build_config across evaluated instance graphs; "
                    f"first={build_config_signature}, current={signature} ({path})"
                )
        scored = [score_template(graph, template) for template in templates]
        scores = [item["score"] for item in scored]
        best_score = max(scores)
        best = [index for index, value in enumerate(scores) if abs(value - best_score) <= args.tie_tolerance]
        if len(best) > 1:
            ties += 1
            tied_by_class[label] += 1
        ordered_indices = sorted(range(n), key=lambda c: scores[c], reverse=True)
        ordered = [scores[index] for index in ordered_indices]
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        margins.append(margin)
        prediction = ordered_indices[0]
        abstention_reasons = []
        if args.min_best_score is not None and best_score < args.min_best_score:
            abstention_reasons.append("best_score_below_threshold")
        if args.min_top2_margin is not None and margin < args.min_top2_margin:
            abstention_reasons.append("top2_margin_below_threshold")
        abstained = bool(abstention_reasons)
        class_counts[label] += 1
        if abstained:
            abstained_count += 1
            abstentions_by_class[label] += 1
        else:
            matrix[label][prediction] += 1
        facts, _ = instance_facts(graph)
        for concept in {concept for _, _, concept in facts}:
            concept_counts[label][concept] += 1

        if diagnostics_handle is not None:
            diagnostics_record = diagnostic_record(
                source=source,
                true_label_id=label,
                scored=scored,
                ordered_indices=ordered_indices,
                prediction=None if abstained else prediction,
                abstained=abstained,
                abstention_reasons=abstention_reasons,
                tie=len(best) > 1,
                best_score=best_score,
                margin=margin,
                min_best_score=args.min_best_score,
                min_top2_margin=args.min_top2_margin,
            )
            diagnostics_handle.write(
                json.dumps(diagnostics_record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    total = sum(class_counts.values())
    if missing and not args.allow_missing:
        raise RuntimeError(f"Missing {missing}/{len(split_lines)} instance graphs; refusing partial-cohort evaluation")
    if total + missing != len(split_lines):
        raise RuntimeError("Evaluated and missing graph counts do not match split size")
    missing_support = [IPN_LABELS[label] for label in range(n) if class_counts[label] == 0]
    if missing_support and not args.allow_partial_labels:
        raise ValueError(
            "Evaluated graph support is missing canonical labels: "
            f"{missing_support}; pass --allow-partial-labels to override"
        )
    correct = sum(matrix[c][c] for c in range(n))
    accepted_count = total - abstained_count
    coverage = accepted_count / max(total, 1)
    selective_accuracy = correct / max(accepted_count, 1)
    rows = f1_table(matrix, class_counts)
    present = [row for row in rows if row["support"] > 0]
    macro_f1 = sum(row["f1"] for row in present) / max(len(present), 1)
    print(f"split={args.split} clips={total} missing_graphs={missing}")
    print(f"accuracy={correct / max(total, 1):.4f} macro_f1={macro_f1:.4f}")
    print(
        f"abstained={abstained_count}/{total} "
        f"abstention_rate={abstained_count / max(total, 1):.4f} "
        f"coverage={coverage:.4f} selective_accuracy={selective_accuracy:.4f}"
    )
    print(f"ties={ties}/{total} tie_rate={ties / max(total, 1):.4f} mean_top2_margin={sum(margins) / max(len(margins), 1):.4f}")
    print("label  support  precision  recall  f1")
    for row in rows:
        print(f"{row['label']:5}  {row['support']:7}  {row['precision']:9.3f}  {row['recall']:6.3f}  {row['f1']:.3f}")

    confusions = sorted(
        ((matrix[r][c], IPN_LABELS[r], IPN_LABELS[c]) for r in range(n) for c in range(n) if r != c and matrix[r][c]),
        reverse=True,
    )
    print("top confusions (true -> pred):")
    for count, truth, pred in confusions[:10]:
        print(f"  {truth} -> {pred}: {count}")

    presence = {
        IPN_LABELS[c]: {
            concept: round(count / class_counts[c], 3)
            for concept, count in concept_counts[c].most_common()
        }
        for c in range(n)
        if class_counts[c]
    }
    if diagnostics_handle is not None:
        diagnostics_handle.flush()
        diagnostics_handle.close()
        diagnostics_temp_path.replace(args.diagnostics_jsonl)
        cleanups.remove(cleanup_diagnostics)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_payload = {
            "split": args.split,
            "split_file": str(split_file),
            "split_sha256": sha256_file(split_file),
            "template_dir": str(template_dir),
            "template_graphs_sha256": sha256_json_directory(template_dir),
            "instance_graph_root": str(instance_graph_root),
            "clips": total,
            "expected_clips": len(split_lines),
            "missing_graphs": missing,
            "allow_legacy_graphs": args.allow_legacy_graphs,
            "allow_partial_labels": args.allow_partial_labels,
            "min_best_score": args.min_best_score,
            "min_top2_margin": args.min_top2_margin,
            "diagnostics_jsonl": str(args.diagnostics_jsonl) if args.diagnostics_jsonl else None,
            "accuracy": correct / max(total, 1),
            "macro_f1": macro_f1,
            "accepted_count": accepted_count,
            "coverage": coverage,
            "selective_accuracy": selective_accuracy,
            "abstained_count": abstained_count,
            "abstention_rate": abstained_count / max(total, 1),
            "abstentions_by_class": {
                IPN_LABELS[label]: abstentions_by_class[label] for label in range(n)
            },
            "tie_count": ties,
            "tie_rate": ties / max(total, 1),
            "mean_top2_margin": sum(margins) / max(len(margins), 1),
            "ties_by_class": {IPN_LABELS[label]: tied_by_class[label] for label in range(n)},
            "per_class": rows,
            "confusion_matrix": matrix,
            "concept_presence_by_class": presence,
        }
        if cohort_predicate_config is not None:
            output_payload["predicate_config"] = cohort_predicate_config
            output_payload["predicate_config_sha256"] = sha256_json(cohort_predicate_config)
        if cohort_build_config is not None:
            output_payload["build_config"] = cohort_build_config
            output_payload["build_config_sha256"] = sha256_json(cohort_build_config)
            # Keep the short ``hash`` spelling as a compatibility affordance
            # for consumers that do not use SHA-256-specific field names.
            output_payload["build_config_hash"] = output_payload["build_config_sha256"]
        args.output.write_text(
            json.dumps(
                output_payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {args.output}")


def main() -> None:
    cleanups: list[Callable[[], None]] = []
    try:
        _run_main(cleanups)
    finally:
        for cleanup in reversed(cleanups):
            cleanup()


if __name__ == "__main__":
    main()
