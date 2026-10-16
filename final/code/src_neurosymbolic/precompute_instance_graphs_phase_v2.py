import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.events.build_events_phase_v2 import build_events_phase_v2
from src_neurosymbolic.kg.build_instance_graph_phase_v2 import build_instance_graph_phase_v2
from src_neurosymbolic.models.keypoint_dataset import (
    KEYPOINT_NAMES,
    KeypointSequenceDataset,
    read_split_file,
    resolve_keypoint_path,
)
from src_neurosymbolic.pose.normalize_keypoints import normalize_video
from src_neurosymbolic.predicates.extract_predicates import (
    DEFAULT_MOTION_THRESHOLD,
    IPN_FINGER_MOTION_THRESHOLD,
    IPN_HAND_MOTION_THRESHOLD,
    IPN_PINCH_THRESHOLD,
    build_predicate_config,
    extract_predicates,
)


def tensor_to_normalized_video(tensor: torch.Tensor, source_path: str) -> dict:
    frames = []
    for frame_id in range(tensor.shape[0]):
        frame_points = {}
        for idx, name in enumerate(KEYPOINT_NAMES):
            x, y, score = tensor[frame_id, idx].tolist()
            frame_points[name] = {"xy": [float(x), float(y)], "score": float(score)}
        frames.append({"frame_id": frame_id, "valid": True, "keypoints": frame_points})
    return {"video_id": Path(source_path).stem, "source_path": source_path, "frames": frames}


def output_path_for_source(source: str, keypoint_root: Path, output_root: Path, split: str) -> Path:
    # Reject parent traversal before any of the fallback marker mapping below.
    # A split entry such as ``../foreign/keypoints/val/...`` must not be
    # reinterpreted as if its ``keypoints`` marker made it trusted.
    source_text = str(source).replace("\\", "/")
    if any(part == ".." for part in source_text.split("/")):
        raise ValueError(f"Unsafe source path for instance graph output: {source}")

    source_path = Path(source)
    try:
        rel = source_path.relative_to(keypoint_root)
    except ValueError:
        root_text = keypoint_root.as_posix().rstrip("/")
        if source_text.startswith(root_text + "/"):
            rel = Path(source_text[len(root_text) + 1 :])
        elif "keypoints" in source_text.split("/"):
            parts = source_text.split("/")
            rel = Path(*parts[parts.index("keypoints") + 1 :])
        elif not source_path.is_absolute():
            rel = source_path
        else:
            raise ValueError(f"Cannot map absolute source outside keypoint root: {source}")

    if rel.is_absolute() or not rel.parts or any(part == ".." for part in rel.parts):
        raise ValueError(f"Unsafe source path for instance graph output: {source}")
    candidate = (output_root / rel.with_suffix(".json")).resolve()
    output_root_resolved = output_root.resolve()
    try:
        candidate.relative_to(output_root_resolved)
    except ValueError as exc:
        raise ValueError(f"Instance graph output escapes output root: {candidate}") from exc

    parts = source.replace("\\", "/").split("/")
    for split_name in ("train", "val", "test"):
        if split_name not in parts:
            continue
        idx = parts.index(split_name)
        if idx + 2 >= len(parts):
            continue
        label = parts[idx + 1]
        filename = Path(parts[-1]).stem
        prefix = f"{label}_"
        rest = filename[len(prefix) :] if filename.startswith(prefix) else filename
        tokens = rest.split("_")
        if len(tokens) >= 3 and tokens[0].startswith("Subject"):
            subject = "_".join(tokens[:2])
            clip_name = "_".join(tokens[2:])
            alternate = (output_root / label / subject / f"{clip_name}.json").resolve()
            alternate.relative_to(output_root_resolved)
            return alternate
    return candidate


def file_fingerprint(path: Path, kind: str, logical_source: str) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "kind": kind,
        "source": logical_source.replace("\\", "/"),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def precompute_split(
    split_file: Path,
    keypoint_root: Path,
    output_root: Path,
    split: str,
    cache_root: Path | None,
    num_frames: int,
    limit: int | None,
    overwrite: bool,
    max_phase_duration: int,
    min_phase_duration: int,
    min_overlap_ratio: float,
    min_overlap_frames: int,
    merge_adjacent: bool,
    motion_threshold: float | None = None,
    hand_motion_threshold: float | None = None,
    finger_motion_threshold: float | None = None,
    pinch_threshold: float | None = None,
    continue_on_error: bool = False,
) -> dict:
    # Keep a legacy aggregate threshold for programmatic callers while making
    # the phase-v2/IPN defaults explicit for new precompute runs.
    legacy_threshold = DEFAULT_MOTION_THRESHOLD if motion_threshold is None else motion_threshold
    if motion_threshold is not None:
        resolved_hand_threshold = motion_threshold if hand_motion_threshold is None else hand_motion_threshold
        resolved_finger_threshold = motion_threshold if finger_motion_threshold is None else finger_motion_threshold
        resolved_pinch_threshold = motion_threshold if pinch_threshold is None else pinch_threshold
    else:
        resolved_hand_threshold = IPN_HAND_MOTION_THRESHOLD if hand_motion_threshold is None else hand_motion_threshold
        resolved_finger_threshold = IPN_FINGER_MOTION_THRESHOLD if finger_motion_threshold is None else finger_motion_threshold
        resolved_pinch_threshold = IPN_PINCH_THRESHOLD if pinch_threshold is None else pinch_threshold
    expected_predicate_config = build_predicate_config(
        min_score=0.3,
        window_size=5,
        motion_threshold=legacy_threshold,
        hand_motion_threshold=resolved_hand_threshold,
        finger_motion_threshold=resolved_finger_threshold,
        pinch_threshold=resolved_pinch_threshold,
    )
    build_config = {
        "schema_version": 2,
        "predicate_config": expected_predicate_config,
        "event_config": {
            "builder": "phase_v2",
            "max_gap": 2,
            "min_confidence": 0.3,
            "max_phase_duration": max_phase_duration,
            "min_phase_duration": min_phase_duration,
            "min_overlap_ratio": min_overlap_ratio,
            "min_overlap_frames": min_overlap_frames,
            "merge_adjacent": merge_adjacent,
        },
    }

    samples = read_split_file(split_file)
    if limit is not None:
        samples = samples[:limit]
    written = 0
    skipped = 0
    failed = 0
    cache_dataset = (
        KeypointSequenceDataset(split_file, keypoint_root, split, num_frames, limit=0, cache_root=cache_root)
        if cache_root
        else None
    )

    for idx, (source, _) in enumerate(samples, start=1):
        try:
            out_path = output_path_for_source(source, keypoint_root, output_root, split)

            try:
                input_path = resolve_keypoint_path(source, keypoint_root, split)
                input_kind = "keypoint_json"
            except FileNotFoundError:
                if cache_dataset is None:
                    raise
                input_path = cache_dataset.cache_path_for_source(source)
                if not input_path or not input_path.exists():
                    raise FileNotFoundError(f"No keypoint JSON or tensor cache for {source}")
                input_kind = "tensor_cache"
            source_fingerprint = file_fingerprint(input_path, input_kind, source)

            if out_path.exists() and not overwrite:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                provenance = existing.get("provenance") or {}
                if (
                    existing.get("graph_type") != "instance_graph"
                    or provenance.get("build_config") != build_config
                    or provenance.get("source_fingerprint") != source_fingerprint
                ):
                    raise ValueError(
                        f"stale instance graph or source at {out_path}; rerun with --overwrite"
                    )
                skipped += 1
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if input_kind == "keypoint_json":
                    normalized_data = normalize_video(input_path)
                else:
                    normalized_data = tensor_to_normalized_video(torch.load(input_path, map_location="cpu"), source)

                predicates_data = extract_predicates(
                    normalized_data,
                    min_score=0.3,
                    window_size=5,
                    motion_threshold=legacy_threshold,
                    hand_motion_threshold=resolved_hand_threshold,
                    finger_motion_threshold=resolved_finger_threshold,
                    pinch_threshold=resolved_pinch_threshold,
                )
                events_data = build_events_phase_v2(
                    predicates_data,
                    max_gap=2,
                    min_confidence=0.3,
                    max_phase_duration=max_phase_duration,
                    min_phase_duration=min_phase_duration,
                    min_overlap_ratio=min_overlap_ratio,
                    min_overlap_frames=min_overlap_frames,
                    merge_adjacent=merge_adjacent,
                )
                instance_graph = build_instance_graph_phase_v2(events_data)
                predicate_config = predicates_data.get("predicate_config", {})
                instance_graph["predicate_config"] = predicate_config
                instance_graph["provenance"] = {
                    "predicate_config": predicate_config,
                    "event_builder": "phase_v2",
                    "build_config": build_config,
                    "source_fingerprint": source_fingerprint,
                }

                with out_path.open("w", encoding="utf-8") as handle:
                    json.dump(instance_graph, handle, ensure_ascii=False)
                written += 1
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {source}: {exc}")
            if not continue_on_error:
                raise

        if idx % 100 == 0 or idx == len(samples):
            print(
                f"{split}: {idx}/{len(samples)} "
                f"written={written} skipped={skipped} failed={failed}"
            )

    return {"split": split, "total": len(samples), "written": written, "skipped": skipped, "failed": failed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute phase-aware instance graphs.")
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/instance_graphs_phase_v2"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-file", type=Path, default=Path("val.txt"))
    parser.add_argument("--test-file", type=Path, default=Path("test.txt"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success even when one or more graphs fail to build.",
    )
    parser.add_argument("--max-phase-duration", type=int, default=16)
    parser.add_argument("--min-phase-duration", type=int, default=4)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.4)
    parser.add_argument("--min-overlap-frames", type=int, default=2)
    parser.add_argument("--no-merge-adjacent", action="store_true")
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=None,
        help="Legacy aggregate threshold; when set, it fills any omitted family threshold.",
    )
    parser.add_argument(
        "--hand-motion-threshold",
        type=float,
        default=None,
        help=f"Threshold for shoulder-normalized hand motion (IPN phase-v2 default: {IPN_HAND_MOTION_THRESHOLD}).",
    )
    parser.add_argument(
        "--finger-motion-threshold",
        type=float,
        default=None,
        help=f"Threshold for hand-root-relative finger motion (IPN phase-v2 default: {IPN_FINGER_MOTION_THRESHOLD}).",
    )
    parser.add_argument(
        "--pinch-threshold",
        "--pinch-motion-threshold",
        dest="pinch_threshold",
        type=float,
        default=None,
        help=f"Threshold for normalized thumb-index pinch distance trends (IPN phase-v2 default: {IPN_PINCH_THRESHOLD}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_files = {
        "train": args.train_file,
        "val": args.val_file,
        "test": args.test_file,
    }
    summaries = []
    for split in args.splits:
        summaries.append(
            precompute_split(
                split_files[split],
                args.keypoint_root,
                args.output_root,
                split,
                args.cache_root,
                args.num_frames,
                args.limit,
                args.overwrite,
                args.max_phase_duration,
                args.min_phase_duration,
                args.min_overlap_ratio,
                args.min_overlap_frames,
                merge_adjacent=not args.no_merge_adjacent,
                motion_threshold=args.motion_threshold,
                hand_motion_threshold=args.hand_motion_threshold,
                finger_motion_threshold=args.finger_motion_threshold,
                pinch_threshold=args.pinch_threshold,
                continue_on_error=args.allow_failures,
            )
        )
    print("Done.")
    for item in summaries:
        print(item)
    failed = sum(item["failed"] for item in summaries)
    if failed and not args.allow_failures:
        raise SystemExit(f"Precompute failed for {failed} sample(s)")


if __name__ == "__main__":
    main()
