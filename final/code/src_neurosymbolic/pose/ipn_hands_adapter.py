"""Adapt the official IPN HandS release to the project's canonical IPN-13 layout.

IPN HandS stores one JSON object per continuous video.  This module segments
those objects with the refined temporal annotations and writes the raw
``[{frame_id, instances: ...}]`` schema consumed by ``normalize_keypoints``.

The current model and symbolic templates use the historical 13-class IPN
taxonomy.  Therefore ``NOG`` and the new ``GRB`` class are intentionally
excluded, while the remaining HandS class codes are mapped to B0A/B0B/G01-G11.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


CLASS_MAP = {
    1: ("B0A", 0),
    2: ("B0B", 1),
    3: ("G01", 2),
    4: ("G02", 3),
    5: ("G03", 4),
    6: ("G04", 5),
    7: ("G05", 6),
    8: ("G06", 7),
    9: ("G07", 8),
    10: ("G08", 9),
    11: ("G09", 10),
    12: ("G10", 11),
    13: ("G11", 12),
}
EXCLUDED_HANDS_CLASSES = {0: "NOG", 14: "GRB"}
GRAB_HANDS_CLASS_ID = 14  # opt-in 14th class (``--include-grab``) so results can be compared with the IPN HandS paper

POSE_NAME_BY_INDEX = {
    "0": "nose",
    "2": "left_eye",
    "5": "right_eye",
    "11": "left_shoulder",
    "12": "right_shoulder",
    "13": "left_elbow",
    "14": "right_elbow",
}
FINGER_NAMES = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")
VIDEO_PATTERN = re.compile(r"^(?P<subject>.+)_[RL]_#\d+$")
DATA_CONTRACT_VERSION = "ipn_hands_official_inputs_v1"
DERIVED_OUTPUT_DIRS = ("keypoint_tensor_cache", "instance_graphs_phase_v2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_xy(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x * 640.0, y * 480.0]


def point(value: object) -> dict | None:
    xy = finite_xy(value)
    return None if xy is None else {"xy": xy, "score": 1.0}


def midpoint_point(a: object, b: object) -> dict | None:
    a_xy = finite_xy(a)
    b_xy = finite_xy(b)
    if a_xy is None or b_xy is None:
        return None
    return {
        "xy": [(a_xy[0] + b_xy[0]) / 2.0, (a_xy[1] + b_xy[1]) / 2.0],
        "score": 1.0,
    }


def hand_names(side: str) -> dict[str, str]:
    names = {"0": f"{side}_hand_root"}
    for finger_index, finger in enumerate(FINGER_NAMES):
        start = 1 + finger_index * 4
        for joint in range(4):
            names[str(start + joint)] = f"{side}_{finger}{joint + 1}"
    return names


HAND_NAME_BY_INDEX = {side: hand_names(side) for side in ("left", "right")}


def hands_by_side(record: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for hand in record.get("hand_landmarks") or []:
        handedness = str(hand.get("handedness", "")).strip().lower()
        if handedness not in HAND_NAME_BY_INDEX:
            raise ValueError(f"Unsupported handedness: {hand.get('handedness')!r}")
        if handedness in result:
            raise ValueError(f"Duplicate {handedness} hand in one frame")
        result[handedness] = hand.get("landmarks") or {}
    return result


def frame_to_canonical(record: dict, frame_id: int) -> dict:
    """Convert one HandS frame to the RTMW-compatible raw frame schema."""

    keypoints: dict[str, dict] = {}
    pose = record.get("pose_landmarks") or {}
    for index, name in POSE_NAME_BY_INDEX.items():
        converted = point(pose.get(index))
        if converted is not None:
            keypoints[name] = converted

    # RTMW ``face-8`` is a lower-face/chin landmark.  HandS does not annotate
    # a chin, so the midpoint of MediaPipe mouth corners 9 and 10 is the
    # closest stable lower-face proxy.  The adapter report records this choice.
    lower_face = midpoint_point(pose.get("9"), pose.get("10"))
    if lower_face is not None:
        keypoints["face-8"] = lower_face

    hands = hands_by_side(record)
    for side, landmarks in hands.items():
        for index, name in HAND_NAME_BY_INDEX[side].items():
            converted = point(landmarks.get(index))
            if converted is not None:
                keypoints[name] = converted
        wrist = point(landmarks.get("0"))
        if wrist is not None:
            keypoints[f"{side}_wrist"] = wrist

    return {
        "frame_id": frame_id,
        "instances": [{"keypoints_by_name": keypoints}] if keypoints else [],
    }


def parse_temporal_annotations(path: Path) -> list[tuple[int, int, int]]:
    segments = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Malformed annotation at {path}:{line_number}: {line!r}")
        class_id, start, end = map(int, parts)
        if class_id not in CLASS_MAP and class_id not in EXCLUDED_HANDS_CLASSES:
            raise ValueError(f"Unknown HandS class {class_id} at {path}:{line_number}")
        if start < 1 or end < start:
            raise ValueError(f"Invalid interval at {path}:{line_number}: {start}-{end}")
        segments.append((class_id, start, end))
    return segments


def video_subject(video_name: str) -> str:
    match = VIDEO_PATTERN.fullmatch(video_name)
    if not match:
        raise ValueError(f"Cannot parse IPN subject from video name: {video_name}")
    return match.group("subject")


def load_subject_protocol(path: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    train = set(data.get("train_subjects") or [])
    validation = set(data.get("validation_subjects") or [])
    test = set(data.get("test_subjects") or [])
    excluded = set(data.get("excluded_subjects") or [])
    if not train or not validation or not test:
        raise ValueError(f"Incomplete subject protocol: {path}")
    groups = [train, validation, test, excluded]
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            if left & right:
                raise ValueError(f"Overlapping subject groups in {path}: {sorted(left & right)}")
    return train, validation, test, excluded


def read_video_list(path: Path) -> set[str]:
    return {Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def split_for_video(
    video_name: str,
    official_train: set[str],
    official_test: set[str],
    train_subjects: set[str],
    validation_subjects: set[str],
    test_subjects: set[str],
    excluded_subjects: set[str],
) -> str | None:
    subject = video_subject(video_name)
    if subject in excluded_subjects:
        return None
    if video_name in official_test:
        if subject not in test_subjects:
            raise ValueError(f"Official test video has unexpected subject: {video_name}")
        return "test"
    if video_name not in official_train:
        raise ValueError(f"Video is absent from official HandS split lists: {video_name}")
    if subject in validation_subjects:
        return "val"
    if subject in train_subjects:
        return "train"
    raise ValueError(f"Official train video has unassigned subject: {video_name}")


def atomic_json_dump(data: object, path: Path, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)
    temporary.replace(path)


def write_split_file(rows: Iterable[dict], path: Path) -> None:
    lines = [f"{row['keypoint_path']} 0 {row['label_id']}" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _input_fingerprint(hands_root: Path, protocol: Path, paths: Iterable[Path], include_grab: bool = False) -> dict:
    files = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(hands_root).as_posix() if item.is_relative_to(hands_root) else str(item)):
        try:
            name = path.relative_to(hands_root).as_posix()
        except ValueError:
            name = str(path)
        files.append({"path": name, "sha256": sha256_file(path)})
    files.append({"path": "subject_protocol", "sha256": sha256_file(protocol)})
    fingerprint_source = json.dumps(
        {"data_contract_version": DATA_CONTRACT_VERSION, "files": files, "include_grab": include_grab},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "data_contract_version": DATA_CONTRACT_VERSION,
        "sha256": hashlib.sha256(fingerprint_source).hexdigest(),
        "files": files,
    }


def _validate_official_inputs(
    official_train: set[str],
    official_test: set[str],
    landmark_paths: list[Path],
    annotations_root: Path,
) -> dict[str, list[tuple[int, int, int]]]:
    if official_train & official_test:
        raise ValueError(f"Official train/test video overlap: {sorted(official_train & official_test)}")
    gt_stems = {path.stem for path in landmark_paths}
    annotation_paths = sorted(annotations_root.glob("*.txt"))
    annotation_stems = {path.stem for path in annotation_paths}
    official = official_train | official_test
    if official != gt_stems:
        raise ValueError(f"Official split/GT video mismatch: missing GT={sorted(official - gt_stems)}, unlisted GT={sorted(gt_stems - official)}")
    if official != annotation_stems:
        raise ValueError(f"Official split/annotation video mismatch: missing annotations={sorted(official - annotation_stems)}, unlisted annotations={sorted(annotation_stems - official)}")
    segments_by_video: dict[str, list[tuple[int, int, int]]] = {}
    for landmark_path in landmark_paths:
        name = landmark_path.stem
        frames = json.loads(landmark_path.read_text(encoding="utf-8"))
        if not isinstance(frames, dict):
            raise ValueError(f"Expected frame-keyed GT object in {landmark_path}")
        expected_keys = {f"{name}_{index:06d}.jpg" for index in range(1, len(frames) + 1)}
        if set(frames) != expected_keys:
            raise ValueError(f"Non-contiguous or malformed GT frame keys in {landmark_path}")
        segments = parse_temporal_annotations(annotations_root / f"{name}.txt")
        cursor = 1
        for _, start, end in segments:
            if start != cursor:
                reason = "gap" if start > cursor else "overlap or unsorted interval"
                raise ValueError(f"Temporal annotation {reason} in {name}: expected start {cursor}, got {start}")
            cursor = end + 1
        if cursor - 1 != len(frames):
            raise ValueError(f"Temporal annotations for {name} end at {cursor - 1}; GT has {len(frames)} frames")
        segments_by_video[name] = segments
    return segments_by_video


def _prepare_dataset(
    hands_root: Path,
    output_root: Path,
    subject_protocol: Path,
    *,
    overwrite: bool = False,
    pretty: bool = False,
    include_grab: bool = False,
) -> dict:
    # Default is the historical 13-class IPN taxonomy; include_grab adds GRB as label id 13 (14 classes).
    class_map = dict(CLASS_MAP)
    excluded_classes = dict(EXCLUDED_HANDS_CLASSES)
    if include_grab:
        class_map[GRAB_HANDS_CLASS_ID] = ("GRB", len(CLASS_MAP))
        excluded_classes.pop(GRAB_HANDS_CLASS_ID)
    landmarks_root = hands_root / "landmarks_gt"
    annotations_root = hands_root / "temporal_annotations"
    for required in (
        landmarks_root,
        annotations_root,
        hands_root / "splits" / "train_list.txt",
        hands_root / "splits" / "test_list.txt",
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required IPN HandS input not found: {required}")
    output_root.mkdir(parents=True, exist_ok=True)

    official_train = read_video_list(hands_root / "splits" / "train_list.txt")
    official_test = read_video_list(hands_root / "splits" / "test_list.txt")
    train_subjects, validation_subjects, test_subjects, excluded_subjects = load_subject_protocol(
        subject_protocol
    )

    manifest_rows: list[dict] = []
    skipped_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = {name: Counter() for name in ("train", "val", "test")}

    landmark_paths = sorted(landmarks_root.glob("*.json"))
    segments_by_video = _validate_official_inputs(
        official_train, official_test, landmark_paths, annotations_root
    )
    fingerprint_paths = [hands_root / "splits" / "train_list.txt", hands_root / "splits" / "test_list.txt"]
    fingerprint_paths.extend(landmark_paths)
    fingerprint_paths.extend(sorted(annotations_root.glob("*.txt")))
    input_fingerprint = _input_fingerprint(hands_root, subject_protocol, fingerprint_paths, include_grab)
    for video_index, landmark_path in enumerate(landmark_paths, 1):
        video_name = landmark_path.stem
        split = split_for_video(
            video_name,
            official_train,
            official_test,
            train_subjects,
            validation_subjects,
            test_subjects,
            excluded_subjects,
        )
        if split is None:
            skipped_counts["excluded_subject_video"] += 1
            continue
        video_counts[split] += 1
        subject = video_subject(video_name)
        # Preflight deliberately releases each continuous-video object before
        # conversion so the 200-video release is never retained in memory at
        # once.  Re-reading one video here bounds peak memory to one JSON.
        frames = json.loads(landmark_path.read_text(encoding="utf-8"))
        segments = segments_by_video[video_name]
        for class_id, start, end in segments:
            if class_id in excluded_classes:
                skipped_counts[excluded_classes[class_id]] += 1
                continue
            label, label_id = class_map[class_id]
            frame_records = []
            for output_frame, source_frame in enumerate(range(start, end + 1)):
                key = f"{video_name}_{source_frame:06d}.jpg"
                if key not in frames:
                    raise KeyError(f"Missing HandS frame {key} for interval {start}-{end}")
                frame_records.append(frame_to_canonical(frames[key], output_frame))

            safe_video = video_name.replace("_#", "_")
            clip_id = f"{safe_video}_{label}_f{start:06d}_f{end:06d}"
            output_path = output_root / "keypoints" / split / label / subject / f"{clip_id}.json"
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Output already exists: {output_path}")
            atomic_json_dump(frame_records, output_path, pretty=pretty)
            logical_path = output_path.as_posix()
            manifest_rows.append(
                {
                    "clip_id": clip_id,
                    "source_video": video_name,
                    "subject": subject,
                    "split": split,
                    "label": label,
                    "label_id": label_id,
                    "hands_class_id": class_id,
                    "t_start": start,
                    "t_end": end,
                    "frames": end - start + 1,
                    "keypoint_path": logical_path,
                }
            )
            class_counts[split][label] += 1
        if video_index % 20 == 0 or video_index == len(landmark_paths):
            print(f"videos={video_index}/{len(landmark_paths)} clips={len(manifest_rows)}")

    manifest_rows.sort(key=lambda row: (row["split"], row["label_id"], row["clip_id"]))
    fieldnames = list(manifest_rows[0]) if manifest_rows else []
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    for split in ("train", "val", "test"):
        write_split_file((row for row in manifest_rows if row["split"] == split), output_root / f"{split}.txt")

    report = {
        "adapter": "ipn_hands_gt_to_project_ipn13_v1",
        "source": str(hands_root),
        "output": str(output_root),
        "subject_protocol": str(subject_protocol),
        "subject_protocol_sha256": sha256_file(subject_protocol),
        "input_fingerprint": input_fingerprint,
        "coordinate_contract": {
            "source_xy": "IPN HandS normalized image coordinates",
            "raw_output_xy": "pixels at 640x480; downstream shoulder normalization remains unchanged",
            "score": (
                "1.0 when a released point exists; hand x/y are manually corrected GT, "
                "while auxiliary pose points are uncorrected MediaPipe outputs; 0/missing otherwise"
            ),
            "handedness": "IPN HandS anatomical Left/Right; no additional mirror swap",
            "face_8_proxy": "midpoint of MediaPipe pose mouth corners 9 and 10",
        },
        "taxonomy": {
            "included": {str(key): {"label": value[0], "label_id": value[1]} for key, value in class_map.items()},
            "excluded": {str(key): value for key, value in excluded_classes.items()},
            "include_grab": include_grab,
        },
        "counts": {
            "videos": dict(sorted(video_counts.items())),
            "clips": dict(Counter(row["split"] for row in manifest_rows)),
            "classes": {split: dict(sorted(counts.items())) for split, counts in class_counts.items()},
            "skipped": dict(sorted(skipped_counts.items())),
        },
    }
    atomic_json_dump(report, output_root / "adapter_report.json", pretty=True)
    return report


def prepare_dataset(
    hands_root: Path,
    output_root: Path,
    subject_protocol: Path,
    *,
    overwrite: bool = False,
    pretty: bool = False,
    include_grab: bool = False,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output is not empty: {output_root}. Pass --overwrite to replace matching files."
        )
    if overwrite:
        derived = [output_root / name for name in DERIVED_OUTPUT_DIRS if (output_root / name).exists()]
        if derived:
            raise FileExistsError(
                f"Refusing --overwrite because derived data exists under {output_root}: {derived}. "
                "Use a new output root or remove derived data first."
            )
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "conversion_status.json"
    atomic_json_dump({"status": "running"}, status_path, pretty=True)
    try:
        report = _prepare_dataset(
            hands_root, output_root, subject_protocol,
            overwrite=overwrite, pretty=pretty, include_grab=include_grab,
        )
    except Exception as exc:
        atomic_json_dump({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, status_path, pretty=True)
        raise
    atomic_json_dump({"status": "complete"}, status_path, pretty=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official IPN HandS GT for the project IPN-13 C/V4 pipeline.")
    parser.add_argument("--hands-root", type=Path, default=Path("external_data/ipn_hands"))
    parser.add_argument("--output-root", type=Path, default=Path("external_data/ipn_hands_prepared"))
    parser.add_argument(
        "--subject-protocol",
        type=Path,
        default=Path("external_data/ipn_processed/split_subjects.json"),
        help="Existing project subject split; keeps experiment C and HandS comparison subject-aligned.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--include-grab",
        action="store_true",
        help="Keep the HandS 'grab' class as label id 13 (14 classes, as in the IPN HandS paper). Default: 13 classes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_dataset(
        args.hands_root,
        args.output_root,
        args.subject_protocol,
        overwrite=args.overwrite,
        pretty=args.pretty,
        include_grab=args.include_grab,
    )
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
