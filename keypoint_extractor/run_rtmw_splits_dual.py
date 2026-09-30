import argparse
import json
import shutil
import sys
from pathlib import Path


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


KEEP_KEYPOINTS = {
    "nose",
    "left_eye",
    "right_eye",
    "face-8",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
}

FINGERS = ["thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger"]
JOINTS = ["1", "2", "3", "4"]

for side in ["left", "right"]:
    KEEP_KEYPOINTS.add(f"{side}_hand_root")
    for finger in FINGERS:
        for joint in JOINTS:
            KEEP_KEYPOINTS.add(f"{side}_{finger}{joint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RTMW-L wholebody keypoints for dynamic gesture splits."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing mmpose/, checkpoints/, and dataset/.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root. Defaults to <repo-root>/dataset.",
    )
    parser.add_argument(
        "--dataset",
        choices=["old", "new", "both"],
        default="both",
        help="Which split folder(s) to process: splits_old, splits_new, or both.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to process.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "splits", "raw-subjects"],
        default="auto",
        help=(
            "Input layout. splits expects splits_old/splits_new. raw-subjects "
            "expects <data-root>/<gesture>/<subject>/<video>."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output root for raw-subjects layout. Defaults to "
            "<data-root>/keypoints_52_mirror_swapped."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='MMPose device, for example "cuda:0" or "cpu".',
    )
    parser.add_argument(
        "--mmpose-root",
        type=Path,
        default=None,
        help="Path to local MMPose checkout. Defaults to <repo-root>/mmpose.",
    )
    parser.add_argument(
        "--pose-config",
        type=Path,
        default=None,
        help="RTMW pose config path. Defaults to the RTMW-L config in mmpose/projects.",
    )
    parser.add_argument(
        "--pose-weights",
        type=Path,
        default=None,
        help="RTMW checkpoint path. Defaults to checkpoints/rtmw-dw-x-l...pth.",
    )
    parser.add_argument(
        "--legacy-cache-dir",
        type=Path,
        default=None,
        help="Optional cache dir to copy existing JSON from. Defaults to <data-root>/keypoints.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not copy JSON from legacy cache before extracting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even when output JSON already exists.",
    )
    parser.add_argument(
        "--no-mirror-swap",
        action="store_true",
        help="Disable left/right keypoint name swap used for mirrored videos.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented JSON. By default JSON is compact to save disk.",
    )
    return parser.parse_args()


def default_pose_config(mmpose_root: Path) -> Path:
    return (
        mmpose_root
        / "projects"
        / "rtmpose"
        / "rtmpose"
        / "wholebody_2d_keypoint"
        / "rtmw-l_8xb320-270e_cocktail14-384x288.py"
    )


def default_pose_weights(repo_root: Path) -> Path:
    return (
        repo_root
        / "checkpoints"
        / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
    )


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_id2name(mmpose_root: Path, mirror_swap: bool) -> list[str]:
    from mmengine import Config

    meta_path = mmpose_root / "configs" / "_base_" / "datasets" / "coco_wholebody.py"
    require_file(meta_path, "COCO-WholeBody metainfo")

    cfg = Config.fromfile(str(meta_path))
    keypoint_info = cfg["dataset_info"]["keypoint_info"]

    names = []
    for idx in range(133):
        name = keypoint_info[idx]["name"]
        if mirror_swap:
            if "left_" in name:
                name = name.replace("left_", "right_")
            elif "right_" in name:
                name = name.replace("right_", "left_")
        names.append(name)
    return names


def make_inferencer(pose_config: Path, pose_weights: Path, device: str):
    from mmpose.apis import MMPoseInferencer

    print(f"Initializing RTMW-L on {device}...")
    return MMPoseInferencer(
        pose2d=str(pose_config),
        pose2d_weights=str(pose_weights),
        device=device,
        show_progress=False,
    )


def find_videos(folder: Path) -> list[Path]:
    videos = []
    for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos.extend(folder.glob(pattern))
    return sorted(videos)


def pick_best_instance(instances: list[dict]) -> dict | None:
    best_instance = None
    best_score = -1.0

    for instance in instances:
        scores = instance.get("keypoint_scores") or []
        if not scores:
            continue
        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_instance = instance

    return best_instance


def frame_to_record(result: dict, frame_id: int, id2name: list[str]) -> dict:
    record = {"frame_id": frame_id, "instances": []}
    predictions = result.get("predictions", [])
    instances = predictions[0] if predictions and isinstance(predictions[0], list) else predictions

    best_instance = pick_best_instance(instances)
    if best_instance is None:
        return record

    keypoints = best_instance.get("keypoints") or []
    scores = best_instance.get("keypoint_scores") or []

    by_name = {
        id2name[idx]: {"xy": keypoints[idx], "score": scores[idx]}
        for idx in range(min(133, len(keypoints), len(scores)))
        if id2name[idx] in KEEP_KEYPOINTS
    }
    record["instances"].append({"keypoints_by_name": by_name})
    return record


def extract_video(video_path: Path, out_json: Path, inferencer, id2name: list[str], pretty: bool) -> None:
    frames = []
    for frame_id, result in enumerate(inferencer(str(video_path), return_vis=False)):
        frames.append(frame_to_record(result, frame_id, id2name))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(frames, handle, ensure_ascii=False, indent=2 if pretty else None)


def copy_from_cache(cache_json: Path, out_json: Path, overwrite: bool) -> bool:
    if not cache_json.is_file():
        return False
    if out_json.exists() and not overwrite:
        return True

    out_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_json, out_json)
    return True


def process_dataset(
    dataset_name: str,
    data_root: Path,
    split_names: list[str],
    legacy_cache_dir: Path | None,
    inferencer_holder: dict,
    id2name: list[str],
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    splits_dir = data_root / f"splits_{dataset_name}"
    output_dir = data_root / f"keypoints_{dataset_name}"

    if not splits_dir.is_dir():
        print(f"Skip missing folder: {splits_dir}")
        return 0, 0, 0, 0

    print("")
    print(f"Processing {splits_dir} -> {output_dir}")

    extracted = 0
    skipped = 0
    copied = 0
    jobs = []

    for split_name in split_names:
        split_dir = splits_dir / split_name
        if not split_dir.is_dir():
            print(f"  Skip missing split: {split_dir}")
            continue

        print(f"  Split: {split_name}")
        for gesture_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            videos = find_videos(gesture_dir)
            if not videos:
                continue

            for video_path in videos:
                out_json = output_dir / split_name / gesture_dir.name / f"{video_path.stem}.json"
                jobs.append((split_name, gesture_dir.name, video_path, out_json))

    total_videos = len(jobs)
    print(f"  Videos found: {total_videos}")

    for index, (split_name, gesture_name, video_path, out_json) in enumerate(jobs, start=1):
        progress = f"[{index}/{total_videos}]"

        if out_json.exists() and not args.overwrite:
            skipped += 1
            print(f"    {progress} Skipping existing: {split_name}/{gesture_name}/{video_path.name}")
            continue

        if legacy_cache_dir is not None and not args.no_cache:
            cache_json = legacy_cache_dir / split_name / gesture_name / f"{video_path.stem}.json"
            if copy_from_cache(cache_json, out_json, args.overwrite):
                copied += 1
                print(f"    {progress} Copied cache: {split_name}/{gesture_name}/{video_path.name}")
                continue

        if "inferencer" not in inferencer_holder:
            inferencer_holder["inferencer"] = make_inferencer(
                args.pose_config,
                args.pose_weights,
                args.device,
            )

        print(f"    {progress} Extracting: {split_name}/{gesture_name}/{video_path.name}")
        try:
            extract_video(
                video_path,
                out_json,
                inferencer_holder["inferencer"],
                id2name,
                args.pretty,
            )
            extracted += 1
        except Exception as exc:
            print(f"    {progress} ERROR: {video_path} -> {exc}")

    return extracted, skipped, copied, total_videos


def looks_like_splits_layout(data_root: Path) -> bool:
    return (data_root / "splits_new").is_dir() or (data_root / "splits_old").is_dir()


def looks_like_raw_subjects_layout(data_root: Path) -> bool:
    for gesture_dir in data_root.iterdir() if data_root.is_dir() else []:
        if not gesture_dir.is_dir() or gesture_dir.name.startswith("."):
            continue
        if gesture_dir.name.startswith(("keypoints", "splits", "cached", "features")):
            continue
        for subject_dir in gesture_dir.iterdir():
            if subject_dir.is_dir() and find_videos(subject_dir):
                return True
    return False


def iter_raw_subject_videos(data_root: Path):
    for gesture_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if gesture_dir.name.startswith("."):
            continue
        if gesture_dir.name.startswith(("keypoints", "splits", "cached", "features")):
            continue
        for subject_dir in sorted(p for p in gesture_dir.iterdir() if p.is_dir()):
            for video_path in find_videos(subject_dir):
                yield gesture_dir.name, subject_dir.name, video_path


def process_raw_subjects(
    data_root: Path,
    output_root: Path,
    inferencer_holder: dict,
    id2name: list[str],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    print("")
    print(f"Processing raw subjects: {data_root} -> {output_root}")

    extracted = 0
    skipped = 0
    jobs = list(iter_raw_subject_videos(data_root))
    total_videos = len(jobs)
    print(f"  Videos found: {total_videos}")

    for index, (gesture, subject, video_path) in enumerate(jobs, start=1):
        progress = f"[{index}/{total_videos}]"
        out_json = output_root / gesture / subject / f"{video_path.stem}.json"

        if out_json.exists() and not args.overwrite:
            skipped += 1
            print(f"  {progress} Skipping existing: {gesture}/{subject}/{video_path.name}")
            continue

        if "inferencer" not in inferencer_holder:
            inferencer_holder["inferencer"] = make_inferencer(
                args.pose_config,
                args.pose_weights,
                args.device,
            )

        print(f"  {progress} Extracting: {gesture}/{subject}/{video_path.name}")
        try:
            extract_video(
                video_path,
                out_json,
                inferencer_holder["inferencer"],
                id2name,
                args.pretty,
            )
            extracted += 1
        except Exception as exc:
            print(f"  {progress} ERROR: {video_path} -> {exc}")

    return extracted, skipped, total_videos


def main() -> None:
    args = parse_args()

    args.repo_root = args.repo_root.resolve()
    data_root = (args.data_root or args.repo_root / "dataset").resolve()
    mmpose_root = (args.mmpose_root or args.repo_root / "mmpose").resolve()
    args.pose_config = (args.pose_config or default_pose_config(mmpose_root)).resolve()
    args.pose_weights = (args.pose_weights or default_pose_weights(args.repo_root)).resolve()
    legacy_cache_dir = None if args.no_cache else (args.legacy_cache_dir or data_root / "keypoints").resolve()

    require_file(args.pose_config, "Pose config")
    require_file(args.pose_weights, "Pose weights")

    split_names = ["train", "val", "test"] if args.split == "all" else [args.split]
    dataset_names = ["old", "new"] if args.dataset == "both" else [args.dataset]
    output_root = (
        args.output_root or data_root / "keypoints_52_mirror_swapped"
    ).resolve()

    print(f"Repo root: {args.repo_root}")
    print(f"Data root: {data_root}")
    print(f"MMPose root: {mmpose_root}")
    print(f"Pose config: {args.pose_config}")
    print(f"Pose weights: {args.pose_weights}")
    print(f"Device: {args.device}")

    id2name = load_id2name(mmpose_root, mirror_swap=not args.no_mirror_swap)
    inferencer_holder = {}

    use_raw_subjects = args.layout == "raw-subjects" or (
        args.layout == "auto"
        and not looks_like_splits_layout(data_root)
        and looks_like_raw_subjects_layout(data_root)
    )

    if use_raw_subjects:
        total_extracted, total_skipped, total_videos = process_raw_subjects(
            data_root,
            output_root,
            inferencer_holder,
            id2name,
            args,
        )
        total_copied = 0
    else:
        total_extracted = 0
        total_skipped = 0
        total_copied = 0
        total_videos = 0

        for dataset_name in dataset_names:
            extracted, skipped, copied, videos = process_dataset(
                dataset_name,
                data_root,
                split_names,
                legacy_cache_dir,
                inferencer_holder,
                id2name,
                args,
            )
            total_extracted += extracted
            total_skipped += skipped
            total_copied += copied
            total_videos += videos

    print("")
    print("Done.")
    print(f"Total videos found: {total_videos}")
    print(f"Extracted: {total_extracted}")
    print(f"Copied from cache: {total_copied}")
    print(f"Skipped existing: {total_skipped}")
    print(f"Available JSON after run: {total_extracted + total_copied + total_skipped}")


if __name__ == "__main__":
    main()
