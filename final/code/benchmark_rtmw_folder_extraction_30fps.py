import argparse
import csv
import json
import time
from pathlib import Path

import torch

from benchmark_rtmw_keypoint_inference_30fps import (
    TARGET_FPS,
    decode_at_30_fps,
    load_extractor,
    synchronize,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def find_videos(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def extract_frame(inferencer, frame) -> None:
    # Fully consume the generator so detector, RTMW-L and prediction
    # postprocessing are finished for this frame.
    for _ in inferencer(frame, return_vis=False):
        pass


def main(args: argparse.Namespace) -> None:
    video_dir = args.video_dir.resolve()
    extractor_path = args.extract_script.resolve()
    mmpose_root = args.mmpose_root.resolve()
    pose_weights = args.pose_weights.resolve()

    for path, description in (
        (video_dir, "video directory"),
        (extractor_path, "extractor script"),
        (mmpose_root, "MMPose root"),
        (pose_weights, "RTMW-L weights"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")

    videos = find_videos(video_dir)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise RuntimeError(f"No videos found under: {video_dir}")

    extractor = load_extractor(extractor_path)
    pose_config = extractor.default_pose_config(mmpose_root).resolve()

    initialization_start = time.perf_counter()
    inferencer = extractor.make_inferencer(pose_config, pose_weights, args.device)
    synchronize(args.device)
    initialization_seconds = time.perf_counter() - initialization_start

    first_frames, _, _ = decode_at_30_fps(videos[0])
    warmup_count = min(args.warmup_frames, len(first_frames))
    for frame in first_frames[:warmup_count]:
        extract_frame(inferencer, frame)
    synchronize(args.device)

    rows = []
    total_seconds = 0.0
    total_frames = 0

    for index, video_path in enumerate(videos, start=1):
        frames, source_fps, source_frame_count = decode_at_30_fps(video_path)
        synchronize(args.device)
        start = time.perf_counter()
        for frame in frames:
            extract_frame(inferencer, frame)
        synchronize(args.device)
        elapsed = time.perf_counter() - start

        frame_count = len(frames)
        total_seconds += elapsed
        total_frames += frame_count
        relative_path = video_path.relative_to(video_dir)
        row = {
            "video": str(relative_path),
            "gesture": relative_path.parent.as_posix(),
            "source_fps": source_fps,
            "source_frames": source_frame_count,
            "frames_at_30fps": frame_count,
            "extraction_seconds": elapsed,
            "latency_ms_per_frame": 1000.0 * elapsed / frame_count,
            "extraction_fps": frame_count / elapsed,
            "real_time_factor_at_30fps": (frame_count / TARGET_FPS) / elapsed,
        }
        rows.append(row)
        print(
            f"[{index}/{len(videos)}] {relative_path}: "
            f"{row['latency_ms_per_frame']:.2f} ms/frame, "
            f"{row['extraction_fps']:.2f} FPS"
        )

    summary = {
        "video_directory": str(video_dir),
        "device": args.device,
        "target_fps": TARGET_FPS,
        "video_count": len(rows),
        "total_frames_at_30fps": total_frames,
        "warmup_frames": warmup_count,
        "model_initialization_seconds": initialization_seconds,
        "total_extraction_seconds": total_seconds,
        "latency_ms_per_frame": 1000.0 * total_seconds / total_frames,
        "extraction_fps": total_frames / total_seconds,
        "real_time_factor_at_30fps": (total_frames / TARGET_FPS) / total_seconds,
        "timing_scope": (
            "Keypoint extraction from already-decoded 30 FPS frames: person "
            "detection, pose preprocessing, RTMW-L forward and prediction "
            "postprocessing; excludes video decoding, FPS resampling, model "
            "initialization, visualization, JSON writing and sequence normalization"
        ),
        "per_video": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({key: value for key, value in summary.items() if key != "per_video"}, indent=2))
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / "final" / "Experiment" / "reports" / "keypoint_timing"
    parser = argparse.ArgumentParser(
        description="Benchmark full keypoint extraction on a video folder at 30 FPS."
    )
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--extract-script", type=Path, default=project_root / "run_rtmw_splits_dual.py")
    parser.add_argument("--mmpose-root", type=Path, default=project_root / "mmpose")
    parser.add_argument(
        "--pose-weights",
        type=Path,
        default=(
            project_root
            / "checkpoints"
            / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
        ),
    )
    parser.add_argument("--output-json", type=Path, default=report_root / "summary.json")
    parser.add_argument("--output-csv", type=Path, default=report_root / "per_video.csv")
    args = parser.parse_args()
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


if __name__ == "__main__":
    main(parse_args())
