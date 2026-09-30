import argparse
import importlib.util
import json
import time
from pathlib import Path

import cv2
import torch


TARGET_FPS = 30.0


def load_extractor(script_path: Path):
    spec = importlib.util.spec_from_file_location("rtmw_extractor", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import extractor: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_at_30_fps(video_path: Path) -> tuple[list, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        raise RuntimeError(f"Video has no decodable frames: {video_path}")
    if source_fps <= 0:
        raise RuntimeError("Source video does not provide a valid FPS value.")

    duration_seconds = len(frames) / source_fps
    target_count = max(1, int(round(duration_seconds * TARGET_FPS)))
    indices = [
        min(len(frames) - 1, int(round(index * source_fps / TARGET_FPS)))
        for index in range(target_count)
    ]
    return [frames[index] for index in indices], source_fps, len(frames)


def pose_model_forward(pose_inferencer, model_inputs):
    return pose_inferencer.forward(model_inputs)


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def main(args: argparse.Namespace) -> None:
    video_path = args.video.resolve()
    repo_root = args.repo_root.resolve()
    extractor_path = args.extract_script.resolve()
    mmpose_root = args.mmpose_root.resolve()
    pose_weights = args.pose_weights.resolve()

    for path, description in (
        (video_path, "video"),
        (extractor_path, "extractor script"),
        (mmpose_root, "MMPose root"),
        (pose_weights, "RTMW-L weights"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")

    # Video decoding and 30 FPS sampling happen before the timed section.
    frames, source_fps, source_frame_count = decode_at_30_fps(video_path)
    extractor = load_extractor(extractor_path)
    pose_config = extractor.default_pose_config(mmpose_root).resolve()

    initialization_start = time.perf_counter()
    inferencer = extractor.make_inferencer(pose_config, pose_weights, args.device)
    synchronize(args.device)
    initialization_seconds = time.perf_counter() - initialization_start

    # Detection, cropping, resizing and tensor construction are intentionally
    # completed before timing. Only Pose2DInferencer.forward(), which calls
    # RTMW-L model.test_step(), is benchmarked below.
    pose_inferencer = inferencer.inferencer
    preprocessed = [
        model_inputs
        for model_inputs, _ in pose_inferencer.preprocess(frames, batch_size=1)
    ]

    warmup_count = min(args.warmup_frames, len(preprocessed))
    for model_inputs in preprocessed[:warmup_count]:
        pose_model_forward(pose_inferencer, model_inputs)
    synchronize(args.device)

    timed_inputs = preprocessed[warmup_count:]
    if not timed_inputs:
        raise RuntimeError("No frames remain after warm-up. Reduce --warmup-frames.")

    inference_seconds = 0.0
    if args.device.startswith("cuda") and torch.cuda.is_available():
        for _ in range(args.repeats):
            for model_inputs in timed_inputs:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                pose_model_forward(pose_inferencer, model_inputs)
                end_event.record()
                end_event.synchronize()
                inference_seconds += start_event.elapsed_time(end_event) / 1000.0
    else:
        for _ in range(args.repeats):
            for model_inputs in timed_inputs:
                start = time.perf_counter()
                pose_model_forward(pose_inferencer, model_inputs)
                inference_seconds += time.perf_counter() - start

    measured_frames = len(timed_inputs) * args.repeats
    result = {
        "video": str(video_path),
        "device": args.device,
        "target_fps": TARGET_FPS,
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "resampled_frame_count": len(frames),
        "warmup_frames": warmup_count,
        "repeats": args.repeats,
        "measured_frames": measured_frames,
        "model_initialization_seconds": initialization_seconds,
        "inference_seconds": inference_seconds,
        "latency_ms_per_frame": 1000.0 * inference_seconds / measured_frames,
        "inference_fps": measured_frames / inference_seconds,
        "real_time_factor_at_30fps": (measured_frames / TARGET_FPS) / inference_seconds,
        "timing_scope": (
            "RTMW-L Pose2D model forward (model.test_step) only; excludes video "
            "decoding, 30 FPS resampling, person detection, crop/resize/tensor "
            "preprocessing, model initialization, visualization, JSON writing, "
            "keypoint subset selection and sequence normalization"
        ),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {args.output}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Benchmark RTMW-L keypoint inference on frames resampled to exactly 30 FPS."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repo-root", type=Path, default=project_root)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "final" / "Experiment" / "reports" / "rtmw_30fps_inference.json",
    )
    args = parser.parse_args()
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


if __name__ == "__main__":
    main(parse_args())
