import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.demo_ui.skeleton_renderer import (
    BODY_COLOR,
    BODY_EDGES,
    FACE_COLOR,
    FACE_EDGES,
    LEFT_COLOR,
    LEFT_HAND_EDGES,
    RIGHT_COLOR,
    RIGHT_HAND_EDGES,
    draw_edges,
    draw_points,
    transcode_browser_video,
)
from src_neurosymbolic.models.keypoint_dataset import (
    KEYPOINT_NAMES,
    KeypointSequenceDataset,
    resolve_keypoint_path,
)
from src_neurosymbolic.pose.normalize_keypoints import normalize_video


def tensor_to_normalized_video(tensor: torch.Tensor, source_path: str) -> dict:
    frames = []
    for frame_id in range(tensor.shape[0]):
        keypoints = {}
        for index, name in enumerate(KEYPOINT_NAMES):
            x, y, score = tensor[frame_id, index].tolist()
            keypoints[name] = {"xy": [float(x), float(y)], "score": float(score)}
        frames.append({"frame_id": frame_id, "valid": True, "keypoints": keypoints})
    return {"video_id": Path(source_path).stem, "source_path": source_path, "frames": frames}


def load_normalized_from_args(args: argparse.Namespace) -> tuple[dict, Path | None]:
    if args.input_keypoints:
        return normalize_video(args.input_keypoints), args.input_keypoints

    if args.input_cache:
        tensor = torch.load(args.input_cache, map_location="cpu")
        return tensor_to_normalized_video(tensor, str(args.input_cache)), args.input_cache

    if not args.source:
        raise ValueError("Use --source, --input-keypoints, or --input-cache.")

    helper = KeypointSequenceDataset(
        args.split_file,
        args.keypoint_root,
        args.split,
        args.num_frames,
        limit=0,
        cache_root=args.cache_root,
    )
    cache_path = helper.cache_path_for_source(args.source)
    if cache_path and cache_path.exists():
        tensor = torch.load(cache_path, map_location="cpu")
        return tensor_to_normalized_video(tensor, args.source), cache_path

    keypoint_path = resolve_keypoint_path(args.source, args.keypoint_root, args.split)
    return normalize_video(keypoint_path), keypoint_path


def canvas(width: int, height: int, background: str) -> np.ndarray:
    if background == "white":
        return np.full((height, width, 3), 255, dtype=np.uint8)
    return np.zeros((height, width, 3), dtype=np.uint8)


def render_keypoints_only_video(
    normalized: dict,
    output_video: Path,
    width: int,
    height: int,
    fps: float,
    background: str,
    confidence_threshold: float,
    title: str,
) -> Path:
    intermediate = output_video.with_name(f"{output_video.stem}_mp4v.mp4")
    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {intermediate}")

    body_names = sorted({name for edge in BODY_EDGES for name in edge})
    face_names = sorted({name for edge in FACE_EDGES for name in edge})
    left_names = sorted({name for edge in LEFT_HAND_EDGES for name in edge})
    right_names = sorted({name for edge in RIGHT_HAND_EDGES for name in edge})
    text_color = (20, 20, 20) if background == "white" else (245, 245, 245)

    frames = normalized.get("frames") or []
    try:
        for index, frame in enumerate(frames):
            image = canvas(width, height, background)
            if frame.get("valid", True):
                draw_edges(image, frame, BODY_EDGES, BODY_COLOR, confidence_threshold)
                draw_edges(image, frame, FACE_EDGES, FACE_COLOR, confidence_threshold)
                draw_edges(image, frame, LEFT_HAND_EDGES, LEFT_COLOR, confidence_threshold)
                draw_edges(image, frame, RIGHT_HAND_EDGES, RIGHT_COLOR, confidence_threshold)
                draw_points(image, frame, body_names, BODY_COLOR, confidence_threshold)
                draw_points(image, frame, face_names, FACE_COLOR, confidence_threshold)
                draw_points(image, frame, left_names, LEFT_COLOR, confidence_threshold)
                draw_points(image, frame, right_names, RIGHT_COLOR, confidence_threshold)

            cv2.putText(
                image,
                f"{title}  frame {index + 1}/{len(frames)}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_color,
                2,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()

    transcode_browser_video(intermediate, output_video)
    intermediate.unlink(missing_ok=True)
    return output_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a keypoint-only MP4 for one sample.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--source", type=str, help="Split source path, e.g. a row from predictions CSV.")
    input_group.add_argument("--input-keypoints", type=Path, help="Raw RTMW keypoint JSON.")
    input_group.add_argument("--input-cache", type=Path, help="Cached keypoint tensor .pt.")
    parser.add_argument("--keypoint-root", type=Path, default=Path("dataset/keypoints_rtmw_l_27cls_s14v3t3_v14"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/keypoint_tensor_cache"))
    parser.add_argument("--split-file", type=Path, default=Path("test.txt"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--background", choices=["black", "white"], default="black")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--title", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized, source_path = load_normalized_from_args(args)
    title = args.title or Path(str(source_path or args.source)).stem
    output = render_keypoints_only_video(
        normalized,
        args.output,
        args.width,
        args.height,
        args.fps,
        args.background,
        args.confidence_threshold,
        title,
    )
    print(f"source={source_path or args.source}")
    print(f"frames={len(normalized.get('frames') or [])}")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
