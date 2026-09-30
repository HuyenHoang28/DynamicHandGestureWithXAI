import argparse
from pathlib import Path

import run_rtmw_splits_dual as extractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RTMW-L keypoints from one video into one JSON file."
    )
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mmpose-root", type=Path)
    parser.add_argument("--pose-config", type=Path)
    parser.add_argument("--pose-weights", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--no-mirror-swap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    mmpose_root = (args.mmpose_root or repo_root / "mmpose").resolve()
    pose_config = (args.pose_config or extractor.default_pose_config(mmpose_root)).resolve()
    pose_weights = (
        args.pose_weights or extractor.default_pose_weights(repo_root)
    ).resolve()

    extractor.require_file(args.input_video, "Input video")
    extractor.require_file(pose_config, "Pose config")
    extractor.require_file(pose_weights, "Pose weights")

    id2name = extractor.load_id2name(
        mmpose_root,
        mirror_swap=not args.no_mirror_swap,
    )
    inferencer = extractor.make_inferencer(pose_config, pose_weights, args.device)
    extractor.extract_video(
        args.input_video.resolve(),
        args.output_json.resolve(),
        inferencer,
        id2name,
        args.pretty,
    )
    print(f"Saved keypoints: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
