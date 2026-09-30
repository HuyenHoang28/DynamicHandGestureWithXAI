import argparse
import json
from pathlib import Path


REQUIRED_ANCHORS = ["left_shoulder", "right_shoulder"]


def load_frames(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of frames: {path}")
    return data


def first_instance_keypoints(frame: dict) -> dict:
    instances = frame.get("instances") or []
    if not instances:
        return {}
    return instances[0].get("keypoints_by_name") or {}


def get_xy_score(keypoints: dict, name: str) -> tuple[list[float] | None, float]:
    item = keypoints.get(name) or {}
    xy = item.get("xy")
    score = max(0.0, min(1.0, float(item.get("score") or 0.0)))
    if not xy or len(xy) != 2:
        return None, score
    return [float(xy[0]), float(xy[1])], score


def midpoint(a: list[float], b: list[float]) -> list[float]:
    return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]


def distance(a: list[float], b: list[float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def normalize_frame(frame: dict, min_anchor_score: float) -> dict:
    raw_keypoints = first_instance_keypoints(frame)
    left_shoulder, left_score = get_xy_score(raw_keypoints, "left_shoulder")
    right_shoulder, right_score = get_xy_score(raw_keypoints, "right_shoulder")

    valid = (
        left_shoulder is not None
        and right_shoulder is not None
        and left_score >= min_anchor_score
        and right_score >= min_anchor_score
    )

    if valid:
        origin = midpoint(left_shoulder, right_shoulder)
        scale = distance(left_shoulder, right_shoulder)
        if scale <= 1e-6:
            valid = False
    else:
        origin = [0.0, 0.0]
        scale = 1.0

    normalized = {}
    for name, item in raw_keypoints.items():
        xy = item.get("xy")
        if not xy or len(xy) != 2:
            continue
        score = max(0.0, min(1.0, float(item.get("score") or 0.0)))
        if valid:
            norm_xy = [(float(xy[0]) - origin[0]) / scale, (float(xy[1]) - origin[1]) / scale]
        else:
            norm_xy = [0.0, 0.0]
        normalized[name] = {"xy": norm_xy, "score": score}

    return {
        "frame_id": int(frame.get("frame_id", 0)),
        "valid": bool(valid),
        "origin_xy": origin,
        "scale": scale,
        "keypoints": normalized,
    }


def normalize_video(input_path: Path, min_anchor_score: float = 0.3) -> dict:
    frames = load_frames(input_path)
    return {
        "video_id": input_path.stem,
        "source_path": str(input_path),
        "normalization": {
            "origin": "shoulder_midpoint",
            "scale": "shoulder_width",
            "mirror_swap_required": True,
        },
        "frames": [normalize_frame(frame, min_anchor_score) for frame in frames],
    }


def write_json(data: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2 if pretty else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize 52-keypoint JSON by shoulder midpoint/width.")
    parser.add_argument("--input", type=Path, required=True, help="Input keypoint JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output normalized JSON.")
    parser.add_argument("--min-anchor-score", type=float, default=0.3)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = normalize_video(args.input, args.min_anchor_score)
    write_json(data, args.output, args.pretty)
    print(f"Saved normalized keypoints: {args.output}")
    print(f"Frames: {len(data['frames'])}")


if __name__ == "__main__":
    main()
