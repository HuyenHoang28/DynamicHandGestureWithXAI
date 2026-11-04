"""Extract the project's 52-keypoint layout from IPN clips with MediaPipe Holistic (2D, mức A).

The output keeps the RTMW JSON schema (``[{frame_id, instances: [{keypoints_by_name}]}]``) and the same
52 names, so tensors, instance graphs and models are unchanged.  The point mapping reuses
``ipn_hands_adapter`` (the IPN HandS conversion of MediaPipe pose + hands), plus two documented choices:

* ``left_wrist`` / ``right_wrist`` come from the hand wrist (hand landmark 0); when the hand is not
  detected the Holistic pose wrist (pose landmark 15/16) is used as a fallback (``--no-wrist-fallback``
  disables it to mimic the HandS adapter exactly).
* ``face-8`` is the midpoint of the pose mouth corners (pose landmarks 9 and 10), as in the adapter.

Scores are 1.0 for every detected point (as in the HandS adapter); missing points are omitted.
Left/right names are swapped by default (``--no-mirror-swap`` to disable), matching the RTMW extraction
for the horizontally mirrored IPN videos; verify the convention on a pilot against RTMW keypoints
(``scripts/compare_keypoint_sources.py``) before a full run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.pose.ipn_hands_adapter import (  # noqa: E402
    HAND_NAME_BY_INDEX,
    POSE_NAME_BY_INDEX,
    midpoint_point,
    point,
)

WRIST_POSE_INDEX = {"left": "15", "right": "16"}


def swap_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


def _landmarks(container: object) -> dict[str, list[float]]:
    """Holistic landmark list -> ``{"index": [x, y]}`` with normalised coordinates."""

    if container is None:
        return {}
    return {str(i): [float(lm.x), float(lm.y)] for i, lm in enumerate(container.landmark)}


def frame_keypoints(results: object, mirror_swap: bool = True, wrist_fallback: bool = True) -> dict[str, dict]:
    pose = _landmarks(getattr(results, "pose_landmarks", None))
    hands = {
        "left": _landmarks(getattr(results, "left_hand_landmarks", None)),
        "right": _landmarks(getattr(results, "right_hand_landmarks", None)),
    }
    keypoints: dict[str, dict] = {}
    for index, name in POSE_NAME_BY_INDEX.items():
        converted = point(pose.get(index))
        if converted is not None:
            keypoints[name] = converted
    lower_face = midpoint_point(pose.get("9"), pose.get("10"))
    if lower_face is not None:
        keypoints["face-8"] = lower_face
    for side, landmarks in hands.items():
        for index, name in HAND_NAME_BY_INDEX[side].items():
            converted = point(landmarks.get(index))
            if converted is not None:
                keypoints[name] = converted
        wrist = point(landmarks.get("0"))
        if wrist is None and wrist_fallback:
            wrist = point(pose.get(WRIST_POSE_INDEX[side]))
        if wrist is not None:
            keypoints[f"{side}_wrist"] = wrist
    if mirror_swap:
        keypoints = {swap_name(name): value for name, value in keypoints.items()}
    return keypoints


def extract_clip(video: Path, output: Path, holistic: object, mirror_swap: bool, wrist_fallback: bool) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    frames = []
    frame_id = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        results = holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        keypoints = frame_keypoints(results, mirror_swap=mirror_swap, wrist_fallback=wrist_fallback)
        frames.append({"frame_id": frame_id, "instances": [{"keypoints_by_name": keypoints}] if keypoints else []})
        frame_id += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(frames), encoding="utf-8")
    tmp.replace(output)
    return len(frames)


def _worker(task: tuple) -> tuple[str, int | str]:
    import mediapipe as mp

    videos, output_root, mirror_swap, wrist_fallback, complexity = task
    done: list[tuple[str, int | str]] = []
    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=complexity, smooth_landmarks=True,
        refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as holistic:
        for count, (video, relative) in enumerate(videos, 1):
            output = Path(output_root) / relative
            try:
                done.append((relative.as_posix(), extract_clip(Path(video), output, holistic, mirror_swap, wrist_fallback)))
            except Exception as exc:  # keep going, report at the end
                done.append((relative.as_posix(), f"ERROR {exc}"))
            if count % 25 == 0:
                print(f"[worker] {count}/{len(videos)} clips", flush=True)
    return done  # type: ignore[return-value]


def collect_videos(clip_roots: list[Path], list_file: Path | None) -> list[tuple[Path, Path]]:
    wanted: set[str] | None = None
    if list_file is not None:
        wanted = set()
        for line in list_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                wanted.add(Path(line.split()[0].replace("\\", "/")).stem)
    videos: dict[str, tuple[Path, Path]] = {}
    for root in clip_roots:
        for video in sorted(root.rglob("*.avi")):
            if wanted is not None and video.stem not in wanted:
                continue
            relative = video.relative_to(root).with_suffix(".json")
            videos.setdefault(relative.as_posix(), (video, relative))
    return list(videos.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-root", type=Path, nargs="+", required=True,
                        help="Directories holding <split>/<label>/<subject>/<clip>.avi; the first root wins on duplicates.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, help="Optional split list; only clips whose stem is listed are processed.")
    parser.add_argument("--limit", type=int, help="Process only the first N clips (pilot).")
    parser.add_argument("--stride", type=int, default=1, help="With --limit, take every k-th clip for a spread sample.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model-complexity", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--no-mirror-swap", action="store_true")
    parser.add_argument("--no-wrist-fallback", action="store_true")
    args = parser.parse_args(argv)

    videos = collect_videos(args.clips_root, args.list_file)
    if args.limit:
        videos = videos[:: max(1, args.stride)][: args.limit]
    todo = [(v, r) for v, r in videos if not (args.output_root / r).exists()]
    print(f"clips={len(videos)} todo={len(todo)} skipped_existing={len(videos) - len(todo)}", flush=True)
    if not todo:
        return 0
    shards = [todo[i :: args.workers] for i in range(args.workers)]
    tasks = [(shard, str(args.output_root), not args.no_mirror_swap, not args.no_wrist_fallback, args.model_complexity)
             for shard in shards if shard]
    errors = 0
    written = 0
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        for batch in pool.map(_worker, tasks):
            for name, outcome in batch:
                if isinstance(outcome, str):
                    errors += 1
                    print(name, outcome, flush=True)
                else:
                    written += 1
    print(f"written={written} errors={errors}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
