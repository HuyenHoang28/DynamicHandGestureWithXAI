"""Benchmark MMPose whole-body extractors on a class-folder video set."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_PATH = ROOT / "keypoint_extractor" / "run_rtmw_splits_dual.py"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--models-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument(
        "--model-name", required=True, help="Named manifest entry; one model per process."
    )
    parser.add_argument(
        "--full-keypoints-classes",
        type=int,
        default=0,
        help="Also extract every decoded frame for the N shortest selected clips.",
    )
    return parser.parse_args()


def resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_extractor_module():
    spec = importlib.util.spec_from_file_location("rtmw_project_extractor", EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import extractor helpers: {EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_video(path: Path) -> tuple[bool, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return False, 0
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    complete = declared <= 0 or abs(decoded - declared) <= 1
    return decoded > 0 and complete, decoded


def select_videos(video_root: Path) -> tuple[list[dict], list[dict]]:
    selected, errors = [], []
    for class_dir in sorted(path for path in video_root.iterdir() if path.is_dir()):
        candidates = []
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            valid, frame_count = probe_video(path)
            if valid:
                candidates.append((frame_count, path.stat().st_size, path))
            else:
                errors.append({"stage": "decode_probe", "class": class_dir.name, "video": str(path)})
        if candidates:
            frame_count, _, path = min(candidates, key=lambda item: (item[0], item[1], item[2].name))
            selected.append({"class": class_dir.name, "video": path, "frame_count": frame_count})
        else:
            errors.append({"stage": "selection", "class": class_dir.name, "error": "no decodable video"})
    return selected, errors


def decode_uniform(path: Path, requested: int) -> tuple[list, list[int]]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        return [], []
    count = min(max(requested, 1), len(frames))
    if count == 1:
        indices = [len(frames) // 2]
    else:
        indices = sorted({round(i * (len(frames) - 1) / (count - 1)) for i in range(count)})
    return [frames[index] for index in indices], indices


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def parameter_count(obj) -> int:
    model = getattr(obj, "model", obj)
    if not hasattr(model, "parameters"):
        return 0
    return sum(parameter.numel() for parameter in model.parameters())


def infer_one(inferencer, frame) -> dict:
    results = list(inferencer(frame, return_vis=False))
    return results[0] if results else {"predictions": []}


def frame_metrics(records: list[dict], threshold: float) -> dict:
    detected = shoulder = left = right = both = 0
    confidences: list[float] = []
    jitters: list[float] = []
    previous: dict | None = None
    for record in records:
        instances = record.get("instances", [])
        points = instances[0].get("keypoints_by_name", {}) if instances else {}
        if points:
            detected += 1
        valid = {name: value for name, value in points.items() if float(value.get("score", 0)) >= threshold}
        confidences.extend(float(value["score"]) for value in valid.values())
        shoulder_ok = all(name in valid for name in ("left_shoulder", "right_shoulder"))
        left_ok, right_ok = "left_hand_root" in valid, "right_hand_root" in valid
        shoulder += int(shoulder_ok)
        left += int(left_ok)
        right += int(right_ok)
        both += int(left_ok and right_ok)
        if previous and shoulder_ok:
            left_xy = valid["left_shoulder"]["xy"]
            right_xy = valid["right_shoulder"]["xy"]
            scale = max(((left_xy[0] - right_xy[0]) ** 2 + (left_xy[1] - right_xy[1]) ** 2) ** 0.5, 1.0)
            distances = []
            for name in sorted(set(previous) & set(valid)):
                a, b = previous[name]["xy"], valid[name]["xy"]
                distances.append((((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5) / scale)
            if distances:
                jitters.append(statistics.mean(distances))
        previous = valid
    total = len(records) or 1
    return {
        "detection_coverage": detected / total,
        "shoulder_coverage": shoulder / total,
        "left_hand_root_coverage": left / total,
        "right_hand_root_coverage": right / total,
        "both_hand_root_coverage": both / total,
        "mean_valid_keypoint_confidence": statistics.mean(confidences) if confidences else 0.0,
        "temporal_normalized_displacement": statistics.mean(jitters) if jitters else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    video_root = resolve_from_root(args.video_root).resolve()
    models_path = resolve_from_root(args.models_json).resolve()
    output_dir = resolve_from_root(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = json.loads(models_path.read_text(encoding="utf-8"))
    if args.model_name:
        models = [model for model in models if model["name"] == args.model_name]
        if not models:
            raise ValueError(f"Model not found in manifest: {args.model_name}")
    extractor = load_extractor_module()
    mmpose_root = ROOT / ".venv" / "Lib" / "site-packages" / "mmpose" / ".mim"
    id2name = extractor.load_id2name(mmpose_root, mirror_swap=True)
    selected, errors = select_videos(video_root)
    selection_errors = [error for error in errors if error["stage"] == "selection"]
    if selection_errors:
        missing = ", ".join(error["class"] for error in selection_errors)
        raise RuntimeError(f"No complete decodable video for classes: {missing}")
    decoded = {}
    for item in selected:
        frames, indices = decode_uniform(item["video"], args.frames_per_video)
        decoded[item["class"]] = (frames, indices)
        item["sampled_indices"] = indices

    from mmpose.apis import MMPoseInferencer

    clip_rows: list[dict] = []
    summaries: list[dict] = []
    for model_spec in models:
        name = model_spec["name"]
        config = resolve_from_root(Path(model_spec["config"])).resolve()
        checkpoint = resolve_from_root(Path(model_spec["checkpoint"])).resolve()
        print(f"\n=== {name} ({args.device}) ===", flush=True)
        sampled_dir = output_dir / "keypoints" / name
        full_dir = output_dir / "keypoints_full" / name
        if any(sampled_dir.glob("*.json")) or any(full_dir.glob("*.json")):
            raise FileExistsError(
                f"Output already contains JSON for {name}; use a fresh --output-dir: {output_dir}"
            )
        started = time.perf_counter()
        inferencer = MMPoseInferencer(
            pose2d=str(config), pose2d_weights=str(checkpoint), device=args.device, show_progress=False
        )
        init_seconds = time.perf_counter() - started
        inner = getattr(inferencer, "inferencer", inferencer)
        pose_params = parameter_count(inner)
        detector_params = parameter_count(getattr(inner, "detector", None))
        warmed = False
        model_latencies: list[float] = []
        model_rows = []
        for item in selected:
            class_name = item["class"]
            frames, indices = decoded[class_name]
            records, latencies = [], []
            if frames and not warmed:
                infer_one(inferencer, frames[0])
                warmed = True
            for frame_id, frame in zip(indices, frames):
                try:
                    tick = time.perf_counter()
                    result = infer_one(inferencer, frame)
                    latencies.append((time.perf_counter() - tick) * 1000)
                    records.append(extractor.frame_to_record(result, frame_id, id2name))
                except Exception as exc:  # retain the failed frame in all denominators
                    records.append({"frame_id": frame_id, "instances": []})
                    errors.append({
                        "stage": "inference", "model": name, "class": class_name,
                        "frame_id": frame_id, "error": repr(exc),
                    })
            artifact = output_dir / "keypoints" / name / f"{class_name}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            metrics = frame_metrics(records, args.confidence_threshold)
            row = {
                "model": name,
                "class": class_name,
                "video": str(item["video"].relative_to(video_root)),
                "source_frames": item["frame_count"],
                "sampled_frames": len(records),
                **metrics,
                "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
                "p95_latency_ms": percentile(latencies, 0.95),
            }
            clip_rows.append(row)
            model_rows.append(row)
            model_latencies.extend(latencies)
            print(f"{name}: {class_name} ({len(records)} frames)", flush=True)
        full_subset = sorted(selected, key=lambda item: (item["frame_count"], item["class"]))[
            : max(0, args.full_keypoints_classes)
        ]
        for item in full_subset:
            class_name = item["class"]
            capture = cv2.VideoCapture(str(item["video"]))
            full_records = []
            frame_id = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                try:
                    result = infer_one(inferencer, frame)
                    full_records.append(extractor.frame_to_record(result, frame_id, id2name))
                except Exception as exc:
                    full_records.append({"frame_id": frame_id, "instances": []})
                    errors.append({
                        "stage": "full_inference", "model": name,
                        "class": class_name, "frame_id": frame_id, "error": repr(exc),
                    })
                frame_id += 1
            capture.release()
            artifact = output_dir / "keypoints_full" / name / f"{class_name}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps(full_records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"{name}: {class_name} full sequence ({len(full_records)} frames)", flush=True)
        weighted_frames = sum(row["sampled_frames"] for row in model_rows) or 1
        summaries.append({
            "model": name,
            "clips": len(model_rows),
            "frames": weighted_frames,
            "pose_parameters": pose_params,
            "detector_parameters": detector_params,
            "total_parameters": pose_params + detector_params,
            "checkpoint_mb": checkpoint.stat().st_size / (1024 * 1024),
            "initialization_seconds": init_seconds,
            "median_latency_ms": statistics.median(model_latencies) if model_latencies else 0.0,
            "p95_latency_ms": percentile(model_latencies, 0.95),
            "fps": 1000.0 / statistics.mean(model_latencies) if model_latencies else 0.0,
            **{key: sum(row[key] * row["sampled_frames"] for row in model_rows) / weighted_frames for key in (
                "detection_coverage", "shoulder_coverage", "left_hand_root_coverage",
                "right_hand_root_coverage", "both_hand_root_coverage",
                "mean_valid_keypoint_confidence", "temporal_normalized_displacement")},
            "official_hand_ap": model_spec.get("official_hand_ap"),
            "official_wholebody_ap": model_spec.get("official_wholebody_ap"),
            "downstream_accuracy": None,
        })
        del inferencer

    write_csv(output_dir / "per_clip.csv", clip_rows)
    write_csv(output_dir / "summary.csv", summaries)
    report = {
        "settings": {
            "video_root": str(video_root), "device": args.device,
            "frames_per_video": args.frames_per_video,
            "confidence_threshold": args.confidence_threshold,
            "selection": "shortest decodable clip per class; uniformly sampled frames",
            "full_keypoints_classes": args.full_keypoints_classes,
        },
        "selected_videos": [{**item, "video": str(item["video"])} for item in selected],
        "summary": summaries,
        "errors": errors,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote benchmark results to {output_dir}")
    runtime_errors = [error for error in errors if error["stage"] in {"inference", "full_inference"}]
    return 2 if runtime_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
