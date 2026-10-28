"""Variable length keypoint cache format, separate from legacy T64 tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.models.keypoint_dataset import (  # noqa: E402
    KEYPOINT_NAMES,
    read_split_file,
    resolve_keypoint_path,
)
from src_neurosymbolic.pose.normalize_keypoints import normalize_video  # noqa: E402

SCHEMA_VERSION = 1
MAX_FRAMES_METADATA = 256


def normalized_frames_to_tensor(normalized: dict) -> torch.Tensor:
    """Convert every normalized input frame, preserving order and count."""
    frames = normalized.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("normalized frames must be a list")
    result = torch.zeros((len(frames), len(KEYPOINT_NAMES), 3), dtype=torch.float32)
    for frame_index, frame in enumerate(frames):
        frame_valid = bool(frame.get("valid", True))
        points = frame.get("keypoints", {})
        for keypoint_index, name in enumerate(KEYPOINT_NAMES):
            item = points.get(name) or {}
            xy = item.get("xy")
            if xy is not None:
                if len(xy) != 2:
                    raise ValueError(f"{name} xy must have two coordinates")
                result[frame_index, keypoint_index, 0] = float(xy[0])
                result[frame_index, keypoint_index, 1] = float(xy[1])
            result[frame_index, keypoint_index, 2] = (
                float(item.get("score") or 0.0) if frame_valid else 0.0
            )
    if not torch.isfinite(result).all():
        raise ValueError("normalized keypoints contain non-finite values")
    return result


def compute_velocity(keypoints: torch.Tensor) -> torch.Tensor:
    """Frame deltas on adjacent real frames; invalid-confidence pairs are zero."""
    if keypoints.ndim != 3 or tuple(keypoints.shape[1:]) != (len(KEYPOINT_NAMES), 3):
        raise ValueError("keypoints must have shape [N, 52, 3]")
    velocity = torch.zeros((*keypoints.shape[:2], 2), dtype=keypoints.dtype)
    if keypoints.shape[0] > 1:
        valid = (keypoints[1:, :, 2] > 0) & (keypoints[:-1, :, 2] > 0)
        deltas = keypoints[1:, :, :2] - keypoints[:-1, :, :2]
        velocity[1:] = torch.where(valid.unsqueeze(-1), deltas, 0)
    return velocity


def build_variable_cache(
    normalized: dict,
    max_frames: int = MAX_FRAMES_METADATA,
    *,
    source_size: int | None = None,
    source_mtime_ns: int | None = None,
    source_sha256: str | None = None,
) -> dict:
    if max_frames <= 0:
        raise ValueError("max_frames metadata threshold must be positive")
    keypoints = normalized_frames_to_tensor(normalized)
    length = int(keypoints.shape[0])
    payload = {
        "keypoints": keypoints,
        "velocity": compute_velocity(keypoints),
        "length": length,
        "source_frames": length,
        "schema_version": SCHEMA_VERSION,
        "exceeds_max_frames": length > max_frames,
    }
    if source_size is not None:
        payload["source_size"] = source_size
    if source_mtime_ns is not None:
        payload["source_mtime_ns"] = source_mtime_ns
    if source_sha256 is not None:
        payload["source_sha256"] = source_sha256
    return payload


def validate_variable_cache(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("cache payload must be a dictionary")
    required = {"keypoints", "velocity", "length", "source_frames", "schema_version"}
    if not required.issubset(payload):
        raise ValueError(f"cache missing required fields: {sorted(required - set(payload))}")
    kp, vel = payload["keypoints"], payload["velocity"]
    if not isinstance(kp, torch.Tensor) or not isinstance(vel, torch.Tensor):
        raise ValueError("keypoints and velocity must be tensors")
    if kp.ndim != 3 or tuple(kp.shape[1:]) != (52, 3):
        raise ValueError("keypoints must have shape [N, 52, 3]")
    if vel.shape != (kp.shape[0], 52, 2):
        raise ValueError("velocity must have shape [N, 52, 2]")
    if not kp.is_floating_point() or not vel.is_floating_point():
        raise ValueError("keypoints and velocity must be floating tensors")
    if not torch.isfinite(kp).all() or not torch.isfinite(vel).all():
        raise ValueError("cache tensors contain non-finite values")
    if type(payload["length"]) is not int or payload["length"] != kp.shape[0]:
        raise ValueError("length does not match keypoints")
    if type(payload["source_frames"]) is not int or payload["source_frames"] != kp.shape[0]:
        raise ValueError("source_frames does not match keypoints")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if "exceeds_max_frames" in payload and type(payload["exceeds_max_frames"]) is not bool:
        raise ValueError("exceeds_max_frames must be boolean")
    for field in ("source_size", "source_mtime_ns"):
        if field in payload and (type(payload[field]) is not int or payload[field] < 0):
            raise ValueError(f"{field} must be a non-negative integer")
    if "source_sha256" in payload:
        digest = payload["source_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    return payload


def save_variable_cache(payload: dict, path: Path) -> None:
    validate_variable_cache(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_variable_cache(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # Fail closed rather than permit unrestricted pickle loading.
        raise RuntimeError("A PyTorch version with torch.load(weights_only=True) is required") from exc
    return validate_variable_cache(payload)


def source_fingerprint(source_path: Path) -> dict:
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = source_path.stat()
    return {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha256": digest.hexdigest(),
    }


def cache_matches_source(payload: dict, source_path: Path) -> bool:
    """Return whether structural validation and the saved source fingerprint match."""
    validate_variable_cache(payload)
    fingerprint = source_fingerprint(source_path)
    return all(payload.get(field) == value for field, value in fingerprint.items())


def precompute_manifest(manifest: Path, keypoint_root: Path, output_root: Path, split: str, max_frames: int = MAX_FRAMES_METADATA) -> dict:
    written = skipped = failed = 0
    canonical_root = keypoint_root.resolve()
    for source, _label in read_split_file(manifest):
        try:
            source_path = resolve_keypoint_path(source, keypoint_root, split).resolve()
            try:
                relative = source_path.relative_to(canonical_root)
            except ValueError as exc:
                raise ValueError(f"Resolved source is outside keypoint root: {source_path}") from exc
            out_path = output_root / "variable_v1" / relative.with_suffix(".pt")
            if out_path.exists():
                try:
                    cached = load_variable_cache(out_path)
                    threshold_matches = cached.get("exceeds_max_frames") == (
                        cached["length"] > max_frames
                    )
                    if cache_matches_source(cached, source_path) and threshold_matches:
                        skipped += 1
                        continue
                except (OSError, RuntimeError, ValueError):
                    pass
            fingerprint = source_fingerprint(source_path)
            payload = build_variable_cache(
                normalize_video(source_path),
                max_frames=max_frames,
                **fingerprint,
            )
            save_variable_cache(payload, out_path)
            written += 1
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {source}: {exc}")
    return {"split": split, "written": written, "skipped": skipped, "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute schema-v1 variable length keypoint caches.")
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True, help="Repeat as --manifest split=path")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_METADATA, help="Metadata threshold only; clips are never truncated")
    args = parser.parse_args()
    for entry in args.manifest:
        if "=" not in str(entry):
            parser.error("--manifest must be split=path; repeat for each split")
    summaries = []
    for entry in args.manifest:
        split, path = str(entry).split("=", 1)
        if split not in {"train", "val", "test"}:
            parser.error(f"unsupported split: {split}")
        summaries.append(precompute_manifest(Path(path), args.keypoint_root, args.output_root, split, args.max_frames))
    print(json.dumps(summaries, indent=2))
    return 1 if any(summary["failed"] for summary in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
