"""Dataset and dynamic collation for schema-v1 variable keypoint caches."""
from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import Dataset

try:
    from src_neurosymbolic.variable_keypoint_cache import load_variable_cache
    from src_neurosymbolic.models.keypoint_dataset import read_split_file
except ImportError:  # Allow direct imports from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src_neurosymbolic.variable_keypoint_cache import load_variable_cache
    from src_neurosymbolic.models.keypoint_dataset import read_split_file


def variable_cache_path(source: str, cache_root: Path) -> Path:
    """Map a split source to variable_v1 while retaining its split-relative tree."""
    normalized = source.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    split_index = next(
        (i for i, part in enumerate(parts) if part.lower() in {"train", "val", "test"}),
        None,
    )
    if split_index is not None:
        relative = parts[split_index:]
    else:
        # Relative paths are already suitable; for absolute Drive paths retain
        # the class and filename when no conventional split component exists.
        relative = parts if not (normalized.startswith("/") or ":/" in normalized) else parts[-2:]
    rel_path = PurePosixPath(*relative).with_suffix(".pt")
    return Path(cache_root) / "variable_v1" / Path(*rel_path.parts)


class VariableKeypointDataset(Dataset):
    def __init__(self, split_file: Path, cache_root: Path, split: str,
                 max_frames: int = 256, limit: int | None = None) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.samples = read_split_file(Path(split_file))
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            self.samples = self.samples[:limit]
        self.cache_root = Path(cache_root)
        self.split = split
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        source, label = self.samples[index]
        if label < 0:
            raise ValueError(f"Invalid negative label {label} in split entry: {source}")
        path = variable_cache_path(source, self.cache_root)
        if not path.is_file():
            raise FileNotFoundError(f"Variable keypoint cache not found for split entry {source!r}: {path}")
        payload = load_variable_cache(path)
        keypoints, velocity, length = payload["keypoints"].float(), payload["velocity"].float(), payload["length"]
        if length <= 0:
            raise ValueError(f"Empty cache clip (length=0): {path}")
        if length > self.max_frames:
            raise ValueError(
                f"Clip has {length} frames, exceeding max_frames={self.max_frames}: {path}. "
                "Step 5 windowing is required for longer clips."
            )
        return {"keypoints": keypoints, "velocity": velocity,
                "length": torch.tensor(length, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long), "path": source}


def collate_variable_batch(items: list[dict], max_frames: int | None = None) -> dict:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    lengths = [int(item["length"]) for item in items]
    if any(length <= 0 for length in lengths):
        raise ValueError("All clips must have positive length")
    for item, length in zip(items, lengths):
        if item["keypoints"].shape != (length, 52, 3):
            raise ValueError(f"keypoints must have shape [{length},52,3]")
        if item["velocity"].shape != (length, 52, 2):
            raise ValueError(f"velocity must have shape [{length},52,2]")
    batch_length = max(lengths)
    if max_frames is not None and batch_length > max_frames:
        raise ValueError(f"Batch length {batch_length} exceeds max_frames={max_frames}")
    batch = len(items)
    keypoints = torch.zeros((batch, batch_length, 52, 3), dtype=items[0]["keypoints"].dtype)
    velocity = torch.zeros((batch, batch_length, 52, 2), dtype=items[0]["velocity"].dtype)
    mask = torch.ones((batch, batch_length), dtype=torch.bool)
    for i, (item, length) in enumerate(zip(items, lengths)):
        if item["keypoints"].dtype != keypoints.dtype or item["velocity"].dtype != velocity.dtype:
            raise ValueError("All clips in a batch must use consistent tensor dtypes")
        keypoints[i, :length] = item["keypoints"]
        velocity[i, :length] = item["velocity"]
        mask[i, :length] = False
    return {"keypoints": keypoints, "velocity": velocity,
            "padding_mask": mask, "lengths": torch.tensor(lengths, dtype=torch.long),
            "label": torch.stack([item["label"].long() for item in items]),
            "path": [item["path"] for item in items]}
