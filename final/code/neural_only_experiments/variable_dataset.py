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


def window_spans(length: int, max_frames: int = 256, stride: int = 128) -> list[tuple[int, int]]:
    """Return deterministic, covering half-open spans for one clip."""
    if length <= 0 or max_frames <= 0 or stride <= 0:
        raise ValueError("length, max_frames, and stride must be positive")
    if length <= max_frames:
        return [(0, length)]
    starts = list(range(0, length - max_frames + 1, stride))
    starts.append(length - max_frames)
    starts = list(dict.fromkeys(starts))
    return [(start, start + max_frames) for start in starts]


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


class WindowedVariableKeypointDataset(Dataset):
    """Flatten each manifest clip into deterministic, overlapping windows."""
    def __init__(self, split_file: Path, cache_root: Path, split: str,
                 max_frames: int = 256, stride: int = 128, limit: int | None = None) -> None:
        if max_frames <= 0 or stride <= 0:
            raise ValueError("max_frames and stride must be positive")
        samples = read_split_file(Path(split_file))
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            samples = samples[:limit]
        self.cache_root, self.split = Path(cache_root), split
        self.max_frames, self.stride = max_frames, stride
        self.samples = samples
        self.windows: list[tuple[int, int, int, int, int]] = []
        for clip_index, (source, label) in enumerate(samples):
            if label < 0:
                raise ValueError(f"Invalid negative label {label} in split entry: {source}")
            path = variable_cache_path(source, self.cache_root)
            if not path.is_file():
                raise FileNotFoundError(f"Variable keypoint cache not found for split entry {source!r}: {path}")
            payload = load_variable_cache(path)
            length = int(payload["length"])
            if length <= 0:
                raise ValueError(f"Empty cache clip (length=0): {path}")
            kp, vel = payload["keypoints"], payload["velocity"]
            if kp.shape != (length, 52, 3) or vel.shape != (length, 52, 2):
                raise ValueError(f"Invalid variable cache schema for {path}")
            spans = window_spans(length, max_frames, stride)
            for window_index, (start, end) in enumerate(spans):
                self.windows.append((clip_index, window_index, len(spans), start, end))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        clip_index, window_index, window_count, start, end = self.windows[index]
        source, label = self.samples[clip_index]
        path = variable_cache_path(source, self.cache_root)
        payload = load_variable_cache(path)
        keypoints = payload["keypoints"].float()[start:end]
        velocity = payload["velocity"].float()[start:end].clone()
        velocity[0].zero_()
        return {"keypoints": keypoints, "velocity": velocity,
                "length": torch.tensor(end - start, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long), "path": source,
                "source": source, "clip_id": source, "split": self.split,
                "clip_index": clip_index, "window_index": window_index,
                "window_count": window_count, "start": start, "end": end,
                "sample_weight": 1.0 / window_count}


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
    result = {"keypoints": keypoints, "velocity": velocity,
            "padding_mask": mask, "lengths": torch.tensor(lengths, dtype=torch.long),
            "label": torch.stack([item["label"].long() for item in items]),
            "path": [item["path"] for item in items]}
    # Preserve the legacy output shape while carrying window metadata when present.
    metadata = ("source", "clip_id", "split", "clip_index", "window_index",
                "window_count", "start", "end")
    if any("window_index" in item for item in items):
        for key in metadata:
            if not all(key in item for item in items):
                raise ValueError(f"Window batch is missing metadata field {key}")
            result[key] = [item[key] for item in items]
        weights = [float(item.get("sample_weight", 1.0)) for item in items]
        result["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
    return result


def aggregate_window_logits(logits: torch.Tensor, metadata: dict | None = None, *,
                            labels=None, sources=None, splits=None, clip_indices=None,
                            window_indices=None, window_counts=None, require_complete: bool = True,
                            reduction: str = "mean") -> dict:
    """Aggregate windows (possibly from shuffled batches) into ordered clip logits."""
    if reduction != "mean":
        raise ValueError("Only mean logit aggregation is supported")
    metadata = metadata or {}
    def field(arg, name):
        return arg if arg is not None else metadata.get(name)
    labels = field(labels, "label"); sources = field(sources, "source") or metadata.get("path")
    splits = field(splits, "split"); clip_indices = field(clip_indices, "clip_index")
    window_indices = field(window_indices, "window_index"); window_counts = field(window_counts, "window_count")
    required = (labels, sources, clip_indices, window_indices, window_counts)
    if any(value is None for value in required):
        raise ValueError("Aggregation requires label, source, clip_index, window_index, and window_count")
    n = logits.shape[0]
    arrays = [labels, sources, clip_indices, window_indices, window_counts]
    if any(len(value) != n for value in arrays) or (splits is not None and len(splits) != n):
        raise ValueError("Logits and window metadata lengths differ")
    groups = {}
    for i in range(n):
        split = splits[i] if splits is not None else None
        key = (split, sources[i])
        label, clip_index, wi, wc = int(labels[i]), int(clip_indices[i]), int(window_indices[i]), int(window_counts[i])
        if wc <= 0 or not 0 <= wi < wc:
            raise ValueError(f"Invalid window index/count for {sources[i]}")
        group = groups.setdefault(key, {"label": label, "clip_index": clip_index, "count": wc, "windows": {}})
        if group["label"] != label:
            raise ValueError(f"Inconsistent labels for clip {sources[i]}")
        if group["clip_index"] != clip_index or group["count"] != wc:
            raise ValueError(f"Inconsistent window metadata for clip {sources[i]}")
        if wi in group["windows"]:
            raise ValueError(f"Duplicate window index {wi} for clip {sources[i]}")
        group["windows"][wi] = logits[i]
    rows = []
    for (split, source), group in groups.items():
        got = group["windows"]
        if require_complete and set(got) != set(range(group["count"])):
            raise ValueError(f"Missing windows for clip {source}")
        rows.append((group["clip_index"], split, source, group["label"], torch.stack([got[k] for k in sorted(got)]).mean(0)))
    rows.sort(key=lambda row: (row[0], str(row[1]), row[2]))
    return {"logits": torch.stack([r[4] for r in rows]) if rows else logits.new_empty((0, *logits.shape[1:])),
            "labels": torch.tensor([r[3] for r in rows], dtype=torch.long),
            "paths": [r[2] for r in rows], "splits": [r[1] for r in rows],
            "clip_indices": [r[0] for r in rows]}
