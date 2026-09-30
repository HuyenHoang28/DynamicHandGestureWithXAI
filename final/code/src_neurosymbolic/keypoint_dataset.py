import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src_neurosymbolic.pose.normalize_keypoints import normalize_video


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "face-8",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand_root",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb4",
    "left_forefinger1",
    "left_forefinger2",
    "left_forefinger3",
    "left_forefinger4",
    "left_middle_finger1",
    "left_middle_finger2",
    "left_middle_finger3",
    "left_middle_finger4",
    "left_ring_finger1",
    "left_ring_finger2",
    "left_ring_finger3",
    "left_ring_finger4",
    "left_pinky_finger1",
    "left_pinky_finger2",
    "left_pinky_finger3",
    "left_pinky_finger4",
    "right_hand_root",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb4",
    "right_forefinger1",
    "right_forefinger2",
    "right_forefinger3",
    "right_forefinger4",
    "right_middle_finger1",
    "right_middle_finger2",
    "right_middle_finger3",
    "right_middle_finger4",
    "right_ring_finger1",
    "right_ring_finger2",
    "right_ring_finger3",
    "right_ring_finger4",
    "right_pinky_finger1",
    "right_pinky_finger2",
    "right_pinky_finger3",
    "right_pinky_finger4",
]


def read_split_file(path: Path) -> list[tuple[str, int]]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(maxsplit=2)
        if len(parts) != 3:
            continue
        source, _, label_text = parts
        samples.append((source, int(label_text)))
    return samples


def label_from_source(source: str, split: str) -> str | None:
    parts = source.replace("\\", "/").split("/")
    if split in parts:
        idx = parts.index(split)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    for known_split in ("train", "val", "test"):
        if known_split in parts:
            idx = parts.index(known_split)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def resolve_keypoint_path(source: str, keypoint_root: Path, split: str) -> Path:
    import re
    direct = Path(source)
    if direct.exists():
        return direct

    label = label_from_source(source, split)
    if label is None:
        raise FileNotFoundError(f"Cannot infer label from split entry: {source}")

    parts = source.replace("\\", "/").split("/")
    filename = parts[-1]
    stem = Path(filename).stem

    # Extract subject if possible (e.g. Subject11_HoaiLinh)
    subject_dir = None
    match = re.search(r"Subject\d+_[A-Za-z0-9]+", stem)
    if match:
        subject_dir = match.group(0)

    # Extract actual filename by stripping "{label}_{subject}_" if present
    actual_filename = filename
    if subject_dir:
        prefix = f"{label}_{subject_dir}_"
        if filename.startswith(prefix):
            actual_filename = filename[len(prefix):]

    # Build potential search paths
    candidates = []
    if subject_dir:
        candidates.extend([
            # With subject directory and stripped filename (Remote server structure)
            keypoint_root / split / label / subject_dir / actual_filename,
            keypoint_root / label / subject_dir / actual_filename,
            # With subject directory and original filename
            keypoint_root / split / label / subject_dir / filename,
            keypoint_root / label / subject_dir / filename,
        ])
    candidates.extend([
        keypoint_root / split / label / filename,
        keypoint_root / label / filename,
    ])

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Keypoint JSON not found for split entry: {source}")


def keypoints_to_tensor(normalized: dict, num_frames: int) -> torch.Tensor:
    frames = normalized.get("frames", [])
    if not frames:
        return torch.zeros(num_frames, len(KEYPOINT_NAMES), 3, dtype=torch.float32)

    if len(frames) == num_frames:
        indices = list(range(num_frames))
    elif len(frames) > num_frames:
        indices = torch.linspace(0, len(frames) - 1, steps=num_frames).round().long().tolist()
    else:
        indices = list(range(len(frames))) + [len(frames) - 1] * (num_frames - len(frames))

    tensor = torch.zeros(num_frames, len(KEYPOINT_NAMES), 3, dtype=torch.float32)
    for out_idx, frame_idx in enumerate(indices):
        frame = frames[frame_idx]
        points = frame.get("keypoints", {})
        for kp_idx, name in enumerate(KEYPOINT_NAMES):
            item = points.get(name) or {}
            xy = item.get("xy")
            if xy and len(xy) == 2:
                tensor[out_idx, kp_idx, 0] = float(xy[0])
                tensor[out_idx, kp_idx, 1] = float(xy[1])
            tensor[out_idx, kp_idx, 2] = float(item.get("score") or 0.0)
    return tensor


class KeypointSequenceDataset(Dataset):
    def __init__(
        self,
        split_file: Path,
        keypoint_root: Path,
        split: str,
        num_frames: int = 64,
        limit: int | None = None,
        cache_root: Path | None = None,
        instance_graph_root: Path | None = None,
    ) -> None:
        self.split_file = split_file
        self.keypoint_root = keypoint_root
        self.split = split
        self.num_frames = num_frames
        self.cache_root = cache_root
        self.instance_graph_root = instance_graph_root
        self.samples = read_split_file(split_file)
        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        source, label = self.samples[index]
        path = Path(source)
        cache_path = self.cache_path_for_source(source)
        if cache_path and cache_path.exists():
            keypoints = torch.load(cache_path, map_location="cpu")
        else:
            path = resolve_keypoint_path(source, self.keypoint_root, self.split)
            cache_path = self.cache_path_for(path)
            if cache_path and cache_path.exists():
                keypoints = torch.load(cache_path, map_location="cpu")
            else:
                normalized = normalize_video(path)
                keypoints = keypoints_to_tensor(normalized, self.num_frames)
            
        item = {
            "keypoints": keypoints,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }
        
        # Load Instance Graph if configured
        if self.instance_graph_root:
            ig_path = self.instance_graph_path_for_source(source)
            if not ig_path or not ig_path.exists():
                ig_path = self.instance_graph_path_for(path)
            if ig_path and ig_path.exists():
                item["instance_graph_json"] = ig_path.read_text(encoding="utf-8")
            else:
                # Return empty graph
                item["instance_graph_json"] = '{"nodes": [], "edges": []}'
                
        return item

    def instance_graph_path_for(self, keypoint_path: Path) -> Path | None:
        if self.instance_graph_root is None:
            return None
        try:
            rel = keypoint_path.relative_to(self.keypoint_root)
        except ValueError:
            rel = Path(keypoint_path.name)
        return self.instance_graph_root / self.split / rel.with_suffix(".json")

    def instance_graph_path_for_source(self, source: str) -> Path | None:
        if self.instance_graph_root is None:
            return None
        candidates = []
        source_path = Path(source)
        try:
            rel = source_path.relative_to(self.keypoint_root)
        except ValueError:
            source_text = source.replace("\\", "/")
            root_text = self.keypoint_root.as_posix().rstrip("/")
            if source_text.startswith(root_text + "/"):
                rel = Path(source_text[len(root_text) + 1 :])
            else:
                rel = source_path
        candidates.append(self.instance_graph_root / rel.with_suffix(".json"))
        candidates.append(self.instance_graph_root / self.split / rel.with_suffix(".json"))

        parts = source.replace("\\", "/").split("/")
        for split_name in ("train", "val", "test"):
            if split_name not in parts:
                continue
            idx = parts.index(split_name)
            if idx + 2 >= len(parts):
                continue
            label = parts[idx + 1]
            filename = Path(parts[-1]).stem
            prefix = f"{label}_"
            rest = filename[len(prefix) :] if filename.startswith(prefix) else filename
            tokens = rest.split("_")
            if len(tokens) >= 3 and tokens[0].startswith("Subject"):
                subject = "_".join(tokens[:2])
                clip_name = "_".join(tokens[2:])
                candidates.append(self.instance_graph_root / label / subject / f"{clip_name}.json")
                candidates.append(self.instance_graph_root / self.split / label / subject / f"{clip_name}.json")
            candidates.append(self.instance_graph_root / label / f"{filename}.json")
            candidates.append(self.instance_graph_root / self.split / label / f"{filename}.json")

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    def cache_path_for(self, keypoint_path: Path) -> Path | None:
        if self.cache_root is None:
            return None
        try:
            rel = keypoint_path.relative_to(self.keypoint_root)
        except ValueError:
            rel = Path(keypoint_path.name)
        return self.cache_root / f"T{self.num_frames}" / rel.with_suffix(".pt")

    def cache_path_for_source(self, source: str) -> Path | None:
        if self.cache_root is None:
            return None
        source_path = Path(source)
        base = self.cache_root / f"T{self.num_frames}"
        candidates = []
        try:
            rel = source_path.relative_to(self.keypoint_root)
        except ValueError:
            source_text = source.replace("\\", "/")
            root_text = self.keypoint_root.as_posix().rstrip("/")
            if source_text.startswith(root_text + "/"):
                rel = Path(source_text[len(root_text) + 1 :])
            else:
                rel = source_path
        candidates.append(base / rel.with_suffix(".pt"))

        parts = source.replace("\\", "/").split("/")
        for split_name in ("train", "val", "test"):
            if split_name not in parts:
                continue
            idx = parts.index(split_name)
            if idx + 2 >= len(parts):
                continue
            label = parts[idx + 1]
            filename = Path(parts[-1]).stem
            prefix = f"{label}_"
            rest = filename[len(prefix) :] if filename.startswith(prefix) else filename
            tokens = rest.split("_")
            if len(tokens) >= 3 and tokens[0].startswith("Subject"):
                subject = "_".join(tokens[:2])
                clip_name = "_".join(tokens[2:])
                candidates.append(base / label / subject / f"{clip_name}.pt")
            candidates.append(base / label / f"{filename}.pt")

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None


def load_label_names(split_file: Path) -> dict[int, str]:
    names = {}
    for source, label in read_split_file(split_file):
        label_name = label_from_source(source, "train")
        if label_name is not None:
            names.setdefault(label, label_name)
    return names
