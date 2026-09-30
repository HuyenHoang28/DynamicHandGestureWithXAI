from pathlib import Path

import torch
from torch.utils.data import Dataset


def read_split_file(path: Path) -> list[tuple[str, int]]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(maxsplit=2)
        if len(parts) == 3:
            samples.append((parts[0], int(parts[2])))
    return samples


def source_info(source: str, fallback_split: str) -> tuple[str, str | None, str]:
    parts = source.replace("\\", "/").split("/")
    split = fallback_split
    label = None
    for candidate in ("train", "val", "test"):
        if candidate in parts:
            split = candidate
            idx = parts.index(candidate)
            if idx + 1 < len(parts):
                label = parts[idx + 1]
            break
    return split, label, Path(parts[-1]).stem


def label_names_from_split(split_file: Path) -> dict[int, str]:
    names = {}
    for source, label in read_split_file(split_file):
        _, label_name, _ = source_info(source, "train")
        if label_name is not None:
            names.setdefault(label, label_name)
    return dict(sorted(names.items()))


def cache_path_for_source(source: str, cache_root: Path, num_frames: int, fallback_split: str) -> Path | None:
    base = cache_root / f"T{num_frames}"
    split, label, filename = source_info(source, fallback_split)
    if label is None:
        return None
    candidates = [
        base / split / label / f"{filename}.pt",
        base / label / f"{filename}.pt",
    ]
    prefix = f"{label}_"
    rest = filename[len(prefix):] if filename.startswith(prefix) else filename
    tokens = rest.split("_")
    if len(tokens) >= 3 and tokens[0].startswith("Subject"):
        subject = "_".join(tokens[:2])
        clip_name = "_".join(tokens[2:])
        candidates.extend(
            [
                base / split / label / subject / f"{clip_name}.pt",
                base / label / subject / f"{clip_name}.pt",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for pattern in [
        f"{split}/{label}/**/{filename}.pt",
        f"{label}/**/{filename}.pt",
        f"**/{filename}.pt",
    ]:
        hits = list(base.glob(pattern))
        if hits:
            return hits[0]
    return None


class NeuralOnlyKeypointDataset(Dataset):
    def __init__(
        self,
        split_file: Path,
        cache_root: Path,
        split: str,
        num_frames: int = 64,
        limit: int | None = None,
    ) -> None:
        self.samples = read_split_file(split_file)
        if limit is not None:
            self.samples = self.samples[:limit]
        self.cache_root = cache_root
        self.split = split
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        source, label = self.samples[index]
        path = cache_path_for_source(source, self.cache_root, self.num_frames, self.split)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Cache tensor not found for split entry: {source}")
        return {
            "keypoints": torch.load(path, map_location="cpu").float(),
            "label": torch.tensor(label, dtype=torch.long),
            "path": source,
        }


def collate_batch(items: list[dict]) -> dict:
    return {
        "keypoints": torch.stack([item["keypoints"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "path": [item["path"] for item in items],
    }
