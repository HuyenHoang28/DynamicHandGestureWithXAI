import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.models.keypoint_dataset import read_split_file, resolve_keypoint_path, keypoints_to_tensor
from src_neurosymbolic.pose.normalize_keypoints import normalize_video


def precompute_split(split_file: Path, keypoint_root: Path, cache_root: Path, split: str, num_frames: int) -> dict:
    samples = read_split_file(split_file)
    written = 0
    skipped = 0
    failed = 0

    for idx, (source, _) in enumerate(samples, start=1):
        try:
            keypoint_path = resolve_keypoint_path(source, keypoint_root, split)
            rel = keypoint_path.relative_to(keypoint_root)
            out_path = cache_root / f"T{num_frames}" / rel.with_suffix(".pt")
            if out_path.exists():
                skipped += 1
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tensor = keypoints_to_tensor(normalize_video(keypoint_path), num_frames)
                torch.save(tensor, out_path)
                written += 1
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {source}: {exc}")

        if idx % 500 == 0 or idx == len(samples):
            print(
                f"{split}: {idx}/{len(samples)} "
                f"written={written} skipped={skipped} failed={failed}"
            )

    return {"split": split, "total": len(samples), "written": written, "skipped": skipped, "failed": failed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute normalized keypoint tensors for faster training.")
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("data/keypoint_tensor_cache"))
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-file", type=Path, default=Path("val.txt"))
    parser.add_argument("--test-file", type=Path, default=Path("test.txt"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_files = {
        "train": args.train_file,
        "val": args.val_file,
        "test": args.test_file,
    }
    summaries = []
    for split in args.splits:
        summaries.append(
            precompute_split(split_files[split], args.keypoint_root, args.cache_root, split, args.num_frames)
        )
    print("Done.")
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
