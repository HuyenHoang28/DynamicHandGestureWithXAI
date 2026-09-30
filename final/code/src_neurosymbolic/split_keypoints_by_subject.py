import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SUBJECT_RE = re.compile(r"(Subject\d+_[^_]+)")


def infer_subject(path: Path) -> str:
    match = SUBJECT_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer subject from filename: {path}")
    return match.group(1)


def load_label_ids(paths: list[Path]) -> dict[str, int]:
    label_ids = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().rsplit(maxsplit=2)
            if len(parts) != 3:
                continue
            source, _, label_text = parts
            try:
                label_id = int(label_text)
            except ValueError:
                continue
            segments = source.replace("\\", "/").split("/")
            for split_name in ("train", "val", "test"):
                if split_name in segments:
                    idx = segments.index(split_name)
                    if idx + 1 < len(segments):
                        label_ids.setdefault(segments[idx + 1], label_id)
                    break
    return label_ids


def frame_count(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        return len(data["frames"])
    return 0


def collect_samples(keypoint_root: Path, label_ids: dict[str, int], default_frames: int) -> list[dict]:
    samples = []
    seen = set()
    for path in sorted(keypoint_root.glob("*/*/*.json")):
        if path.name == "split_audit.json":
            continue
        label = path.parent.name
        if label not in label_ids:
            raise KeyError(f"Missing label id for label: {label}")
        subject = infer_subject(path)
        key = (label, path.name)
        if key in seen:
            continue
        seen.add(key)
        samples.append(
            {
                "path": path,
                "label": label,
                "label_id": label_ids[label],
                "subject": subject,
                "frames": default_frames,
            }
        )
    return samples


def write_split(path: Path, samples: list[dict]) -> None:
    lines = [
        f"{sample['path'].as_posix()} {sample['frames']} {sample['label_id']}"
        for sample in sorted(samples, key=lambda item: (item["label_id"], item["subject"], item["path"].name))
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def summarize(samples: list[dict]) -> dict:
    by_subject = Counter(sample["subject"] for sample in samples)
    by_label = Counter(sample["label"] for sample in samples)
    return {
        "video_count": len(samples),
        "subject_count": len(by_subject),
        "subjects": dict(sorted(by_subject.items())),
        "labels": dict(sorted(by_label.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test split files from keypoint JSON by subject.")
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-out", type=Path, default=Path("val.txt"))
    parser.add_argument("--test-out", type=Path, default=Path("test.txt"))
    parser.add_argument("--audit-out", type=Path, default=Path("scratch/subject_split_audit.json"))
    parser.add_argument("--label-source", type=Path, nargs="+", default=[Path("train.txt"), Path("val.txt"), Path("test.txt")])
    parser.add_argument("--test-subjects", nargs="+", required=True)
    parser.add_argument("--val-subjects", nargs="+", required=True)
    parser.add_argument("--default-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_ids = load_label_ids(args.label_source)
    samples = collect_samples(args.keypoint_root, label_ids, args.default_frames)

    test_subjects = set(args.test_subjects)
    val_subjects = set(args.val_subjects)
    overlap = test_subjects.intersection(val_subjects)
    if overlap:
        raise ValueError(f"Subjects cannot be both val and test: {sorted(overlap)}")

    splits = defaultdict(list)
    for sample in samples:
        if sample["subject"] in test_subjects:
            splits["test"].append(sample)
        elif sample["subject"] in val_subjects:
            splits["val"].append(sample)
        else:
            splits["train"].append(sample)

    write_split(args.train_out, splits["train"])
    write_split(args.val_out, splits["val"])
    write_split(args.test_out, splits["test"])

    audit = {
        "keypoint_root": str(args.keypoint_root),
        "test_subjects": sorted(test_subjects),
        "val_subjects": sorted(val_subjects),
        "train": summarize(splits["train"]),
        "val": summarize(splits["val"]),
        "test": summarize(splits["test"]),
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in ("train", "val", "test"):
        summary = audit[name]
        print(f"{name}: videos={summary['video_count']} subjects={summary['subject_count']}")
        print("  " + ", ".join(summary["subjects"].keys()))
    print(f"Saved audit: {args.audit_out}")


if __name__ == "__main__":
    main()
