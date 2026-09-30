import argparse
import sys
import json
from pathlib import Path
import traceback
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.models.keypoint_dataset import KEYPOINT_NAMES, KeypointSequenceDataset, read_split_file, resolve_keypoint_path
from src_neurosymbolic.pose.normalize_keypoints import normalize_video
from src_neurosymbolic.predicates.extract_predicates import extract_predicates
from src_neurosymbolic.events.build_events import build_events
from src_neurosymbolic.kg.build_instance_graph import build_instance_graph


def tensor_to_normalized_video(tensor: torch.Tensor, source_path: str) -> dict:
    frames = []
    for frame_id in range(tensor.shape[0]):
        frame_points = {}
        for idx, name in enumerate(KEYPOINT_NAMES):
            x, y, score = tensor[frame_id, idx].tolist()
            frame_points[name] = {"xy": [float(x), float(y)], "score": float(score)}
        frames.append({"frame_id": frame_id, "valid": True, "keypoints": frame_points})
    return {"video_id": Path(source_path).stem, "source_path": source_path, "frames": frames}


def output_path_for_source(source: str, keypoint_root: Path, output_root: Path, split: str) -> Path:
    source_path = Path(source)
    candidates = []
    try:
        rel = source_path.relative_to(keypoint_root)
    except ValueError:
        source_text = source.replace("\\", "/")
        root_text = keypoint_root.as_posix().rstrip("/")
        if source_text.startswith(root_text + "/"):
            rel = Path(source_text[len(root_text) + 1 :])
        else:
            rel = source_path
    candidates.append(output_root / rel.with_suffix(".json"))

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
            return output_root / label / subject / f"{clip_name}.json"
    return candidates[0]


def precompute_split(split_file: Path, keypoint_root: Path, output_root: Path, split: str, cache_root: Path | None, num_frames: int, limit: int | None) -> dict:
    samples = read_split_file(split_file)
    if limit is not None:
        samples = samples[:limit]
    written = 0
    skipped = 0
    failed = 0
    cache_dataset = KeypointSequenceDataset(split_file, keypoint_root, split, num_frames, limit=0, cache_root=cache_root) if cache_root else None

    for idx, (source, _) in enumerate(samples, start=1):
        try:
            out_path = output_path_for_source(source, keypoint_root, output_root, split)
            
            if out_path.exists():
                skipped += 1
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    keypoint_path = resolve_keypoint_path(source, keypoint_root, split)
                    normalized_data = normalize_video(keypoint_path)
                except FileNotFoundError:
                    if cache_dataset is None:
                        raise
                    cache_path = cache_dataset.cache_path_for_source(source)
                    if not cache_path or not cache_path.exists():
                        raise
                    normalized_data = tensor_to_normalized_video(torch.load(cache_path, map_location="cpu"), source)
                
                # 2. Extract Predicates
                predicates_data = extract_predicates(
                    normalized_data, 
                    min_score=0.3, 
                    window_size=5, 
                    motion_threshold=0.08
                )
                
                # 3. Build Composite Events
                events_data = build_events(
                    predicates_data, 
                    max_gap=2, 
                    min_confidence=0.3, 
                    composite_gap=4
                )
                
                # 4. Build Instance Graph
                instance_graph = build_instance_graph(events_data)
                
                # Write to file
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(instance_graph, f, ensure_ascii=False)
                
                written += 1
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {source}: {exc}")
            # Uncomment for debugging:
            # traceback.print_exc()

        if idx % 100 == 0 or idx == len(samples):
            print(
                f"{split}: {idx}/{len(samples)} "
                f"written={written} skipped={skipped} failed={failed}"
            )

    return {"split": split, "total": len(samples), "written": written, "skipped": skipped, "failed": failed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Instance Graphs for all videos.")
    parser.add_argument("--keypoint-root", type=Path, required=True, help="Path to raw keypoints")
    parser.add_argument("--output-root", type=Path, default=Path("data/instance_graphs"), help="Output directory for instance graphs")
    parser.add_argument("--cache-root", type=Path, help="Optional tensor cache root used when raw keypoint JSON is unavailable")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-file", type=Path, default=Path("val.txt"))
    parser.add_argument("--test-file", type=Path, default=Path("test.txt"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int)
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
            precompute_split(split_files[split], args.keypoint_root, args.output_root, split, args.cache_root, args.num_frames, args.limit)
        )
    print("Done.")
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
