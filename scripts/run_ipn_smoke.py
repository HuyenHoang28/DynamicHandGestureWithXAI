"""Run existing V4 demo on a manifest, reusing RTMW in one CPU process."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["WIN_PD_OVERRIDE_LOCAL_APPDATA"] = str(ROOT / ".gitignore")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(ROOT / "final/code"))
    from src_neurosymbolic.demo_video_v4 import run_demo
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    started = time.perf_counter()
    for index, item in enumerate(manifest, 1):
        dest = output / item["clip_id"]
        print(f"[{index}/{len(manifest)}] {item['clip_id']}", flush=True)
        source = Path(item["clip_path"])
        if not source.is_absolute():
            source = ROOT / source
        params = argparse.Namespace(
            input_video=source, input_keypoints=None, input_cache=None,
            checkpoint=ROOT / "final/data/Output/checkpoints/cross_attention_v4_template170_retrain_next/best_test.pt",
            template_dir=ROOT / "final/data/Input/template_graphs", output_dir=dest,
            device=args.device, top_k=5, disable_symbolic_match=False,
            min_score=0.3, window_size=5, motion_threshold=0.08,
            max_gap=2, min_event_confidence=0.3, composite_gap=4,
            repo_root=ROOT, extract_script=ROOT / "keypoint_extractor/run_rtmw_splits_dual.py",
            mmpose_root=ROOT / ".venv/Lib/site-packages/mmpose/.mim",
            pose_weights=ROOT / "final/data/Output/checkpoints/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth",
        )
        began = time.perf_counter()
        row = dict(item)
        try:
            reused = args.resume and (dest / "demo_report.json").exists() and (dest / "summary.md").exists()
            if not reused:
                run_demo(params)
            row.update(status="reused" if reused else "ok", elapsed_s=None if reused else time.perf_counter() - began,
                       predictions=json.loads((dest / "prediction.json").read_text(encoding="utf-8")))
        except Exception as error:
            traceback.print_exc()
            row.update(status="error", elapsed_s=time.perf_counter() - began, error=str(error))
        results.append(row)
        (output / "batch_summary.json").write_text(json.dumps(dict(device=args.device, elapsed_s=time.perf_counter()-started, clips=results), indent=2), encoding="utf-8")
        print(f"Finished {index}/{len(manifest)}: {row['status']}; seconds={row['elapsed_s']}", flush=True)
    if any(r["status"] == "error" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
