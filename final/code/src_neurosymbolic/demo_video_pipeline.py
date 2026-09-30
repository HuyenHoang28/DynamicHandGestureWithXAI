import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.events.build_events import build_events
from src_neurosymbolic.kg.build_instance_graph import build_instance_graph
from src_neurosymbolic.kg.graph_matching import load_template_graphs, match_instance_to_templates
from src_neurosymbolic.models.cross_attention_fusion import NeuralKeypointTemplateCrossAttention
from src_neurosymbolic.models.keypoint_dataset import keypoints_to_tensor
from src_neurosymbolic.pose.normalize_keypoints import normalize_video
from src_neurosymbolic.predicates.extract_predicates import extract_predicates
from src_neurosymbolic.precompute_instance_graphs import tensor_to_normalized_video


def extract_keypoints_from_video(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.extract_script is None:
        raise ValueError("--input-video requires --extract-script or use --input-keypoints for demo from existing keypoint JSON.")
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_data_root = output_dir / "raw_video_demo"
    video_label_dir = demo_data_root / "demo" / "SubjectDemo"
    video_label_dir.mkdir(parents=True, exist_ok=True)
    video_copy = video_label_dir / Path(args.input_video).name
    if not video_copy.exists():
        shutil.copy2(args.input_video, video_copy)

    command = [
        sys.executable,
        str(args.extract_script),
        "--repo-root",
        str(args.repo_root),
        "--data-root",
        str(demo_data_root),
        "--layout",
        "raw-subjects",
        "--output-root",
        str(output_dir / "extracted_keypoints"),
        "--mmpose-root",
        str(args.mmpose_root),
        "--pose-weights",
        str(args.pose_weights),
        "--device",
        args.device,
        "--overwrite",
    ]
    subprocess.run(command, check=True)
    candidates = sorted((output_dir / "extracted_keypoints").rglob(f"{video_copy.stem}.json"))
    if not candidates:
        candidates = sorted((output_dir / "extracted_keypoints").rglob("*.json"))
    if not candidates:
        raise FileNotFoundError("Extractor finished but no keypoint JSON was found.")
    return candidates[0]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[NeuralKeypointTemplateCrossAttention, argparse.Namespace, dict[int, str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = argparse.Namespace(**checkpoint["args"])
    model = NeuralKeypointTemplateCrossAttention(
        num_classes=model_args.num_classes,
        d_model=model_args.d_model,
        num_layers=model_args.layers,
        num_heads=model_args.heads,
        template_tokens=model_args.template_tokens,
        dropout=model_args.dropout,
        max_frames=model_args.num_frames,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    label_names = {int(k): v for k, v in (checkpoint.get("label_names") or {}).items()}
    return model, model_args, label_names


def predict(
    model: NeuralKeypointTemplateCrossAttention,
    model_args: argparse.Namespace,
    label_names: dict[int, str],
    normalized: dict,
    instance_graph: dict,
    device: torch.device,
    top_k: int,
) -> dict:
    tensor = keypoints_to_tensor(normalized, model_args.num_frames).unsqueeze(0).to(device)
    instance_json = json.dumps(instance_graph, ensure_ascii=False)
    with torch.no_grad():
        logits, attention = model(tensor, instance_graph_json=[instance_json], return_attention=True)
        probs = logits.softmax(dim=1)[0].cpu()
    values, indices = probs.topk(k=min(top_k, probs.numel()))
    return {
        "prediction_id": int(indices[0].item()),
        "prediction_label": label_names.get(int(indices[0].item()), str(int(indices[0].item()))),
        "confidence": float(values[0].item()),
        "top_k": [
            {
                "label_id": int(idx.item()),
                "label_name": label_names.get(int(idx.item()), str(int(idx.item()))),
                "probability": float(value.item()),
            }
            for idx, value in zip(indices, values)
        ],
        "attention_shape": list(attention.shape),
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_demo(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    if args.input_keypoints is not None:
        keypoint_path = args.input_keypoints
        normalized = normalize_video(keypoint_path)
    elif args.input_cache is not None:
        keypoint_path = args.input_cache
        normalized = tensor_to_normalized_video(torch.load(args.input_cache, map_location="cpu"), str(args.input_cache))
    else:
        keypoint_path = extract_keypoints_from_video(args, args.output_dir)
        normalized = normalize_video(keypoint_path)
    predicates = extract_predicates(normalized)
    events = build_events(predicates)
    instance_graph = build_instance_graph(events)

    model, model_args, label_names = load_model(args.checkpoint, device)
    neural_prediction = predict(model, model_args, label_names, normalized, instance_graph, device, args.top_k)

    symbolic_result = None
    if args.template_dir is not None and args.template_dir.exists():
        template_graphs = load_template_graphs(args.template_dir)
        symbolic_result = match_instance_to_templates(instance_graph, template_graphs, top_k=args.top_k)

    write_json(args.output_dir / "normalized_keypoints.json", normalized)
    write_json(args.output_dir / "predicates.json", predicates)
    write_json(args.output_dir / "events.json", events)
    write_json(args.output_dir / "instance_graph.json", instance_graph)
    write_json(args.output_dir / "prediction.json", {"neural": neural_prediction, "symbolic": symbolic_result})

    report = {
        "input_keypoints": str(keypoint_path),
        "checkpoint": str(args.checkpoint),
        "neural_prediction": neural_prediction,
        "symbolic_prediction": {
            "prediction": symbolic_result["prediction"],
            "prediction_label_id": symbolic_result["prediction_label_id"],
            "top_k": symbolic_result["top_k"],
        }
        if symbolic_result
        else None,
        "artifacts": {
            "normalized_keypoints": "normalized_keypoints.json",
            "predicates": "predicates.json",
            "events": "events.json",
            "instance_graph": "instance_graph.json",
            "prediction": "prediction.json",
        },
    }
    write_json(args.output_dir / "demo_report.json", report)

    print("Demo prediction")
    print(f"  Neural:   {neural_prediction['prediction_label']} ({neural_prediction['confidence']:.4f})")
    if symbolic_result:
        print(f"  Symbolic: {symbolic_result['prediction']} ({symbolic_result['prediction_label_id']})")
    print(f"Saved demo artifacts: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end demo pipeline for one gesture video/keypoint JSON.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-keypoints", type=Path, help="Existing RTMW 52-keypoint JSON.")
    input_group.add_argument("--input-cache", type=Path, help="Precomputed keypoint tensor .pt file.")
    input_group.add_argument("--input-video", type=Path, help="Raw video file. Requires --extract-script and MMPose args.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_video_pipeline"))
    parser.add_argument("--template-dir", type=Path, default=Path("data/template_graphs"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--extract-script", type=Path, default=Path("run_rtmw_splits_dual.py"))
    parser.add_argument("--mmpose-root", type=Path, default=Path("mmpose"))
    parser.add_argument("--pose-weights", type=Path, default=Path("checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"))
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
