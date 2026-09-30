import argparse
import importlib.util
import json
import shutil
import sys
import threading
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.events.build_events import build_events
from src_neurosymbolic.kg.build_instance_graph import build_instance_graph
from src_neurosymbolic.kg.graph_matching import load_template_graphs, match_instance_to_templates
from src_neurosymbolic.models.cross_attention_v2 import REQUIREMENT_LEVELS, load_checkpoint_compat
from src_neurosymbolic.models.cross_attention_v4 import DualCrossAttentionV4NeuroSymbolicModel
from src_neurosymbolic.models.keypoint_dataset import keypoints_to_tensor
from src_neurosymbolic.pose.normalize_keypoints import normalize_video
from src_neurosymbolic.predicates.extract_predicates import extract_predicates
from src_neurosymbolic.precompute_instance_graphs import tensor_to_normalized_video


REQUIREMENT_NAMES = {value: key for key, value in REQUIREMENT_LEVELS.items()}
_RTMW_MODULES: dict[Path, object] = {}
_RTMW_INFERENCERS: dict[tuple[str, str, str], tuple[object, list[str]]] = {}
_RTMW_LOCK = threading.RLock()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def label_name(label_names: dict, label_id: int) -> str:
    return str(label_names.get(label_id, label_names.get(str(label_id), label_id)))


def select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {requested} to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def load_rtmw_extractor(script_path: Path):
    script_path = script_path.resolve()
    module = _RTMW_MODULES.get(script_path)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(
        f"_demo_rtmw_extractor_{abs(hash(script_path))}",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load RTMW extraction module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RTMW_MODULES[script_path] = module
    return module


def initialize_rtmw(
    extract_script: Path,
    mmpose_root: Path,
    pose_weights: Path,
    device: str,
) -> None:
    extractor = load_rtmw_extractor(extract_script)
    pose_config = extractor.default_pose_config(mmpose_root).resolve()
    pose_weights = pose_weights.resolve()
    cache_key = (str(pose_config), str(pose_weights), device)

    with _RTMW_LOCK:
        if cache_key in _RTMW_INFERENCERS:
            return
        print(f"Preloading persistent RTMW-L on {device}...")
        inferencer = extractor.make_inferencer(
            pose_config,
            pose_weights,
            device,
        )
        id2name = extractor.load_id2name(
            mmpose_root.resolve(),
            mirror_swap=True,
        )
        _RTMW_INFERENCERS[cache_key] = (inferencer, id2name)
        print(f"Persistent RTMW-L is ready on {device}.")


def extract_keypoints_from_video(args: argparse.Namespace) -> Path:
    for path, description in (
        (args.input_video, "input video"),
        (args.extract_script, "RTMW extraction script"),
        (args.repo_root, "repository root"),
        (args.mmpose_root, "MMPose root"),
        (args.pose_weights, "RTMW pose weights"),
    ):
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")

    extraction_root = args.output_dir / "keypoint_extraction"
    input_dir = extraction_root / "raw" / "demo" / "SubjectDemo"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied_video = input_dir / args.input_video.name
    shutil.copy2(args.input_video, copied_video)
    output_json = (
        extraction_root
        / "keypoints"
        / "demo"
        / "SubjectDemo"
        / f"{copied_video.stem}.json"
    )

    extractor = load_rtmw_extractor(args.extract_script)
    pose_config = extractor.default_pose_config(args.mmpose_root).resolve()
    cache_key = (str(pose_config), str(args.pose_weights.resolve()), args.device)

    # MMPoseInferencer is reused between videos, but inference itself is serialized
    # because its internal visualizer and pipeline state are not thread-safe.
    with _RTMW_LOCK:
        cached = _RTMW_INFERENCERS.get(cache_key)
        if cached is None:
            initialize_rtmw(
                args.extract_script,
                args.mmpose_root,
                args.pose_weights,
                args.device,
            )
            inferencer, id2name = _RTMW_INFERENCERS[cache_key]
        else:
            print(f"Reusing persistent RTMW-L on {args.device}.")
            inferencer, id2name = cached

        print(f"Extracting RTMW keypoints: {args.input_video.name}")
        extractor.extract_video(
            copied_video,
            output_json,
            inferencer,
            id2name,
            pretty=False,
        )

    if not output_json.is_file():
        raise FileNotFoundError(
            f"RTMW extractor completed but produced no keypoint JSON: {output_json}"
        )
    return output_json


def load_input(args: argparse.Namespace) -> tuple[dict, Path]:
    if args.input_video is not None:
        source = extract_keypoints_from_video(args)
        return normalize_video(source), source
    if args.input_keypoints is not None:
        return normalize_video(args.input_keypoints), args.input_keypoints

    cached = torch.load(args.input_cache, map_location="cpu")
    if isinstance(cached, dict):
        cached = cached.get("keypoints", cached.get("tensor"))
    if not isinstance(cached, torch.Tensor):
        raise TypeError(f"Unsupported keypoint cache content: {type(cached).__name__}")
    return tensor_to_normalized_video(cached, str(args.input_cache)), args.input_cache


def load_v4_model(
    checkpoint_path: Path,
    template_dir: Path,
    device: torch.device,
) -> tuple[DualCrossAttentionV4NeuroSymbolicModel, dict, dict]:
    checkpoint = load_checkpoint_compat(checkpoint_path)
    version = checkpoint.get("model_version")
    if version != "v4_structured_requirement_dual_cross_attention":
        raise ValueError(
            "This demo requires a V4 checkpoint, but checkpoint model_version is "
            f"{version!r}."
        )

    saved = checkpoint["args"]
    model = DualCrossAttentionV4NeuroSymbolicModel(
        template_dir=template_dir,
        num_classes=int(saved["num_classes"]),
        num_frames=int(saved["num_frames"]),
        d_model=int(saved["d_model"]),
        layers=int(saved["layers"]),
        heads=int(saved["heads"]),
        graph_layers=int(saved["graph_layers"]),
        decoder_layers=int(saved["decoder_layers"]),
        instance_tokens=int(saved["instance_tokens"]),
        dropout=float(saved["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    return model, saved, checkpoint


def branch_prediction(
    logits: torch.Tensor,
    label_names: dict,
    top_k: int,
) -> dict:
    probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()
    values, indices = probabilities.topk(min(top_k, probabilities.numel()))
    rows = [
        {
            "label_id": int(index),
            "label_name": label_name(label_names, int(index)),
            "probability": float(value),
        }
        for value, index in zip(values, indices)
    ]
    return {
        "prediction_id": rows[0]["label_id"],
        "prediction_label": rows[0]["label_name"],
        "confidence": rows[0]["probability"],
        "top_k": rows,
    }


def requirement_evidence(
    model: DualCrossAttentionV4NeuroSymbolicModel,
    outputs: dict,
    class_id: int,
) -> list[dict]:
    requirement_ids = outputs["requirement_ids"][0, class_id].detach().cpu()
    weights = outputs["requirement_weights"][0, class_id].detach().cpu()
    presence = outputs["presence"][0, class_id].detach().cpu()
    targets = outputs["requirement_targets"][0, class_id].detach().cpu()
    concepts = model.graph_encoder.template_concept_names[class_id]
    facts = model.graph_encoder.template_requirement_facts[class_id]

    rows = []
    for index, requirement_id_tensor in enumerate(requirement_ids):
        requirement_id = int(requirement_id_tensor)
        weight = float(weights[index])
        if requirement_id not in (
            REQUIREMENT_LEVELS["required"],
            REQUIREMENT_LEVELS["preferred"],
            REQUIREMENT_LEVELS["optional"],
            REQUIREMENT_LEVELS["negative"],
        ) or weight <= 0:
            continue
        observed = float(targets[index]) >= 0.5
        level = REQUIREMENT_NAMES[requirement_id]
        rows.append(
            {
                "node_index": index,
                "concept": concepts[index],
                "fact": list(facts[index]) if facts[index] is not None else None,
                "level": level,
                "weight": weight,
                "model_presence": float(presence[index]),
                "observed_in_instance_graph": observed,
                "satisfied": (not observed) if level == "negative" else observed,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            not row["satisfied"],
            -row["weight"],
            -row["model_presence"],
        ),
    )


def attention_summary(
    outputs: dict,
    class_id: int,
    top_k: int,
) -> dict:
    result = {}
    neural_attention = outputs.get("neural_attention")
    if neural_attention is not None:
        values = neural_attention[0, class_id].mean(dim=(0, 1)).detach().cpu()
        scores, indices = values.topk(min(top_k, values.numel()))
        result["important_frames"] = [
            {"frame_index": int(index), "attention": float(score)}
            for score, index in zip(scores, indices)
        ]
    graph_attention = outputs.get("graph_attention")
    if graph_attention is not None:
        values = graph_attention[0, class_id].mean(dim=(0, 1)).detach().cpu()
        scores, indices = values.topk(min(top_k, values.numel()))
        result["important_graph_tokens"] = [
            {"token_index": int(index), "attention": float(score)}
            for score, index in zip(scores, indices)
        ]
    return result


@torch.no_grad()
def infer(
    model: DualCrossAttentionV4NeuroSymbolicModel,
    saved_args: dict,
    checkpoint: dict,
    normalized: dict,
    instance_graph: dict,
    device: torch.device,
    top_k: int,
) -> tuple[dict, dict]:
    keypoints = keypoints_to_tensor(
        normalized,
        int(saved_args["num_frames"]),
    ).unsqueeze(0).to(device)
    outputs = model(
        keypoints,
        [json.dumps(instance_graph, ensure_ascii=False)],
        return_attention=True,
    )
    label_names = checkpoint.get("label_names", {})
    predictions = {
        "final": branch_prediction(outputs["logits"], label_names, top_k),
        "neural": branch_prediction(outputs["neural_logits"], label_names, top_k),
        "kg": branch_prediction(outputs["kg_logits"], label_names, top_k),
    }
    final_id = predictions["final"]["prediction_id"]
    evidence = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation": checkpoint.get("val"),
        "final_class_id": final_id,
        "final_class_name": predictions["final"]["prediction_label"],
        "final_class_neural_gate": float(outputs["gate"][0, final_id]),
        "mean_neural_gate": float(outputs["gate"][0].mean()),
        "neural_temperature": float(outputs["neural_temperature"].detach().cpu()),
        "kg_temperature": float(outputs["kg_temperature"].detach().cpu()),
        "requirements": requirement_evidence(model, outputs, final_id),
        "attention": attention_summary(outputs, final_id, top_k),
    }
    return predictions, evidence


def render_markdown(
    predictions: dict,
    evidence: dict,
    symbolic: dict | None,
) -> str:
    lines = [
        "# V4 Single-Video Demo",
        "",
        f"- Final: **{predictions['final']['prediction_label']}** "
        f"({predictions['final']['confidence']:.4f})",
        f"- Neural: **{predictions['neural']['prediction_label']}** "
        f"({predictions['neural']['confidence']:.4f})",
        f"- KG: **{predictions['kg']['prediction_label']}** "
        f"({predictions['kg']['confidence']:.4f})",
        f"- Neural gate for final class: {evidence['final_class_neural_gate']:.4f}",
        "",
        "## Final Top-K",
        "",
    ]
    for row in predictions["final"]["top_k"]:
        lines.append(
            f"- {row['label_name']}: {row['probability']:.4f}"
        )
    if symbolic is not None:
        lines.extend(
            [
                "",
                "## Rule-Based Symbolic Match",
                "",
                f"- Prediction: **{symbolic['prediction']}**",
            ]
        )
    satisfied = [row for row in evidence["requirements"] if row["satisfied"]]
    missing = [row for row in evidence["requirements"] if not row["satisfied"]]
    lines.extend(["", "## Requirement Evidence", ""])
    for row in satisfied[:12]:
        fact = " / ".join(row["fact"] or [row["concept"]])
        lines.append(f"- [hit] {row['level']}: {fact}")
    for row in missing[:8]:
        fact = " / ".join(row["fact"] or [row["concept"]])
        lines.append(f"- [miss] {row['level']}: {fact}")
    return "\n".join(lines) + "\n"


def run_demo(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    args.device = str(device)
    normalized, source_path = load_input(args)

    predicates = extract_predicates(
        normalized,
        min_score=args.min_score,
        window_size=args.window_size,
        motion_threshold=args.motion_threshold,
    )
    events = build_events(
        predicates,
        max_gap=args.max_gap,
        min_confidence=args.min_event_confidence,
        composite_gap=args.composite_gap,
    )
    instance_graph = build_instance_graph(events)
    model, saved_args, checkpoint = load_v4_model(
        args.checkpoint,
        args.template_dir,
        device,
    )
    predictions, model_evidence = infer(
        model,
        saved_args,
        checkpoint,
        normalized,
        instance_graph,
        device,
        args.top_k,
    )

    symbolic = None
    if not args.disable_symbolic_match:
        symbolic = match_instance_to_templates(
            instance_graph,
            load_template_graphs(args.template_dir),
            top_k=args.top_k,
        )

    write_json(args.output_dir / "normalized_keypoints.json", normalized)
    write_json(args.output_dir / "predicates.json", predicates)
    write_json(args.output_dir / "events.json", events)
    write_json(args.output_dir / "instance_graph.json", instance_graph)
    write_json(args.output_dir / "prediction.json", predictions)
    write_json(args.output_dir / "model_evidence.json", model_evidence)
    if symbolic is not None:
        write_json(args.output_dir / "symbolic_match.json", symbolic)

    report = {
        "input": str(source_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_version": checkpoint.get("model_version"),
        "device": str(device),
        "predictions": predictions,
        "artifacts": {
            "normalized_keypoints": "normalized_keypoints.json",
            "predicates": "predicates.json",
            "events": "events.json",
            "instance_graph": "instance_graph.json",
            "prediction": "prediction.json",
            "model_evidence": "model_evidence.json",
            "symbolic_match": "symbolic_match.json" if symbolic is not None else None,
            "summary": "summary.md",
        },
    }
    write_json(args.output_dir / "demo_report.json", report)
    (args.output_dir / "summary.md").write_text(
        render_markdown(predictions, model_evidence, symbolic),
        encoding="utf-8",
    )

    print("V4 single-video prediction")
    for branch in ("final", "neural", "kg"):
        row = predictions[branch]
        print(
            f"  {branch:6s}: {row['prediction_label']} "
            f"({row['confidence']:.4f})"
        )
    print(
        "  gate  : "
        f"{model_evidence['final_class_neural_gate']:.4f} neural / "
        f"{1.0 - model_evidence['final_class_neural_gate']:.4f} KG"
    )
    print(f"Saved demo artifacts to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete one-video demo pipeline with a V4 checkpoint."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-video", type=Path)
    inputs.add_argument("--input-keypoints", type=Path)
    inputs.add_argument("--input-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, default=Path("data/template_graphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_video_v4"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--disable-symbolic-match", action="store_true")

    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--motion-threshold", type=float, default=0.08)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-event-confidence", type=float, default=0.3)
    parser.add_argument("--composite-gap", type=int, default=4)

    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--extract-script", type=Path)
    parser.add_argument("--mmpose-root", type=Path)
    parser.add_argument("--pose-weights", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
