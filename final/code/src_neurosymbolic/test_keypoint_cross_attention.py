import argparse
import json
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_neurosymbolic.models.cross_attention_fusion import NeuralKeypointTemplateCrossAttention
from src_neurosymbolic.models.keypoint_dataset import KeypointSequenceDataset

def collate_batch(items: list[dict]) -> dict:
    batch = {
        "keypoints": torch.stack([item["keypoints"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "path": [item["path"] for item in items],
    }
    if "instance_graph_json" in items[0]:
        batch["instance_graph_json"] = [item["instance_graph_json"] for item in items]
    return batch

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        x = batch["keypoints"].to(device)
        y = batch["label"].to(device)
        logits = model(x, instance_graph_json=batch.get("instance_graph_json"))
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * y.numel()
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "total": total,
    }

def test(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = argparse.Namespace(**checkpoint["args"])
    
    keypoint_root = args.keypoint_root if args.keypoint_root is not None else model_args.keypoint_root
    cache_root = args.cache_root if args.cache_root is not None else model_args.cache_root
    num_frames = model_args.num_frames
    
    instance_graph_root = args.instance_graph_root if args.instance_graph_root is not None else getattr(model_args, "instance_graph_root", None)
    test_set = KeypointSequenceDataset(args.test_file, keypoint_root, "test", num_frames, None, cache_root, instance_graph_root)
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    
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
    
    print("Evaluating on test set...")
    metrics = evaluate(model, test_loader, device)
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total samples: {metrics['total']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics: {args.output}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test neural keypoint + template cross-attention baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--test-file", type=Path, default=Path("test.txt"))
    parser.add_argument("--keypoint-root", type=Path, help="Override keypoint root from args")
    parser.add_argument("--cache-root", type=Path, help="Override cache root from args")
    parser.add_argument("--instance-graph-root", type=Path, help="Optional instance graph root for 3-branch model.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=Path, help="Optional path to save test metrics JSON.")
    return parser.parse_args()

if __name__ == "__main__":
    test(parse_args())
