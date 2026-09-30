from pathlib import Path


REQUIRED_PATHS = [
    "final/code/src_neurosymbolic/train_cross_attention_v4.py",
    "final/code/src_neurosymbolic/test_cross_attention_v2.py",
    "final/code/neural_only_experiments/train_neural_only.py",
    "final/data/Input/keypoint_tensor_cache",
    "final/data/Input/instance_graphs",
    "final/data/Input/template_graphs",
    "final/data/Input/train.txt",
    "final/data/Input/val.txt",
    "final/data/Input/test_1229.txt",
    "final/data/Output/checkpoints/cross_attention_v4_template170_retrain_next/best_test.pt",
    "final/data/Output/checkpoints/kg_cleaned_3branch/last.pt",
    "final/data/Output/checkpoints/neural_only_transformer_30epoch/last.pt",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = []
    for relative_path in REQUIRED_PATHS:
        path = root / relative_path
        if not path.exists():
            missing.append(relative_path)

    if missing:
        print("Missing required paths:")
        for relative_path in missing:
            print(f"- {relative_path}")
        return 1

    print("OK: required code, data, and checkpoints are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
