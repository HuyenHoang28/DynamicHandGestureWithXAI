"""CPU smoke benchmark of existing classifiers; random weights, not accuracy."""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'final/code/neural_only_experiments'))
from models import NeuralOnlyClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 20 or args.threads < 1:
        parser.error('Require iterations >= 20 and threads >= 1')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    x = torch.randn(1, 64, 52, 3)
    rows = []
    for width, layers in [(128, 3), (64, 2)]:
        for backbone in ['tcn', 'bigru', 'transformer', 'mlp_mixer']:
            model = NeuralOnlyClassifier(
                backbone=backbone, num_classes=13, d_model=width, layers=layers
            ).eval()
            with torch.inference_mode():
                for _ in range(20):
                    model(x)
                times = []
                for _ in range(args.iterations):
                    start = time.perf_counter_ns()
                    result = model(x)
                    times.append((time.perf_counter_ns() - start) / 1e6)
            if result.shape != (1, 13) or not torch.isfinite(result).all():
                raise RuntimeError(f'Invalid forward output for {backbone}')
            row = dict(backbone=backbone, width=width, layers=layers,
                       parameters=sum(p.numel() for p in model.parameters()),
                       median_ms=statistics.median(times),
                       p95_ms=sorted(times)[int(0.95 * (len(times) - 1))])
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(
        scope='Synthetic CPU forward only, random weights; excludes extraction, graph, IO; not edge-device performance or accuracy.',
        torch_version=torch.__version__, platform=platform.platform(),
        processor=os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
        threads=args.threads, shape=list(x.shape), warmup=20,
        iterations=args.iterations, seed=42, results=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
