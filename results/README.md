# Result layout

Generated artifacts are separated from source code and raw datasets:

```text
results/
├── runs/          fresh reproducible evaluations and training runs
├── final/         validation-selected final deliverable
└── experiments/   historical, exploratory, smoke, ablation, and IPN artifacts
```

Các ZIP trong `experiments/ipn13/artifact_zips/` là artifact lịch sử của IPN,
bao gồm các lần chạy rule-based symbolic và neural ablation. Không suy ra rằng
chúng chứa learned V4 `kg_logits`. Run Colab mới được lưu riêng theo run ID dưới
`MyDrive/ipn-extract-data/runs/ipn13_kg_logits/`; chỉ copy về `runs/ipn13/` khi
cần lưu local.

`runs/`, `final/`, and `experiments/` are local artifacts ignored by Git. The
tracked documentation under `docs/` records provenance, metrics, compatibility,
and validation. Raw datasets remain under `external_data/` and runtime inputs
remain under `final/data/` so existing loaders keep working.

The canonical DATE 27-class result is
`results/final/date27_v4_validation_selected/`. A test-selected checkpoint may
be retained under experiments, but must never be reported as an unbiased
holdout result.
