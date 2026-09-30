# Neural-only backbone experiments

Folder nay dung de thu rieng nhanh neural, khong dung Instance KG va Template KG.

Input van la tensor keypoint:

```text
[64, 52, 3]
```

Backbone co the thu:

```text
transformer
tcn
bigru
mlp_mixer
```

## Train tren local/remote

Vi du train TCN:

```bash
python neural_only_experiments/train_neural_only.py \
  --backbone tcn \
  --cache-root data/keypoint_tensor_cache \
  --train-file train.txt \
  --val-file val.txt \
  --output-dir outputs/neural_only_tcn \
  --batch-size 128 \
  --epochs 20 \
  --device cuda:0
```

Train Transformer baseline:

```bash
python neural_only_experiments/train_neural_only.py \
  --backbone transformer \
  --cache-root data/keypoint_tensor_cache \
  --train-file train.txt \
  --val-file val.txt \
  --output-dir outputs/neural_only_transformer \
  --batch-size 128 \
  --epochs 20 \
  --device cuda:0
```

Train BiGRU:

```bash
python neural_only_experiments/train_neural_only.py \
  --backbone bigru \
  --cache-root data/keypoint_tensor_cache \
  --train-file train.txt \
  --val-file val.txt \
  --output-dir outputs/neural_only_bigru \
  --batch-size 128 \
  --epochs 20 \
  --device cuda:0
```

Train MLP-Mixer:

```bash
python neural_only_experiments/train_neural_only.py \
  --backbone mlp_mixer \
  --cache-root data/keypoint_tensor_cache \
  --train-file train.txt \
  --val-file val.txt \
  --output-dir outputs/neural_only_mlp_mixer \
  --batch-size 128 \
  --epochs 20 \
  --device cuda:0
```

## Test

```bash
python neural_only_experiments/test_neural_only.py \
  --checkpoint outputs/neural_only_tcn/best.pt \
  --cache-root data/keypoint_tensor_cache \
  --test-file test.txt \
  --batch-size 128 \
  --device cuda:0 \
  --output outputs/neural_only_tcn/test_metrics.json
```

## Muc dich bao cao

So sanh:

```text
Neural-only Transformer
Neural-only TCN
Neural-only BiGRU
Neural-only MLP-Mixer
```

Sau do so voi:

```text
Neural + Instance KG + Template KG Cross-Attention
```

De chung minh KG branch co cai thien so voi neural-only hay khong.
