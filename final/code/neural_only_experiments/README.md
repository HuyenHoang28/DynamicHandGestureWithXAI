# Neural-only backbone experiments

Folder nay dung de thu rieng nhanh neural, khong dung Instance KG va Template KG.

Google Colab notebook: [`../../../notebooks/Neural_Only_Improved_Colab.ipynb`](../../../notebooks/Neural_Only_Improved_Colab.ipynb).

Input van la tensor keypoint:

```text
[64, 52, 3]
```

Baseline mac dinh moi tap trung vao kha nang tong quat, khong gan voi mot bo du lieu cu the:

- them van toc `(dx, dy)` cua tung keypoint de hoc chuyen dong;
- attention pooling bo qua frame khong co keypoint hop le thay cho mean pooling mu;
- jitter chi tren keypoint hop le, frame masking va keypoint dropout;
- label smoothing va class-weighted loss theo tan suat lop;
- chon checkpoint theo validation macro-F1, van ghi accuracy de doi chieu;
- fail-fast khi split/cache sai, tensor sai shape, co NaN/Inf, hoac cache bi trung ten.

Nhung thay doi nay van can duoc kiem chung bang nhieu seed tren tung dataset. Khong dung test
set de chon tham so hay checkpoint.

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
  --epochs 50 \
  --device cuda:0
```

Nhung tham so regularization quan trong (gia tri duoi day cung la mac dinh):

```bash
  --pooling attention \
  --label-smoothing 0.05 \
  --class-weight-power 0.5 \
  --jitter-std 0.005 \
  --frame-mask-probability 0.05 \
  --keypoint-dropout-probability 0.03 \
  --select-metric macro_f1
```

`--class-weight-power 0` tat class weighting. `--disable-motion-features` va
`--pooling mean` tai tao kien truc cu de chay ablation. Khi so sanh, giu nguyen split theo
subject/group, training budget va danh sach seed (nen it nhat 3 seed).

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

Moi bao cao test sinh `accuracy`, `macro_f1`, precision/recall/F1 va support theo lop.
Nen tong hop mean +/- standard deviation qua nhieu seed; mot lan chay don le khong du de
ket luan cai tien co tong quat hay khong.
