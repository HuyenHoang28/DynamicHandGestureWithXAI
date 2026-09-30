# Huong dan tai hien ket qua va chay demo

Chay tat ca lenh tu thu muc goc project:

```powershell
cd C:\Thanh\HUST\20252\DATN\w3\test_method
```

## 1. Cai dat

Dung moi truong Python da co:

```powershell
.\.venv\Scripts\activate
```

Neu tao moi moi truong:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision torchaudio gradio opencv-python numpy pandas pyyaml tqdm matplotlib scikit-learn networkx
```

Can co cac thu muc/file sau:

```text
final/code
final/data/Input/keypoint_tensor_cache
final/data/Input/instance_graphs
final/data/Input/template_graphs
final/data/Input/train.txt
final/data/Input/val.txt
final/data/Input/test_1229.txt
final/data/Output/checkpoints
```

Neu chay demo tu video goc, can them:

```text
run_rtmw_splits_dual.py
mmpose
checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth
```

## 2. Tai hien ket qua training

Kich ban 1 - Neural-only:

```powershell
.\.venv\Scripts\python.exe final\code\neural_only_experiments\train_neural_only.py `
  --backbone transformer `
  --cache-root final\data\Input\keypoint_tensor_cache `
  --train-file final\data\Input\train.txt `
  --val-file final\data\Input\val.txt `
  --output-dir final\outputs\train_neural_only `
  --epochs 30 `
  --device cuda:0
```

Kich ban 2 - V4 KG-only:

```powershell
.\.venv\Scripts\python.exe final\code\src_neurosymbolic\train_cross_attention_v4.py `
  --keypoint-root final\data\Input\keypoint_tensor_cache `
  --cache-root final\data\Input\keypoint_tensor_cache `
  --instance-graph-root final\data\Input\instance_graphs `
  --template-dir final\data\Input\template_graphs `
  --train-file final\data\Input\train.txt `
  --val-file final\data\Input\val.txt `
  --output-dir final\outputs\train_v4_kg_only `
  --logit-branch kg `
  --freeze-neural-for-kg `
  --epochs 30 `
  --device cuda:0
```

Kich ban 3 - Neuro-symbolic fusion:

```powershell
.\.venv\Scripts\python.exe final\code\src_neurosymbolic\train_cross_attention_v4.py `
  --keypoint-root final\data\Input\keypoint_tensor_cache `
  --cache-root final\data\Input\keypoint_tensor_cache `
  --instance-graph-root final\data\Input\instance_graphs `
  --template-dir final\data\Input\template_graphs `
  --train-file final\data\Input\train.txt `
  --val-file final\data\Input\val.txt `
  --output-dir final\outputs\train_v4_fusion `
  --logit-branch final `
  --epochs 30 `
  --device cuda:0
```

Checkpoint sau train nam trong thu muc `output-dir`, gom `best.pt`, `last.pt`, `history.json`.

## 3. Tai hien ket qua test

Kich ban 1 - Neural-only:

```powershell
.\.venv\Scripts\python.exe final\code\neural_only_experiments\test_neural_only.py `
  --checkpoint final\data\Output\checkpoints\neural_only_transformer_30epoch\last.pt `
  --cache-root final\data\Input\keypoint_tensor_cache `
  --test-file final\data\Input\test_1229.txt `
  --batch-size 128 `
  --num-workers 0 `
  --device cuda:0 `
  --output final\Experiment\reports\packaged_test_neural_only.json
```

Kich ban 2 - V4 KG-only:

```powershell
.\.venv\Scripts\python.exe final\code\src_neurosymbolic\test_cross_attention_v2.py `
  --checkpoint final\data\Output\checkpoints\kg_cleaned_3branch\last.pt `
  --keypoint-root final\data\Input\keypoint_tensor_cache `
  --cache-root final\data\Input\keypoint_tensor_cache `
  --instance-graph-root final\data\Input\instance_graphs `
  --template-dir final\data\Input\template_graphs `
  --test-file final\data\Input\test_1229.txt `
  --batch-size 16 `
  --num-workers 0 `
  --device cuda:0 `
  --eval-branch kg `
  --output final\Experiment\reports\packaged_test_kg_guided
```

Kich ban 3 - Neuro-symbolic fusion:

```powershell
.\.venv\Scripts\python.exe final\code\src_neurosymbolic\test_cross_attention_v2.py `
  --checkpoint final\data\Output\checkpoints\cross_attention_v4_template170_retrain_next\best_test.pt `
  --keypoint-root final\data\Input\keypoint_tensor_cache `
  --cache-root final\data\Input\keypoint_tensor_cache `
  --instance-graph-root final\data\Input\instance_graphs `
  --template-dir final\data\Input\template_graphs `
  --test-file final\data\Input\test_1229.txt `
  --batch-size 16 `
  --num-workers 0 `
  --eval-branch final `
  --device cuda:0 `
  --output final\Experiment\reports\packaged_test_fusion_best_test
```

Moi lenh test tao them file prediction, per-class va confusion matrix trong `final/results`.

## 4. Chay demo UI

```powershell
.\.venv\Scripts\python.exe final\code\src_neurosymbolic\demo_ui\app.py `
  --server-name 127.0.0.1 `
  --server-port 7860
```

Mo trinh duyet:

```text
http://127.0.0.1:7860
```

Trong UI, chon mot trong ba kich ban:

```text
Kich ban 1 - Neural-only
Kich ban 2 - V4 KG-only
Kich ban 3 - Neuro-symbolic fusion
```

Sau do upload video va bam `Run Model Prediction`.

## 5. Chay demo bang lenh

```powershell
.\.venv\Scripts\python.exe final\code\src_neurosymbolic\demo_video_v4.py `
  --input-video "C:\Users\ADMIN\Pictures\Camera Roll\WIN_20260706_12_16_48_Pro.mp4" `
  --checkpoint final\data\Output\checkpoints\cross_attention_v4_template170_retrain_next\best_test.pt `
  --template-dir final\data\Input\template_graphs `
  --output-dir final\outputs\demo_video_v4 `
  --device cuda:0 `
  --extract-script run_rtmw_splits_dual.py `
  --mmpose-root mmpose `
  --pose-weights checkpoints\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth
```

Ket qua demo luu trong `final/outputs/demo_video_v4` hoac `final/outputs/demo_ui_v4`.

## 6. Luu y

- Neu khong co GPU, doi `cuda:0` thanh `cpu`.
- Buoc RTMW extract keypoint tu video goc se cham hon buoc du doan.
- Neu UI van hien giao dien cu, tat server va refresh trinh duyet bang `Ctrl + F5`.
