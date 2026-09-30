# Hướng Dẫn Tái Hiện Kết Quả Và Chạy Demo

Chạy tất cả lệnh từ thư mục gốc project:

```powershell
cd C:\path\to\dynamic_hand_gesture_recognition
```

## 1. Cài Đặt

Dùng môi trường Python:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu dùng GPU NVIDIA, nên cài PyTorch theo đúng bản CUDA của máy. Sau đó kiểm tra:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. Chuẩn Bị Data Và Checkpoint

Repo mặc định không commit data/checkpoint để tránh quá nặng. Trước khi train/test/demo, cần copy hoặc giải nén dữ liệu vào đúng cấu trúc:

```text
final/data/Input/keypoint_tensor_cache/
final/data/Input/instance_graphs/
final/data/Input/template_graphs/
final/data/Input/train.txt
final/data/Input/val.txt
final/data/Input/test_1229.txt
final/data/Output/checkpoints/
```

Kiểm tra nhanh:

```powershell
python scripts\verify_repo.py
```

Nếu đủ file cần thiết, script sẽ báo `OK`.

## 3. Train Neural-Only

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

Nếu không có GPU, đổi `cuda:0` thành `cpu`.

## 4. Train KG-Only

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

## 5. Train Neuro-Symbolic Fusion

Đây là kịch bản chính của mô hình cuối:

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

Hoặc chạy script gọn ở thư mục gốc:

```powershell
.\scripts\run_train_fusion.ps1
```

Checkpoint sau khi train nằm trong `output-dir`, gồm:

```text
best.pt
last.pt
history.json
```

## 6. Test Neural-Only

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

## 7. Test KG-Only

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

## 8. Test Neuro-Symbolic Fusion

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

Hoặc chạy script gọn ở thư mục gốc:

```powershell
.\scripts\run_test_fusion.ps1
```

## 9. Chạy Demo UI

```powershell
.\scripts\run_demo_ui.ps1
```

Sau đó mở trình duyệt:

```text
http://127.0.0.1:7860
```

## 10. Trích Xuất Keypoint Từ Video Gốc

Nếu muốn chạy từ video gốc, xem hướng dẫn trong:

```text
keypoint_extractor/README.md
```

Cần chuẩn bị thêm MMPose và weight RTMW-L:

```text
keypoint_extractor/mmpose/
keypoint_extractor/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth
```

## 11. Lưu Ý

- Nếu không có GPU, đổi `cuda:0` thành `cpu`.
- Bước trích xuất keypoint từ video gốc chậm hơn nhiều so với bước train/test trên cache.
- Nếu UI vẫn hiện giao diện cũ, tắt server và refresh trình duyệt bằng `Ctrl + F5`.
