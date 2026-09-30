# Neuro-Symbolic Gesture Recognition

Repo này chứa bản code gọn để người khác clone về chạy lại phần train/test/demo trong thư mục `final`.

Data cache và checkpoint không được commit trực tiếp vào GitHub để repo nhẹ, dễ clone, và tránh giới hạn dung lượng. Sau khi clone, người dùng cần tải/copy data vào đúng đường dẫn như phần bên dưới.

## Nội dung repo

```text
final/code/                         Code train, test, demo
final/data/Input/                   Không commit; cần tải/copy riêng
final/data/Output/checkpoints/      Không commit; cần tải/copy riêng nếu muốn test/demo ngay
final/scripts/                      Script vẽ confusion matrix
requirements.txt                    Thư viện Python cần cài
scripts/verify_repo.py              Kiểm tra nhanh dữ liệu/checkpoint bắt buộc
scripts/run_train_fusion.ps1        Chạy lại train mô hình fusion chính
scripts/run_test_fusion.ps1         Test checkpoint fusion
scripts/run_demo_ui.ps1             Mở demo UI
```

## Chuẩn bị data/checkpoint

Trước khi chạy train/test/demo, đặt dữ liệu theo cấu trúc:

```text
final/data/Input/keypoint_tensor_cache/
final/data/Input/instance_graphs/
final/data/Input/template_graphs/
final/data/Input/train.txt
final/data/Input/val.txt
final/data/Input/test_1229.txt
final/data/Output/checkpoints/
```

Gợi ý cách chia sẻ:

- Upload `final/data/Input` lên Google Drive/Kaggle/Zenodo/GitHub Release.
- Upload `final/data/Output/checkpoints` nếu muốn người khác test/demo không cần train lại.
- Sau khi tải về, giải nén/copy đúng cấu trúc trên.

Nếu bạn thật sự muốn commit data lên GitHub, hãy dùng Git LFS. Tuy nhiên bản repo mặc định này đã ignore `final/data/`.

## Cài đặt

Chạy từ thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu dùng GPU NVIDIA, cài PyTorch theo bản CUDA phù hợp từ trang PyTorch. Sau đó kiểm tra:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Kiểm tra repo sau khi clone

```powershell
python scripts\verify_repo.py
```

Nếu mọi thứ đủ, script sẽ báo `OK`.

## Chạy lại train chính

Mô hình chính cuối cùng là V4 neuro-symbolic fusion:

```powershell
.\scripts\run_train_fusion.ps1
```

Kết quả lưu ở:

```text
final/outputs/train_v4_fusion_rerun/
```

Trong đó có:

```text
best.pt
last.pt
history.json
```

Nếu không có GPU:

```powershell
.\scripts\run_train_fusion.ps1 -Device cpu
```

## Test checkpoint fusion có sẵn

```powershell
.\scripts\run_test_fusion.ps1
```

Kết quả test lưu ở:

```text
final/Experiment/reports/packaged_test_fusion_best_test/
```

## Chạy demo UI

```powershell
.\scripts\run_demo_ui.ps1
```

Sau đó mở:

```text
http://127.0.0.1:7860
```

## Các kịch bản train trong repo

Kịch bản chính:

```text
final/code/src_neurosymbolic/train_cross_attention_v4.py
```

Baseline neural-only:

```text
final/code/neural_only_experiments/train_neural_only.py
```

Ablation:

```text
final/code/RUN_GRAPH_ENCODER_LAYER_ABLATION_1_TO_6.ps1
final/code/neural_only_experiments/RUN_TRANSFORMER_LAYER_ABLATION_1_TO_10.ps1
```

Chi tiết lệnh thủ công nằm trong:

```text
final/README.md
```
