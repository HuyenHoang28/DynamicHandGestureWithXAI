# Neuro-Symbolic Gesture Recognition

Repo này chứa bản code gọn để người khác clone về chạy lại phần train/test/demo trong thư mục `final`.

Data cache và checkpoint không được commit trực tiếp vào GitHub để repo nhẹ, dễ clone, và tránh giới hạn dung lượng. Sau khi clone, người dùng cần tải/copy data vào đúng đường dẫn như phần bên dưới.

Để hiểu pipeline IPN 13 lớp hiện tại từ dữ liệu, neural, pooling và augmentation đến symbolic/fusion, đọc [Handbook toàn diện IPN neuro-symbolic](docs/HANDBOOK_TOAN_DIEN_IPN_NEUROSYMBOLIC_VI.md). Quy trình viết, kiểm chứng và tích hợp template nằm trong [Playbook template IPN 13 lớp](docs/IPN_TEMPLATE_AUTHORING_PLAYBOOK_VI.md).

Notebook Colab để huấn luyện/đánh giá learned `kg_logits` trên IPN-13 và so
sánh với neural-only baseline nằm tại
[`notebooks/IPN13_KG_Logits_Baseline_Comparison_Colab.ipynb`](notebooks/IPN13_KG_Logits_Baseline_Comparison_Colab.ipynb).
Protocol chống test leakage và cách đọc kết quả được mô tả tại
[`docs/IPN13_KG_LOGITS_EXPERIMENT_PROTOCOL_VI.md`](docs/IPN13_KG_LOGITS_EXPERIMENT_PROTOCOL_VI.md).

## Nội dung repo

```text
final/code/                         Code train, test, demo
final/data/Input/                   Không commit; cần tải/copy riêng
final/data/Output/checkpoints/      Không commit; cần tải/copy riêng nếu muốn test/demo ngay
final/scripts/                      Script vẽ confusion matrix
results/runs/                       Kết quả chạy mới, có thể tái lập
results/final/                      Bundle checkpoint/evidence chính
results/experiments/                Thử nghiệm, ablation và artifact lịch sử
keypoint_extractor/                 Script trích xuất keypoint RTMW-L từ video
requirements.txt                    Thư viện Python cần cài
docs/HANDBOOK_TOAN_DIEN_IPN_NEUROSYMBOLIC_VI.md  Handbook IPN 13 lớp hiện tại
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

## Extract keypoint từ video gốc

Nếu chỉ train/test từ `final/data/Input/keypoint_tensor_cache` có sẵn thì không cần bước này.

Nếu muốn tự trích xuất keypoint từ video gốc, dùng thư mục:

```text
keypoint_extractor/
```

Thư mục này đã có:

```text
extract_single_video.py
run_rtmw_splits_dual.py
requirements.txt
README.md
```

Không commit kèm `mmpose/` và weight RTMW-L vì khá nặng. Người dùng cần tự chuẩn bị:

```text
keypoint_extractor/mmpose/
keypoint_extractor/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth
```

Xem hướng dẫn chi tiết trong:

```text
keypoint_extractor/README.md
```

Ví dụ chạy một video:

```powershell
cd keypoint_extractor
python extract_single_video.py `
  --input-video "C:\path\to\video.mp4" `
  --output-json ".\outputs\video_keypoints.json" `
  --device cuda:0 `
  --pretty
```

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

Chạy unit test:

```powershell
python -m pytest -q
```

Nếu mọi thứ đủ, script sẽ báo `OK`.

## Chạy lại train chính

Mô hình chính cuối cùng là V4 neuro-symbolic fusion:

```powershell
.\scripts\run_train_fusion.ps1
```

Kết quả lưu ở:

```text
results/runs/date27_v4_fusion_train/
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
results/runs/date27_v4_validation_selected/
```

Wrapper mặc định dùng `best.pt`. `best_test.pt` đã dùng test để chọn checkpoint và chỉ được phép chạy khi truyền rõ `-AllowTestSelected`; không dùng kết quả đó làm headline.

Bundle cuối cùng để báo cáo nằm tại `results/final/date27_v4_validation_selected/`.

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
