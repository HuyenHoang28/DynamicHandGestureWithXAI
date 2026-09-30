# RTMW-L Keypoint Extractor

Thư mục này chứa các script cần thiết để trích xuất keypoint từ video bằng RTMW-L/MMPose.

## Nội Dung Trong Repo

- `run_rtmw_splits_dual.py`: script chính, dùng để trích xuất keypoint cho cả dataset.
- `extract_single_video.py`: wrapper gọn để chạy một video và xuất ra một file JSON.
- `requirements.txt`: các gói Python phụ trợ. `torch`, `mmcv`, `mmdet`, `mmengine` nên được cài theo đúng phiên bản CUDA của máy.

Hai thành phần nặng sau không được commit lên GitHub. Cần tải hoặc copy riêng trước khi extract:

- `mmpose/`: source MMPose cần cho `MMPoseInferencer`, config RTMW-L và metadata COCO-WholeBody.
- `checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth`: weight RTMW-L.

Script chỉ giữ 52 keypoint cần dùng: mặt/mũi/vai/khuỷu tay/cổ tay và 21 khớp mỗi bàn tay. Output JSON có dạng theo frame:

```json
[
  {
    "frame_id": 0,
    "instances": [
      {
        "keypoints_by_name": {
          "left_wrist": {"xy": [x, y], "score": 0.99}
        }
      }
    ]
  }
]
```

## Cài Đặt Môi Trường

Nên dùng Python 3.10 hoặc 3.11. Ví dụ trên Windows PowerShell:

```powershell
cd keypoint_extractor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Nếu máy có GPU NVIDIA, cài PyTorch bản CUDA phù hợp. Ví dụ CUDA 12.1:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Nếu chỉ chạy CPU:

```powershell
python -m pip install torch torchvision
```

Cài các gói OpenMMLab và MMPose local:

```powershell
python -m pip install -U openmim
mim install "mmengine" "mmcv>=2.0.0" "mmdet>=3.0.0"
python -m pip install -r requirements.txt
python -m pip install -e .\mmpose
```

Kiểm tra nhanh:

```powershell
python -c "from mmpose.apis import MMPoseInferencer; print('MMPose OK')"
```

## Chạy Một Video

Đặt video ở đâu cũng được, rồi chạy:

```powershell
python extract_single_video.py `
  --input-video "C:\path\to\video.mp4" `
  --output-json ".\outputs\video_keypoints.json" `
  --device cuda:0 `
  --pretty
```

Nếu không có GPU, đổi `--device cuda:0` thành:

```powershell
--device cpu
```

Mặc định script sẽ tìm:

- MMPose tại `.\mmpose`
- weight tại `.\checkpoints\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth`
- config tại `.\mmpose\projects\rtmpose\rtmpose\wholebody_2d_keypoint\rtmw-l_8xb320-270e_cocktail14-384x288.py`

Nếu đặt các file ở nơi khác, truyền thêm:

```powershell
--mmpose-root "C:\path\to\mmpose" `
--pose-weights "C:\path\to\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
```

## Chạy Cả Dataset

Script gốc hỗ trợ hai kiểu layout.

### Layout `raw-subjects`

Input:

```text
dataset_root/
  gesture_name/
    subject_name/
      video_1.mp4
      video_2.mp4
```

Lệnh chạy:

```powershell
python run_rtmw_splits_dual.py `
  --repo-root . `
  --data-root "C:\path\to\dataset_root" `
  --layout raw-subjects `
  --output-root "C:\path\to\output_keypoints" `
  --device cuda:0 `
  --pretty
```

Output:

```text
output_keypoints/
  gesture_name/
    subject_name/
      video_1.json
```

### Layout `splits`

Input:

```text
dataset_root/
  splits_old/
    train/
      gesture_name/
        video.mp4
    val/
    test/
  splits_new/
    train/
    val/
    test/
```

Lệnh chạy:

```powershell
python run_rtmw_splits_dual.py `
  --repo-root . `
  --data-root "C:\path\to\dataset_root" `
  --layout splits `
  --dataset both `
  --split all `
  --device cuda:0 `
  --pretty
```

Output mặc định:

```text
dataset_root/
  keypoints_old/
  keypoints_new/
```

## Ghi Chú

- Nếu file JSON đã tồn tại, script sẽ bỏ qua. Thêm `--overwrite` để trích xuất lại.
- Mặc định có swap tên left/right cho video bị mirror. Thêm `--no-mirror-swap` nếu không muốn swap.
- Video hỗ trợ: `.mp4`, `.avi`, `.mov`, `.mkv`.
- Chạy bằng GPU sẽ nhanh hơn CPU rất nhiều.
