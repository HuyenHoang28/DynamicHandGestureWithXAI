# RTMW-L Keypoint Extractor

Thu muc nay gom cac file can thiet de trich xuat keypoint tu video bang RTMW-L/MMPose.

## Thu muc trong repo

- `run_rtmw_splits_dual.py`: script chinh, dung cho ca dataset.
- `extract_single_video.py`: wrapper gon de chay mot video ra mot file JSON.
- `requirements.txt`: goi Python phu tro. `torch`, `mmcv`, `mmdet`, `mmengine` nen cai theo dung CUDA cua may.

Hai thanh phan nang sau khong commit len GitHub. Can tai/copy rieng truoc khi extract:

- `mmpose/`: source MMPose can cho `MMPoseInferencer`, config RTMW-L va metainfo COCO-WholeBody.
- `checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth`: weight RTMW-L.

Script chi giu 52 keypoint can dung: mat/mui/vai/khuu tay/co tay va 21 khop moi ban tay. Output JSON co dang theo frame:

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

## Cai dat moi truong

Nen dung Python 3.10 hoac 3.11. Vi du tren Windows PowerShell:

```powershell
cd keypoint_extractor_package
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Neu may co GPU NVIDIA, cai PyTorch ban CUDA phu hop. Vi du CUDA 12.1:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Neu chi chay CPU:

```powershell
python -m pip install torch torchvision
```

Cai cac goi OpenMMLab va MMPose local:

```powershell
python -m pip install -U openmim
mim install "mmengine" "mmcv>=2.0.0" "mmdet>=3.0.0"
python -m pip install -r requirements.txt
python -m pip install -e .\mmpose
```

Kiem tra nhanh:

```powershell
python -c "from mmpose.apis import MMPoseInferencer; print('MMPose OK')"
```

## Chay mot video

Dat video o dau cung duoc, roi chay:

```powershell
python extract_single_video.py `
  --input-video "C:\path\to\video.mp4" `
  --output-json ".\outputs\video_keypoints.json" `
  --device cuda:0 `
  --pretty
```

Neu khong co GPU, doi `--device cuda:0` thanh:

```powershell
--device cpu
```

Mac dinh script se tim:

- MMPose tai `.\mmpose`
- weight tai `.\checkpoints\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth`
- config tai `.\mmpose\projects\rtmpose\rtmpose\wholebody_2d_keypoint\rtmw-l_8xb320-270e_cocktail14-384x288.py`

Neu dat cac file o noi khac, truyen them:

```powershell
--mmpose-root "C:\path\to\mmpose" `
--pose-weights "C:\path\to\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
```

## Chay ca dataset

Script goc ho tro hai kieu layout.

### Layout raw-subjects

Input:

```text
dataset_root/
  gesture_name/
    subject_name/
      video_1.mp4
      video_2.mp4
```

Lenh chay:

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

### Layout splits

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

Lenh chay:

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

Output mac dinh:

```text
dataset_root/
  keypoints_old/
  keypoints_new/
```

## Ghi chu

- Neu file JSON da ton tai, script se bo qua. Them `--overwrite` de trich lai.
- Mac dinh co swap ten left/right cho video bi mirror. Them `--no-mirror-swap` neu khong muon swap.
- Video ho tro: `.mp4`, `.avi`, `.mov`, `.mkv`.
- Chay GPU se nhanh hon CPU rat nhieu.
