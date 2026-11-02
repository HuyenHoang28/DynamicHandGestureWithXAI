# Adapter IPN HandS và quy trình huấn luyện lại C → V4

Ngày cập nhật: 2026-11-02

## Mục tiêu

Tài liệu này khóa data contract để dùng bản IPN HandS chính thức với kiến trúc trong bản thảo: nhánh neural Transformer của thí nghiệm C, nhánh symbolic/graph và đầu ra decision fusion của V4. Dữ liệu HandS phải được chuyển và dựng lại cache/instance graph; không được tái sử dụng tensor hoặc instance graph RTMW cũ.

## Thay đổi

- `final/code/src_neurosymbolic/pose/ipn_hands_adapter.py` chuyển landmark GT liên tục theo video và temporal annotation của IPN HandS thành JSON theo từng clip mà pipeline hiện tại đọc được.
- `scripts/prepare_ipn_hands.py` là entry point chạy từ thư mục gốc repository.
- Adapter giữ protocol subject hiện có trong `external_data/ipn_processed/split_subjects.json`, vì vậy train/validation/test vẫn so sánh được với thí nghiệm trước và không bị trộn subject.
- Taxonomy HandS được ánh xạ về IPN-13 hiện tại: `1..13 → B0A, B0B, G01..G11`. `NOG` (0) và lớp mới `GRB` (14) bị loại có chủ đích vì head neural và template symbolic hiện có chỉ có 13 lớp.

## Data contract và giả định tương thích

- HandS cung cấp tọa độ `x/y` chuẩn hóa. Adapter đổi về pixel 640×480, sau đó giữ nguyên bước shoulder normalization của pipeline.
- Các điểm pose dùng trực tiếp: nose, hai mắt, hai vai và hai khuỷu. Hai wrist lấy từ hand landmark 0 vì bộ pose HandS không phát hành pose index 15/16.
- 21 điểm mỗi tay được giữ theo thứ tự MediaPipe và theo handedness giải phẫu `Left/Right` của HandS; adapter không mirror-swap lần nữa.
- HandS không có confidence cho từng điểm. Điểm được phát hành tồn tại được gán `score=1.0`; điểm không tồn tại được để thiếu. Đây là cờ hiện diện nhị phân, không phải xác suất của detector. Chỉ `x/y` của hand landmark là GT đã sửa thủ công; các pose point phụ trợ vẫn là đầu ra MediaPipe chưa sửa.
- Canonical RTMW có `face-8` nhưng HandS không có chin tương ứng. Adapter dùng trung điểm hai mouth corner MediaPipe pose 9/10 làm proxy lower-face và ghi quyết định này vào `adapter_report.json`.
- Mỗi clip dùng interval đóng `[t_start, t_end]` của temporal annotation chính thức.

Các khác biệt trên có thể làm metric thay đổi so với RTMW. Kết quả mới phải được gọi là kết quả trên **IPN HandS GT adapter**, không trộn trực tiếp với checkpoint/cache RTMW như thể cùng một đầu vào.

## Chuyển dữ liệu

Từ thư mục gốc repository:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\prepare_ipn_hands.py
```

Đầu ra mặc định nằm ở `external_data/ipn_hands_prepared/` gồm `keypoints/`, `manifest.csv`, ba split file, `conversion_status.json` và `adapter_report.json`. Adapter preflight toàn bộ danh sách video, frame coverage và annotation trước khi ghi clip; report chứa fingerprint SHA-256 gắn với version data contract. Adapter từ chối thư mục đầu ra không rỗng; `--overwrite` cũng bị từ chối nếu đã có tensor cache/instance graph. Khi input hoặc adapter contract đổi, dùng output root mới rồi dựng lại toàn bộ cache/graph, không trộn artifact cũ.

Lần chuyển ngày 2026-11-02 tạo 4.540 clip: 2.649 train, 650 validation và 1.241 test; tương ứng 29/7/13 subject, không giao nhau. Adapter loại 2.545 interval `NOG`, 534 interval `GRB` và bốn video của subject đã bị loại trong protocol trước.

## Dựng cache và instance graph mới

```powershell
$env:PYTHONPATH='final\code'
.\.venv\Scripts\python.exe -u final\code\src_neurosymbolic\precompute_keypoint_tensors.py `
  --keypoint-root external_data\ipn_hands_prepared\keypoints `
  --cache-root external_data\ipn_hands_prepared\keypoint_tensor_cache `
  --train-file external_data\ipn_hands_prepared\train.txt `
  --val-file external_data\ipn_hands_prepared\val.txt `
  --test-file external_data\ipn_hands_prepared\test.txt

.\.venv\Scripts\python.exe -u final\code\src_neurosymbolic\precompute_instance_graphs_phase_v2.py `
  --keypoint-root external_data\ipn_hands_prepared\keypoints `
  --cache-root external_data\ipn_hands_prepared\keypoint_tensor_cache `
  --output-root external_data\ipn_hands_prepared\instance_graphs_phase_v2 `
  --train-file external_data\ipn_hands_prepared\train.txt `
  --val-file external_data\ipn_hands_prepared\val.txt `
  --test-file external_data\ipn_hands_prepared\test.txt
```

Template graph 13 lớp vẫn có thể dùng từ `external_data/ipn_processed/template_graphs_v2`; instance graph phải dựng lại từ HandS.

## Huấn luyện và đánh giá

Thứ tự bắt buộc để trả lời đúng câu hỏi nghiên cứu:

1. Huấn luyện neural-only C từ đầu trên cache HandS với motion features và `masked_mean`.
2. Chỉ chọn checkpoint bằng validation Macro-F1; đánh giá test một lần sau khi cấu hình đã khóa.
3. Khởi tạo V4 từ checkpoint C **vừa huấn luyện trên HandS**, bật `--neural-motion-features --neural-pooling masked_mean`, rồi huấn luyện graph/cross-attention/fusion.
4. Báo riêng neural logits, KG logits và final fusion logits trên cùng locked test split. Không chỉ báo final accuracy.

Cấu hình C tương thích thử nghiệm trước: 52 điểm × (x, y, score, vx, vy), T=64, Transformer `d_model=192`, 4 layer, 8 head, dropout 0,15, label smoothing 0,05, class-weight power 0,5, jitter 0,005, keypoint dropout 0,02, seed 42, chọn Macro-F1 và patience 12. Để có claim khoa học, chạy tối thiểu ba seed và báo mean ± std; một seed chỉ là kiểm tra tái lập/pipeline.

V4 dùng decision fusion của bản thảo:

`final_logits = gate × neural_logits / temperature_neural + (1 - gate) × kg_logits / temperature_graph`

Trong code, `gate` là trọng số neural. Khi khởi tạo từ C, dùng `--freeze-neural-epochs 3` để nhánh graph khởi động mà không phá ngay encoder neural, và giữ việc chọn checkpoint hoàn toàn trên validation.

## An toàn và validation

- Adapter ghi atomic từng JSON, ghi trạng thái `running/failed/complete`, và từ chối input thiếu video/frame/annotation, interval hở/chồng, class lạ, handedness lạ hoặc hand trùng trong một frame.
- Unit test bao phủ mapping 21 điểm tay, layout canonical 52 điểm, missing/duplicate hand, taxonomy, parser annotation/subject, preflight video/coverage, overwrite/cache guard và ca chuyển đổi end-to-end.
- Validation ngày 2026-11-02: `python -m pytest tests/test_ipn_hands_adapter.py -q` → 13 passed; manifest 4.540 dòng, đủ 13 label ID và không giao subject giữa ba split.
- Dữ liệu sinh, tensor cache, graph và checkpoint là artifact cục bộ; không stage/version chúng nếu không có yêu cầu riêng.

## 2026-11-02: bộ dữ liệu đầy đủ (không loại subject) và tùy chọn 14 lớp

### Thay đổi

- `final/code/src_neurosymbolic/pose/ipn_hands_adapter.py`: thêm tùy chọn `--include-grab` (tham số `include_grab`, mặc định tắt). Khi bật, lớp HandS `GRB` (id 14) được giữ với `label_id = 13`, tổng 14 lớp như paper IPN HandS; `NOG` vẫn bị loại. Báo cáo ghi `taxonomy.include_grab`, và cờ này nằm trong `input_fingerprint` nên hai chế độ không bị trộn artifact. Chạy mặc định không đổi.
- `tests/test_ipn_hands_adapter.py`: thêm test end-to-end cho `include_grab` (GRB có label 13, `NOG` vẫn bị loại, mặc định vẫn 13 lớp, fingerprint khác nhau).
- Dữ liệu cục bộ (không version hóa): `external_data/split_subjects_full.json` là bản của `split_subjects.json` với `1CM42_3` chuyển từ `excluded_subjects` sang `train_subjects` (validation và test không đổi); `external_data/ipn_hands_full/` (13 lớp, không loại subject) và `external_data/ipn_hands_full14/` (14 lớp).

### Lý do

Các baseline dùng đủ 50 subject, và paper IPN HandS dùng 14 lớp gồm `grab`. Bộ trước loại subject `1CM42_3` (mất một vai trong keypoint RTMW) và lớp `GRB` vì template symbolic/mô hình ban đầu là 13 lớp, không phải vì lý do khoa học. Trong dữ liệu HandS, `1CM42_3` có hai vai: vai phải nằm ngoài khung ở 95% frame và là tọa độ ngoại suy của MediaPipe; độ rộng vai ngoại suy 0,379 so với 0,342 khi vai nằm trong ảnh, mũi lệch trung điểm vai 0,004 so với 0,007, nên ước lượng dùng được. Không cần viết bộ ước lượng vai riêng cho bộ HandS. `1CM42_3` thuộc train chính thức (4 video) nên được đưa vào train, giữ val và test như cũ (subject không xuất hiện ở hai split).

### Kết quả dựng dữ liệu

| Bộ | Video | Train | Val | Test |
|---|---:|---:|---:|---:|
| `ipn_hands_prepared` (cũ, 13 lớp) | 196 | 2.649 | 650 | 1.241 |
| `ipn_hands_full` (13 lớp) | 200 | 2.746 | 650 | 1.241 |
| `ipn_hands_full14` (14 lớp) | 200 | 3.066 | 725 | 1.380 |

Val và test của `ipn_hands_full` giống hệt bộ cũ (so sánh danh sách clip). Test của `ipn_hands_full14` là 1.380 clip, khớp số test trong paper IPN HandS; train + val là 3.791 so với 3.797 trong paper (lệch 6 clip, chưa rõ lý do). Có 534 clip `GRB` (320 train, 75 val, 139 test). Cache tensor và đồ thị của bộ 14 lớp tái sử dụng bản 13 lớp (các clip trùng), chỉ tính thêm cho `GRB`. Không clip nào lỗi.

### Tương thích và hạn chế

- Neural C chạy được với `--num-classes 14`. V4 **chưa** chạy được ở 14 lớp: cần thêm template graph cho `GRB` (label_id 13) vì encoder lấy số template theo lớp. Template hiện có (`external_data/ipn_processed/template_graphs_v2`) chỉ có 13 lớp.
- Mốc thời gian của lớp G03/G04 (throw) trong HandS đã bỏ phần grab đầu, nên khác IPN gốc.
- Dùng `python -m unittest`/shim thay pytest vì `.venv` không có pytest: 13/14 test của adapter chạy; test còn lại (`pytest.approx` trên list) không chạy được với shim, không liên quan đến thay đổi này. Cần chạy lại bằng pytest thật.
