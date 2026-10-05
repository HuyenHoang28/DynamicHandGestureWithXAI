# Kế hoạch 3 người: chạy project trên IPN Hand

## Mục tiêu và phạm vi

Mục tiêu trước mắt là **đo được mô hình nhận dạng đúng bao nhiêu cử chỉ IPN**, không chỉ kiểm tra video có chạy qua pipeline hay không. Làm bài toán **phân loại clip cử chỉ đã cắt** trước: 13 lớp `B0A`, `B0B`, `G01`–`G11`. Tạm không đưa `D0X` (đoạn không cử chỉ) vào classifier 13 lớp. Sau khi có kết quả này mới mở rộng sang 14 lớp có `D0X` hoặc phát hiện cử chỉ trong video liên tục.

Checkpoint V4 hiện tại được huấn luyện cho 27 nhãn khác IPN. Không được dùng dự đoán của checkpoint đó để tính accuracy IPN. Thử nghiệm 16 clip đã làm chỉ là kiểm tra pipeline và hành vi ngoài miền; báo cáo thử nghiệm nằm tại `final/outputs/ipn_smoke_20261005/BAO_CAO_KET_QUA_THU_NGHIEM_IPN_HAND.md` trên máy có dữ liệu (artifact này đang được Git ignore).

Nguồn chính thức: [IPN Hand](https://gibranbenitez.github.io/IPN_Hand/) và [benchmark code](https://github.com/GibranBenitez/IPN-hand). IPN đã có annotation cho từng khoảng frame, nên nhóm **không cần gán nhãn lại bằng tay**. Cần chuyển annotation thành clip/manifest và kiểm tra nhãn bằng mắt trên một tập mẫu.

## Quy ước chung — thống nhất trước khi chia việc

| Nhãn IPN | ID dùng để train | Ý nghĩa |
|---|---:|---|
| B0A | 0 | Trỏ bằng một ngón |
| B0B | 1 | Trỏ bằng hai ngón |
| G01 | 2 | Click bằng một ngón |
| G02 | 3 | Click bằng hai ngón |
| G03 | 4 | Đẩy lên |
| G04 | 5 | Đẩy xuống |
| G05 | 6 | Đẩy sang trái |
| G06 | 7 | Đẩy sang phải |
| G07 | 8 | Mở hai lần |
| G08 | 9 | Double click bằng một ngón |
| G09 | 10 | Double click bằng hai ngón |
| G10 | 11 | Zoom vào |
| G11 | 12 | Zoom ra |

`classIdx.txt` gốc của IPN đánh ID từ 1 đến 14 và có `D0X=1`. Bảng trên là **mapping mới 0–12 của thí nghiệm 13 lớp**. Mọi manifest, checkpoint, template graph và báo cáo phải dùng đúng cùng mapping. Lưu mapping này trong tài liệu, không suy đoán ID từ thứ tự tên file.

Giữ train/test chính thức của IPN: **148 video từ 37 người train, 52 video từ 13 người test**. Tạo validation bằng cách chọn người từ 37 người train; không chia ngẫu nhiên clip của cùng người sang cả train và validation, và không dùng test để chọn epoch hay sửa template.

Đặt tên clip duy nhất, ví dụ `1CM1_1_R_217_G10_f000029_f000072.avi`. Manifest tối thiểu phải lưu: video nguồn, subject, split, nhãn IPN, ID 0–12, frame bắt đầu/kết thúc, đường dẫn clip và đường dẫn keypoint. Mọi output mới của IPN đặt ở thư mục riêng; không ghi đè dữ liệu và template 27 lớp hiện có.

## Người 1 — video, split và chất lượng dữ liệu

**Việc làm**

1. Kiểm kê và giải nén dữ liệu. Trên máy hiện tại, `external_data` đã có ba ZIP video chứa `videos01.tgz`–`videos05.tgz` và một ZIP annotation. Chỉ tải thêm từ [trang chính thức](https://gibranbenitez.github.io/IPN_Hand/) nếu thiếu file; tránh tải lại khoảng 4,6 GB không cần thiết.
2. Đối chiếu video giải nén với `Video_TrainList.txt`, `Video_TestList.txt` và `metadata.csv`. Mục tiêu: 200 video mở được, 148 train, 52 test, không trùng video/người giữa hai split.
3. Chọn validation theo người từ nhóm train. Ghi danh sách subject và seed/quy tắc chọn để cả nhóm tái lập được.
4. Khảo sát video khó: tay trái/phải, ánh sáng tối, nền lộn xộn, nhiều người; gửi ví dụ cho người 2 và 3 để kiểm tra pipeline/template.

**Lệnh giải nén tham khảo (PowerShell, chạy từ thư mục gốc project):**

```powershell
$ipnRoot = 'external_data\ipn_full'
New-Item -ItemType Directory -Force -Path $ipnRoot | Out-Null
Get-ChildItem 'external_data' -Filter 'videos-*.zip' -File |
  ForEach-Object { Expand-Archive -LiteralPath $_.FullName -DestinationPath $ipnRoot }
Expand-Archive -LiteralPath 'external_data\annotations-20261005T055143Z-1-001.zip' -DestinationPath $ipnRoot
Get-ChildItem "$ipnRoot\videos" -Filter '*.tgz' -File |
  ForEach-Object { tar -xzf $_.FullName -C $ipnRoot }
(Get-ChildItem "$ipnRoot\videos" -Filter '*.avi' -File).Count
```

Lệnh cuối kỳ vọng `200`. Các tệp nguồn trên máy hiện có đuôi `.avi` bên trong `.tgz`; kiểm tra file thực tế thay vì giả định đuôi `.mp4` theo mô tả trang tải.

**Bàn giao/nghiệm thu:** danh sách 200 video với split và subject; danh sách validation theo người; biên bản video thiếu/hỏng (nếu có); vài video mẫu cho mỗi điều kiện khó. Không bắt đầu đánh giá nếu train/test có subject trùng nhau.

## Người 2 — cắt clip, nhãn, keypoint và baseline

**Việc làm**

1. Đọc `Annot_TrainList.txt`/`Annot_TestList.txt`. Mỗi dòng có dạng `video,label,id,t_start,t_end,frames`. Với 13-class isolated classification, bỏ `D0X`, giữ đủ các dòng B0A/B0B/G01–G11.
2. Cắt clip từ `t_start` **đến cả** `t_end`. Annotation đánh số từ 1; nếu dùng OpenCV thì seek đến frame `t_start - 1`. Xác nhận `decoded_frames == t_end - t_start + 1 == frames` trên mọi clip.
3. Tạo `train`, `val`, `test` theo danh sách subject của người 1. Không tách validation từ test. Tạo manifest với mapping ID ở trên và tên file duy nhất.
4. Mở và kiểm tra bằng mắt ít nhất vài clip mỗi lớp, ưu tiên cặp dễ nhầm: G01/G08, G02/G09, G03/G04, G05/G06, G10/G11. Kiểm tra tay không bị mất ở đầu/cuối clip.
5. Trích xuất RTMW keypoint, ghi tỷ lệ frame không phát hiện người/không đủ hai vai, rồi precompute tensor 64 frame. Không gọi `valid=100%` là keypoint chính xác 100%; đó chỉ là điều kiện để chuẩn hóa.
6. Train **neural-only 13 lớp** làm baseline. Chọn checkpoint theo validation, sau đó chạy test một lần và báo accuracy, macro-F1, per-class precision/recall và confusion matrix.

Thư mục clip nên theo layout mà extractor hỗ trợ, ví dụ `external_data/ipn_processed/clips/train/G10/Subject01/<clip>.avi`; tương tự cho `val` và `test`. Với máy đã cài dependencies và checkpoint RTMW như workspace hiện tại, lệnh trích một split là:

```powershell
.\.venv\Scripts\python.exe keypoint_extractor\run_rtmw_splits_dual.py `
  --repo-root keypoint_extractor `
  --data-root external_data\ipn_processed\clips\train `
  --layout raw-subjects `
  --output-root external_data\ipn_processed\keypoints\train `
  --mmpose-root .venv\Lib\site-packages\mmpose\.mim `
  --pose-weights final\data\Output\checkpoints\rtmw\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth `
  --device cuda:0
```

Lặp lại với `val` và `test`. Kiểm tra trên vài clip xem có cần thay đổi `--no-mirror-swap`; mặc định script hoán đổi tên keypoint trái/phải cho video gương. Chỉ dùng một quy ước nhất quán cho toàn bộ train/val/test.

**Ví dụ quy ước manifest ba cột mà code hiện tại đọc:**

```text
external_data/ipn_processed/keypoints/train/B0A/Subject01/clip_001.json 0 0
external_data/ipn_processed/keypoints/train/G10/Subject03/clip_002.json 0 11
```

Cột cuối là label ID. Đường dẫn cần chứa `train/<nhãn>/...`, `val/<nhãn>/...` hoặc `test/<nhãn>/...`, vì code đọc tên nhãn từ đường dẫn. Dùng đúng một mapping cho cả ba split. Hai file train/val/test thực tế phải được tạo từ annotation; không lấy `train.txt` hiện có của bộ 27 lớp.

**Lệnh train baseline sau khi đã có keypoint cache và manifest:**

```powershell
.\.venv\Scripts\python.exe final\code\neural_only_experiments\train_neural_only.py `
  --backbone transformer `
  --cache-root external_data\ipn_processed\keypoint_tensor_cache `
  --train-file external_data\ipn_processed\train.txt `
  --val-file external_data\ipn_processed\val.txt `
  --num-classes 13 --num-frames 64 `
  --output-dir final\outputs\ipn_neural_only `
  --device cuda:0
```

Máy hiện tại chưa phát hiện CUDA; nếu chạy ở máy này, thay `cuda:0` bằng `cpu`, nhưng trích xuất và train toàn bộ IPN sẽ chậm. Script thử 16 clip `scripts/prepare_ipn_smoke.py` **không** chuẩn bị toàn bộ 5.649 annotation, không được dùng nó làm pipeline train chính thức.

**Bàn giao/nghiệm thu:** clip và manifest đúng frame/nhãn; thống kê số clip mỗi lớp và split; tỷ lệ keypoint hợp lệ; checkpoint baseline chọn bằng validation; báo cáo test kèm prediction từng clip. Không báo accuracy nếu chưa kiểm tra nhãn và không gian đầu ra.

## Người 3 — thiết kế và kiểm định template graph IPN

**Việc làm**

1. Xem ví dụ video của từng nhãn từ [IPN Hand](https://gibranbenitez.github.io/IPN_Hand/) và clip đã cắt. Viết mô tả gồm tư thế đầu, chuyển động, tư thế cuối và tín hiệu phân biệt với lớp gần giống. Ví dụ G01/G08 khác ở số lần click; G10/G11 khác ở chiều biến đổi khoảng cách/động tác zoom.
2. Kiểm tra [extract_predicates.py](../final/code/src_neurosymbolic/predicates/extract_predicates.py) và [build_events.py](../final/code/src_neurosymbolic/events/build_events.py): chỉ dùng predicate/event mà code **thực sự tạo được**. Nếu thiếu tín hiệu để phân biệt một lớp, ghi rõ khoảng trống; không tự đặt tên event mới trong template rồi kỳ vọng matcher tìm thấy.
3. Viết file `gesture_templates.yaml` cho 13 nhãn IPN theo cú pháp mà [build_template_graph.py](../final/code/src_neurosymbolic/kg/build_template_graph.py) hỗ trợ: `gestures`, `expected_events`, `subject`, nhóm `motions`/`hand_shapes`/`finger_motions`/`positions`..., mức `required`/`preferred`/`optional`, và `negative_evidence` khi có căn cứ. File nguồn YAML cho 27 nhãn cũ **không có trong workspace hiện tại**; có thể xem các JSON cũ trong `final/data/Input/template_graphs` để tham khảo cấu trúc, nhưng không sao chép rồi đổi tên.
4. Sinh 13 graph vào `external_data/ipn_processed/template_graphs`, không ghi đè `final/data/Input/template_graphs`. Cấp đúng `label_id` từ manifest; không tin rằng `labels.yaml` tự quyết định ID, vì loader hiện tại lấy ID từ các split file.
5. Kiểm tra `index.json` có đúng 13 nhãn, ID 0–12 liên tục, mỗi graph có event/requirement hợp lệ. Chạy thử matcher trên clip mẫu của lớp đó và các lớp dễ nhầm; ghi cả trường hợp template thất bại.

Ví dụ cú pháp tối thiểu cho một phần file YAML, **chỉ để minh họa định dạng**; cần kiểm tra event này thực sự xuất hiện trong clip G03 trước khi dùng làm template cuối:

```yaml
version: 1
gestures:
  G03:
    description: Throw hand upward
    expected_events:
      - subject: any
        motions:
          required:
            - hand_moves_up
```

**Lệnh sinh graph sau khi người 2 bàn giao manifest:**

```powershell
.\.venv\Scripts\python.exe final\code\src_neurosymbolic\kg\build_template_graph.py `
  --templates external_data\ipn_processed\gesture_templates.yaml `
  --list-file external_data\ipn_processed\train.txt `
  --list-file external_data\ipn_processed\val.txt `
  --list-file external_data\ipn_processed\test.txt `
  --output-dir external_data\ipn_processed\template_graphs `
  --pretty
```

**Bàn giao/nghiệm thu:** bảng nhãn–định nghĩa–predicate/event; 13 graph và index đúng mapping; bằng chứng template nhận ra ví dụ đúng và phân biệt lớp gần giống; danh sách predicate/event còn thiếu nếu có. Không đánh giá template bằng cách sửa theo kết quả trên test split.

## Mốc ghép việc và báo cáo cuối

1. **Mốc A — dữ liệu:** Người 1 xác nhận video và split; người 2 bàn giao ít nhất một clip chuẩn mỗi lớp. Người 3 có thể bắt đầu viết mô tả cử chỉ ngay từ đây.
2. **Mốc B — baseline:** Người 2 hoàn tất keypoint/manifest, train neural-only, báo kết quả validation và test. Đây là mốc tối thiểu để trả lời thầy “model nhận đúng bao nhiêu trên IPN”.
3. **Mốc C — V4:** Người 3 bàn giao 13 template đã kiểm định. Nhóm train V4 trên cùng train/val/test và so sánh với neural-only; giữ cách chọn checkpoint theo validation.
4. **Mốc D — báo cáo:** Ghi kích thước từng split, mapping nhãn, số frame/keypoint lỗi, hyperparameter, accuracy, macro-F1, confusion matrix, lỗi điển hình và giới hạn. Phân biệt rõ kết quả **isolated classification** với bài toán **continuous recognition**.

Tiêu chí hoàn thành: manifest–checkpoint–template thống nhất 13 nhãn; không rò rỉ người giữa các split; clip đúng annotation; metric được tính trên test chính thức; thí nghiệm chạy lại được bằng lệnh và cấu hình đã lưu.

## Lưu ý khi chia sẻ qua Git

File hướng dẫn này nằm trong `docs/` và **không bị `.gitignore` bỏ qua**, có thể commit. Dữ liệu `external_data/`, checkpoint và output thử nghiệm đang bị Git ignore để không đẩy hàng GB video lên repository. Nếu ba người làm trên các máy khác nhau, chia sẻ dữ liệu qua ổ chung/Drive và giữ manifest, mapping, phiên bản script giống nhau; đừng giả định `git pull` sẽ mang theo video đã giải nén.
