# Kế hoạch 3 người: thử nghiệm project trên IPN Hand

## Mục tiêu và phạm vi

Mục tiêu trước mắt là **đo được mô hình nhận dạng đúng bao nhiêu cử chỉ IPN**, không chỉ kiểm tra video có chạy qua pipeline hay không. Làm bài toán **phân loại clip cử chỉ đã cắt** trước: 13 lớp `B0A`, `B0B`, `G01`–`G11`. Tạm không đưa `D0X` (đoạn không cử chỉ) vào classifier 13 lớp. Sau khi có kết quả này mới mở rộng sang 14 lớp có `D0X` hoặc phát hiện cử chỉ trong video liên tục.

Checkpoint V4 hiện tại được huấn luyện cho 27 nhãn khác IPN. Không được dùng dự đoán của checkpoint đó để tính accuracy IPN. Thử nghiệm 16 clip đã làm chỉ là kiểm tra pipeline và hành vi ngoài miền; báo cáo thử nghiệm nằm tại `final/outputs/ipn_smoke_20261005/BAO_CAO_KET_QUA_THU_NGHIEM_IPN_HAND.md` trên máy có dữ liệu (artifact này đang được Git ignore).

Nguồn chính thức: [IPN Hand](https://gibranbenitez.github.io/IPN_Hand/) và [benchmark code](https://github.com/GibranBenitez/IPN-hand). IPN đã có annotation cho từng khoảng frame, nên nhóm **không cần gán nhãn lại bằng tay**. Cần chuyển annotation thành clip/manifest và kiểm tra nhãn bằng mắt trên một tập mẫu.

## Bước 0 — cả ba tự tải dữ liệu về máy riêng

**Hai thành viên còn lại chưa có dữ liệu: mỗi người cần tải video và annotation về máy mình.** Dữ liệu đã có trên một máy không tự xuất hiện trên máy khác khi clone/pull Git. Cả ba có thể tải và chuẩn bị môi trường song song; người đã có đủ file chỉ cần kiểm tra lại.

1. Clone/pull project về máy riêng.
2. Vào [trang tải chính thức IPN Hand](https://gibranbenitez.github.io/IPN_Hand/), tải hai mục:
   - **MP4 videos:** đủ `videos01.tgz` đến `videos05.tgz`, tổng khoảng 4,6 GB, chứa 200 video. [Thư mục video](https://drive.google.com/drive/folders/1O4Fn_jbAEcKIksXHmQIMcHTyaDw4wt9x?usp=drive_open).
   - **Annotations:** tải toàn bộ thư mục, gồm `Annot_List.txt`, `Annot_TrainList.txt`, `Annot_TestList.txt`, `Video_TrainList.txt`, `Video_TestList.txt`, `classIdx.txt`, `metadata.csv`. [Thư mục annotation](https://drive.google.com/drive/folders/1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD?usp=drive_open).
3. Nếu Google Drive trả nhiều file ZIP, tải đủ các phần và giải nén lớp ZIP trước. Tên ZIP theo ngày tải có thể khác giữa ba máy; không đổi tên các video hoặc annotation bên trong.
4. Đặt năm file `.tgz` vào `external_data/ipn_archives/`. Đặt các file annotation đã giải nén vào `external_data/ipn_full/annotations/`. Giải nén video theo hướng dẫn bên dưới rồi kiểm tra đủ 200 video.
5. Chỉ cần video RGB và annotation cho pipeline RTMW này; chưa cần tải thêm RGB frames, optical flow hay segmentation masks.
6. Cài môi trường project và chuẩn bị checkpoint RTMW trên từng máy chạy trích keypoint. `.venv`, dữ liệu và checkpoint không được mang theo qua Git; các đường dẫn trong lệnh dưới đây là vị trí cần có sau khi thiết lập, không phải tài nguyên đã được tải tự động.

Mỗi thành viên báo lại: đã có đủ năm `.tgz`, giải nén được 200 video, mở được một video mẫu và đọc được annotation. Người 1 tổng hợp checklist của cả nhóm và thống nhất split/manifest; người 2 và 3 không tự tạo một cách chia train/val/test khác.

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

## Phân công cân bằng và cách làm song song

| Người | Chịu trách nhiệm chính | Bàn giao cuối |
|---|---|---|
| 1 — Dữ liệu | Video, split, công cụ cắt toàn bộ clip, trích keypoint và kiểm tra chất lượng | Clip/keypoint/manifest train–val–test có thể tái tạo |
| 2 — Mô hình và đo kết quả | Train neural-only 13 lớp, chọn checkpoint, tính metric và phân tích lỗi | Baseline cùng báo cáo đúng/sai trên test |
| 3 — Template và V4 | Thiết kế 13 template, kiểm tra predicate/event, train V4 và so sánh với baseline | Graph IPN, checkpoint V4 và bảng so sánh |

Ba người **bắt đầu song song bằng bước tải và kiểm tra dữ liệu trên máy riêng**. Người 1 xây quy trình cắt và bàn giao sớm danh sách frame/lệnh tạo ít nhất một clip mỗi lớp; người 2 và 3 dùng cùng quy trình với video đã tải trên máy mình. Trong lúc đó người 2 chuẩn bị pipeline train/metric trên một tập nhỏ, người 3 mô tả động tác và kiểm tra predicate/event. Việc train toàn bộ của người 2 chỉ chờ manifest/keypoint hoàn chỉnh; việc train V4 của người 3 chờ cả dữ liệu lẫn template được xác nhận. Keypoint/cache có thể nhận qua ổ chung/Drive để tránh chạy lại RTMW trên cả ba máy; đây là dữ liệu xử lý bổ sung, không thay thế bước mỗi người tải video và annotation.

Phần xem clip bằng mắt chia đều theo **4–4–5 lớp**, thay vì dồn cho một người: người 1 kiểm tra B0A, B0B, G01, G02; người 2 kiểm tra G03–G06; người 3 kiểm tra G07–G11. Mỗi người ghi các clip có nhãn/cắt sai hoặc keypoint bất thường; người 1 tổng hợp lỗi dữ liệu, người 3 tổng hợp lỗi template. Đây là kiểm tra mẫu, không thay thế kiểm tra tự động toàn bộ clip.

## Người 1 — dữ liệu, clip và keypoint

**Việc làm**

1. Tải, giải nén dữ liệu trên máy mình theo bước 0 và kiểm kê tình trạng tải của hai thành viên còn lại. Người 1 chịu trách nhiệm quy trình dữ liệu chung, không phải người duy nhất tải dữ liệu cho cả nhóm.
2. Đối chiếu video giải nén với `Video_TrainList.txt`, `Video_TestList.txt` và `metadata.csv`. Mục tiêu: 200 video mở được, 148 train, 52 test, không trùng video/người giữa hai split.
3. Chọn validation theo người từ nhóm train. Ghi danh sách subject và seed/quy tắc chọn để cả nhóm tái lập được.
4. Đọc `Annot_TrainList.txt`/`Annot_TestList.txt`. Mỗi dòng có dạng `video,label,id,t_start,t_end,frames`. Với 13 lớp, bỏ `D0X`, giữ B0A/B0B/G01–G11. Cắt từ `t_start` **đến cả** `t_end`; nếu dùng OpenCV thì seek đến frame `t_start - 1`. Kiểm tra tự động `decoded_frames == t_end - t_start + 1 == frames` cho mọi clip.
5. Tạo clip/manifest train–val–test theo subject đã chốt. Đặt tên clip duy nhất và giữ mapping ID 0–12 ở trên. Bàn giao **một clip mỗi lớp càng sớm càng tốt** để người 2 và 3 tiếp tục làm song song.
6. Trích xuất RTMW keypoint, ghi tỷ lệ frame không phát hiện người/không đủ hai vai, rồi precompute tensor 64 frame. Kiểm tra trên vài clip xem có cần `--no-mirror-swap`; dùng cùng quy ước trái/phải cho ba split. `valid=100%` chỉ nói đủ anchor chuẩn hóa, không chứng minh keypoint chính xác 100%.
7. Khảo sát video khó: tay trái/phải, ánh sáng tối, nền lộn xộn, nhiều người; chia sẻ ví dụ và lỗi dữ liệu cho cả nhóm.

**Lệnh giải nén cho từng máy (PowerShell, chạy từ thư mục gốc project):**

Trước khi chạy, đã giải nén lớp ZIP nếu có, đặt đủ năm `.tgz` vào `external_data/ipn_archives/` và annotation vào `external_data/ipn_full/annotations/`. Dùng tên file bên trong gói tải, không phụ thuộc tên ZIP có timestamp do Google Drive tạo.

```powershell
$ipnRoot = 'external_data\ipn_full'
New-Item -ItemType Directory -Force -Path $ipnRoot | Out-Null
$ipnArchives = @(Get-ChildItem 'external_data\ipn_archives' -Filter 'videos*.tgz' -File)
if ($ipnArchives.Count -ne 5) { throw 'Can du 5 file videos01.tgz den videos05.tgz' }
$ipnArchives |
  ForEach-Object { tar -xzf $_.FullName -C $ipnRoot }
@(Get-ChildItem "$ipnRoot\videos" -File |
  Where-Object { $_.Extension -in '.avi', '.mp4' }).Count
```

Lệnh cuối kỳ vọng `200`. Các tệp nguồn trên máy hiện có đuôi `.avi` bên trong `.tgz`; kiểm tra file thực tế thay vì giả định đuôi `.mp4` theo mô tả trang tải.

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

Cột cuối là label ID. Đường dẫn cần chứa `train/<nhãn>/...`, `val/<nhãn>/...` hoặc `test/<nhãn>/...`, vì code đọc tên nhãn từ đường dẫn. Dùng đúng một mapping cho cả ba split. Ba file train/val/test thực tế phải được tạo từ annotation; không lấy `train.txt` hiện có của bộ 27 lớp.

**Bàn giao/nghiệm thu:** danh sách 200 video với split và subject; clip và manifest đúng frame/nhãn; thống kê số clip mỗi lớp và split; tỷ lệ keypoint hợp lệ; danh sách video/clip lỗi. Không bắt đầu đánh giá nếu train/test có subject trùng nhau hoặc số frame không khớp annotation.

## Người 2 — baseline neural-only và đánh giá

**Việc làm**

1. Kiểm tra mapping ID, số mẫu mỗi lớp và đường dẫn train/val/test trong manifest do người 1 bàn giao. Chuẩn bị lệnh train và bộ tính metric trên tập clip mẫu ngay khi nhận được một clip mỗi lớp.
2. Xây báo cáo đánh giá: accuracy, macro-F1, precision/recall từng lớp, confusion matrix và prediction từng clip. Bổ sung phép kiểm tra để không so sánh nhầm ID 0–12 với ID gốc 1–14 của IPN.
3. Train **neural-only 13 lớp** trên train, chọn checkpoint bằng validation. Chỉ khi đã chốt checkpoint và cách tính metric mới chạy test chính thức; không chỉnh hyperparameter dựa trên kết quả test.
4. Phân tích các cặp dễ nhầm và kiểm tra bằng mắt các lớp G03–G06 đã nhận phân công. Gửi trường hợp lỗi điển hình cho người 3 đối chiếu template.
5. Bàn giao checkpoint, cấu hình/lệnh chạy, seed và báo cáo để người 3 dùng cùng split khi so sánh V4.

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

Mỗi thành viên kiểm tra CUDA trên máy mình và chọn thiết bị phù hợp. Máy đã chạy thử 16 clip trước đó chưa phát hiện CUDA; điều này không nói lên cấu hình của hai máy còn lại. Nếu không có CUDA, thay `cuda:0` bằng `cpu`, nhưng trích xuất và train toàn bộ IPN sẽ chậm. Script thử 16 clip `scripts/prepare_ipn_smoke.py` chỉ có trên máy thử nghiệm và **không** chuẩn bị toàn bộ 5.649 annotation; không được dùng nó làm pipeline train chính thức.

**Bàn giao/nghiệm thu:** checkpoint baseline chọn bằng validation; báo cáo test với prediction từng clip và ít nhất accuracy, macro-F1, metric từng lớp, confusion matrix. Không báo accuracy nếu nhãn đầu ra không cùng hệ với IPN.

## Người 3 — template graph IPN và đánh giá V4

**Việc làm**

1. Xem ví dụ video của từng nhãn từ [IPN Hand](https://gibranbenitez.github.io/IPN_Hand/) và clip đã cắt. Viết mô tả gồm tư thế đầu, chuyển động, tư thế cuối và tín hiệu phân biệt với lớp gần giống. Ví dụ G01/G08 khác ở số lần click; G10/G11 khác ở chiều biến đổi khoảng cách/động tác zoom.
2. Kiểm tra [extract_predicates.py](../../final/code/src_neurosymbolic/predicates/extract_predicates.py) và [build_events.py](../../final/code/src_neurosymbolic/events/build_events.py): chỉ dùng predicate/event mà code **thực sự tạo được**. Nếu thiếu tín hiệu để phân biệt một lớp, ghi rõ khoảng trống; không tự đặt tên event mới trong template rồi kỳ vọng matcher tìm thấy.
3. Viết file `gesture_templates.yaml` cho 13 nhãn IPN theo cú pháp mà [build_template_graph.py](../../final/code/src_neurosymbolic/kg/build_template_graph.py) hỗ trợ: `gestures`, `expected_events`, `subject`, nhóm `motions`/`hand_shapes`/`finger_motions`/`positions`..., mức `required`/`preferred`/`optional`, và `negative_evidence` khi có căn cứ. File nguồn YAML cho 27 nhãn cũ **không có trong workspace hiện tại**; có thể xem các JSON cũ trong `final/data/Input/template_graphs` để tham khảo cấu trúc, nhưng không sao chép rồi đổi tên.
4. Sinh 13 graph vào `external_data/ipn_processed/template_graphs`, không ghi đè `final/data/Input/template_graphs`. Cấp đúng `label_id` từ manifest; không tin rằng `labels.yaml` tự quyết định ID, vì loader hiện tại lấy ID từ các split file.
5. Kiểm tra `index.json` có đúng 13 nhãn, ID 0–12 liên tục, mỗi graph có event/requirement hợp lệ. Chạy thử matcher trên clip mẫu của lớp đó và các lớp dễ nhầm; ghi cả trường hợp template thất bại.
6. Sau khi người 1 bàn giao dữ liệu và người 2 có baseline, train V4 với 13 nhãn/template trên **đúng cùng train/val/test**. Chọn checkpoint theo validation, chấm test bằng metric giống người 2 rồi báo phần cải thiện/suy giảm theo từng lớp.

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

**Lệnh sinh graph sau khi người 1 bàn giao manifest:**

```powershell
.\.venv\Scripts\python.exe final\code\src_neurosymbolic\kg\build_template_graph.py `
  --templates external_data\ipn_processed\gesture_templates.yaml `
  --list-file external_data\ipn_processed\train.txt `
  --list-file external_data\ipn_processed\val.txt `
  --list-file external_data\ipn_processed\test.txt `
  --output-dir external_data\ipn_processed\template_graphs `
  --pretty
```

**Bàn giao/nghiệm thu:** bảng nhãn–định nghĩa–predicate/event; 13 graph và index đúng mapping; bằng chứng template nhận ra ví dụ đúng và phân biệt lớp gần giống; danh sách predicate/event còn thiếu nếu có; checkpoint V4 và bảng so sánh metric với neural-only. Không sửa template dựa trên kết quả test split.

## Mốc ghép việc và báo cáo cuối

1. **Mốc 0 — tải dữ liệu:** Cả ba tự tải và kiểm tra đủ video/annotation trên máy riêng; gửi checklist cho người 1 tổng hợp.
2. **Mốc A — dữ liệu mẫu:** Người 1 xác nhận split rồi bàn giao quy trình tạo ít nhất một clip chuẩn mỗi lớp. Người 2 thử pipeline train/metric; người 3 kiểm tra tín hiệu và phác thảo template song song.
3. **Mốc B — toàn bộ dữ liệu:** Người 1 bàn giao clip, keypoint và manifest train/val/test. Người 2 train neural-only, báo kết quả validation và test. Đây là mốc tối thiểu để trả lời thầy “model nhận đúng bao nhiêu trên IPN”.
4. **Mốc C — V4:** Người 3 bàn giao 13 template đã kiểm định, train V4 trên cùng split rồi so sánh với neural-only; giữ cách chọn checkpoint theo validation.
5. **Mốc D — báo cáo:** Ghi kích thước từng split, mapping nhãn, số frame/keypoint lỗi, hyperparameter, accuracy, macro-F1, confusion matrix, lỗi điển hình và giới hạn. Phân biệt rõ kết quả **isolated classification** với bài toán **continuous recognition**.

Tiêu chí hoàn thành: manifest–checkpoint–template thống nhất 13 nhãn; không rò rỉ người giữa các split; clip đúng annotation; metric được tính trên test chính thức; thí nghiệm chạy lại được bằng lệnh và cấu hình đã lưu.

## Lưu ý khi chia sẻ qua Git

File hướng dẫn này nằm trong `docs/test_thu_voi_bo_IPN/` và **không bị `.gitignore` bỏ qua**, có thể commit. Bản cập nhật Git này chỉ chứa tài liệu; dữ liệu, môi trường và checkpoint chưa được đưa lên Git. Cả ba tự tải video/annotation theo bước 0; dùng chung split, mapping và phiên bản script xử lý. Có thể chuyển keypoint/cache và checkpoint giữa các máy qua ổ chung/Drive để tiết kiệm thời gian. Trước khi stage ở máy mới, kiểm tra `.gitignore` hoặc `.git/info/exclude` có chặn `external_data/` cùng các output lớn; quy tắc ignore cục bộ ở máy khác không tự được mang theo nếu chưa commit.
