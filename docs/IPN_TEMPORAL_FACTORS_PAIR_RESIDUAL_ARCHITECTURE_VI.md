# Kiến trúc temporal factor và pair-residual cho IPN-13

Ngày bắt đầu: 2026-10-27  
Trạng thái: triển khai theo giai đoạn; chưa dùng kết quả test để chọn threshold.

## 1. Mục tiêu

Giảm các lỗi có cấu trúc của neural baseline trong ba cặp:

- `G01/G08`: một click và hai click bằng một ngón;
- `G02/G09`: một click và hai click bằng hai ngón;
- `G10/G11`: mở và khép khoảng cách thumb-index.

Thiết kế không thay thế classifier 13 lớp. Neural ensemble vẫn là nguồn dự đoán chính; temporal factor chỉ được phép tạo hiệu chỉnh thưa trong đúng cặp lớp khi evidence đủ chất lượng.

## 2. Vì sao pipeline V2 chưa đủ

Predicate V2 phát chuyển động từ các cửa sổ trượt rồi merge các cửa sổ cùng hướng. Đơn vị này là một directional run, không phải một chu kỳ click hoàn chỉnh. Vì vậy hai click gần nhau có thể nhập thành một event, còn nhiễu có thể tạo nhiều event.

Predicate pinch V2 so endpoint của từng cửa sổ. Một clip có cả action và return có thể phát đồng thời hai hướng. Mẫu số palm scale cũng có thể thay đổi khi tập keypoint visible thay đổi.

V4 hiện học `kg_logits` bằng cross-attention và trộn chúng với neural logits trên cả 13 lớp. Cơ chế này không đảm bảo một thay đổi dự đoán phải được hỗ trợ bởi factor phân biệt đúng cặp; kết quả audit trước đây cho thấy KG sửa 17 lỗi nhưng gây 25 lỗi mới.

## 3. Luồng kiến trúc mới

```text
keypoint sequence
  +-- click-cycle detector
  |     +-- index cycle count: 0 / 1 / 2+ / unknown
  |     +-- synchronized index-middle cycle count
  +-- pinch-transition detector
  |     +-- opening / closing / unknown
  +-- evidence quality
        +-- visibility, amplitude, duration, monotonicity, synchronization

neural probabilities + temporal factors
  -> pair eligibility
  -> calibrated pair score
  -> zero-initialized residual on two logits only
  -> final probabilities
```

Các lớp ngoài cặp được xét phải giữ nguyên logits. Khi evidence thiếu, mâu thuẫn hoặc dưới threshold, hệ thống abstain và trả nguyên dự đoán neural.

## 4. Temporal factor contract

### 4.1 Click cycle

Một click hợp lệ là chu kỳ trạng thái có hysteresis:

```text
OPEN -> PRESSING -> PRESSED -> RELEASING -> OPEN
```

Detector nhận chuỗi scalar đã chuẩn hóa theo kích thước bàn tay, cờ valid, confidence và `frame_id`. Chỉ một chu kỳ hoàn tất mới sinh occurrence. Mỗi occurrence phải có ID, biên frame, amplitude, duration và confidence riêng; không merge hai occurrence chỉ vì khoảng cách frame nhỏ.

Hai-ngón click yêu cầu occurrence index và middle overlap hoặc lệch tâm trong tolerance cấu hình. Nếu middle finger thiếu visibility hoặc không đồng bộ, detector trả evidence một ngón hoặc `unknown`, không tự suy diễn hai ngón.

### 4.2 Pinch transition

Chuỗi pinch là khoảng cách thumb-index chia cho một palm scale ổn định. Detector chọn dominant monotonic excursion và trả:

- `opening`: excursion tăng đủ lớn;
- `closing`: excursion giảm đủ lớn;
- `unknown`: amplitude nhỏ, coverage thấp, hoặc hai hướng gần tương đương.

Pha return được giữ trong diagnostics nhưng không được tự động coi là action chính.

### 4.3 Evidence quality

Mọi factor phải có quality/confidence riêng. Threshold cuối chỉ được chọn trên train/validation, có phân tích theo subject và camera. Không chỉnh detector hoặc template dựa trên test IPN đã mở.

## 5. Template V3

Template mục tiêu dùng factor trực tiếp:

| Lớp | Evidence bắt buộc | Occurrence |
|---|---|---:|
| G01 | `index_click_cycle` | đúng 1 |
| G08 | `index_click_cycle` | đúng 2 |
| G02 | `two_finger_click_cycle` | đúng 1 |
| G09 | `two_finger_click_cycle` | đúng 2 |
| G10 | `pinch_opening` | 1 active transition |
| G11 | `pinch_closing` | 1 active transition |

Các directional predicate V2 vẫn được giữ để tương thích và giải thích, nhưng không dùng làm count ground truth cho V3.

## 6. Pair-residual contract

Residual chỉ được xét khi neural top-2 đúng là một cặp cấu hình. Với cặp `(a, b)`, pair score `s` chỉ thay đổi:

```text
logit[a] += alpha * s
logit[b] -= alpha * s
```

Điều kiện kích hoạt gồm neural margin, factor quality, pair confidence và trạng thái không `unknown`. `alpha` khởi tạo bằng 0 để mô hình ban đầu tái hiện chính xác neural baseline.

Không dùng global 13-way blend cho temporal factor. Pair head được huấn luyện bằng factor label có sẵn:

- click count: G01/G02 = 1, G08/G09 = 2;
- finger count: G01/G08 = 1, G02/G09 = 2;
- pinch direction: G10 = opening, G11 = closing.

## 7. Tương thích và an toàn

- Pipeline V2 mặc định phải giữ nguyên cho tới khi V3 vượt validation gate.
- Predicate/template V3 dùng version và provenance riêng; graph V2 không được coi là graph V3.
- Checkpoint cũ không được nạp như checkpoint pair-residual nếu thiếu metadata kiến trúc.
- Không thay đổi logits ngoài cặp mục tiêu trong chế độ residual.
- Missing/ambiguous evidence phải dẫn tới abstention.

## 8. Kế hoạch validation

1. Unit test synthetic: một/hai click, hai click gần nhau, click dang dở, hai ngón lệch pha, mất visibility, pinch opening/closing, action + return và trường hợp ambiguous.
2. Train audit: phân phối amplitude/duration/coverage theo lớp, subject và camera; chọn giả thuyết detector.
3. Validation-only: đo exact count/direction accuracy, coverage, pair decision precision, beneficial/harmful/net corrections và macro-F1.
4. Chạy ít nhất ba seed cho tích hợp; P5 đầy đủ dùng ít nhất năm seed, paired CI và shuffled/empty-factor control.
5. Chỉ chấp nhận khi net correction dương ổn định, lớp ngoài cặp không giảm và macro-F1 tăng ít nhất 0,01 so với neural control mạnh nhất.
6. Freeze code/config/checkpoint trước lần đánh giá test tiếp theo.

## 9. Giới hạn dữ liệu hiện tại

Workspace local chỉ có 40 raw IPN keypoint JSON; full cache dùng cho sweep nằm ở Colab/Drive. Vì vậy local phù hợp để triển khai và unit-test contract, còn threshold/coverage và gain thật phải được đo trên train/validation đầy đủ ở môi trường có dữ liệu.

## 10. Trạng thái triển khai ngày 2026-10-27

Đã triển khai lát cắt đầu tiên:

- module scalar temporal factor không phụ thuộc NumPy/SciPy tại `final/code/src_neurosymbolic/predicates/temporal_factors.py`;
- click-cycle hysteresis với occurrence ID và biên frame riêng;
- ghép chu kỳ index/middle một-một theo độ lệch start và overlap;
- dominant pinch transition `opening/closing/unknown` với quality;
- adapter từ normalized keypoint sang stable clip-level palm scale, closure signal và pinch signal;
- predicate opt-in `index_click_cycle`, `two_finger_click_cycle`, `pinch_opening`, `pinch_closing`;
- provenance `ipn_predicate_v3` cùng toàn bộ threshold temporal factor;
- event builder giữ occurrence V3 tách biệt bằng `occurrence_id`;
- occurrence ID và factor diagnostics được bảo toàn từ predicate qua atomic/phase event tới graph evidence;
- detector chỉ arm sau inactive baseline, không nối chu kỳ/xu hướng qua occlusion dài, và trả `unknown` khi hai hướng pinch gần tương đương;
- CLI extraction/precompute có `--enable-temporal-factors`; mặc định tắt nên V2 không đổi.

Chưa triển khai pair-residual classifier hoặc bật template V3 trong production. Các threshold mặc định hiện chỉ là giá trị khởi đầu cho smoke/unit test, chưa phải threshold đã được calibration trên IPN đầy đủ.

Đã thêm pipeline calibration tái lập:

- `scripts/calibrate_ipn_temporal_factors.py` chỉ nhận `train.txt` và `val.txt`; CLI không có test option;
- đánh giá conditionally trong từng cặp ground-truth để đo chất lượng factor, chưa mô phỏng neural routing;
- chọn threshold chỉ từ validation, dùng per-pair coverage gate; train chỉ là diagnostics;
- chọn dominant hand thay vì cộng occurrence từ hai tay;
- chỉ ánh xạ occurrence chính xác: `1` cho single, `2` cho double; `>2` phải abstain;
- xuất per-pair/per-subject metrics, confusion có `unknown`, CSV từng mẫu, selected config, hashes và run manifest;
- resolver dùng đúng layout cache IPN `T64/<split>/<label>/<subject>/<clip>.pt`;
- CLI từ chối path không khai báo split hoặc có split khác file bao (`train`/`val`) trước khi resolve input, nên manifest bị trộn không thể mở tensor test;
- `notebooks/IPN13_Temporal_Factor_V3_Calibration_Colab.ipynb` preflight toàn bộ tensor của sáu lớp mục tiêu, chạy grid 18 cấu hình, calibration và đóng gói ZIP lên Drive;
- artifact `input_tensors.jsonl` ghi path, kích thước và SHA-256 của từng tensor thực sự được đọc, để phát hiện cache trộn hoặc thay đổi giữa hai lần chạy.

Validation code: focused suite `30 passed`; full repository suite `176 passed`, `32 warnings` PyTorch đã biết; notebook JSON và toàn bộ code cell compile thành công. Chưa có kết luận về accuracy vì full train/validation IPN không có trong workspace.

### Chạy trên Colab

1. Commit/push code V3, mở notebook `IPN13_Temporal_Factor_V3_Calibration_Colab.ipynb`.
2. Sửa `DRIVE_DATA_ROOT`, `REPO_REF`; nên ghim `REPO_COMMIT`.
3. Có thể đặt `SMOKE_TARGET_LIMIT = 24` để smoke, sau đó đặt lại `None` để chạy full.
4. Chạy tất cả cell và gửi ZIP trong `MyDrive/ipn-extract-data/runs/ipn13_temporal_factor_v3/` để audit.
5. Không dùng selected config cho pair-residual nếu report có `feasible=false`.

