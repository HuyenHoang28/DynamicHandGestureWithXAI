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

## 11. Cập nhật 2026-10-27: lát cắt pair-residual tất định

### Thay đổi

- `final/code/src_neurosymbolic/pair_residual.py`: `apply_pair_residual` thực hiện đúng hợp đồng mục 6 như một lớp hậu xử lý trên logits của neural. Chỉ hành động khi neural top-2 đúng bằng một cặp cấu hình, neural margin nhỏ hơn ngưỡng, và factor cho đáp án chắc (khác `unknown`, thuộc cặp, quality đủ lớn). Khi đó chỉ hai logit đổi: `logit[a] += alpha·q·s`, `logit[b] −= alpha·q·s` (s = +1 nếu factor chọn lớp thứ nhất của cặp, −1 nếu chọn lớp thứ hai); mọi logit khác giữ nguyên. `alpha = 0` là phép đồng nhất, tái hiện đúng neural. Kèm `correction_effect` đếm beneficial/harmful/neutral/net.
- `scripts/extract_pair_factors.py`: tính quyết định của **cả ba họ factor cho mọi clip** (kể cả B0A, B0B và các cử chỉ khác) bằng `selected_config.json`, **không dùng nhãn thật**. Lý do: `calibrate_ipn_temporal_factors.py` chỉ chấm các clip thuộc sáu lớp mục tiêu và chọn họ factor theo nhãn thật, nên không thể dùng để mô phỏng pair-residual thật. CLI chỉ nhận `--split train|val`, không có tùy chọn test.
- `scripts/eval_pair_residual.py`: nhận CSV dự đoán neural (`top_labels`/`top_probs`, định dạng của `test_neural_only.py` và nay cả `ensemble_predictions.csv`) cùng CSV factor; từ chối mọi dòng factor có split không phải `train`/`val`; quét `alpha`, margin, quality floor và báo cáo beneficial/harmful/net, accuracy và macro-F1 trước và sau, theo từng cặp.
- `ensemble_neural_only.py`: `ensemble_predictions.csv` có thêm hai cột `top_labels`, `top_probs` (cột cũ không đổi).
- `notebooks/IPN13_KG_Logits_V4_FromC_Colab.ipynb`: ghim `REPO_COMMIT` vào `3901f74`, vì kiểm tra "mã dựng graph không đổi" so sánh theo file và đầu nhánh đã có thêm mã temporal factor (mặc định tắt).

### Điều chưa làm (có chủ ý)

- Pair head **học** từ factor label (mục 6 đoạn cuối): bản này chọn `alpha`, margin và quality bằng lưới trên validation thay vì huấn luyện. Chỉ nên làm khi bản tất định cho net dương ổn định.
- Bật template V3 trong production: cần `selected_config.json` đã hiệu chỉnh trên train/validation đầy đủ (chưa có; cần chạy notebook calibration trên Colab).

### Cách chạy sau khi có `selected_config.json`

```bash
python scripts/extract_pair_factors.py --split val --split-file <val.txt> --manifest-csv <val_manifest.csv> \
  --keypoint-root <keypoints> --cache-root <keypoint_tensor_cache> --config <selected_config.json> --output val_factors.csv
python scripts/eval_pair_residual.py --predictions <ensemble_predictions.csv hoặc val_eval/predictions.csv> \
  --factors val_factors.csv --output pair_residual_val.json
```

Chọn cấu hình trên validation, đóng băng, rồi mới quyết định có xem test hay không. Nếu `best_by_net` không dương, để residual tắt.

### Quan sát và rủi ro cần kiểm chứng

- Chạy thử trên 40 clip B0A cục bộ với cấu hình mẫu: họ hai ngón phát `G09` ở 11/40 clip và `G02` ở 5/40 clip dù không phải cử chỉ (dương tính giả trên clip không có cử chỉ). Residual chỉ chạm tới clip mà neural top-2 là một cặp nên phơi nhiễm nhỏ, nhưng cho thấy cần đo tỉ lệ dương tính giả trên toàn validation, không chỉ trên sáu lớp mục tiêu. Đây là cấu hình mẫu chưa hiệu chỉnh, chỉ là smoke test.
- Detector chạy trên tensor đã lấy mẫu về 64 frame, nên `temporal_max_gap` và độ rộng chu kỳ tính theo frame đã lấy mẫu: một clip dài 140 frame có bước lấy mẫu 2,2 nên khoảng trống 2 frame tương đương khoảng 4 đến 5 frame gốc. Thời lượng cử chỉ cùng lớp thay đổi nhiều giữa subject (xem tài liệu audit mục 3.3), nên threshold nên được kiểm tra theo độ dài clip.
- Lợi ích kỳ vọng nhỏ so với dao động giữa các seed (macro-F1 validation của C có độ lệch chuẩn 2,7 điểm); chỉ kết luận khi net dương ổn định trên ba seed trở lên.

### Kiểm chứng

`tests/test_pair_residual.py` (19 test): identity khi `alpha=0`; chỉ hai logit di chuyển, ngược chiều; định tuyến đúng từng họ; abstain khi top-2 không phải cặp, neural tự tin, `unknown`, quality thấp, factor của cặp khác, đáp án ngoài cặp, hoặc thiếu evidence; trộn batch; quyết định factor không dùng nhãn (hai chu kỳ là G08, một chu kỳ là G02, opening là G10; hơn hai chu kỳ hoặc hai hướng gần nhau thì `unknown`); evaluator đếm đúng beneficial/harmful và từ chối dòng split `test`; CLI không có tùy chọn test. Chạy thử `extract_pair_factors.py` từ đầu đến cuối trên 40 clip thật cục bộ. Kết hợp với kiểm tra: graph dựng ở HEAD giống hệt từng byte với `e817f16` trên 40 clip thật khi dùng cấu hình mặc định (temporal factor tắt).

## 12. Kết quả calibration trên Colab (run `20261027T065209Z-331f625514`)

Nguồn: `results/experiments/ipn13/artifact_zips/ipn13-temporal-factor.zip`. Provenance đã kiểm tra: commit `c73ea72`, `scope = train_validation_only`, không giới hạn mẫu (864 tensor = 6 lớp × (116 train + 28 val), mỗi tensor có kích thước và SHA-256), 18 cấu hình, `min_coverage = 0,25`, không có đường dẫn test.

### Kết quả theo họ factor (cấu hình được chọn: chỉ số 15)

| Cặp | Coverage val | Độ chính xác trên phần được phủ (val / train) | Nhận xét |
|---|---:|---:|---|
| G10/G11 (pinch) | 0,89 | **0,92 / 0,90** | Ổn định giữa train và validation và giữa các camera (0,87 đến 1,00) |
| G01/G08 (một ngón) | 0,48 | 0,70 / **0,54** | Train gần ngẫu nhiên; con số val (n = 27) nhiễu |
| G02/G09 (hai ngón) | 0,45 | **0,40 / 0,46** | Dưới mức ngẫu nhiên: 13/28 clip G09 bị phát hiện là một lần click |

### Vì sao họ click yếu (đo từ `per_sample.csv`, train, cấu hình 15)

Số chu kỳ click phát hiện được chồng lấn mạnh giữa lớp một click và hai click: G01 (một click) chỉ 29% clip phát hiện đúng 1 chu kỳ (35% không phát hiện chu kỳ nào, 36% phát hiện từ 2 chu kỳ trở lên); G08 (hai click) chỉ 18% phát hiện đúng 2 chu kỳ và 39% phát hiện từ 3 chu kỳ trở lên (detector đếm dao động chứ không đếm click). Họ hai ngón: 70% clip G02 không có chu kỳ đồng bộ nào.

Số chu kỳ còn phụ thuộc độ dài clip (giả thuyết về độ phân giải thời gian, chưa kiểm chứng trực tiếp): với G01 ngắn hơn 45 frame, 53% không phát hiện chu kỳ nào, còn ở 45 đến 65 frame thì 52% phát hiện đúng một chu kỳ; clip dài hơn có nhiều chu kỳ giả. Detector chạy trên tensor đã lấy mẫu về 64 frame nên cùng một ngưỡng tương ứng các tốc độ vật lý khác nhau.

### Vấn đề của quy trình chọn cấu hình

- Chỉ 6/18 cấu hình thỏa cổng coverage; cả sáu đều là các cấu hình nghiêm ngặt nhất. Cấu hình được chọn nằm **ở biên lưới** ở nhiều chiều (`click_active_threshold` 0,62 là giá trị lớn nhất, `temporal_max_gap` 4 và `two_finger_start_delta` 5 là lớn nhất, `pinch_min_excursion` 0,08 là nhỏ nhất), nên lưới chưa hội tụ.
- Mục tiêu là độ chính xác tổng với abstain tính là sai, tức thưởng coverage hơn là độ chính xác khi quyết định; cổng coverage lại ép chọn cấu hình có độ chính xác kém ở hai cặp click. Với pair-residual cần chọn **theo từng cặp**, ưu tiên precision trên phần được phủ.

### Ước tính lợi ích của pair-residual trên validation

Phương pháp: dùng quyết định factor của cấu hình 15 cho các clip thuộc sáu lớp mục tiêu (đầy đủ, vì clip ngoài cặp chỉ có thể cho thay đổi trung tính) cộng dự đoán validation của ba seed `c_ref`; chất lượng click đặt bằng 1,0 (không có trong file), chất lượng pinch lấy từ `best_confidence`; quét `alpha`, margin, quality; số liệu dưới đây là kết quả **tốt nhất trên lưới ngay trên validation (lạc quan)**.

| Họ factor | Net clip sửa được (seed 42 / 123 / 3407, 589 clip) |
|---|---|
| pinch | 0 / +1 / 0 |
| một ngón | +3 / +3 / +3 (cố định `alpha=2, margin=0,4, quality=0,3`: 0 / +3 / +3) |
| hai ngón | 0 / −1 / −1 (cố định: 0 / −3 / −4) |
| cả ba | +1 / +1 / 0 |

Kết luận: với các detector hiện tại pair-residual cho lợi ích gần bằng 0 (cao nhất +3 clip trên 589, tức khoảng 0,5 điểm accuracy, nhỏ hơn nhiều so với dao động giữa các seed) và họ hai ngón có hại. Họ pinch tốt về chất lượng nhưng neural đã đúng ở các clip đó nên không có gì để sửa. **Không bật residual và không bật template V3.**

### Hướng tiếp theo nếu muốn tiếp tục nhánh này

1. Mở rộng lưới (ngưỡng active cao hơn, `temporal_max_gap` lớn hơn, `start_delta` lớn hơn) và chọn cấu hình theo từng cặp bằng precision có cổng coverage riêng.
2. Tính detector trên tần số khung hình gốc hoặc chuẩn hóa ngưỡng theo độ dài clip thay vì trên tensor 64 frame.
3. Thử quy tắc "số chu kỳ ≥ 2 so với ≤ 1" thay cho "đúng 1 so với đúng 2": trên train (G01/G08, bỏ qua clip không có chu kỳ), quy tắc "từ 2 chu kỳ là G08, đúng 1 chu kỳ là G01" cho khoảng 58% chính xác trên phần được phủ so với khoảng 54% hiện tại, vẫn thấp.
4. Chỉ làm tiếp nếu sau các bước trên họ click đạt độ chính xác trên phần được phủ ổn định trên cả train và validation; nếu không, tập trung vào ensemble và dữ liệu (mục 7 của tài liệu audit).

