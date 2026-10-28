# Báo cáo tổng hợp việc đã làm trên IPN-13 (2026-10-27)

Viết cho: người phụ trách dự án (để báo cáo tiến độ và quyết định bước tiếp theo). Tổng hợp từ các tài liệu đã có trong `docs/` (nguồn ở cuối). Nhãn bằng chứng: **[Đã đo]** là số đọc từ file kết quả; **[Lập luận]** là suy luận chưa kiểm chứng; **[Chưa biết]** là chưa có dữ liệu.

## 1. Tóm tắt trong một phút

1. IPN-13 (13 lớp, test 1.101 clip): neural C hiện là mô hình mạnh nhất. Một seed tốt nhất đạt **84,65% accuracy / 77,24% macro-F1**, nhưng đó là seed may mắn; trung bình ba seed chỉ là **82,6% / 74,2%**.
2. **Ensemble ba seed** (trung bình xác suất) đạt **85,01% / 77,36%** trên test, là cải thiện **duy nhất** đã đo được và ổn định.
3. Nhánh symbolic (`kg_logits`) **chưa có bằng chứng giúp** trên IPN: ở lần chạy V4 duy nhất nó sửa 17 lỗi nhưng gây 25 lỗi mới. Lần chạy đó chỉ có một seed và khởi tạo thiếu vận tốc, nên chưa đủ để kết luận cả hai hướng.
4. Thêm vận tốc, đặc trưng hình học bàn tay và augmentation đã được code xong và kiểm thử; **chỉ ensemble mới có bằng chứng giúp**, các thay đổi còn lại chưa chứng minh được vì dao động giữa các seed (khoảng 2,7 điểm macro-F1) lớn hơn chênh lệch cần đo.
5. Việc tiếp theo chính: chạy notebook V4 nhiều seed để trả lời "symbolic có giúp dự đoán không", rồi làm phần XAI (giải thích) cho kiến trúc symbolic hỗ trợ.

## 2. Ensemble là gì

**Ý tưởng.** Huấn luyện cùng một mô hình nhiều lần với seed khác nhau (khởi tạo trọng số và thứ tự dữ liệu khác nhau). Mỗi mô hình sai ở những clip hơi khác nhau. Khi dự đoán, lấy **trung bình xác suất softmax** của các mô hình rồi chọn lớp có xác suất trung bình cao nhất.

Ví dụ một clip G08 (double click): mô hình 1 cho G01 0,5 / G08 0,4; mô hình 2 cho G08 0,6 / G01 0,3; mô hình 3 cho G08 0,5 / G01 0,4. Mô hình 1 sai nếu đứng một mình, nhưng trung bình G08 = 0,50 và G01 = 0,40 nên ensemble đúng.

**Vì sao hợp với bài toán này.** Seed làm macro-F1 validation lệch chuẩn khoảng 2,7 điểm, tức phần lớn "nhiễu" là do may rủi của từng lần huấn luyện. Trung bình nhiều mô hình triệt tiêu phần may rủi đó, đặc biệt ở các lớp click vốn dễ lật qua lại.

**Kết quả đã đo** [Đã đo, neural C, ba seed 42/123/3407]:

| | Val acc | Val macro-F1 | Test acc | Test macro-F1 |
|---|---:|---:|---:|---:|
| Trung bình một seed | 87,32% | 80,78% | 82,59% | 74,17% |
| Seed tốt nhất (42) | 89,13% | 83,76% | 84,65% | 77,24% |
| **Ensemble 3 seed** | **89,47%** | **83,95%** | **85,01%** | **77,36%** |

Hiểu đúng: ensemble hơn mức trung bình **+2,4 điểm accuracy, +3,2 điểm macro-F1** trên test, và ngang seed tốt nhất nhưng **không phụ thuộc việc chọn đúng seed**. Cái giá: tốn thời gian huấn luyện và suy luận gấp số seed. Test IPN đã được xem nhiều lần nên số test chỉ là bằng chứng exploratory; số validation là số dùng để quyết định.

Lưu ý: ensemble không tự nhiên tạo giải thích; nó chỉ là bộ trộn các dự đoán.

## 3. Các mô hình đã thử nghiệm

| Mô hình | Mô tả ngắn | Trạng thái |
|---|---|---|
| **Neural C** (mô hình neural chủ lực) | Transformer d192, 4 lớp, 8 head, masked-mean pooling, vận tốc keypoint, label smoothing 0,05, class-weight power 0,5, jitter 0,005, keypoint dropout 0,02, chọn checkpoint theo macro-F1 | Đã chạy 3 seed + ensemble |
| Neural A/B (ablation cũ) | Cấu hình cũ hơn dùng để chọn C (mean pooling, không vận tốc...) | Đã chạy, C được chọn |
| Neural không vận tốc | Cùng siêu tham số, tắt vận tốc | Chỉ 1 seed (chưa đủ kết luận) |
| Neural + `hand-features` | Thêm 20 đặc trưng hình học mỗi bàn tay và hiệu theo frame (40 chiều) | Đã train 3 seed; chưa chạy bước đánh giá đầy đủ |
| Neural + augmentation (`aug`, `hand_aug`) | Cắt/lấy mẫu lại thời gian, scale jitter, dịch vị trí | Notebook sẵn, **chưa chạy** |
| **Symbolic thuần luật** (V2) | Predicate, event, instance graph khớp template | Accuracy thô khoảng 23% trên IPN; chỉ dùng làm nhánh phụ |
| **V4** (`DualCrossAttentionV4`) | Encoder neural dùng chung với đầu tri thức; cross-attention hai nguồn (neural + đồ thị template); đầu `kg_logits`; fusion có gate theo từng lớp | Đã chạy 1 seed, khởi tạo thiếu vận tốc |
| **V4 khởi tạo từ C, nhiều seed** | V4 từ checkpoint C của từng seed, so cặp với chính checkpoint khởi tạo | Notebook sẵn, **chưa chạy trên IPN thật** |
| Temporal factors V3 | Bộ phát hiện chu kỳ click và pinch mở/đóng, theo đường tùy chọn | Đã calibration |
| Pair-residual | Dịch nhẹ hai logits khi neural top-2 là một cặp khó | Ước tính trên validation, không bật |

Công thức fusion V4: `final = gate * neural + (1 - gate) * kg`, gate theo từng lớp (khởi tạo 0,8 nghiêng về neural). Loss gồm CE(final) + 0,3 CE(neural) + 0,3 CE(kg) + 0,05 BCE(requirement) + 0,01 (gate − 0,8)².

## 4. Kết quả đã chạy được

### 4.1 DATE-27 (1.229 clip test) [Đã đo]

| Mô hình | Accuracy |
|---|---:|
| V4 final (chọn theo validation) | 94,87% (macro-F1 94,52%) |
| V4 `kg_logits` | 95,04% |
| V4 neural head | 94,63% |
| Neural-only độc lập | 95,28% |
| ProtoGCN / DATE Transformer (paper) | 93,28% / 94,22% |
| Proposed mean (paper) | 96,40% (cần xác minh nguồn gốc, xem cảnh báo bên dưới) |

Cảnh báo: run 3 của cột Proposed trong paper (96,18%) trùng đúng accuracy của `best_test.pt`, tức checkpoint chọn theo test; cần xác minh trước khi dùng 96,40% làm số tái lập.

### 4.2 IPN-13, lần chạy V4 đầu tiên (1.101 clip test, một seed) [Đã đo]

| Mô hình | Accuracy | Macro-F1 |
|---|---:|---:|
| Neural C (seed 42) | 84,65% | 77,24% |
| Neural không vận tốc (checkpoint khởi tạo V4) | 82,56% | 73,65% |
| V4 neural head | 83,02% | 74,40% |
| V4 `kg_logits` | 82,29% | 73,22% |
| V4 final | 82,56% | 73,84% |

`kg_logits` trên test: sửa 17 lỗi, gây 25 lỗi mới (net −8).

### 4.3 Sweep ba seed (neural C) [Đã đo]

Xem bảng ensemble ở mục 2. Chi tiết theo seed: test accuracy 84,65 / 81,02 / 82,11; val macro-F1 83,76 / 78,47 / 80,13. Với `hand-features`, validation macro-F1 79,48 ± 1,32 so với 80,78 ± 2,71 của C: chênh lệch nằm trong nhiễu, **không kết luận được**.

### 4.4 Cấu trúc lỗi của IPN [Đã đo]

- Trong 169 lỗi của C trên test: 57 lỗi nằm trong nhóm click (G01/G02/G08/G09), 63 lỗi trong ba cặp khó (G01↔G08, G02↔G09, G10↔G11).
- Camera 1CM1 (chỉ 1 subject train, 0 subject val) có accuracy thấp nhất, 77,2%, so với 89,9% ở camera 4CM11. Validation không có 1CM1 nên che mất điểm yếu này.
- Số clip dùng đúng: test 1.101 clip đầy đủ; train mất 85 clip do loại subject `1CM42_3`; lớp D0X không dùng. Bộ IPN-13 vẫn nhỏ hơn DATE-27 nhiều (116 clip train mỗi cử chỉ so với 336).
- Chưa có số baseline IPN skeleton cùng giao thức để so sánh trực tiếp, nên **chưa tuyên bố vượt baseline công bố**.

### 4.5 Temporal factors và pair-residual [Đã đo]

- Pinch (G10/G11): độ chính xác khi phủ 0,92 trên validation, ổn định giữa các camera. Nhưng neural đã đúng ở các clip đó nên lợi ích thêm gần 0.
- Click (G01/G08, G02/G09): gần ngẫu nhiên (0,40 đến 0,54 trên train); detector đếm dao động chứ không đếm click.
- Pair-residual tốt nhất trên lưới (lạc quan) chỉ sửa tối đa +3 clip trên 589; họ hai ngón có hại. **Kết luận: không bật residual và không bật template V3.**

## 5. Các phần đã thêm vào code

Tất cả đều tùy chọn và mặc định giữ hành vi cũ (kiểm chứng: adapter V4 cho logits giống hệt neural-only ở 12 tổ hợp cấu hình; 128/128 clip DATE-27 cho dự đoán giống hệt lần chạy đã đóng gói). Bộ test đầy đủ: **199 passed**.

| Thêm vào | Làm gì | Vì sao thêm |
|---|---|---|
| **Vận tốc keypoint** `(dx, dy)` | Input mỗi frame từ 52×3 lên 52×5 = 260 chiều | Click hai lần và hướng pinch là chuyển động theo thời gian; vị trí tĩnh không đủ. Chưa có bằng chứng ba seed về mức đóng góp |
| Pooling `mean` / `masked_mean` / `attention` | Chọn cách gộp theo thời gian; `masked_mean` bỏ frame rỗng | Clip ngắn bị lấp frame lặp, pooling thường tính cả frame giả |
| V4 nhận vận tốc, pooling, hand-features (`--neural-motion-features`, `--neural-pooling`, `--neural-hand-features`) | Cho phép khởi tạo V4 từ checkpoint C | Lần chạy V4 đầu bị mất vận tốc vì adapter không hỗ trợ |
| `--hand-features` (`hand_geometry.py`) | Khoảng cách đầu ngón, pinch, độ cao tương đối chuẩn hóa theo cỡ bàn tay, cộng hiệu theo thời gian | Biến cử động ngón thành tín hiệu sạch, bất biến theo khoảng cách camera [Lập luận, chưa chứng minh] |
| Augmentation (`--time-crop-min`, `--scale-jitter`, `--shift-std`) | Cắt và lấy mẫu lại thời gian, zoom, dịch | Giảm domain shift giữa subject và camera |
| `ensemble_neural_only.py` | Trung bình softmax nhiều seed neural | Giảm nhiễu giữa các seed (mục 2) |
| `--save-logits` ở `test_cross_attention_v2.py` và `scripts/ensemble_v4_logits.py` | Lưu logits `final/neural/kg` và ensemble các V4 | So V4 ensemble với neural ensemble |
| Temporal factors V3 (`--enable-temporal-factors`), pair-residual (`pair_residual.py`, `extract_pair_factors.py`, `eval_pair_residual.py`) | Phát hiện chu kỳ click/pinch; dịch nhẹ logits theo cặp khó | Hướng sửa cặp lỗi lớn nhất; hiện không bật |
| `load_checkpoint_compat` | Đọc checkpoint Colab (PosixPath) trên Windows | Phân tích cục bộ bị vỡ nếu thiếu |
| Notebook | `IPN13_Neural_Improvement_Sweep_Colab`, `IPN13_KG_Logits_V4_FromC_Colab`, `IPN13_KG_Logits_V4_MultiSeed_Colab`, `IPN13_Temporal_Factor_V3_Calibration_Colab` | Chạy trên Colab, đóng gói ZIP có provenance |

## 6. Điều đã hiệu chỉnh (để báo cáo trung thực)

- Lúc đầu mình kết luận "V4 thấp hơn C 2,1 điểm" và "vận tốc đóng góp +2,1 điểm". Cả hai đều so sánh với **seed tốt nhất** của C. Sau sweep ba seed, V4 final (82,56%) ngang trung bình ba seed của C (82,59%), nên không còn là bằng chứng V4 kém hơn; tương tự, chênh lệch vận tốc nằm trong dao động seed.
- Mốc neural baseline nên là **trung bình ba seed (82,6%) hoặc ensemble (85,0%)**, không phải 84,65%.
- Một nhận định sai về fusion V4 (nhầm với lớp V2 cũ) đã được sửa trong `GIAI_THICH_THAY_DOI_VA_THU_NGHIEM_VI.md`.

## 7. Đề xuất cải thiện (theo thứ tự ưu tiên)

1. **Chạy notebook V4 nhiều seed** (`IPN13_KG_Logits_V4_MultiSeed_Colab.ipynb`, `RUN_LOCKED_TEST=False`). Trả lời có kiểm soát: V4 final so với chính checkpoint khởi tạo của nó, cùng clip, cùng seed; số clip KG sửa và hại; V4 ensemble so với neural ensemble. Chỉ kết luận "symbolic giúp" khi mọi seed đều tăng và mức tăng trung bình lớn hơn 0,005 macro-F1.
2. **Dùng ensemble làm neural baseline** và tăng lên 5 seed. Đây là thay đổi duy nhất có bằng chứng.
3. **Xây phần XAI cho kiến trúc symbolic hỗ trợ**: xuất requirement nào của template khớp, attention vào frame và pha nào, và chạy **ablation nhân quả** (template xáo trộn, đồ thị rỗng) để phân biệt giải thích đáng tin cậy với dấu vết trực quan. Hiện các tín hiệu này chỉ là dấu vết, chưa phải bằng chứng nhân quả.
4. **Đánh giá nốt các biến thể đã train hoặc có notebook**: `hand` (đã có checkpoint), `aug`, `hand_aug`; chỉ chấp nhận khi chênh lệch lớn hơn độ lệch chuẩn giữa các seed.
5. **Ablation vận tốc ba seed** (`--disable-motion-features`) để biết vận tốc thật sự đóng góp bao nhiêu.
6. **Nhãn phân rã cho các cặp click**: đầu phụ dự đoán "số lần click" (1 hay 2) và "một hay hai ngón", chia sẻ dữ liệu giữa các lớp (khoảng 464 clip mỗi lớp-số-lần so với 232 mỗi lớp) [Lập luận].
7. **Huấn luyện trên train + validation** sau khi chốt cấu hình bằng validation (thêm 7 subject, khoảng +24% dữ liệu), kèm chế độ số epoch cố định.
8. **Validation tốt hơn**: K-fold theo subject trên train+validation và đánh giá riêng camera 1CM1, vì validation hiện thiếu camera yếu nhất.
9. **Nâng chất lượng tri thức** nếu muốn đóng góp symbolic trên IPN: đưa đặc trưng flexion vào predicate và template của G01/G08/G02/G09, và tính detector trên tần số khung hình gốc thay vì tensor 64 frame.

Không nên làm lúc này: chọn siêu tham số theo test; bật pair-residual hoặc template V3; thêm lớp mới vào mô hình trước khi có kết quả V4 nhiều seed.

## 8. Việc cần bạn làm tiếp

1. Chạy `notebooks/IPN13_KG_Logits_V4_MultiSeed_Colab.ipynb` trên Colab (cần checkpoint `c_ref_seed42/123/3407` từ lần sweep trên Drive, code đã ở commit `f6124ca` của nhánh `HuyenHoang`).
2. Gửi lại ZIP trong `MyDrive/ipn-extract-data/runs/ipn13_v4_multiseed/`.
3. Báo mình có làm phần XAI theo hướng nào ở mục 7.3 sau khi xem kết quả.

## Nguồn trong `docs/`

`BAO_CAO_TIEN_DO_2026-10-24_VI.md`, `AUDIT_IPN13_KET_QUA_VA_CAI_THIEN_2026-10-24_VI.md` (kể cả mục 7), `GIAI_THICH_THAY_DOI_VA_THU_NGHIEM_VI.md`, `IPN13_KG_LOGITS_EXPERIMENT_PROTOCOL_VI.md`, `IPN_TEMPORAL_FACTORS_PAIR_RESIDUAL_ARCHITECTURE_VI.md` (mục 11 và 12).

## Validation của tài liệu

Tài liệu chỉ tổng hợp: mọi con số được chép từ các tài liệu nguồn ở trên (đã đối chiếu với file kết quả khi viết chúng). Không có thay đổi code hay cấu hình trong mục này, nên không cần chạy lại test. Kết quả V4 nhiều seed chưa tồn tại; mục 7.1 là kế hoạch, không phải kết quả.
