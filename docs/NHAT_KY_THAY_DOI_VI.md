# Nhật ký thay đổi và thí nghiệm, từ đầu dự án đến 2026-11-04

Tài liệu này thay thế các ghi chú riêng lẻ trước đây trong `docs/` (handbook, báo cáo tiến độ, changelog symbolic, giao thức thí nghiệm, kế hoạch variable-length, v.v.); các bản đó vẫn còn trong lịch sử git (xem mục 8). Kiến trúc mô tả ở `KIEN_TRUC_CHI_TIET_VI.md`.

**Quy ước.** [Đã đo] có file kết quả; [Lập luận] chưa có thí nghiệm tách riêng; [Chưa biết] chưa đo. Mọi con số IPN dưới đây thuộc một trong hai loại: *validation* (dùng để chọn cấu hình) hoặc *test* (mở một lần cho mỗi checkpoint đã cố định). Test của DATE-27 và IPN-13 cũ đã bị xem trong lúc phát triển nên chỉ là bằng chứng thăm dò (exploratory).

## 1. Dòng thời gian tóm tắt

| Ngày | Việc | Kết quả chính |
|---|---|---|
| 2026-09-30 | Đóng gói code tái lập (`final/`), script trích keypoint RTMW nhẹ | Hệ thống DATE-27: V4 final 94,87% trên 1.229 clip test (checkpoint chọn theo validation) |
| 2026-10-05 → 10-09 | Bắt đầu IPN Hand: kế hoạch 3 người, audit video/annotation, cắt clip, manifest, template 13 lớp | 4.133 clip IPN-13, chia theo subject |
| 2026-10-16 | Symbolic V2 (predicate → event → instance graph → template) chạy trên IPN | Thô: 20,03% accuracy, 6,26% macro-F1 (nhiều hòa điểm, 448/589 clip đoán B0A) |
| 2026-10-19 → 10-21 | Cải tiến neural, ablation A/B/C, handbook, sửa template | Neural C: 89,13% / 83,76% macro-F1 trên validation |
| 2026-10-21 → 10-22 | Làm cứng symbolic: lọc trend, abstention, active/return, soft role, audit theo cặp | Symbolic thuần ~23% accuracy thô; override cặp chưa có lợi → tắt |
| 2026-10-23 → 10-24 | Giao thức V4 `kg_logits`, adapter V4 nhận vận tốc, lần chạy V4 đầu trên IPN-13 | V4 final 82,56% (1 seed, test); phát hiện nhầm lẫn "V2 so với V4" ở tài liệu |
| 2026-10-27 | Sweep 3 seed, ensemble, hand-features, temporal factor V3, pair-residual | C trung bình 82,6% (seed 42 là seed may mắn 84,65%); ensemble 85,01% |
| 2026-10-28 → 10-29 | Variable-length, calibration click, aux heads, backbone LSTM, phân tích mốc cắt | Các hướng này đều **không** cải thiện (mục 4) |
| 2026-11-02 | Adapter IPN HandS (14 lớp), trích RTMW cho subject `1CM42_3` | HandS: 4.540 clip (13 lớp) / bản 14 lớp thêm lớp GRB |
| 2026-11-03 | Dựng đủ 4.218 clip IPN gốc, chạy C và V4 3 seed; trích MediaPipe 52 điểm | RTMW V4 83,17 ± 0,87; MediaPipe V4 85,17 ± 0,38 |
| 2026-11-04 | Sửa tiêu chí chọn checkpoint V4 (`final_kg_mean_macro_f1`), chạy lại; **phát hiện V4 chạy với đồ thị instance rỗng** (mục 7.0) | Mọi kết quả V4 trên bộ 4.218 clip và HandS 14 lớp **không còn dùng được cho kết luận về symbolic**; sửa lỗi, chưa chạy lại |

## 2. Dữ liệu và quy trình

### 2.1 IPN gốc (13 lớp)

- 13 lớp: B0A, B0B (hai tư thế "không có cử chỉ", rất nhiều clip) và G01–G11. Bỏ lớp D0X.
- Chia theo **subject** (train/val/test không giao nhau). Bản đầu 4.133 clip; từ 2026-11-03 dùng **4.218 clip** (train 2.528, val 589, test 1.101) sau khi thêm 85 clip của subject `1CM42_3` vào train. Validation/test giống hệt bản cũ.
- Đã loại 9 file khỏi bản tải từ Drive: 7 bản sao hậu tố `(1)` và 2 file `1CM42_18_..._B0A` nằm sót trong `train/` trong khi subject thuộc validation (giữ lại sẽ rò rỉ subject val vào train).
- Cặp khó (chỉ khác nhau theo thời gian): G01↔G08 (click một ngón một lần/hai lần), G02↔G09 (hai ngón), G10↔G11 (pinch tăng/giảm); G03–G06 phụ thuộc hướng chuyển động.
- Kiểm tra mốc cắt: cả 4.133 dòng manifest trùng đúng khoảng trong `Annot_List.txt` gốc. Clip G01/G02 bị đoán thành hai-click thường dài hơn (61 so với 34 frame) vì khoảng annotation gốc của IPN chứa cả pha giơ/thu tay giống động tác click [Lập luận từ quan sát video; chưa đếm hệ thống].

### 2.2 IPN HandS (14 lớp)

Annotation làm lại thời gian và landmark bàn tay đã sửa tay, 14 lớp (thêm GRB). Adapter `pose/ipn_hands_adapter.py` chuyển landmark thành đúng 52 tên và schema JSON của RTMW. Điểm tin cậy HandS không phát hành nên đặt 1,0.

### 2.3 Nguyên tắc chống rò rỉ

Train học trọng số; validation chọn cấu hình và checkpoint; test mở **một lần** cho mỗi checkpoint đã cố định. Wrapper đánh giá mặc định dùng `best.pt`; `best_test.pt` cần cờ riêng và không được báo cáo như kết quả khách quan. Trình nạp dữ liệu dừng ngay (thay vì đoán) khi manifest sai, tensor trùng tên, shape sai hoặc có NaN.

## 3. Thay đổi phía neural

| Thay đổi | Lý do | Bằng chứng |
|---|---|---|
| Vận tốc keypoint (dx, dy): 156 → 260 giá trị mỗi frame, = 0 khi mất điểm | Cử chỉ động được xác định bởi chuyển động; không tạo vận tốc giả khi mất detection | 1 seed, test: +2,1 điểm accuracy, +3,6 macro-F1, recall G08 36,5% → 71,2%. Sweep 3 seed cho thấy dao động giữa seed (81,0–84,65%) lớn hơn chênh lệch → **chưa kết luận** |
| Pooling `mean` / `masked_mean` / `attention` | Clip ngắn bị lặp frame; trung bình thường tính cả frame giả | A so với B: attention tốt hơn masked_mean khi augmentation mạnh [1 seed] |
| Bỏ frame masking, keypoint dropout 0,03 → 0,02 (cấu hình C) | Click là sự kiện ngắn; che đúng frame có thể biến double thành single click | B → C: +3,2 điểm accuracy, +3,8 macro-F1 (1 seed; chưa tách hai yếu tố) |
| Class weight power 0,5, label smoothing 0,05 | Mất cân bằng B0A/B0B (140 clip) so với mỗi G (28) | [Lập luận], chưa ablation riêng |
| Chọn checkpoint theo macro-F1 | B0A+B0B = 47,7% validation: đoán đúng hai lớp này đã đạt 47,7% accuracy nhưng chỉ ~15% macro-F1 | — |
| Backbone `lstm`, `bigru`, `tcn`, `mlp_mixer` | Baseline skeleton cùng dữ liệu | Notebook sẵn sàng, **chưa chạy trên 52 điểm** (xem mục 7) |

### Thí nghiệm A/B/C (validation, 1 seed, 80 epoch, patience 12)

| Run | Pooling | Frame masking | Keypoint dropout | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Cũ (d128, 3 lớp) | mean | — | — | 87,10 | 81,09 |
| A (d192, 4 lớp, 8 head) | attention | 0,05 | 0,03 | 87,61 | 81,57 |
| B | masked_mean | 0,05 | 0,03 | 85,91 | 79,96 |
| **C** | masked_mean | 0 | 0,02 | **89,13** | **83,76** |

Mô hình lớn hơn chỉ +0,5 điểm → sức chứa không phải nút thắt. Bảng thiếu ô "attention + không frame masking".

### Sweep 3 seed và ensemble (neural C, test 1.101 clip)

| | Accuracy | Macro-F1 |
|---|---:|---:|
| Trung bình một seed | 82,59 | 74,17 |
| Seed 42 (may mắn) | 84,65 | 77,24 |
| Ensemble 3 seed (trung bình softmax) | 85,01 | 77,36 |

Ensemble là cải thiện **duy nhất** ổn định đã đo trên bộ 4.133 clip cũ.

## 4. Các hướng đã thử nhưng không cải thiện (kết quả âm tính)

Ghi lại để không lặp và để có thể trích dẫn trung thực.

| Hướng | Kết quả (validation, 3 seed) | Kết luận |
|---|---|---|
| Hand-features (20 đặc trưng hình học mỗi tay + hiệu theo frame) | macro-F1 79,48 ± 1,32 so với 80,78 ± 2,71 | Trong nhiễu |
| Aux heads (số lần click, số ngón) | Hiệu aux − c_ref: macro-F1 −0,0406 / −0,0016 / −0,0125; số clip nhầm G01↔G08 45→49, G02↔G09 35→40 | Không giúp; mặc định tắt |
| Variable-length (giữ độ dài thật, window 256) | macro-F1 0,8098 ± 0,0041 so với 0,8078 ± 0,0271; ensemble thấp hơn 0,0205; lớp click không cải thiện | Giữ fixed-64 |
| Detector click ở tốc độ khung gốc | G01/G08 coverage 0,43, độ chính xác trên phần được phủ 0,79; G02/G09 0,43 / 0,42 (dưới ngẫu nhiên); 11/28 G01 và 17/28 G02 không phát hiện được click | Tiêu chí đặt trước không đạt; tín hiệu click trong 2D yếu |
| Temporal factors V3 + pair-residual | Pinch phủ 0,92 nhưng neural đã đúng; residual tốt nhất chỉ sửa tối đa +3 clip/589 | Không bật |
| Override symbolic theo cặp | Quét margin neural 0,02–0,50, symbolic 0,02–0,20: không cấu hình nào net dương | Chưa bật; vai trò là audit |
| Role cứng `action_role: active` trong template | F1 của cả G01–G11 về 0 (63,2% cạnh bị gắn `ambiguous`), dù accuracy thô tăng nhẹ vì đoán B0A nhiều hơn | Gỡ, thay bằng soft role (hệ số 0,85) |

**Bài học phương pháp:** accuracy thô từng che giấu hồi quy (case role cứng); luôn xem macro-F1 và F1 từng lớp.

## 5. Thay đổi phía symbolic

- Lọc trend nhiễu: `trend_consistency = |dịch chuyển ròng| / tổng biến thiên`, ngưỡng 0,6.
- Abstention: chỉ chấp nhận khi điểm top-1 ≥ 0,15 và khoảng cách top-1/top-2 ≥ 0,02; báo cáo coverage và selective accuracy.
- Sequence nhiều pha: các pha phải cùng chủ thể, tuần tự, không dùng lại cùng occurrence.
- Soft role alignment: hệ số `0,85 + 0,15 × tỉ lệ active`; rescore 589 graph validation: accuracy thô 21,73% → 23,43%, macro-F1 8,68% → 9,01%, coverage 20,7% → 27,8% [Đã đo].
- `negative_evidence` để trống vì bộ khớp phạt trên toàn clip, phạt oan lớp đúng.
- Kết luận: symbolic thuần luật chưa là classifier dùng được trên IPN (~23% accuracy thô, ~9% macro-F1); vai trò là chẩn đoán/an toàn và là nguồn graph + template cho `kg_logits` học được.

## 6. Kết quả hiện tại (test 1.101 clip, 3 seed 42/123/3407, mỗi checkpoint mở test một lần)

> **Cảnh báo (2026-11-04, xem mục 7.0).** Các dòng **V4** dưới đây (RTMW 13 lớp, MediaPipe 13 lớp, HandS 14 lớp) được huấn luyện và đánh giá khi dataset **không nạp được đồ thị instance** (đồ thị rỗng): nhánh KG chỉ thấy neural memory và template, không thấy event. Chúng vẫn đúng là kết quả của "neural + template, không có bằng chứng đồ thị", nhưng **không được trình bày như kết quả của kiến trúc đầy đủ**. Các dòng **Neural C** không bị ảnh hưởng (nhánh neural không dùng đồ thị).

### 6.1 IPN gốc 13 lớp, 4.218 clip

| Nguồn keypoint | Mô hình | Accuracy | Macro-F1 |
|---|---|---:|---:|
| RTMW 52 điểm | Neural C | 82,59 ± 1,36 | 73,82 ± 2,34 |
| RTMW 52 điểm | V4 neural / KG / **final** | 83,14 / 81,29 / **83,17 ± 0,87** | 74,56 / 71,57 / 74,42 |
| MediaPipe 52 điểm | V4 neural / KG / **final** (chọn checkpoint theo `final`) | 84,86 / 72,00 ± 12,41 / **85,17 ± 0,38** | 76,94 / 54,60 / 77,42 |
| MediaPipe 52 điểm | V4 final, chọn theo `final_kg_mean_macro_f1`, seed 42 / 123 | 85,83 / 85,83 (KG 85,74 / 85,83) | seed 42: 78,88 |

Cùng bộ clip, cùng nhãn, cùng split; khác duy nhất là nguồn keypoint (RTMW luôn xuất cả hai bàn tay, MediaPipe chỉ xuất tay phát hiện được). Giả thuyết "tay phụ của RTMW là nhiễu" **chưa được kiểm chứng** [Chưa biết]. Hàng chạy lại với tiêu chí mới cho RTMW và seed 3407 của MediaPipe đang chạy tại thời điểm viết; cập nhật vào đây khi có.

### 6.2 IPN HandS 14 lớp

| Mô hình | Accuracy |
|---|---:|
| Neural C | 92,90 ± 0,94 |
| V4 final | 92,97 ± 0,38 |

So với LSTM trong bài HandS gốc (91,2 ± 1,4; split 37/13 subject): cao hơn khoảng 1,7 điểm, nhưng chưa xác minh trùng danh sách subject; giữ lớp GRB.

### 6.3 So với bài báo (isolated, IPN gốc 13 lớp)

| Mô hình | Đầu vào | Accuracy |
|---|---|---:|
| AG-MAE (2025) | 21 điểm 3D MediaPipe | 93,70 |
| SkelMAE + ST-GCN (2024) | 21 điểm 3D | 92,80 |
| ASCA-STGCN (2026) / ST-SGCN (2024) | skeleton | 89,43 / 89,00 |
| MediaPipe + RNN (2025) | 21 điểm 2D, split 37/13 | 86,38 ± 2,03 |
| ResNeXt-101 (2021) | RGB + optical flow | 86,32 |
| Mình, V4 final (MediaPipe 52 điểm 2D) | 2D | 85,17 ± 0,38 |
| Mình, V4 final (RTMW 52 điểm 2D) | 2D | 83,17 ± 0,87 |

Không phải bảng xếp hạng chính xác: split, số lần chạy và 2D/3D khác nhau. Các bài ASCA/SkelMAE/AG-MAE không nêu rõ số seed; hầu hết dùng 3D.

## 7. Chẩn đoán phát hiện ngày 2026-11-04

### 7.0 Lỗi: V4 chạy với đồ thị instance rỗng (nghiêm trọng)

- **Triệu chứng.** Khi chạy script phân tích XAI (`xai_v4_analysis.py`), kết quả "xóa đồ thị" giống hệt baseline và `events_removed = 0`. Kiểm tra trực tiếp `KeypointSequenceDataset`: 0 trên 66 (val MediaPipe), 0 trên 61 (train RTMW), 0 trên 62 (test RTMW) mẫu có đồ thị không rỗng. Tổn thất `req_loss` (BCE yêu cầu) giảm về ~0,0005–0,005, phù hợp với mục tiêu toàn số 0 [Đã đo].
- **Nguyên nhân.** Các file `train/val/test.txt` ghi đường dẫn **tương đối so với thư mục gốc repo** (`external_data/.../keypoints/<split>/<label>/<subject>/<clip>.json`), còn `--keypoint-root` truyền vào là đường dẫn tuyệt đối. `instance_graph_path_for_source` tính đường dẫn đồ thị từ phần tương đối so với `keypoint_root` nên ghép sai (`<graph_root>/external_data/.../*.json`), không tồn tại, và dataset lặng lẽ trả đồ thị rỗng (`{"nodes": [], "edges": []}`) không báo lỗi.
- **Phạm vi ảnh hưởng.** V4 trên: IPN 13 lớp RTMW (4.218 clip), IPN 13 lớp MediaPipe, IPN HandS 14 lớp; cả huấn luyện lẫn đánh giá, mọi seed. Tensor cache, đồ thị đã tính trước và checkpoint neural C **không** bị ảnh hưởng. Các kết quả cũ chạy trên Colab (bộ 4.133 clip) và DATE-27 dùng cấu trúc đường dẫn khác, **chưa được kiểm tra** xem đồ thị có được nạp không [Chưa biết]; cần kiểm tra trước khi dùng chúng trong bài.
- **Hệ quả cần đọc lại.** (a) Việc nhánh KG đạt ~85,8% ở MediaPipe và ngang nhánh neural là kết quả của neural memory + template, không phải của bằng chứng sự kiện. (b) Kết luận "symbolic không cộng thêm điểm" trên IPN chỉ áp dụng cho cấu hình này. (c) Việc nhánh KG sập ở epoch 1 vẫn đúng như mô tả (tiêu chí chọn checkpoint), nhưng "KG sau khi sửa" cũng chưa dùng đồ thị. (d) Phân tích mật độ sự kiện ở 7.2 đo trên **file đồ thị đã tính trước**, không phải trên đầu vào mô hình, nên vẫn đúng như phân tích dữ liệu.
- **Sửa.** `models/keypoint_dataset.py` (`instance_graph_path_for_source`) thêm ứng viên `<graph_root>/<split>/<label>/<subject>/<clip>.json` suy từ đường dẫn danh sách; ứng viên cũ giữ nguyên thứ tự nên bố cục đường dẫn khác không đổi. Kiểm tra sau sửa: 66/66, 61/61, 62/62 mẫu có đồ thị không rỗng; 3 test hồi quy trong `tests/test_instance_graph_path_resolution.py`. Hàng đợi chạy lại V4 đã **dừng thủ công** (MediaPipe seed 3407 đang ở epoch ~50, RTMW chưa bắt đầu); các thư mục `results/runs/*_selkg` và `results/runs/ipn_*_full*/v4_*` giữ nguyên làm bằng chứng nhưng đã vô hiệu cho kết luận symbolic.
- **Việc phải làm.** Chạy lại V4 (3 seed) cho RTMW 13 lớp, MediaPipe 13 lớp và HandS 14 lớp với đồ thị được nạp, dùng tiêu chí chọn `final_kg_mean_macro_f1`; `scripts/check_instance_graphs.py` (exit 1 nếu mẫu nào ra đồ thị rỗng; có kiểm chứng âm tính với thư mục đồ thị sai) được gọi ở đầu hàng đợi `results/runs/RUN_QUEUE_V4_GRAPH.ps1`, nên lỗi kiểu này dừng hàng đợi thay vì chạy tiếp. Hàng đợi chạy lại (MediaPipe rồi RTMW, 3 seed, thư mục `*_graph`) bắt đầu 17:05 ngày 2026-11-04. HandS 14 lớp được xếp hàng đợi riêng `results/runs/RUN_QUEUE_V4_GRAPH14.ps1` (đợi hàng đợi trên ghi `ALL COMPLETE`, có cùng bước kiểm tra đồ thị, khởi tạo từ neural C HandS gốc, kết quả vào `results/runs/ipn_hands_full14_graph`). Để dùng làm ablation "không có đồ thị" cùng tiêu chí chọn checkpoint, hàng đợi `results/runs/RUN_QUEUE_V4_NOGRAPH_ABL.ps1` (đợi HandS 14 lớp ghi `V4 GRAPH 14 COMPLETE`) chạy nốt các run `*_selkg` còn thiếu (MediaPipe seed 3407, RTMW ba seed; các run MediaPipe seed 42/123 đã có và được bỏ qua nhờ dấu `DONE`). Các run này được tạo khi lỗi đường dẫn đồ thị còn tồn tại nên chỉ dùng làm ablation, không dùng làm kết quả chính.


1. **(Đọc cùng 7.0: các số KG ở đây chưa dùng đồ thị.)** **Nhánh KG của MediaPipe sập ở 2 trong 3 seed** (59,85% và 71,48% test) vì tiêu chí chọn checkpoint chỉ xét macro-F1 của `final`; encoder đã mạnh từ C nên `final` không cải thiện thêm, và epoch 1 (khi nhánh KG gần như chưa được học, KG ~60% trên validation) thắng. Đã thêm `--select-metric final_kg_mean_macro_f1` (trung bình macro-F1 validation của `final` và `kg`, `train_cross_attention_v4.py`); hai seed chạy lại cho KG ~85,8%, final 85,83% (seed 42: +1,09; seed 123: +0,36) [Đã đo; tăng của final chưa đủ 3 seed]. Sau khi sửa, final ≈ neural: symbolic không hại nhưng chưa chứng minh cộng thêm điểm.
2. Giả thuyết "ngưỡng symbolic hiệu chỉnh cho RTMW nên sai với MediaPipe" bị bác: hạ ba ngưỡng chuyển động về 0,7× và 0,5× hầu như không đổi số sự kiện mỗi clip (64,3 → 65,1 → 65,7 ở B0A). Số sự kiện ít hơn RTMW ~2,4 lần do MediaPipe không sinh sự kiện cho bàn tay không hiện diện; mật độ trên mỗi tay gần như bằng nhau [Đã đo].
3. Lớp yếu nhất vẫn là cặp click (G02 recall 42%, G01 50%, G11 52%, G03 69% ở một seed MediaPipe; phân tích trên test nên chỉ dùng chọn hướng).

## 8. Quy trình, tài liệu và dọn dẹp repo (2026-11-04)

- `AGENTS.md` yêu cầu mọi thay đổi code/cấu hình/hợp đồng dữ liệu có mục trong `docs/`; nay các mục đó ghi vào **tài liệu này** (kiến trúc ghi vào `KIEN_TRUC_CHI_TIET_VI.md`).
- Đã gộp ~30 tài liệu trong `docs/` vào hai tệp này và xóa toàn bộ notebook (`notebooks/`) cùng các bản tải PDF/báo cáo sinh tự động; mọi thứ đã xóa còn trong lịch sử git (commit trước thay đổi dọn dẹp). Bản tổng hợp bài báo so sánh nằm ở `docs/paper/ipn_hand_baselines/README_VI.md`.
- Thêm `keypoint_extractor/extract_mediapipe_holistic.py`, `scripts/compare_keypoint_sources.py`, `tests/test_extract_mediapipe_holistic.py`, `tests/test_v4_selection_score.py`, `--select-metric final_kg_mean_macro_f1`.
- Thêm `xai_v4_analysis.py` (phân tích độ trung thực của giải thích: presence so với fact thật, thay đồ thị rỗng, xáo trộn template, xóa event hỗ trợ so với xóa ngẫu nhiên) và `tests/test_xai_v4_analysis.py`, `tests/test_instance_graph_path_resolution.py`.
- Validation: 5 test ánh xạ MediaPipe, 2 test tiêu chí chọn checkpoint, 5 test XAI, 3 test đường dẫn đồ thị (toàn bộ bộ test 262 pass trước khi thêm các test cuối).

## 9. Việc chưa làm / câu hỏi mở

- **Chạy lại V4 với đồ thị được nạp** (mục 7.0) trước mọi việc khác.
- Ensemble 3 seed V4 trên **test** (mới tính trên validation: accuracy 88,46, macro-F1 82,65 cho RTMW).
- V4-frozen (đóng băng encoder, neural ≡ C) để tách đóng góp của symbolic khỏi việc encoder được huấn luyện thêm.
- Baseline skeleton (LSTM/BiGRU/TCN/MLP-Mixer) trên 52 điểm cho 13 lớp và HandS.
- Ablation nhân quả cho `kg_logits` (template xáo trộn, đồ thị rỗng).
- Xác minh nguồn gốc 10 run của bản thảo DATE (96,40% trung bình; run 3 = 96,18% trùng `best_test.pt`).
- Đưa kênh z (3D) vào nhánh neural, giữ symbolic 2D: chưa làm.
