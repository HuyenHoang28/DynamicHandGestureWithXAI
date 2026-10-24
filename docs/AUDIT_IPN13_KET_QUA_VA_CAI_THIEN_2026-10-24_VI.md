# Audit kết quả IPN-13 (2026-10-24) và kế hoạch cải thiện

Nguồn dữ liệu: gói kết quả `results/experiments/ipn13/artifact_zips/20261024T075203Z-2c6b5c8d30.zip` (run Colab V4 `kg_logits` trên IPN-13, commit `e817f16`), manifest `external_data/ipn_processed/manifest.csv`, ZIP annotation gốc của IPN, và các ZIP ablation A/B/C. Mọi nhận định dưới đây gắn nhãn **[Đã đo]**, **[Lập luận]** hay **[Chưa biết]** như trong `GIAI_THICH_THAY_DOI_VA_THU_NGHIEM_VI.md`.

**Lưu ý phương pháp quan trọng:** test IPN đã bị mở hai lần (neural-only ở các lần trước và lần chạy này). Các phát hiện bên dưới dùng test để *chẩn đoán lỗi* (không sinh quyết định cấu hình). Mọi lựa chọn cải thiện sau này phải dựa trên validation nhiều seed; nếu xem test lần nữa thì chỉ là bằng chứng exploratory.

## 1. Kết quả của lần chạy (1.101 clip test)

| Mô hình | Accuracy | Macro-F1 |
|---|---:|---:|
| Neural-only C (có vận tốc) | **84,65%** | **77,24%** |
| Neural-only cùng siêu tham số, không vận tốc (checkpoint khởi tạo V4) | 82,56% | 73,65% |
| V4 neural head | 83,02% | 74,40% |
| V4 `kg_logits` | 82,29% | 73,22% |
| V4 final (neural + symbolic) | 82,56% | 73,84% |

Hai kết luận trực tiếp [Đã đo]: (1) V4 final **bằng đúng** checkpoint khởi tạo của nó (82,56% so với 82,56%) và thấp hơn C 2,1 điểm; (2) `kg_logits` trên test sửa 17 lỗi của neural head nhưng gây 25 lỗi mới (net −8). Vì vậy kết quả này **không có bằng chứng rằng nhánh symbolic giúp** trên IPN, và V4 không "nhỉnh hơn" baseline nào cùng điều kiện.

## 2. "Ít mẫu hơn" có đúng không

Đối chiếu manifest với annotation gốc của IPN [Đã đo]:

| Hạng mục | Gốc IPN | Đang dùng | Ghi chú |
|---|---:|---:|---|
| Test (13 cử chỉ) | 1.101 | **1.101** | Dùng trọn vẹn test chính thức |
| Train chính thức (13 cử chỉ) | 3.117 | 3.032 (2.443 train + 589 val) | Thiếu 85 clip do **loại subject `1CM42_3`** |
| Lớp `D0X` (không phải cử chỉ) | 922 train + 509 test | **không dùng** | Nhãn gốc có 14 lớp; ta dùng 13 |

Nên bộ IPN-13 không bị cắt bớt một cách bí mật: test đủ, train mất 2,7% do một subject bị loại (theo protocol đã khóa; các docs mình đọc ghi việc loại nhưng không ghi rõ lý do kỹ thuật, nên đây là điểm cần xác minh). Nhưng **nhỏ hơn nhiều so với DATE-27** (train 9.097, test 1.229 với 27 lớp, mỗi lớp 336 clip train so với 116 clip mỗi cử chỉ IPN), và chỉ có 13 subject trong test, nên chênh lệch nhỏ giữa các mô hình rất dễ nằm trong nhiễu (1 clip = 0,09 điểm accuracy; 52 clip mỗi cử chỉ nên 1 clip = 1,9 điểm recall).

Về "baseline": mình tìm số IPN skeleton đã công bố nhưng **không tìm được con số kiểm chứng được** (kết quả tìm kiếm chỉ cho các giao thức khác nhau, ví dụ 80–82% với MediaPipe và subject-wise CV, không so trực tiếp được). Vì vậy không nên tuyên bố vượt baseline công bố nào trên IPN cho tới khi có một bảng baseline cùng giao thức.

## 3. Nguyên nhân lỗi (đã đo từ file dự đoán)

### 3.1 Vận tốc là yếu tố mạnh nhất, và V4 bị mất nó

Neural có và không có vận tốc, cùng siêu tham số, cùng seed [Đã đo, test, một seed]:

| | Accuracy | Macro-F1 | G08 recall | G09 recall |
|---|---:|---:|---:|---:|
| Có vận tốc (C) | 84,65% | 77,24% | 71,2% | 65,4% |
| Không vận tốc | 82,56% | 73,65% | **36,5%** | 50,0% |

Chênh lệch tập trung ở double click: không vận tốc thì G08 gần như bị đoán thành G01. V4 khởi tạo từ bản không vận tốc nên bị chặn trên ở mức này; đây là nguyên nhân lớn nhất khiến V4 thua C và là lỗi thiết kế của lần chạy, không phải lỗi của ý tưởng template.

### 3.2 Lỗi tập trung ở các cặp đếm click và chiều pinch

Trong 169 lỗi của C trên test [Đã đo]: 57 lỗi nằm trong nhóm click (G01/G02/G08/G09), 63 lỗi nằm trong ba cặp khó (G01↔G08, G02↔G09, G10↔G11), chỉ 32 lỗi liên quan B0A/B0B. Các nhầm lẫn lớn nhất: G01→G08 17, G02→G09 15, G09→G02 12, G08→G01 9, G10→G11 6.

### 3.3 Khoảng cách camera/subject là nguồn lỗi chính (domain shift)

Accuracy của C trên test theo camera [Đã đo]:

| Camera | Clip test | Subject test | Clip train | Subject train | Accuracy C |
|---|---:|---:|---:|---:|---:|
| 1CM1 | 254 | 3 | **84** | **1** | **77,2%** |
| 1CM42 | 257 | 3 | 761 | 9 | 83,7% |
| 1CV12 | 253 | 3 | 588 | 7 | 86,2% |
| 4CM11 | 337 | 4 | 1.010 | 12 | 89,9% |

Camera 1CM1 có 3 subject trong test nhưng **chỉ 1 subject trong train và 0 subject trong validation**, nên validation không thể nhìn thấy vấn đề này. Accuracy theo từng subject test dao động từ 67% đến 94%.

Mỗi camera còn có **kiểu lỗi riêng** [Đã đo]: ở 4CM11, recall G01 và G02 chỉ 31% (single click bị đoán thành double click: 11 clip thành G08, 9 clip thành G09) trong khi G08/G09 đạt 94–100%; ở 1CM42, G10 chỉ đạt recall 42% và 13 lỗi của camera này bị đoán thành G11; ở 1CV12, G10 và G11 cùng chỉ đạt 50%. Các nhầm lẫn này thiên về một hướng theo camera, nghĩa là mô hình đang dựa vào đặc điểm của từng camera/subject, không chỉ vào cử chỉ.

Thời lượng clip cũng không phải dấu hiệu đáng tin [Đã đo]: với nhiều subject, clip double click (G08) **không dài hơn** clip single click (G01): tỉ lệ độ dài trung vị G08/G01 theo subject nằm từ 0,75 đến 1,96, nhiều subject xấp xỉ 1,0 (ví dụ `1CV12_12`: 97 so với 98 frame). Có nghĩa là người làm double click nhanh gấp đôi, và mô hình phải *đếm* chứ không thể dựa vào độ dài. Ngoài ra độ dài cùng một cử chỉ thay đổi rất nhiều giữa subject (G01 của `1CM42_12` có trung vị 143 frame so với 30–60 ở các subject khác), nên sau khi lấy mẫu đều về 64 frame, tốc độ sự kiện trong tensor khác nhau tới vài lần.

### 3.4 Khoảng cách validation–test và nhiễu chọn checkpoint

[Đã đo] Neural C: validation 89,13% / macro-F1 83,76% so với test 84,65% / 77,24% (−4,5 và −6,5 điểm). Giảm mạnh ở các lớp hướng và pinch: F1 G11 −18, G03 −14, G04 −12, G07 −8, G02 −8. Validation chỉ có 7 subject (28 clip mỗi cử chỉ) và thiếu camera 1CM1, nên là một thước đo nhiễu và lệch.

[Đã đo] V4 trên IPN: validation dao động 80–86% giữa các epoch liên tiếp, learning rate bị giảm một nửa từ epoch 4, patience 5 làm huấn luyện dừng ở epoch 17 (best epoch 12). Checkpoint được chọn vì vậy một phần là may rủi.

### 3.5 Còn bao nhiêu chỗ để cải thiện

[Đã đo, chẩn đoán] Nếu một "oracle" chọn đúng trong năm đầu ra (C, không vận tốc, V4 neural, KG, final) thì được 90,6%; chỉ 104 clip (9,4%) sai ở cả năm đầu ra. Bỏ phiếu đa số giữa C, không vận tốc và V4 final chỉ được 84,7% (không tăng so với C), nên trộn *các mô hình hiện có* không giúp; cần mô hình mới có thông tin khác hoặc nhiều seed.

## 4. Những gì đã triển khai (đều tùy chọn, mặc định tắt)

Mỗi thay đổi được kiểm thử và không làm đổi hành vi của checkpoint cũ (kiểm tra: 128/128 clip DATE-27 cho dự đoán giống hệt lần chạy đã đóng gói). Bộ test đầy đủ: **153 passed** (123 cũ và 30 mới).

| Thay đổi | Nhắm vào | Cách dùng |
|---|---|---|
| Adapter neural của V4 nhận vận tốc, pooling (`mean`, `masked_mean`, `attention`) và đặc trưng hình học | Mất vận tốc ở V4 (3.1) | `train_cross_attention_v4.py --neural-motion-features --neural-pooling masked_mean` |
| Đặc trưng hình học bàn tay (`--hand-features`): khoảng cách đầu ngón và pinch chuẩn hóa theo cỡ bàn tay, cộng thay đổi theo thời gian (40 chiều mỗi frame) | Đếm click và chiều pinch (3.2): biến cử động ngón thành tín hiệu một chiều sạch, bất biến theo khoảng cách camera | `train_neural_only.py --hand-features` |
| Augmentation thời gian (`--time-crop-min`) và hình học (`--scale-jitter`, `--shift-std`) | Khác biệt tốc độ, zoom, vị trí giữa subject và camera (3.3); không che frame nên không phá nhịp click | `train_neural_only.py` |
| `ensemble_neural_only.py`: trung bình xác suất nhiều seed | Nhiễu giữa các seed và nhiễu chọn checkpoint (3.4) | script riêng |
| `load_checkpoint_compat` đọc được checkpoint Colab (PosixPath) trên Windows | Vặt vãnh nhưng làm mọi phân tích cục bộ vỡ | tự động |
| Notebook `IPN13_Neural_Improvement_Sweep_Colab.ipynb` | Chọn biến thể theo validation ba seed | Colab |

Kiểm chứng cụ thể: adapter V4 cho logits **giống hệt** `NeuralOnlyClassifier` với 12 tổ hợp cấu hình (khớp strict-load); checkpoint C thật nạp vào V4 không lỗi và neural head của V4 tái hiện đúng logits của C; notebook sweep chạy hết các ô trên dữ liệu thử nhỏ (huấn luyện, đánh giá, ensemble, tóm tắt, đóng gói ZIP).

**Hiệu quả trên IPN thật của các thay đổi này [Chưa biết]**: máy này không có GPU và chỉ có 40 file keypoint IPN, nên việc huấn luyện IPN phải chạy trên Colab. Các giả thuyết ở bảng trên (đặc biệt `--hand-features` và augmentation) chưa được đo.

## 5. Đề xuất chưa làm, theo thứ tự ưu tiên

1. **Chạy sweep neural trên Colab** (notebook đã có): 4 biến thể × 3 seed, chọn theo validation, báo cáo trung bình ± độ lệch chuẩn và ensemble. Đây là cách nhanh nhất để biết thay đổi nào có thật sự giúp.
2. **V4 khởi tạo trực tiếp từ C (có vận tốc)** với `--neural-motion-features --neural-pooling masked_mean`, thêm `--freeze-neural-epochs 3` để nhánh KG khởi động mà không làm hỏng encoder, và `--early-stop-patience 12 --scheduler-patience 4` cho validation nhiễu. Chỉ để trả lời câu hỏi "KG có giúp khi neural đã mạnh hay không".
3. **Nhãn phân rã cho các cặp đếm click:** G01, G02, G08, G09 có cấu trúc (ngón: một hay hai ngón) × (số lần: 1 hay 2). Thêm đầu phụ dự đoán "số lần" (G01 và G02 là 1, G08 và G09 là 2) dùng chung 464 clip cho mỗi lớp-số-lần thay vì 232 clip cho mỗi lớp riêng, và một đầu phụ cho "một ngón hay hai ngón". Giả thuyết: tăng gấp đôi dữ liệu cho khái niệm "đếm". *[Lập luận]*
4. **Huấn luyện trên train + validation** sau khi chốt cấu hình bằng validation: thêm 7 subject (tăng 24% dữ liệu và thêm đa dạng subject) rất có lợi khi lỗi chính là domain shift. Cần thêm tùy chọn không dùng validation (số epoch cố định) vì hiện script yêu cầu validation tách rời.
5. **Validation tốt hơn:** validation hiện không có camera 1CM1 nên che mất điểm yếu nhất. Dùng subject-grouped K-fold trên train+validation để chọn cấu hình, hoặc đưa subject duy nhất của 1CM1 vào đánh giá riêng.
6. **Sửa nguồn gốc tốc độ/thời lượng:** tính vận tốc trên tần số khung hình gốc trước khi lấy mẫu về 64 frame, thêm độ dài clip như một đặc trưng, và mask phần padding (clip ngắn hơn 64 frame hiện bị lấp bằng cách lặp lại frame cuối nên pooling tính cả những frame đứng yên giả này).
7. **Nâng chất lượng nhánh tri thức** nếu muốn bài báo IPN có đóng góp symbolic: predicate hiện chỉ cho 23% accuracy thô; đưa đặc trưng flexion (như hình học bàn tay ở trên) vào predicate và template cho G01/G08/G02/G09. Nếu sau bước 2 mà KG vẫn không giúp, cần ghi nhận trung thực trong bài báo rằng trên IPN lợi ích đến từ phần neural.
8. **Không nên làm lúc này:** chọn checkpoint hay siêu tham số dựa trên test; chạy ensemble các mô hình hiện có (đã đo, không tăng); thêm lớp mới vào mô hình trước khi biết bước 1 và 2.

## 6. Cách chạy

1. Commit và push code mới lên nhánh `HuyenHoang` (notebook clone từ GitHub).
2. Chạy `notebooks/IPN13_Neural_Improvement_Sweep_Colab.ipynb` (`RUN_LOCKED_TEST=False`). Gửi lại ZIP trong `MyDrive/ipn-extract-data/runs/ipn13_neural_sweep/`.
3. Dựa trên bảng validation, chọn biến thể; sau đó mới quyết định có xem test lần nữa không (và gọi đúng tên là lần xem thứ hai).

## Validation của tài liệu

Số liệu đọc trực tiếp từ gói kết quả, manifest, ZIP annotation và các ZIP ablation bằng script phân tích; các giả thuyết gắn nhãn [Lập luận]. Thay đổi code được kiểm thử bằng `pytest` (153 passed), kiểm tra tương thích ngược trên DATE-27 và smoke test huấn luyện trên dữ liệu nhỏ.
