# Báo cáo Người 3 — Template graph IPN và mô hình V4

Người thực hiện: Tín. Thí nghiệm: phân loại clip cử chỉ đã cắt, IPN Hand 13 lớp (B0A, B0B, G01–G11, ID 0–12), cùng split train/val/test với Người 1 và Người 2.

## 1. Tóm tắt

| Mô hình | Val acc | Val macro-F1 | Test acc | Test macro-F1 |
|---|---:|---:|---:|---:|
| Template thuần luật (không học) | 0,054 | 0,070 | — | — |
| Neural-only (Người 2) | 0,835 | 0,743 | 0,794 | 0,688 |
| Neural-only, train giống V4, bỏ KG (đối chứng) | 0,857 | 0,781 | 0,805 | 0,701 |
| **V4 neuro-symbolic** | **0,854** | **0,784** | **0,807** | **0,705** |

- V4 cao hơn baseline của Người 2 **+1,3 điểm accuracy, +1,7 điểm macro-F1** trên test.
- Nhưng mô hình đối chứng (cùng cách train, không có KG) đạt gần như bằng V4 (kém 0,4 điểm macro-F1, khoảng 2 clip). **Phần tăng đến từ cách train, không phải từ knowledge graph.**
- Template thuần luật chỉ đạt macro-F1 0,07 (đoán ngẫu nhiên ≈ 0,077): **predicate hiện tại không phân biệt được các lớp IPN**.
- Các cặp khó nhất vẫn chưa giải quyết được: click 1 lần / 2 lần (G01/G08, G02/G09) và zoom vào / ra (G10/G11).

## 2. Việc đã làm

1. Viết template cho 13 lớp IPN ([configs/ipn/gesture_templates.yaml](../../configs/ipn/gesture_templates.yaml)), chỉ dùng predicate mà `extract_predicates.py` sinh được.
2. Sinh 13 template graph theo manifest của Người 1, ID 0–12 khớp với Người 2.
3. Sinh instance graph cho toàn bộ 4.133 clip (2.443 train, 589 val, 1.101 test), không clip nào lỗi.
4. Chấm template thuần luật trên val ([scripts/eval_ipn_template_matching.py](../../scripts/eval_ipn_template_matching.py)).
5. Thêm tùy chọn `--select-metric macro_f1` vào `train_cross_attention_v4.py` để chọn checkpoint theo val macro-F1 giống Người 2 (mặc định vẫn là accuracy).
6. Train V4 và mô hình đối chứng, chấm test **một lần** cho mỗi mô hình.

## 3. Template 13 lớp

| ID | Lớp | Cử chỉ | Predicate bắt buộc | Mức phân biệt dự kiến |
|---:|---|---|---|---|
| 0 | B0A | Trỏ 1 ngón | `index_extended` | Tạm |
| 1 | B0B | Trỏ 2 ngón | `index_middle_extended` | Tạm |
| 2 | G01 | Click 1 ngón | `index_extended`, `index_finger_moves_down` | Yếu |
| 3 | G02 | Click 2 ngón | `index_middle_extended`, `index_finger_moves_down` | Yếu |
| 4–7 | G03–G06 | Hất lên / xuống / trái / phải | `hand_moves_up/down/left/right` | Tốt |
| 8 | G07 | Mở 2 lần | `hand_closed`, `hand_open` | Yếu |
| 9 | G08 | Double click 1 ngón | `index_extended`, `index_finger_moves_down` | Rất yếu |
| 10 | G09 | Double click 2 ngón | `index_middle_extended`, `index_finger_moves_down` | Rất yếu |
| 11 | G10 | Zoom vào | `fingers_spread_increasing` | Yếu |
| 12 | G11 | Zoom ra | `fingers_spread_decreasing` | Yếu |

Mỗi template còn có predicate nên có / tùy chọn và dấu hiệu loại trừ (xem file YAML). Mô tả cử chỉ G07–G11 chưa được đối chiếu bằng mắt với clip.

**Khoảng trống của tầng symbolic:**
- Matcher (`graph_matching.score_template`) chỉ kiểm tra predicate có xuất hiện hay không, **không đếm số lần và không xét thứ tự** → không tách được 1 lần / 2 lần (G01/G08, G02/G09, G07).
- Không có predicate khoảng cách ngón cái – ngón trỏ (pinch) → zoom vào / ra khó phân biệt.
- Không có predicate chuyển động riêng của ngón giữa → click 2 ngón dùng tạm chuyển động ngón trỏ.

## 4. Vì sao template thuần luật thất bại

Tỉ lệ clip val của mỗi lớp có chứa predicate (trích):

| Lớp | `index_extended` | `thumb_extended` | `hand_moves_up` | `hand_moves_left` | `hand_open` |
|---|---:|---:|---:|---:|---:|
| B0A (trỏ, gần như đứng yên) | 1,00 | 1,00 | 1,00 | 1,00 | 0,77 |
| G01 (click 1 ngón) | 1,00 | 1,00 | 0,79 | 0,71 | 0,32 |
| G03 (hất lên) | 1,00 | 1,00 | 0,93 | 0,86 | 0,96 |
| G05 (hất trái) | 1,00 | 1,00 | 0,86 | 0,93 | 1,00 |
| G10 (zoom) | 1,00 | 1,00 | 0,68 | 0,68 | 0,32 |

Predicate "bật" ở gần như mọi lớp: `index_extended` và `thumb_extended` có ở 100% clip của cả 13 lớp; predicate hướng chuyển động có ở 60–100% clip kể cả lớp trỏ đứng yên. Nguyên nhân:
- Ngưỡng predicate (chuẩn hóa theo vai, `motion_threshold=0.08`) được đặt cho bộ 27 lớp cử chỉ cánh tay cũ, quá nhạy với chuyển động tay nhỏ trước webcam của IPN.
- Predicate được tính trên toàn clip; clip dài (B0A/B0B vài trăm frame) chứa gần như mọi predicate ở một thời điểm nào đó.

Vì tín hiệu đầu vào không phân biệt được lớp, **sửa template không khắc phục được**; phải sửa ở tầng predicate.

## 5. Kết quả V4

**Thiết lập:** `train_cross_attention_v4.py --logit-branch final`; nhánh neural Transformer d_model 128 / 3 layer / 4 head giống Người 2, cộng graph encoder, cross-attention và gate fusion. Seed 42, batch 32, tối đa 50 epoch, early stopping patience 8, chọn checkpoint theo val macro-F1. Giữ mặc định V4: neural lr 2e-4, kg lr 3e-4, dropout 0,15, label smoothing 0,05, augmentation (jitter, che frame, che keypoint). Dừng ở epoch 36, checkpoint tốt nhất epoch 28. Máy RTX 3050 6GB, PyTorch 2.6 (Người 2 chạy Colab, PyTorch 2.11).

**Mô hình đối chứng:** cùng script, cùng mọi tham số, `--logit-branch neural` (chỉ còn Transformer, không dùng graph). Dừng ở epoch 36, tốt nhất epoch 28.

**F1 từng lớp trên test:**

| Lớp | Neural-only (Người 2) | Đối chứng (không KG) | V4 |
|---|---:|---:|---:|
| B0A | 0,954 | 0,958 | 0,958 |
| B0B | 0,972 | 0,974 | 0,976 |
| G01 | 0,504 | 0,512 | 0,576 |
| G02 | 0,477 | 0,466 | 0,419 |
| G03 | 0,703 | 0,766 | 0,763 |
| G04 | 0,844 | 0,862 | 0,789 |
| G05 | 0,940 | 0,898 | 0,907 |
| G06 | 0,764 | 0,817 | 0,836 |
| G07 | 0,717 | 0,774 | 0,796 |
| G08 | 0,433 | 0,353 | 0,477 |
| G09 | 0,442 | 0,421 | 0,583 |
| G10 | 0,648 | 0,700 | 0,607 |
| G11 | 0,545 | 0,617 | 0,481 |
| **Macro-F1** | **0,688** | **0,701** | **0,705** |

**Cặp nhầm nhiều nhất của V4** (thật → đoán): G08→G01 27, G02→G09 21, G01→G08 11, G11→G10 9, B0A→B0B 7. Giống baseline của Người 2 (G08→G01 27, G09→G02 24, G01→G08 18, G02→G09 17).

**So V4 với đối chứng trên từng clip test:** V4 đúng mà đối chứng sai 55 clip; đối chứng đúng mà V4 sai 53 clip. V4 khá hơn ở G08/G09 nhưng kém hơn ở G10/G11 và G04, bù trừ nhau. Mỗi lớp G chỉ có 52 clip test và mới chạy một seed, nên chênh lệch từng lớp nằm trong mức dao động thông thường.

**Lưu ý khi đọc "nhánh KG":** trong cùng checkpoint V4, nếu chỉ đọc đầu ra nhánh KG thì test đạt acc 0,816 / macro-F1 0,721. Con số này **không** phản ánh năng lực của predicate, vì nhánh KG trong V4 dùng cross-attention vào cả token keypoint của nhánh neural. Năng lực thật của predicate + template là kết quả thuần luật ở mục 4.

## 6. Kết luận và đề xuất

Trên IPN, với predicate và template hiện tại, knowledge graph **không cải thiện độ chính xác** so với Transformer neural-only được train cùng cách. Mức tăng so với baseline của Người 2 đến từ công thức train (augmentation, label smoothing, dropout, learning rate).

Đề xuất:
1. Hiệu chỉnh ngưỡng predicate cho cử chỉ ngón tay của IPN, dựa trên thống kê train/val; yêu cầu predicate kéo dài tối thiểu một phần clip thay vì chỉ cần xuất hiện một lần.
2. Thêm predicate: khoảng cách ngón cái – ngón trỏ (zoom), đếm số nhịp click, chuyển động ngón giữa; hoặc cho matcher xét thứ tự và số lần event.
3. Chạy thêm 2–3 seed cho V4 và đối chứng để có độ lệch chuẩn trước khi kết luận chênh lệch từng lớp.
4. Có thể đưa công thức train của V4 (augmentation, label smoothing) vào baseline neural-only, vì đó là phần tạo ra cải thiện.

**Hạn chế:** một seed; mô tả template G07–G11 chưa đối chiếu bằng mắt; chưa kiểm tra lại các video lệch frame do Người 1 ghi nhận.

## 7. File và cách chạy lại

**Trong repo (branch này):**
- `configs/ipn/gesture_templates.yaml`: template 13 lớp.
- `scripts/eval_ipn_template_matching.py`: chấm template thuần luật.
- `final/code/src_neurosymbolic/train_cross_attention_v4.py`: thêm `--select-metric`.

**Trên Drive, folder `ipn-extract-data/ipn_v4_seed42`:** checkpoint (`best.pt`, `last.pt`), log, `history.json`, metric val/test, dự đoán từng clip, confusion matrix của V4 (`fusion_seed42`) và đối chứng (`neural_same_recipe_seed42`), 13 template graph, kết quả chấm template.

**Lệnh** (PowerShell, chạy từ thư mục gốc repo, dữ liệu của Người 1 đặt ở `external_data/ipn_processed`):

```powershell
$P = "external_data/ipn_processed"
$env:PYTHONPATH = "final/code"

# 1. Template graph
python final/code/src_neurosymbolic/kg/build_template_graph.py `
  --templates configs/ipn/gesture_templates.yaml `
  --list-file $P/train.txt --list-file $P/val.txt --list-file $P/test.txt `
  --output-dir $P/template_graphs --pretty

# 2. Instance graph (mỗi split, khoảng 10–25 phút)
python final/code/src_neurosymbolic/precompute_instance_graphs_phase_v2.py `
  --keypoint-root $P/keypoints --cache-root $P/keypoint_tensor_cache `
  --output-root $P/instance_graphs `
  --train-file $P/train.txt --val-file $P/val.txt --test-file $P/test.txt `
  --splits train val test

# 3. Chấm template thuần luật trên val
python scripts/eval_ipn_template_matching.py --split val

# 4. Train V4 (đổi --logit-branch neural và --output-dir để chạy đối chứng)
python final/code/src_neurosymbolic/train_cross_attention_v4.py `
  --keypoint-root $P/keypoints --cache-root $P/keypoint_tensor_cache `
  --instance-graph-root $P/instance_graphs --template-dir $P/template_graphs `
  --train-file $P/train.txt --val-file $P/val.txt `
  --output-dir final/outputs/ipn_v4/fusion_seed42 `
  --logit-branch final --num-classes 13 --select-metric macro_f1 `
  --batch-size 32 --epochs 50 --early-stop-patience 8 --seed 42

# 5. Test một lần (đối chứng: --eval-branch neural)
python final/code/src_neurosymbolic/test_cross_attention_v2.py `
  --checkpoint final/outputs/ipn_v4/fusion_seed42/best.pt `
  --keypoint-root $P/keypoints --cache-root $P/keypoint_tensor_cache `
  --instance-graph-root $P/instance_graphs --template-dir $P/template_graphs `
  --test-file $P/test.txt --split test --batch-size 32 --eval-branch final `
  --output final/outputs/ipn_v4/fusion_seed42/test_eval
```
