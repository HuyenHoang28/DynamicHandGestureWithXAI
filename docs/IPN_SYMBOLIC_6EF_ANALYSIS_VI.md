# Phân tích validation artifact 6ef

Ngày phân tích: 2026-10-22

Artifact: `results/experiments/ipn13/artifact_zips/6efa8ad55dde2909-20261022T041458Z-1-001.zip`

Commit nguồn: `160fdd6015450bb19d4573df3c35c0577ad3f54f`

## Tính toàn vẹn

- ZIP có 3.059 thành viên và không có entry hỏng.
- `completed_run.json` ở trạng thái `complete`; hash run config và sáu payload đều khớp.
- Có đủ 2.443 graph train và 589 graph validation, thiếu 0; diagnostics validation có 589 dòng.
- Run dùng `phase_v3_action_roles`, `directional_reversal_v1`, threshold symbolic `0.15/0.02`.
- `RUN_TEST=false`; không có test graph/evaluation. Pairwise audit trong notebook bị tắt.

## So sánh với artifact 0351

### Validation có abstention

| Metric | 0351 | 6ef | Nhận xét |
|---|---:|---:|---|
| Accuracy toàn cohort | 0.06621 | 0.10526 | tăng do coverage/B0A |
| Macro-F1 | 0.04673 | 0.03278 | giảm |
| Coverage | 0.21053 | 0.33277 | tăng |
| Selective accuracy | 0.31452 | 0.31633 | gần như không đổi |
| Abstain | 465 | 393 | giảm 72 |
| Tie | 283 | 255 | giảm 28 |
| Mean top-2 margin | 0.01321 | 0.02816 | tăng |

Trong 62 dự đoán đúng được chấp nhận của 6ef, 60 là B0A và 2 là B0B. F1 thresholded của toàn bộ G01–G11 bằng 0.

### Raw 13-way không abstain

| Metric | 0351 | 6ef |
|---|---:|---:|
| Accuracy | 0.21732 | 0.22241 |
| Macro-F1 | 0.08708 | 0.06091 |
| Dự đoán B0A | 461/589 | 515/589 |

Raw accuracy chỉ tăng 0,51 điểm phần trăm nhưng Macro-F1 giảm 2,62 điểm và tỷ lệ B0A tăng từ 78,3% lên 87,4%. Vì vậy không coi accuracy tổng tăng là cải thiện symbolic gesture.

## Regression do hard action-role gate

| True template | Mean score 0351 | Mean score 6ef | Non-zero 0351 | Non-zero 6ef |
|---|---:|---:|---:|---:|
| G01 | 0.3088 | 0.1777 | 28 | 16 |
| G08 | 0.2773 | 0.0692 | 28 | 11 |
| G02 | 0.3051 | 0.1150 | 25 | 9 |
| G09 | 0.3096 | 0.0730 | 28 | 13 |
| G10 | 0.3890 | 0.1810 | 28 | 15 |
| G11 | 0.3952 | 0.1585 | 28 | 14 |

Trong 274.904 role-tagged edges validation, `ambiguous` chiếm 173.669 (63,2%), `active` 72.698 (26,4%) và `return` 28.537 (10,4%). Nguyên nhân chính là v1 đánh dấu cả hai directional run là ambiguous khi interval chỉ cần chồng nhau, trong khi sliding windows thường chồng tại điểm đổi chiều. Hard `action_role: active` vì vậy loại nhiều true evidence.

G10/G11 cũng chưa được tách sạch: 10/28 true-G10 và 7/28 true-G11 có active interval cho cả hai template.

## Pairwise và neural C

| Cặp | Pair accuracy 0351 | Pair accuracy 6ef | Selective 0351 | Selective 6ef |
|---|---:|---:|---:|---:|
| G01/G08 | 0.6071 | 0.2321 | 0.6154 | 0.4815 |
| G02/G09 | 0.6429 | 0.1607 | 0.6735 | 0.4091 |
| G10/G11 | 0.5714 | 0.4107 | 0.6757 | 0.4490 |

Audit độc lập với neural C tại `max_neural_margin=0.20`, symbolic score/margin `0.15/0.02`:

- neural baseline accuracy: `0.89134`;
- proposed override: 2;
- beneficial: 0; harmful: 2; net: -2.

Sweep neural margin 0.02–0.50 và symbolic margin 0.02–0.20 không tìm được cấu hình có net dương. Mở toàn bộ neural margin tạo 10 proposal: 1 beneficial và 9 harmful.

## Quyết định

- Không bật override và không mở locked test.
- Không giữ hard action-role gate của 6ef.
- Revision tiếp theo dùng role như diagnostics/arbitration gate, không làm điều kiện bắt buộc của classifier.
- Role v2 phải cho phép overlap có thứ tự tại crossover; chỉ đánh ambiguous khi hai run gần như đồng thời/không xác định được thứ tự.
- Chỉ đánh giá lại sau khi dựng graph train/validation mới; không dùng số test để chỉnh tiếp.

## Kiểm tra bản sửa trên graph 6ef hiện có

Để cô lập tác động của hard role gate, 589 graph validation trong 6ef được chấm lại bằng template `action_role: any` và matcher mới, chưa cần dựng lại graph:

- accuracy toàn cohort: `0.06621`;
- Macro-F1: `0.04686`;
- coverage: `0.20713`;
- selective accuracy: `0.31967`;
- tie: `283`; mean top-2 margin: `0.01317`.

Kết quả quay gần như đúng mốc 0351 (`0.06621 / 0.04673 / 0.21053 / 0.31452`), xác nhận phần regression chính đến từ việc dùng role làm hard template requirement. Pairwise audit sau rescore còn 1 proposed override harmful tại neural margin `0.20`, nên quyết định giữ override tắt không đổi.

Đây mới là kiểm tra hồi quy trên graph v3. Hiệu quả của `directional_reversal_v2` vẫn phải được đo bằng một run train/validation mới với graph `phase_v4_action_role_audit`.
