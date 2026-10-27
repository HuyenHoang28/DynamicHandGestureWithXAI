# DATE 27 lớp — đánh giá final và so sánh baseline

## Phạm vi

Đánh giá dùng đúng `test_1229.txt`, gồm 1.229 clip và 27 lớp subject-disjoint. Checkpoint chính là `cross_attention_v4_template170_retrain_next/best.pt`, epoch 19, được chọn theo validation. Đây là test đã được mở trong quá trình phát triển, vì vậy nên gọi là mốc tham chiếu validation-selected, không gọi là locked holdout mới.

## Kết quả tái lập ngày 2026-10-23

| Mô hình/head | Correct | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Neural-only độc lập, checkpoint legacy epoch 23 | 1.171/1.229 | **95,28%** | **95,13%** |
| Neural head trong checkpoint V4 | 1.163/1.229 | 94,63% | 94,29% |
| Template-guided/KG head trong checkpoint V4 | 1.168/1.229 | 95,04% | 94,67% |
| Final neural + symbolic V4 | 1.166/1.229 | 94,87% | 94,52% |

KG head sửa 9 lỗi của neural head và gây 4 lỗi mới, net `+5`. Final fusion chỉ tăng 3 mẫu đúng so với neural head và thấp hơn KG head 2 mẫu. So với neural-only độc lập, Final thấp hơn `0,41` điểm phần trăm accuracy và `0,61` điểm phần trăm Macro-F1. Do checkpoint neural-only hiện chỉ còn file tên `last.pt` và metadata cho biết best validation ở epoch 18, con số neural-only phải được ghi là baseline legacy, không khẳng định là checkpoint chọn theo validation tốt nhất.

## Baseline trong manuscript

| Phương pháp | Accuracy |
|---|---:|
| SignBart | 88,06 ± 0,76% |
| Sparse-ST-GCN | 92,94 ± 0,40% |
| ProtoGCN | 93,28 ± 0,27% |
| DATE Transformer encoder đã công bố | 94,22% |
| Proposed Fusion trong manuscript | **96,40 ± 0,17%** |

Single checkpoint Final `94,87%` cao hơn ProtoGCN `1,59` điểm phần trăm và DATE Transformer `0,65` điểm phần trăm, nhưng thấp hơn mean Proposed Fusion trong manuscript `1,53` điểm phần trăm. Không so trực tiếp Macro-F1 vì bảng paper chỉ công bố accuracy. Các baseline RTMW trong paper là mean ± sample SD của 10 entry; run local ở đây là một checkpoint, nên không được trình bày như một phép kiểm định thống kê tương đương.

## Kết luận dùng khi báo cáo

- Kết quả hợp lệ để demo/tái lập: Final `94,87%`, Macro-F1 `94,52%`.
- Head tốt nhất trong cùng checkpoint: template-guided/KG `95,04%`.
- Fusion chưa chứng minh vượt baseline neural-only độc lập `95,28%`.
- `best_test.pt` đạt `96,18%` nhưng được chọn theo test; chỉ để exploratory, không dùng làm headline.
- Manuscript claim `96,40 ± 0,17%` cần đủ 10 run provenance/checkpoint để tái lập độc lập; repo hiện chỉ đóng gói chắc chắn một checkpoint validation-selected.

## Evidence và validation

- Run V4: `results/runs/date27_v4_validation_selected_2026-10-23/`.
- Run neural-only: `results/runs/date27_neural_only_legacy_2026-10-23/`.
- Final bundle: `results/final/date27_v4_validation_selected/`.
- Repo verification: `OK`.
- Unit tests: `123 passed`, 3 warning PyTorch đã biết.
