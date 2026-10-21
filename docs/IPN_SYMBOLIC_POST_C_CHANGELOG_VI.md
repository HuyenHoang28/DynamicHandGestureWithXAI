# Ghi chú thay đổi Symbolic V2 sau validation C

Ngày cập nhật: 2026-10-21

## Mốc đối chiếu

Artifact đầu vào: `external_data/ipn_symbolic_v2_post_c-20261021T140256Z-1-001.zip`.

- Validation: accuracy `0.20034`, macro-F1 `0.06262`.
- `250/589` clip hòa điểm; top-2 margin trung bình `0.01156`.
- `448/589` clip bị dự đoán thành B0A do tie/điểm thấp và quy tắc argmax cũ.
- Gesture-only G01-G11 chỉ đúng `9/308` clip.
- Các predicate đối chiều, đặc biệt pinch tăng/giảm, xuất hiện gần như toàn bộ clip.

## Thay đổi

### 1. Lọc trend nhiễu trước khi dựng graph

Predicate chuyển động có thêm `trend_consistency`, tính bằng:

`abs(net movement theo chiều đang xét) / total variation`

Phase-V2 dùng mặc định `0.6`. Một cửa sổ có endpoint đủ xa nhưng đảo chiều mạnh ở giữa sẽ không còn phát predicate directional. Gate áp dụng cho:

- chuyển động tay theo trục x/y;
- chuyển động ngón tay;
- khoảng cách pinch;
- độ spread của các đầu ngón;
- coordination hai tay.

Giá trị được ghi vào `predicate_config` và provenance để graph cũ không bị tái sử dụng nhầm. Direct Python callers vẫn mặc định `0.0` để giữ tương thích; pipeline IPN/Colab truyền rõ `0.6`.

Fingerprint nguồn giờ lưu định danh graph tương đối dưới `instance_graph_root` thay vì
đường dẫn input/staging tuyệt đối. Vì vậy cả keypoint JSON và tensor cache có thể tạo
định danh ổn định, và graph train/validation đã xác minh có thể được
tái sử dụng an toàn khi chuyển từ attempt `train_val` sang `full_with_test`; size
và SHA-256 của nội dung vẫn phải khớp.

### 2. Sequence temporal không tái sử dụng evidence

Matcher nhiều phase giờ yêu cầu:

- tất cả phase thuộc cùng một subject;
- phase sau bắt đầu sau khi phase trước kết thúc;
- không tái sử dụng occurrence/atomic evidence giữa hai bước;
- ưu tiên candidate kết thúc sớm để dành không gian cho các phase tiếp theo.

Report score có thêm `sequence_subject` và `sequence_matches`. Thay đổi này chủ yếu bảo vệ G07 khỏi match giả do phase overlap hoặc một atomic event bị chia nhỏ.

### 3. Abstention thay cho tie-break về B0A

Evaluator có hai ngưỡng tùy chọn:

- `--min-best-score`;
- `--min-top2-margin`.

Notebook dùng cấu hình ban đầu `min_best_score=0.15`, `min_top2_margin=0.02`. Mẫu dưới ngưỡng được đánh dấu `abstain`, không tự động gán B0A. Report bổ sung:

- `abstained_count`, `abstention_rate`, `abstentions_by_class`;
- `accepted_count`, `coverage`, `selective_accuracy`.

Accuracy vẫn chia trên toàn cohort; `selective_accuracy` chỉ đo các mẫu symbolic thực sự nhận dự đoán.

### 4. Diagnostic theo từng clip

Validation sinh thêm `val_diagnostics.jsonl`, gồm:

- true/predicted label hoặc abstain;
- 13 template scores, thứ hạng, tie và top-2 margin;
- lý do abstain;
- occurrence count, temporal score và event matches của từng template.

File này là đầu vào để chỉnh threshold/template vòng kế tiếp, không cần đọc test set.

## Cách chạy vòng train/validation tiếp theo

Chạy lại toàn bộ notebook `notebooks/IPN_Symbolic_V2_Colab.ipynb` với:

- `RUN_TEST=False`;
- `OVERWRITE=False` là đủ vì run-config/provenance mới làm attempt và graph cache cũ không tương thích;
- dùng commit chứa thay đổi này, không dùng artifact graph cũ để kết luận hiệu quả trend gate.

Các file cần lấy về sau khi chạy:

- `train_evaluation.json`;
- `val_evaluation.json`;
- `val_diagnostics.jsonl`;
- `completed_run.json`.

## Tiêu chí trước khi bật neural arbitration

- Không dùng symbolic làm classifier thay neural C.
- Không override khi symbolic abstain, tie hoặc margin thấp.
- Chỉ bật override cho từng cặp G01/G08, G02/G09, G10/G11 sau khi đo precision trên validation.
- Không mở/chỉnh template theo test trước khi train/validation đạt tiêu chí đã chốt.

Ngưỡng `0.15/0.02` là cấu hình an toàn ban đầu, chưa phải ngưỡng tối ưu. Chỉ hiệu chỉnh bằng train/validation diagnostics.

Kiểm tra ngưỡng này trên **graph cũ** cho coverage `18.34%` và selective accuracy
`32.41%` (`481/589` abstain). Con số này xác nhận abstention đã chặn phần lớn
tie/low-margin nhưng vẫn tuyệt đối chưa đủ để bật override. Cần chạy lại graph
với trend gate mới rồi mới đánh giá lại hai ngưỡng.
