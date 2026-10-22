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

---

## Cập nhật 2026-10-22 — active action và pair-conditioned audit

### Nguồn quyết định

Artifact được phân tích: `external_data/035179b821061ab7-20261022T015729Z-1-001.zip`, commit `e0a1e50`.

- CRC, số graph và toàn bộ hash payload hợp lệ; train `2443`, validation `589`, thiếu graph `0`.
- Raw 13-way validation tăng nhẹ từ accuracy `0.20034` lên `0.21732`, macro-F1 từ `0.06262` lên `0.08708`.
- Với ngưỡng symbolic `0.15/0.02`: coverage `124/589 = 21.05%`, selective accuracy `39/124 = 31.45%`, abstain `465/589 = 78.95%`.
- Cả `465` abstention đều do top-2 margin dưới `0.02`; ngưỡng best-score `0.15` không loại thêm mẫu nào.
- Raw prediction vẫn lệch mạnh về B0A: `461/589 = 78.27%`. Vì vậy không dùng symbolic 13-way argmax để thay neural C.

Đối chiếu neural C `masked_mean_light_dropout` với diagnostics 0351 cho baseline validation accuracy `0.891341`. Mô phỏng pairwise ở neural margin tối đa `0.20`, symbolic score/margin tối thiểu `0.15/0.02` tạo đúng một override thực sự; override đó sai, làm accuracy còn `0.889643`. Kết luận: **chưa bật override**.

### 1. Phân biệt hành động chính và pha trả về

- Directional atomic event được nhóm theo `subject` và cặp hướng đối nhau.
- Run không chồng lấn đầu tiên là `active`; run đối hướng kế tiếp là `return`; chuỗi A–B–A được gắn `active–return–active`, nên double click không mất lần nhấn thứ hai.
- Run đối hướng chồng lấn bị gắn `ambiguous`; template `active` không dùng evidence này.
- Isolated directional run vẫn là `active`, tránh làm mất recall khi extractor không quan sát được pha trả về.
- Vai trò được truyền từ atomic event qua `concept_action_roles`, edge `action_roles`, matcher và diagnostics `active_intervals`; atomic ID được giữ nguyên để dedup occurrence vẫn đúng.
- Event builder/provenance đổi thành `phase_v3_action_roles` với thuật toán `directional_reversal_v1`, làm graph cache cũ bị nhận diện là không tương thích.

Template G01–G06 và G08–G11 dùng `action_role: active`. B0A, B0B và chuỗi shape G07 giữ `any`. Parser chỉ chấp nhận `any|active|return` và mặc định `any` để tương thích template cũ.

Giới hạn còn lại: role hiện dựa trên thứ tự và đối hướng, chưa dùng motion magnitude. Vì vậy đây là gate bảo thủ; mọi override vẫn phải được đo precision trên validation.

### 2. Arbitration chỉ trong neural candidate pair

Thêm `scripts/eval_ipn_pair_arbitration.py`:

- join neural `predictions.csv` với symbolic diagnostics bằng suffix chuẩn hóa từ `keypoints/`, không so đường dẫn tuyệt đối Colab/local;
- kiểm tra đủ contract 13 label ID/name và join one-to-one;
- chỉ xem xét khi neural top-2 đúng một trong ba cặp G01/G08, G02/G09, G10/G11;
- dùng score/margin, required evidence, temporal score và occurrence gate trong đúng cặp; global 13-way tie/abstain không được dùng để loại một pair hợp lệ do class thứ ba;
- G10/G11 yêu cầu đúng một chiều có `active_intervals`, từ chối evidence đối hướng active hoặc chỉ có return;
- mặc định `audit_only`, không đổi prediction nhưng vẫn báo số override được đề xuất, final đúng/sai, tác động beneficial/harmful/neutral và net change; `--enable-override` là mô phỏng có chủ ý và bắt buộc ngưỡng neural margin đã chọn trên validation;
- exact symbolic tie luôn abstain kể cả khi cấu hình margin bằng `0`; CSV neural phải có xác suất trong `[0,1]`, giảm dần và `confidence` khớp top-1;
- xuất report tổng hợp và JSONL giải thích từng quyết định.

Notebook `notebooks/IPN_Symbolic_V2_Colab.ipynb` có tùy chọn `NEURAL_C_PREDICTIONS_CSV`. Khi được đặt, notebook ghi SHA-256 của CSV vào run config và sinh `val_pairwise_audit.json` cùng `val_pairwise_audit.jsonl`; notebook luôn giữ override tắt.

Ví dụ audit:

```bash
python scripts/eval_ipn_pair_arbitration.py \
  --neural-predictions /path/to/val_eval/predictions.csv \
  --symbolic-diagnostics /path/to/val_diagnostics.jsonl \
  --max-neural-margin 0.20 \
  --output val_pairwise_audit.json \
  --audit-jsonl val_pairwise_audit.jsonl
```

### 3. Quy tắc documentation từ vòng này

Thêm `AGENTS.md` ở root. Mọi thay đổi code, config, template, notebook, data contract hoặc runtime behavior từ nay phải cập nhật file phù hợp trong `docs/` trong cùng commit. Entry phải nêu thay đổi, lý do, ảnh hưởng tương thích/an toàn và bằng chứng kiểm thử. Artifact sinh ra và notebook kết quả không được stage nếu chưa có yêu cầu rõ ràng.

### 4. Validation trước commit

- Focused suite cho matcher, predicate, precompute, evaluator, pair arbitration và notebook: `59 passed`.
- Full repository suite sau review/fix: `113 passed`, `3` warning PyTorch đã có sẵn.
- Review độc lập đã được xử lý: exact-tie bắt buộc abstain; kiểm tra consistency của neural CSV; phân biệt final đúng/sai với tác động beneficial/harmful/neutral; metadata thiếu dùng `event_builder=unknown` thay vì tự nhận là builder mới.
- Pair arbitration unit tests: path canonicalization, pair-only margin, global tie independence, occurrence gate, G10/G11 opposite-active rejection, audit-only behavior, explicit override và non-pair hold.
- Chưa có kết quả validation mới cho graph `phase_v3_action_roles`; cần chạy lại Colab train/val trước khi đánh giá hiệu quả của active-role gate.
