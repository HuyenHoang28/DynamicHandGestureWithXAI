# Ghi chú thay đổi Symbolic V2 sau validation C

Ngày cập nhật: 2026-10-21

## Mốc đối chiếu

Artifact đầu vào: `results/experiments/ipn13/artifact_zips/ipn_symbolic_v2_post_c-20261021T140256Z-1-001.zip`.

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

Artifact được phân tích: `results/experiments/ipn13/artifact_zips/035179b821061ab7-20261022T015729Z-1-001.zip`, commit `e0a1e50`.

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

---

## Cập nhật 2026-10-22 — sửa regression phát hiện từ artifact 6ef

Phân tích đầy đủ được lưu tại `docs/IPN_SYMBOLIC_6EF_ANALYSIS_VI.md`.

Artifact 6ef chứng minh hard `action_role: active` làm giảm mạnh true-template coverage và pair discrimination: raw Macro-F1 giảm từ `0.08708` xuống `0.06091`; tất cả G01–G11 có thresholded F1 bằng 0. Pairwise audit với neural C đề xuất 2 override và cả 2 đều harmful.

Revision kế tiếp thực hiện:

- trả template hành động về `action_role: any`, để role không còn xóa evidence khỏi classifier;
- vẫn giữ role trên graph và xuất `role_intervals`/`active_intervals` làm diagnostic cho arbitration;
- thay rule “có overlap là ambiguous” bằng phân loại dựa trên thứ tự tâm interval; chỉ trường hợp gần như đồng thời mới ambiguous;
- ambiguity chỉ áp dụng cho hai run đối hướng; hai run cùng hướng gần đồng thời vẫn giữ role active;
- đổi provenance sang `phase_v4_action_role_audit` / `directional_reversal_v2`, bắt buộc dựng graph mới;
- tiếp tục giữ pairwise ở `audit_only`, không bật override, không chạy test.

Rescore 589 graph 6ef với hard role gate đã bỏ cho accuracy `0.06621`, Macro-F1 `0.04686`, coverage `0.20713`, selective accuracy `0.31967`; gần như phục hồi mốc 0351 và xác nhận đúng nguyên nhân regression. Full repository suite sau thay đổi: `115 passed`, `3` warning PyTorch đã có sẵn; focused suite sau review fix: `73 passed`.

---

## Cập nhật 2026-10-22 — soft action-role alignment sau artifact d25

### Nguồn quyết định

Artifact `results/experiments/ipn13/artifact_zips/d255086f928a0615-20261022T075132Z-1-001.zip` hợp lệ, đúng commit `1ef544c`, đủ `2443` train và `589` validation graph, không thiếu graph. `directional_reversal_v2` giảm role `ambiguous` từ `63,2%` ở 6ef xuống `8186/274894 = 2,98%`, nhưng kết quả classifier gần như trở lại mốc 0351: validation accuracy `0.06621`, Macro-F1 `0.04686`, coverage `0.20713` và selective accuracy `0.31967`.

Hard `action_role: active` không được phục hồi vì đã chứng minh làm mất recall. Thay vào đó matcher áp dụng `active_share_soft_v1` trên các interval có role quyết định:

`active_share = active / (active + return)`

`multiplier = floor + (1 - floor) * active_share`

- `role_alignment_floor` được cấu hình tại `template_policy.matching`, revision này dùng `0.85`; vì vậy role chỉ có thể hạ tối đa 15% điểm template, không thể xóa evidence.
- `ambiguous` và `unknown` được báo cáo nhưng không tham gia mẫu số và không làm giảm điểm.
- Template V2 cũ hoặc graph V2 không có policy mặc định dùng floor `1.0`, giữ nguyên điểm số cũ. Template V1 chỉ chấp nhận `1.0`; cấu hình V1 khác `1.0` bị từ chối để không quảng bá một policy mà nhánh matcher V1 không thực thi.
- Mỗi dòng `score_template` bổ sung `role_alignment` gồm algorithm, floor, số interval theo role, active share, multiplier và trạng thái applied.
- Template graph và index bổ sung `matching_policy.role_alignment_floor`; parser từ chối bool, NaN/Infinity và giá trị ngoài `[0,1]`.
- Template tiếp tục dùng `action_role: any`; pairwise neural/symbolic vẫn `audit_only`, override vẫn tắt.

### Validation an toàn

Chấm lại trực tiếp toàn bộ `589` validation graph của d25 bằng template graph mới, cùng threshold `0.15/0.02`:

| Metric | d25 trước | Soft role 0.85 |
|---|---:|---:|
| Raw accuracy | `0.21732` | `0.23430` |
| Raw Macro-F1 | `0.08679` | `0.09009` |
| Thresholded accuracy | `0.06621` | `0.10017` |
| Thresholded Macro-F1 | `0.04686` | `0.06182` |
| Coverage | `0.20713` | `0.27844` |
| Selective accuracy | `0.31967` | `0.35976` |
| Tie rate | `0.47878` | `0.43294` |
| Mean top-2 margin | `0.01320` | `0.01750` |

Floor `0.85` là lựa chọn calibration bảo thủ trên validation, chưa phải hằng số đã xác nhận bằng locked test. Sweep lân cận dùng cùng graph, template và threshold:

| Floor | Raw acc. | Raw Macro-F1 | Thresholded acc. | Thresholded Macro-F1 | Coverage | Selective acc. |
|---:|---:|---:|---:|---:|---:|---:|
| `0.800` | `0.23090` | `0.08354` | `0.10187` | `0.06155` | `0.28862` | `0.35294` |
| `0.825` | `0.23090` | `0.08354` | `0.10187` | `0.06166` | `0.28693` | `0.35503` |
| `0.850` | `0.23430` | `0.09009` | `0.10017` | `0.06182` | `0.27844` | `0.35976` |
| `0.875` | `0.23430` | `0.09049` | `0.09677` | `0.05649` | `0.27334` | `0.35404` |
| `0.900` | `0.23260` | `0.08708` | `0.09338` | `0.06023` | `0.26655` | `0.35032` |

`0.85` được chọn vì giới hạn penalty tối đa ở 15%, đồng hạng raw accuracy tốt nhất trong vùng khảo sát, đồng thời đạt Thresholded Macro-F1 và selective accuracy tốt nhất trong năm điểm trên. Đây vẫn là lựa chọn theo validation; locked test tiếp tục đóng.

Audit với neural C tại neural margin `0.20` vẫn đề xuất đúng một override và override đó harmful (`net -1`), nên thay đổi này chưa cấp quyền bật fusion. Locked test không được chạy.

Validation cuối: focused suite `55 passed`; full repository suite `123 passed`, `3` warning PyTorch đã có sẵn. Review độc lập không còn blocker sau khi khóa config production bằng test, thêm contract V1 và ghi rõ tiêu chí chọn floor từ sweep lân cận.
## Cập nhật 2026-10-27 — bắt đầu temporal factor V3 và pair-residual

- Thay đổi: thêm tài liệu kiến trúc cho click-cycle, synchronized two-finger cycle, dominant pinch transition và pair-specific residual; bổ sung quy tắc authoring V3. Triển khai module temporal factor, adapter normalized-keypoint, predicate/provenance V3 opt-in và bảo toàn occurrence ID trong event builder.
- Lý do: audit run ensemble 3 seed cho thấy lỗi vẫn tập trung ở `G01/G08`, `G02/G09`, `G10/G11`; predicate cửa sổ trượt không đại diện ổn định cho click count, còn global KG fusion từng gây net harm.
- Tương thích/an toàn: V2 tiếp tục là mặc định; V3 phải có predicate config, graph provenance và template version riêng. Không tune bằng test đã mở, evidence không chắc phải abstain, và residual không được đổi logits ngoài cặp mục tiêu.
- Review/safety: detector yêu cầu inactive baseline trước click, reset qua occlusion dài, từ chối zero-overlap two-finger match, trả `unknown` cho hai hướng pinch gần tương đương; factor metadata được giữ xuyên predicate → event → graph và factor ngắn hợp lệ không còn bị duration filter V2 loại nhầm. Calibration chỉ nhận đúng `1` hoặc `2` occurrence; từ `3` trở lên abstain thay vì bị tính nhầm là double-click.
- Calibration/Colab: thêm CLI train/validation-only và notebook `IPN13_Temporal_Factor_V3_Calibration_Colab.ipynb`; chọn config duy nhất bằng validation coverage/accuracy, xuất diagnostics từng cặp/subject và artifact có hash/provenance. Notebook không có đường chạy test. Sửa shared dataset resolver để nhận cache IPN `T64/<split>/<label>/<subject>/<clip>.pt`; CLI và preflight từ chối mọi source path thiếu split hoặc lệch khỏi file `train`/`val`, nên manifest bị trộn không thể mở tensor test. Notebook preflight toàn bộ target cache trước sweep và ZIP chứa `input_tensors.jsonl` với path/kích thước/SHA-256 của từng tensor thực sự đã đọc.
- Tương thích: nhánh resolve cache cũ vẫn được giữ làm fallback; thay đổi mới chỉ bổ sung candidate canonical. Preflight cố ý từ chối raw JSON fallback trong notebook calibration để không vô tình trộn raw và tensor cache.
- Validation tại thời điểm ghi: focused suite `30 passed`; full repository suite `176 passed`, `32 warnings` PyTorch đã biết; notebook JSON/code cells compile thành công. Full train/validation chưa chạy do workspace chỉ có raw B0A cục bộ và không có full tensor cache; threshold V3 chưa được calibration và pair-residual chưa được huấn luyện.

## 2026-10-28: calibration detector click ở tốc độ khung gốc

- Thay đổi: `scripts/calibrate_ipn_temporal_factors.py` thêm `--variable-cache-root` để chạy detector trên cache `variable_v1` (giữ nguyên tốc độ khung, không resample về 64 frame) và `fullrate_grid()` (12 cấu hình, `temporal_max_gap`/`two_finger_start_delta` tính theo frame thật; dùng làm grid mặc định khi có cờ này). Thêm notebook `notebooks/IPN13_Click_Detector_FullRate_Colab.ipynb` chạy song song chế độ `t64` (đối chứng, grid 18 cấu hình cũ) và `fullrate`, in coverage/accuracy từng cặp, phân bố số click đếm được theo nhãn thật và kết luận theo tiêu chí đặt trước (coverage ≥ 0,50 và accuracy trên phần được phủ ≥ 0,75 cho cả G01/G08 và G02/G09).
- Lý do: detector click gần ngẫu nhiên (0,40–0,54 trên train) khi chạy trên tensor T64; tài liệu temporal factor đã nêu rủi ro là resample làm méo chu kỳ. Cần tách nguyên nhân "mất tín hiệu do resample" khỏi "tín hiệu closure không phân biệt 1 và 2 click" trước khi đầu tư vào pair-residual và fusion.
- Tương thích/an toàn: mặc định không đổi (không có cờ thì dùng tensor T64 và grid cũ). Script vẫn chỉ nhận `train.txt`/`val.txt`, không có đường đọc test. Chọn config chỉ bằng validation; ngưỡng chấp nhận đặt trước khi xem kết quả. Pair-residual và V3 vẫn tắt.
- Validation: `python -m unittest tests.test_calibrate_variable_cache` pass (đọc đúng cache full-length không cần keypoint JSON, grid có cùng khóa với grid cũ); notebook parse và mọi cell code qua `ast`. Chưa có số liệu accuracy vì cache đầy đủ chỉ có trên Drive; cần chạy notebook trên Colab (push nhánh trước).
