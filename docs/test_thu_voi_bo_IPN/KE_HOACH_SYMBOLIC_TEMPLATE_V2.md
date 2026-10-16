# Kế hoạch cải thiện symbolic template graph V2

Ngày audit: 15/10/2026. Nhánh triển khai: `codex/symbolic-template-v2`, xuất phát từ `nguoi3/ipn-template-v4`.

## Mục tiêu và nguyên tắc đánh giá

Mục tiêu là làm tầng symbolic phân biệt được các nhãn dễ nhầm bằng tín hiệu có ý nghĩa: click một/hai lần, click một/hai ngón, hai chu kỳ đóng-mở và pinch tăng/giảm. Không dùng test để sửa template. Chỉ kết luận symbolic có giá trị khi full model thắng control cùng recipe/capacity qua nhiều seed và ablation shuffled/empty template làm kết quả giảm.

## Kế hoạch thực hiện

| Pha | Công việc | Tiêu chí nghiệm thu | Trạng thái |
|---|---|---|---|
| P0 | Khóa split, label, exclusion, hash và coverage | `2443/589/1101`, ID `0..12`, subject-disjoint, loại `1CM42_3`, đủ 13 template, 100% graph | Đã có validator fail-fast; artifact hiện thiếu 1.101 test graph và cả ba split hash đều sai |
| P1 | Sửa schema và provenance | YAML chuẩn, weight thực sự có hiệu lực, graph ghi predicate config, không bỏ qua graph stale/lỗi | Đã triển khai và có unit test |
| P2 | Predicate V2 | Tách threshold hand/finger/pinch; middle-finger relative motion; pinch chuẩn hóa theo palm scale; không suppress finger khi tay di chuyển | Đã triển khai; cần regenerate toàn bộ graph |
| P3 | Matcher temporal | Hỗ trợ `min/max_occurrences`, event-local facts và thứ tự phase | Đã triển khai; G01/G08, G02/G09, G07 có template count/order |
| P4 | Rule-only gate | Macro-F1 val tối thiểu `0,25`; không thiếu graph; tie rate và top-2 margin được lưu | Smoke 26 mẫu đã chạy và thất bại (`0,0000` val macro-F1); full 589 clip chưa chạy được vì thiếu raw keypoint/cache đầy đủ |
| P5 | Causal ablation | Ít nhất 5 seed; full hơn strongest non-symbolic control ≥ 0,01 macro-F1 và paired CI 95% > 0; shuffled/empty giảm | Chỉ chạy sau khi qua P4 |

## Thay đổi kỹ thuật V2

- `gesture_templates.yaml`: weight schema rõ ràng; G01/G02 giới hạn một occurrence, G08/G09 yêu cầu hai; G07 là `closed → open → closed → open`; G10/G11 dùng pinch-distance trái chiều.
- Template graph dùng concept node riêng theo event và cạnh `BEFORE`, tránh concept dùng chung làm mất ordinal/requirement.
- Matcher chấm requirement theo từng event, đếm distinct atomic occurrence trên cùng subject và xét chuỗi cùng subject; phase split của một atomic event không bị tính thành nhiều click. Template schema cũ vẫn giữ cách chấm cũ.
- Predicate ghi `ipn_predicate_v2` cùng toàn bộ threshold vào instance graph. Precompute mặc định fail khi có lỗi hoặc gặp graph stale; chỉ cho phép bỏ qua lỗi bằng cờ chủ động.
- Evaluator mặc định từ chối cohort thiếu graph, lưu split SHA-256, số tie, tie rate và top-2 margin.

## Kết quả kiểm tra hiện tại

Trên 589 instance graph **cũ** trong package Người 3:

| Cấu hình | Accuracy | Macro-F1 | Tie rate | Ý nghĩa |
|---|---:|---:|---:|---|
| Template cũ + graph cũ | 0,2139 | 0,0595 | 0,8981 | Baseline artifact tái chạy được, nhưng khác report lịch sử |
| Template V2 + graph cũ | 0,0407 | 0,0406 | 0,0306 | Chỉ là compatibility check (`--allow-legacy-graphs`); graph cũ thiếu predicate/config V2 |

Tie giảm cho thấy count/order tạo score phân biệt hơn, nhưng accuracy V2 trên graph cũ không dùng để đánh giá chất lượng. Chạy hợp lệ đòi hỏi regenerate graph từ raw keypoint hoặc tensor cache đầy đủ.

Smoke bổ sung ngày 16/10/2026 tái tạo toàn bộ predicate/event/graph V2 từ một mẫu train và một mẫu validation cho mỗi lớp (26 clip). Kết quả train accuracy `0,0769`, macro-F1 `0,0256`; validation accuracy và macro-F1 đều `0`. Visual review xác nhận template G10/G11 đúng chiều và G05/G06 phải đổi theo tọa độ ảnh. Số predicate/event rất lớn và nhiều chiều đối nghịch cùng xuất hiện trong một clip, vì vậy bước tiếp theo là sửa chọn active phase/dominant hand ở predicate-event, không tiếp tục tune YAML theo 26 mẫu nhỏ.

## Lệnh chạy tiếp khi có đủ dữ liệu

Nếu keypoint JSON đang ở Google Drive, dùng notebook symbolic-only [IPN_Symbolic_V2_Colab.ipynb](../../notebooks/IPN_Symbolic_V2_Colab.ipynb). Notebook copy các split đang chạy sang SSD Colab, hỗ trợ resume graph từ Drive, mặc định chỉ chấm train/validation và khóa test bằng `RUN_TEST=False`.

Smoke cục bộ hiện có thể chạy ngay, không cần checkpoint:

```powershell
python scripts/eval_ipn_symbolic_examples_v2.py `
  --output final/outputs/ipn_symbolic_v2_smoke/report.json `
  --artifact-dir final/outputs/ipn_symbolic_v2_smoke/artifacts `
  --min-val-macro-f1 0.25
```

Exit code `2` nghĩa là chưa qua gate; đây là trạng thái hiện tại. Khi có đủ dữ liệu, chạy full cohort như sau:

```powershell
$P = "external_data/ipn_processed"
$env:PYTHONPATH = "final/code"

python final/code/src_neurosymbolic/kg/build_template_graph.py `
  --templates configs/ipn/gesture_templates.yaml `
  --list-file $P/train.txt --list-file $P/val.txt --list-file $P/test.txt `
  --output-dir $P/template_graphs_v2 --pretty

python final/code/src_neurosymbolic/precompute_instance_graphs_phase_v2.py `
  --keypoint-root $P/keypoints --cache-root $P/keypoint_tensor_cache `
  --output-root $P/instance_graphs_v2 --overwrite `
  --train-file $P/train.txt --val-file $P/val.txt --test-file $P/test.txt `
  --splits train val test

python scripts/validate_symbolic_inputs.py `
  --data-root $P --template-dir $P/template_graphs_v2 `
  --instance-graph-root $P/instance_graphs_v2 --require-hashes

python scripts/eval_ipn_template_matching.py `
  --data-root $P --split val --template-dir $P/template_graphs_v2 `
  --instance-graph-root $P/instance_graphs_v2 `
  --output final/outputs/ipn_symbolic_v2/rule_val.json
```

Nếu rule-only chưa đạt gate P4, dừng ở audit predicate/event theo train rồi sửa extractor/template; không train V4 để tránh lấy capacity neural che chất lượng symbolic.
