# Protocol đánh giá `kg_logits` trên IPN-13

Ngày cập nhật: 2026-10-23

## Mục tiêu

Thí nghiệm này mở rộng bài báo hiện tại sang IPN-13. Notebook chính là
`notebooks/IPN13_KG_Logits_Baseline_Comparison_Colab.ipynb`. Nó huấn luyện và
đánh giá mô hình V4 có ba đầu ra `neural_logits`, `kg_logits` và `logits`
(fusion cuối), rồi so sánh với một checkpoint neural-only độc lập trên cùng
split IPN.

`kg_logits` là learned template-guided head: nó dùng cả neural memory của clip
và graph memory của instance graph để truy vấn template. Vì vậy không được gọi
nó là pure symbolic hoặc đồng nhất với điểm của
`scripts/eval_ipn_template_matching.py`.

## Dữ liệu và artifact đầu vào

- Split cố định: 2.443 train, 589 validation và 1.101 test, tổng cộng 13 lớp.
- ZIP `c46e0e469d2cf3b6-20261023T141655Z-1-001.zip` là artifact symbolic IPN
  train/validation: 3.032 instance graph, 13 template và provenance ở commit
  `e817f16e0d504e95064b889288f06231bcf692ab`. ZIP này không chứa checkpoint
  V4 và không chứa test graph.
- Cần cung cấp checkpoint neural-only IPN 13 lớp, toàn bộ keypoint/cache và ba
  manifest trên Google Drive. Notebook đọc metadata checkpoint để cấu hình V4;
  không hardcode kiến trúc 128/3/4.
- Candidate neural baseline mạnh nhất hiện có theo validation là ablation C
  (`d_model=192`, 4 layer, 8 head, `masked_mean`), validation accuracy
  0,891341 và macro-F1 0,837578. Đây chưa phải kết quả locked test.

## Protocol chống leakage

1. Kiểm tra commit, hash, template IDs, split counts và cấu hình graph của
   artifact train/validation.
2. Chỉ dựng 1.101 test graph còn thiếu bằng đúng builder/ngưỡng ghi trong
   artifact. Thiếu keypoint, cache hoặc graph phải dừng; không chấp nhận graph
   rỗng do fallback của dataset loader.
3. Huấn luyện V4 bằng train/validation, khởi tạo neural branch từ neural-only
   checkpoint. Không truyền `--test-file` và không dùng `--save-best-test`.
4. Chọn `best.pt` theo validation macro-F1.
5. Giữ `RUN_LOCKED_TEST=False` trong lúc phát triển. Khi cấu hình đã đóng băng,
   bật gate một lần: evaluator V4 xuất đồng thời neural/KG/final; evaluator
   neural-only chạy độc lập một lần trên cùng test manifest.
6. Báo cáo accuracy, macro-F1, per-class F1, confusion matrix, hash checkpoint,
   hash manifest, commit và toàn bộ argv. Không chọn checkpoint hoặc sửa
   template/hyperparameter dựa trên test.

## Cách so sánh với baseline

Baseline trực tiếp và công bằng là neural-only trên cùng IPN split. Bảng chính
nên có ít nhất: neural-only độc lập, V4 neural branch, learned `kg_logits`, và
V4 final fusion. Kết quả DATE-27 trong bản thảo hiện tại thuộc dataset/split
khác nên chỉ dùng làm bối cảnh của phương pháp, không dùng để tuyên bố cải thiện
định lượng trên IPN.

Hiện chưa có số locked-test cho learned IPN `kg_logits` trong workspace: không
có checkpoint V4 IPN và artifact c46 chưa có test graph. Không được lấy metric
rule matcher trong các ZIP symbolic để điền thay.

## Output và an toàn tương thích

Mỗi lần chạy tạo một thư mục và ZIP duy nhất dưới
`MyDrive/ipn-extract-data/runs/ipn13_kg_logits/<run_id>`. Gói kết quả chứa
checkpoint V4 validation-selected, metric/prediction/per-class/confusion nếu
đã mở locked test, hash, config và command provenance; không nhúng raw keypoint,
graph cache hoặc output notebook. Tên run có timestamp và UUID, và notebook từ
chối overwrite.

Thay đổi này chỉ thêm notebook/protocol đánh giá, không đổi format dữ liệu,
model runtime hoặc checkpoint cũ. Notebook có chế độ `EVAL_ONLY` để đánh giá
một checkpoint V4 IPN hợp lệ mà không huấn luyện lại.

## Validation đã thực hiện

- Notebook là JSON hợp lệ, mọi code cell parse được bằng Python AST và không có
  output nhúng.
- Các cờ CLI được đối chiếu với `train_cross_attention_v4.py`,
  `test_cross_attention_v2.py`, `test_neural_only.py` và
  `precompute_instance_graphs_phase_v2.py`.
- Metadata artifact c46 được kiểm tra trực tiếp: train 2.443, validation 589,
  `run_test=false`, builder `phase_v4_action_role_audit` và thuật toán
  `directional_reversal_v2`.


## Cập nhật 2026-10-23: notebook chạy full test một lần

- **Thay đổi:** `RUN_LOCKED_TEST` mặc định `True` theo yêu cầu chạy full test lấy kết quả báo cáo; đặt `False` nếu chỉ muốn train/validation. Khi `IPN_NEURAL_BASELINE_CHECKPOINT=None`, notebook tự dò trên Drive các checkpoint neural-only 13 lớp tương thích V4, in danh sách và chỉ tiếp tục nếu đúng một ứng viên; ngược lại dừng và yêu cầu gán thủ công.
- **Lý do:** các ZIP IPN trong repo chỉ chứa checkpoint ablation có validation accuracy 0,859–0,876, không chứa ablation C (0,891) nên checkpoint phải lấy từ Drive.
- **An toàn:** train vẫn không dùng test; `best.pt` chọn theo validation macro-F1; test chỉ chạy một lần với cấu hình đã khóa. Không thay đổi code runtime hay format dữ liệu.
- **Validation:** notebook parse JSON hợp lệ, mọi code cell qua `ast.parse`, không có output nhúng.

## Cập nhật 2026-10-23: gán cố định checkpoint neural C

- **Thay đổi:** cell cấu hình notebook gán `IPN_NEURAL_BASELINE_CHECKPOINT` tới `MyDrive/ipn-extract-data/outputs/neural_ablation_C_masked_mean_light_dropout/transformer_seed42/best.pt`.
- **Lý do:** thư mục C trên Drive còn chứa `bigru_*` và `transformer_seed42_incomplete_*`; bước tự dò sẽ thấy nhiều ứng viên tương thích và dừng. Run hoàn chỉnh khớp ZIP local (validation accuracy 0,891341, macro-F1 0,837578).
- **An toàn:** chỉ đổi giá trị cấu hình; các kiểm tra kiến trúc/13 lớp trong notebook vẫn chạy. Validation: cell parse bằng `ast.parse`.

## Cập nhật 2026-10-23: sửa lỗi velocity khi dùng neural C

- **Phát hiện:** `command.json` của ablation C không có `--disable-motion-features`, nên C được train **có velocity** (input `52*5`). V4 adapter (`cross_attention_v2.py`) chỉ có projection `52*3`; notebook trước đó assert "không có motion" nên sẽ dừng ngay khi nạp C.
- **Thay đổi:** thêm bước 4b train một checkpoint neural **cùng siêu tham số C nhưng `--disable-motion-features`** (chỉ train/validation, chọn theo validation macro-F1, lưu tại `outputs/ipn13_v4_init_no_motion/transformer_seed42/best.pt`, không overwrite) để khởi tạo V4. C vẫn là neural-only baseline độc lập. Locked test đánh giá cả hai neural-only (C có velocity và no-motion) cùng V4 neural/KG/final; manifest Drive gốc được dùng cho evaluator neural-only như lúc train.
- **Lý do:** tránh sửa kiến trúc V4 ngay trước báo cáo; so sánh vẫn công bằng vì có baseline no-motion cùng cấu hình.
- **An toàn:** train không dùng test; không đổi code runtime hay checkpoint cũ.
- **Validation:** mọi code cell qua `ast.parse`, không có output nhúng; full suite `123 passed`. Chưa chạy end-to-end trên Colab.

## Cập nhật 2026-10-23: nhận artifact c46 dạng thư mục Drive

- **Thay đổi:** `TRAIN_VAL_ARTIFACT_ZIP` mặc định trỏ tới thư mục `MyDrive/ipn-extract-data/runs/ipn_symbolic_v2_post_c/train_val/c46e0e469d2cf3b6` (cách artifact thực sự nằm trên Drive); vẫn chấp nhận file `.zip`. Thư mục được copy về `/content` trước khi dùng vì graph test được ghi thêm vào đó. Hash đóng gói ghi đường dẫn nguồn và SHA-256 (của ZIP, hoặc của `run_config.json` khi là thư mục).
- **Lý do:** artifact trên Drive là thư mục, không phải ZIP; bản trước sẽ không tìm thấy file.
- **An toàn:** mọi hard gate (`completed_run.json`, hash `run_config`, số graph 2.443/589, commit, không có test graph hoặc `best.pt`) giữ nguyên; không overwrite thư mục đích.
- **Validation:** mọi code cell qua `ast.parse`; chưa chạy end-to-end trên Colab.

## Cập nhật 2026-10-24: V4 train với augmentation của neural C

- **Thay đổi:** lệnh train V4 trong notebook thêm `--frame-mask-probability 0` và `--keypoint-dropout-probability 0.02`.
- **Lý do:** mặc định của V4 là 0,05 và 0,03 (cấu hình B). Thí nghiệm A/B/C cho thấy bỏ frame masking giúp neural C hơn B khoảng 3,2 điểm accuracy ở các lớp click. Để V4 không bị bất lợi riêng và so sánh với neural C công bằng hơn, V4 dùng cùng augmentation với C. Instance graph vẫn dựng từ keypoint gốc nên augmentation chỉ tác động phía neural.
- **Compatibility và an toàn:** chỉ thêm hai tham số CLI đã có sẵn; không đổi code model hay format dữ liệu. Lần chạy DATE-27 đã đóng gói dùng mặc định cũ (0,05 và 0,03), nên không so trực tiếp cấu hình huấn luyện giữa hai bộ dữ liệu.
- **Validation:** các code cell parse được bằng `ast.parse`, không có output nhúng. Bản trên Colab được người chạy sửa tay thêm đúng hai dòng này trước khi chạy; hai cờ sẽ có trong `commands.json` của ZIP kết quả.

## Cập nhật 2026-10-24: kết quả run đầu, audit và code cải thiện

### Kết quả run `20261024T075203Z-2c6b5c8d30` (commit `e817f16`, test 1.101 clip, một seed)

| Mô hình | Accuracy | Macro-F1 |
|---|---:|---:|
| Neural-only C (có vận tốc) | 84,65% | 77,24% |
| Neural-only không vận tốc (khởi tạo V4) | 82,56% | 73,65% |
| V4 neural | 83,02% | 74,40% |
| V4 `kg_logits` | 82,29% | 73,22% |
| V4 final | 82,56% | 73,84% |

V4 final bằng đúng checkpoint khởi tạo và thấp hơn C; `kg_logits` sửa 17 lỗi, gây 25 lỗi mới (net −8). Hai nguyên nhân thiết kế của run này: adapter V4 chỉ nhận 3 đặc trưng mỗi điểm (không vận tốc) và dùng mean pooling cho trọng số train với masked_mean. Phân tích đầy đủ: `AUDIT_IPN13_KET_QUA_VA_CAI_THIEN_2026-10-24_VI.md`. Notebook cũ `IPN13_KG_Logits_Baseline_Comparison_Colab.ipynb` được giữ nguyên làm bản ghi của run này.

### Thay đổi code (tùy chọn, mặc định tắt)

- `NeuralBaselineAdapter` (V4) nhận `use_motion_features`, `pooling`, `hand_features`; `DualCrossAttentionNeuroSymbolicModel` và `DualCrossAttentionV4NeuroSymbolicModel` truyền tiếp (`neural_pooling`); `train_cross_attention_v4.py` thêm `--neural-motion-features`, `--neural-pooling`, `--neural-hand-features`; `test_cross_attention_v2.py` đọc lại từ checkpoint (checkpoint cũ không có các khóa này nên dựng lại adapter 3 đặc trưng, mean pooling như trước).
- `neural_only_experiments`: `hand_geometry.py` (đặc trưng hình học bàn tay, bản sao ở `src_neurosymbolic/models/`), `--hand-features`, `--time-crop-min`, `--scale-jitter`, `--shift-std` trong `train_neural_only.py`; `ensemble_neural_only.py` (trung bình xác suất nhiều seed); `test_neural_only.py` đọc `hand_features` từ checkpoint.
- `load_checkpoint_compat` đọc được checkpoint Colab trên Windows (alias `PosixPath` tạm thời, có khôi phục).
- Thêm `tests/test_ipn_improvements.py` (30 test): hình học (đếm click từ tín hiệu flexion, bất biến theo scale, khớp lệch khi thiếu keypoint), augmentation (chỉ sao chép frame thật, không động vào điểm thiếu, mặc định giống hệt cũ), adapter V4 khớp `NeuralOnlyClassifier` với 12 tổ hợp, từ chối checkpoint lệch kiến trúc, ensemble.

### Notebook mới

- `notebooks/IPN13_Neural_Improvement_Sweep_Colab.ipynb`: 4 biến thể (`c_ref`, `hand`, `aug`, `hand_aug`) × 3 seed, chọn theo macro-F1 validation trung bình; ensemble từng biến thể; `RUN_LOCKED_TEST=False` mặc định (test đã bị mở, nếu bật là lần xem thứ hai, exploratory); ZIP duy nhất, từ chối ghi đè. Đã chạy thử toàn bộ các ô trên dữ liệu DATE nhỏ cục bộ.
- `notebooks/IPN13_KG_Logits_V4_FromC_Colab.ipynb`: V4 khởi tạo trực tiếp từ C (vận tốc, masked_mean suy ra từ metadata), `--freeze-neural-epochs 3`, patience 12, scheduler patience 4, kiểm tra code dựng graph không đổi so với `e817f16`; test khóa mặc định.
- Cả hai notebook clone nhánh `HuyenHoang` (cần commit và push code mới trước) và ghi lại hash đã dùng.

### Compatibility và an toàn

- Mặc định mọi tùy chọn đều tắt; kiểm tra: 128/128 clip DATE-27 đầu tiên cho dự đoán (final, neural, KG) giống hệt lần chạy đã đóng gói; checkpoint C thật nạp strict vào V4 và neural head tái hiện đúng logits của C.
- Hiệu quả của `--hand-features`, augmentation và V4-từ-C trên IPN thật chưa đo (không có GPU và keypoint IPN đầy đủ ở máy cục bộ).
- Đính chính: bản đầu của tài liệu giải thích mô tả nhầm công thức trộn của V4 (lớp V2 cũ). V4 dùng `ConfidenceLogitFusion` với gate là trọng số của neural, khởi tạo 0,8; đã sửa trong `GIAI_THICH_THAY_DOI_VA_THU_NGHIEM_VI.md` và báo cáo.

### Validation

`pytest`: 153 passed; smoke huấn luyện/đánh giá neural-only, huấn luyện V4 (có `--freeze-neural-epochs`) và đánh giá V4 với các cờ mới trên dữ liệu DATE nhỏ; các code cell của hai notebook qua `ast.parse`; notebook sweep chạy end-to-end cục bộ.
