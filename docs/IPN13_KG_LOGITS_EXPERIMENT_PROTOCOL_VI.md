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

## Cập nhật 2026-10-27: V4 nhiều seed, so sánh cặp với checkpoint khởi tạo

### Vì sao
Mục tiêu của dự án là đầu symbolic (`kg_logits`) **hỗ trợ dự đoán** trong kiến trúc V4. Lần chạy một seed không phân biệt được tác dụng của đầu symbolic với dao động giữa các seed (macro-F1 validation của neural C có độ lệch chuẩn khoảng 2,7 điểm; seed 42 là seed tốt nhất). Phép so sánh đúng là **cặp theo seed**: V4 khởi tạo từ `c_ref_seed<seed>` được so với chính checkpoint khởi tạo đó, cùng clip, cùng seed.

### Thay đổi
- `notebooks/IPN13_KG_Logits_V4_MultiSeed_Colab.ipynb`: với mỗi seed (42, 123, 3407) khởi tạo V4 từ `outputs/ipn13_neural_sweep/c_ref_seed<seed>/best.pt` (cấu hình C: vận tốc, masked_mean), đóng băng encoder 3 epoch, augmentation như C, patience 12; đánh giá validation cho `neural`, `kg`, `final` của V4 và cho checkpoint khởi tạo; bảng Δ(final − init) theo seed, số clip KG sửa và hại, gate, F1 các lớp click và pinch; ensemble V4 (final) so với ensemble các checkpoint khởi tạo; test khóa mặc định (`RUN_LOCKED_TEST=False`, nếu bật là lần xem thứ N, exploratory); ZIP duy nhất, từ chối ghi đè; run dở dang đổi tên chứ không xóa. Kết luận in ra chỉ là "có dấu hiệu giúp" khi mọi seed đều tăng macro-F1 và mức tăng trung bình lớn hơn 0,005.
- `test_cross_attention_v2.py`: thêm `--save-logits` ghi `test_logits.pt` (logits thô của `final`, `neural`, `kg` và thứ tự clip). Mặc định tắt; không đổi đầu ra hiện có.
- `scripts/ensemble_v4_logits.py`: trung bình xác suất softmax của một nhánh qua nhiều checkpoint V4; từ chối khi các member đánh giá khác clip hoặc khác thứ tự.
- Notebook này không ghim `REPO_COMMIT` (cần code mới) và không so sánh mã dựng graph theo file (đầu nhánh có mã temporal factor mặc định tắt; đã kiểm tra graph dựng giống hệt từng byte với `e817f16` trên 40 clip thật). Nó chỉ dựng test graph khi bật test.

### Compatibility và an toàn
Chỉ thêm tùy chọn và script mới; checkpoint và đầu ra cũ không đổi. Train không dùng test; `best.pt` chọn theo macro-F1 validation; kiểm tra kiến trúc từng checkpoint khởi tạo trước khi nạp.

### Validation
`tests/test_v4_ensemble.py` (4 test) và bộ test đầy đủ; chạy thử các ô huấn luyện, đánh giá, bảng cặp, ensemble, bỏ qua test và đóng gói ZIP trên dữ liệu DATE nhỏ cục bộ (2 seed, kiến trúc nhỏ); `--save-logits` và ensemble chạy từ đầu đến cuối. Chưa chạy trên IPN thật (cần Colab).

## 2026-10-27: notebook đánh giá `--hand-features`

**Thay đổi.** Thêm `notebooks/IPN13_Hand_Features_Eval_Colab.ipynb`, bản rút gọn của notebook sweep chỉ gồm `c_ref` và `hand`. Dùng lại các run đã có `SWEEP_RUN_COMPLETE.json` trên Drive, đánh giá validation từng seed và ensemble, so sánh theo cặp từng seed (Δ macro-F1, Δ accuracy), in F1 trung bình các lớp G01/G02/G08/G09/G10/G11, và chỉ in "có dấu hiệu giúp" khi mọi seed tăng và trung bình lớn hơn 0,01 macro-F1.

**Lý do.** Checkpoint `hand` ba seed đã train nhưng chưa có đánh giá validation đầy đủ; số hiện có lấy từ `history.json` và nằm trong dao động seed.

**Tương thích và an toàn.** Không đổi code Python; test mặc định khóa (`RUN_LOCKED_TEST=False`), ZIP ra `runs/ipn13_hand_features/`, từ chối ghi đè.

**Validation.** Notebook sinh từ notebook sweep đã kiểm thử; mọi ô code biên dịch được. Chưa chạy trên Colab với dữ liệu IPN thật.

## 2026-10-28: nhãn phụ số lần click / số ngón (`--aux-heads`)

- Thay đổi: `NeuralOnlyClassifier` có tham số `aux_heads` (mặc định tắt) thêm hai đầu tuyến tính trên vector pooled; `forward(..., return_aux=True)` trả `(logits, {"count","fingers"})`. `train_neural_only.py` thêm `--aux-heads` và `--aux-weight` (mặc định 0,3): loss = CE(13 lớp) + w × (CE số lần click + CE số ngón), tính chỉ trên clip G01/G02/G08/G09 (số lần click: G01,G02 = 1, G08,G09 = 2; số ngón: G01,G08 = 1, G02,G09 = 2); lớp khác bị bỏ qua. Chỉ dùng được với IPN-13 (id cố định). Checkpoint lưu **không** có đầu phụ (`aux_heads=False`, thêm `trained_with_aux_heads`), nên `test_neural_only.py`, `ensemble_neural_only.py` và mọi loader cũ chạy nguyên. Thêm notebook `notebooks/IPN13_Aux_Heads_Eval_Colab.ipynb` (c_ref dùng lại từ Drive, `aux` 3 seed 42/123/3407, so theo cặp seed, F1 lớp click/pinch, tổng clip nhầm G01↔G08 và G02↔G09 qua 3 seed).
- Lý do: detector click bằng luật không đạt ở cả T64 lẫn tốc độ gốc; thử học "đếm" trực tiếp bằng cách chia sẻ dữ liệu giữa các lớp (khoảng 464 clip mỗi giá trị nhãn phụ thay vì 232 mỗi lớp) [Lập luận].
- Tương thích/an toàn: mặc định tắt, huấn luyện c_ref không đổi (đầu phụ khởi tạo sau classifier nên không đổi trọng số khởi tạo của phần còn lại). Chỉ validation; test khóa. Không thêm nhãn mới vào dữ liệu: nhãn phụ suy ra từ nhãn lớp.
- Validation: `python -m unittest tests.test_aux_heads tests.test_train_variable_length` pass 9/9 (ánh xạ nhãn, loss bỏ qua lớp khác, shape đầu phụ, state dict đã lọc nạp được vào model thường, cờ CLI). Chưa train trên dữ liệu thật. Tiêu chí kết luận: macro-F1 dương ở cả 3 seed và trung bình > 0,01; sd giữa seed khoảng 0,027 nên chênh nhỏ hơn không kết luận.

## 2026-10-29: baseline skeleton chạy lại trên dữ liệu của project

- Thay đổi: thêm backbone `lstm` (LSTM một chiều nhiều tầng, hidden bằng `d_model`, LayerNorm đầu ra) vào `NeuralOnlyClassifier`/`--backbone`; mặc định và các backbone khác không đổi. Thêm notebook `notebooks/IPN13_Skeleton_Baselines_Colab.ipynb` chạy 3 seed (42/123/3407) cho `lstm_paper` (cấu hình IPN HandS: 3 tầng, hidden 256, dropout 0,3, lr 1e-3, batch 64, 30 epoch, CE chuẩn, không augmentation), `lstm_c`, `bigru_c`, `tcn_c`, `mlp_mixer_c` (recipe C, đổi backbone) và so với `c_ref` (dùng lại run Transformer có sẵn). Báo cáo accuracy, macro-F1, average recall, số tham số, ensemble, hiệu theo cặp seed và F1 các lớp khó. Chỉ validation; test khóa cho lần so sánh cuối.
- Lý do: cần baseline skeleton so sánh được với bài. Số của các paper (AG-MAE, SkelMAE, ST-SGCN, ASCA-STGCN, MediaPipe+RNN, IPN HandS) dùng protocol, số lớp, nguồn keypoint và số mẫu khác nên không so trực tiếp; chạy lại các kiến trúc đơn giản trên cùng dữ liệu và split của project cho so sánh công bằng.
- Tương thích/an toàn: checkpoint cũ không đổi; `backbone=lstm` chỉ chạy với 52 keypoint, 64 frame resample, đầu vào 52 điểm thay vì 21 điểm một tay như paper; pooling `masked_mean` thay vì trạng thái ẩn cuối. `lstm_paper` giữ siêu tham số của paper nhưng không tái tạo đầu vào 21 điểm một tay (chưa làm). Chưa dò siêu tham số riêng cho từng baseline.
- Validation: `python -m unittest tests.test_lstm_backbone tests.test_aux_heads` pass; kiểm tra forward của lstm/bigru/tcn/mlp_mixer/transformer. Chưa train trên dữ liệu thật.

### Kết quả `--aux-heads` (validation, 3 seed 42/123/3407, 2026-10-29)

- Hiệu aux − c_ref theo seed: macro-F1 −0,0406 / −0,0016 / −0,0125 (trung bình −0,0182), accuracy −0,0204 / −0,0102 / −0,0068. Âm ở cả 3 seed nhưng độ lớn nằm trong dao động seed (sd macro-F1 khoảng 0,027) [Đã đo]: không có bằng chứng giúp; chưa đủ để nói hại chắc chắn.
- F1 lớp khó (c_ref → aux): G01 0,592→0,498, G02 0,608→0,593, G08 0,664→0,630, G09 0,685→0,671, G10 0,762→0,750, G11 0,788→0,767. Số clip nhầm cộng 3 seed: G01↔G08 45→49, G02↔G09 35→40, tức không giảm ở cả hai cặp.
- Kết luận: giả thuyết "chia sẻ dữ liệu qua nhãn phụ số lần click/số ngón giúp đếm" không được ủng hộ. Mặc định giữ `--aux-heads` tắt.
- Tổng hợp các thử nghiệm riêng cho click đến nay, đều không cải thiện cặp G01↔G08 và G02↔G09: đặc trưng hình học bàn tay (`--hand-features`), chuỗi biến độ dài 256 frame, detector click bằng luật ở tốc độ gốc, nhãn phụ số lần click. Gợi ý rằng giới hạn nằm ở thông tin trong keypoint 2D hoặc ở nhãn/mốc thời gian, không phải ở thủ thuật mô hình [Lập luận]. Paper IPN HandS nêu mốc thời gian gốc của IPN thường cắt không đủ chuyển động ở các cử chỉ ngắn, tinh tế, nên nhiễu nhãn là giả thuyết cần kiểm tra bằng xem lại các clip bị nhầm.

### Phân tích lỗi click trên validation bằng predictions của c_ref 3 seed (2026-10-29)

Nguồn: `val_eval/predictions.csv` của `c_ref_seed42/123/3407` (trong `results/experiments/ipn13/neural_sweep_upload/`), 112 clip G01/G02/G08/G09 (28 mỗi lớp). Độ dài clip lấy từ khoảng frame trong tên file. Không có thay đổi code.

- Có 26/112 clip mà cả 3 seed đều sai. Chia theo lớp: G01 9/28, G02 10/28, G08 3/28, G09 4/28 [Đã đo]. Hầu hết bị đoán thành lớp "hai lần" tương ứng (G01→G08 6/9, G02→G09 6/10).
- Độ dài trung bình (frame): G01 luôn đúng 34, luôn sai 61; G02 luôn đúng 39, luôn sai 65; G08 luôn đúng 56, luôn sai 80; G09 luôn đúng 53, luôn sai 88 [Đã đo]. Clip một-click dài bị đoán thành hai-click, clip hai-click dài khác thường bị đoán sai theo hướng khác.
- Phân bố độ dài của hai lớp chồng lấn nhiều: G01 p25/p50/p75 = 27/43/51 so với G08 43/56/70; G02 33/46/63 so với G09 43/52/67 [Đã đo].
- Chỉ dùng độ dài clip với ngưỡng chọn tốt nhất trên chính validation (cận trên lạc quan) đạt 0,661 ở cả G01/G08 và G02/G09; neural c_ref đúng nhãn trên cùng 56 clip mỗi cặp ở mức 0,619 và 0,637 (trung bình 3 seed) [Đã đo]. Gợi ý (chưa chứng minh) rằng c_ref chủ yếu dùng độ dài như dấu hiệu "đếm click", không đọc được số lần bấm.
- Theo camera, recall trung bình của clip một-click thấp ở 4CM11 và 1CM42 (G01: 0,29 và 0,50; G02: 0,42 và 0,44) so với 1CV12 (0,75 và 0,83) [Đã đo; n = 8–12 clip mỗi ô].
- Chưa xem được nội dung video; nhãn đúng/sai của từng clip chưa kiểm chứng bằng mắt.

- Quan sát xem tay trên video (người dùng, 2026-10-29): ở các clip G01/G02 bị đoán thành hai-click, người thực hiện giơ tay rồi gập tay xuống ngay trong khoảng clip, và chuyển động gập xuống này giống động tác click. Nghĩa là mốc thời gian của clip gồm cả pha chuẩn bị/thu tay, không chỉ pha click, nên clip dài hơn và chứa chuyển động giống click thừa. Phù hợp với nhận xét của paper IPN HandS về mốc thời gian gốc của IPN [Lập luận; chưa đếm tỉ lệ trên toàn bộ 26 clip luôn sai].

- Kiểm tra mốc cắt (2026-10-29, không đổi code): cả 4.133 dòng `external_data/ipn_processed/manifest.csv` có `(video, t_start, t_end)` trùng chính xác một khoảng trong `annotations/Annot_List.txt` gốc của IPN (0 dòng lệch) và `frames = t_end − t_start + 1` ở mọi dòng [Đã đo]. `scripts/cut_ipn_clips.py` giải mã tuần tự đến frame bắt đầu (không seek theo keyframe) và kiểm tra số frame ghi/giải mã bằng số frame mong đợi, nên khoảng cắt khớp annotation gốc. Do đó phần chuyển động thừa (giơ tay, gập tay) nằm trong chính mốc của IPN chứ không do bước cắt của project [Lập luận từ kiểm tra trên và quan sát video của người dùng].

- Kiểm tra một clip cụ thể (2026-10-29, không đổi code): `val/G01/4CM11_14/4CM11_14_R_24_G01_f002201_f002291` (91 frame; G01 trung vị toàn bộ annotation là 48 frame). So pixel clip với video gốc `4CM11_14_R_#24.avi`: frame đầu của clip khớp frame 2201 (1-based), frame cuối khớp frame 2291 (sai số trung bình khoảng 1,2 mức xám do nén lại MJPG), đủ 91 frame giải mã [Đã đo], tức bước cắt trung thực với annotation. Xem 12 frame trải đều: những frame đầu (2201–2209) là tư thế chỉ một ngón (giống B0A), tiếp theo có chuyển động hạ tay/ngón xuống, ngón giơ lại (2241–2258), một chuyển động xuống nữa (2266–2274), rồi hai ngón giơ (2282, giống B0B) [quan sát bằng mắt từ một contact sheet, chưa kiểm hệ thống]. Annotation gốc liền kề: B0A 2039–2200, G01 2201–2291, B0B 2292–2467. Cả 3 seed c_ref đoán clip này là B0A. Clip có nhãn G01 nhưng chứa cả đoạn đầu giống B0A, nhiều hơn một chuyển động xuống, và đoạn chuyển sang hai ngón [Lập luận: nguồn gốc nhiễu nằm ở mốc annotation].

### Kiểm tra chuyển động của mọi clip click (2026-10-29)

Thêm `scripts/analyze_click_clip_motion.py`: với mọi clip G01/G02/G08/G09 trong `external_data/ipn_processed/clips` (train, val, test; 784 clip, 196 mỗi lớp) đo năng lượng chuyển động toàn khung theo thời gian bằng hiệu hai frame liên tiếp (ảnh xám 160×120, làm mượt 5 frame), rồi tính: số frame đứng yên ở đầu/đuôi, tỉ lệ frame hoạt động (> 30% đỉnh) và số đợt chuyển động (đỉnh ≥ 40% đỉnh lớn nhất, cách nhau ≥ 7 frame). Chỉ là phân tích mô tả chất lượng dữ liệu, không dùng để huấn luyện hay chọn mô hình. Đây là proxy thô (chuyển động cả thân cũng được tính), không phải detector click.

- Số đợt chuyển động trung bình: G01 3,56 so với G08 4,68; G02 3,58 so với G09 4,79. Tỉ lệ clip có ≥ 3 đợt: G01 0,68, G08 0,92, G02 0,71, G09 0,95 [Đã đo]. Hầu hết clip một-click đã có từ 3 đợt trở lên, nên số đợt không phân biệt rõ một và hai click.
- Validation: 26 clip luôn sai (3 seed) so với 86 clip còn lại: độ dài 69,0 so với 51,5 frame; số frame đứng yên ở đầu 4,15 so với 0,83; tỉ lệ frame hoạt động 0,63 so với 0,81 [Đã đo]. Clip G01 luôn sai có 4,22 đợt so với 2,89 ở clip còn lại (n = 9 và 19).
- 108/392 clip một-click (27,6%) dài ≥ 90 frame hoặc có ≥ 5 đợt chuyển động (train 67, val 12, test 29); ngưỡng tùy ý, chỉ để định cỡ [Đã đo].
- Giới hạn: chưa xem bằng mắt từng clip; chỉ số trên là proxy từ khác biệt pixel.

## 2026-11-02: notebook trích RTMW cho subject 1CM42_3

- Thay đổi: thêm `notebooks/IPN_RTMW_Extract_1CM42_3_Colab.ipynb`, trích RTMW cho 85 clip của subject `1CM42_3` bằng đúng `run_rtmw_splits_dual.py`, đúng checkpoint `rtmw-dw-x-l_simcc-cocktail14_270e-384x288` và cờ mặc định (mirror-swap) như lần trích gốc. Đầu vào là `clips_1CM42_3.zip` (tạo bằng `create_ipn_manifest.py --include-all-subjects` rồi `cut_ipn_clips.py` cho 85 dòng của `1CM42_3`, 12.702 frame). Kết quả ghi vào thư mục tạm `ipn-extract-data/keypoints_1CM42_3_staging/` trên Drive, không đụng `keypoints/`; có cell thống kê vai (điểm tin cậy, tỉ lệ frame đúng một vai < 0,3, độ rộng vai theo pixel) đối chiếu với subject `1CM42_4`; cell ghép vào `keypoints/` mặc định tắt (`MERGE = False`).
- Lý do: kiểm tra trên Drive cho thấy không có keypoint nào của `1CM42_3` (manifest gốc loại subject này), nên bộ IPN gốc 13 lớp đang thiếu 85 clip (4.133 so với 4.218 instance trong paper). Cần keypoint RTMW của subject này để so sánh với các baseline dùng đủ 4.218 clip, và để quyết định hạ ngưỡng điểm vai hay dùng bộ ước lượng vai.
- Tương thích/an toàn: không sửa code hay file hiện có; val/test không đổi (đã kiểm tra: subject validation và test giống hệt khi bỏ loại trừ, train thêm 85 clip: 2.443 → 2.528). Notebook từ chối ghi đè staging nếu đã tồn tại.
- Validation: notebook parse hợp lệ, cell code Python qua `ast`; chưa chạy trên Colab (cần GPU và file zip trên Drive).
