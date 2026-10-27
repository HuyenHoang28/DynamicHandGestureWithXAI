# Báo cáo tiến độ — Neuro-symbolic nhận dạng cử chỉ động trên IPN Hand

Ngày báo cáo: 2026-10-24. Phạm vi: công việc từ 2026-10-06 đến 2026-10-24.

Tài liệu này tổng hợp từ các note đã có trong `docs/` (nguồn ở Phụ lục A). Mọi con số được đối chiếu với file gốc; số nào chưa có được ghi rõ là chưa có. Kết quả test learned `kg_logits` trên IPN-13 (run 2026-10-24) đã có ở mục 8; phân tích lỗi và kế hoạch cải thiện nằm ở `AUDIT_IPN13_KET_QUA_VA_CAI_THIEN_2026-10-24_VI.md`.

## 1. Tóm tắt

1. **Neural baseline IPN-13:** cấu hình C (Transformer d192, 4 lớp, 8 head, masked-mean pooling, vận tốc keypoint, không frame masking) đạt validation accuracy **89,13%** và macro-F1 **83,76%** (589 clip). Bốn lớp click G01, G02, G08, G09 vẫn là điểm yếu (F1 0,63–0,70).
2. **Symbolic thuần luật trên IPN** chưa đủ làm classifier: accuracy thô 13 lớp khoảng **23%** (macro-F1 9%). Quy trình audit phát hiện và gỡ một regression (role cứng); override theo cặp lớp với neural C chưa có cấu hình nào có lợi nên được giữ tắt.
3. **DATE-27 (1.229 clip test):** checkpoint V4 chọn theo validation đạt Final **94,87%** (macro-F1 94,52%), KG head 95,04%, neural head 94,63%; neural-only độc lập 95,28%. Final cao hơn ProtoGCN (93,28%) và DATE Transformer (94,22%) trong paper, nhưng thấp hơn mean Proposed Fusion của paper (96,40%) và chưa vượt neural-only.
4. **Cảnh báo về paper:** run 3 của cột Proposed (96,18%) trùng đúng accuracy của `best_test.pt` (checkpoint chọn theo test). Cần xác minh provenance 10 run trước khi dùng 96,40% làm claim tái lập.
5. **IPN-13 test (run 2026-10-24, 1.101 clip):** neural-only C đạt **84,65%** accuracy / **77,24%** macro-F1; V4 final chỉ đạt **82,56% / 73,84%**, bằng đúng checkpoint khởi tạo của nó (không vận tốc) và `kg_logits` có net −8 clip. Chưa có bằng chứng nhánh symbolic giúp trên IPN.
6. **Phát hiện chính từ audit:** vận tốc đóng góp +2,1 điểm accuracy và +3,6 điểm macro-F1 (recall G08 từ 36,5% lên 71,2%), nhưng V4 không dùng được vận tốc nên bị mất lợi thế này; lỗi còn lại tập trung ở các cặp đếm click và chênh lệch giữa camera (accuracy 77% đến 90%).
7. **Đã triển khai (chưa chạy trên Colab):** V4 nhận vận tốc, đặc trưng hình học bàn tay, augmentation thời gian và hình học, ensemble nhiều seed, notebook sweep chọn theo validation. 153 test pass; tương thích ngược DATE-27 đã kiểm chứng.

## 2. Bối cảnh và dữ liệu

| Bộ dữ liệu | Số lớp | Train | Validation | Test | Ghi chú |
|---|---:|---:|---:|---:|---|
| DATE | 27 | 9.097 | 1.956 | 1.229 | Subject-disjoint; bộ dùng trong manuscript |
| IPN Hand (13 lớp) | 13 | 2.443 | 589 | 1.101 | B0A, B0B và G01–G11; đang phát triển bài báo trên bộ này |

Support IPN: validation B0A 140, B0B 141, mỗi G 28 clip; test B0A 265, B0B 264, mỗi G 52 clip. Vì mỗi cử chỉ chỉ có 28 clip validation, một clip sai làm F1 lớp đó lệch khoảng 3,5 điểm.

Ý nghĩa các lớp dễ nhầm (theo `IPN_TEMPLATE_AUTHORING_PLAYBOOK_VI.md`): G01/G08 là click một ngón một lần/hai lần; G02/G09 là click hai ngón một lần/hai lần; G10/G11 là pinch tăng/giảm; G07 là chuỗi 4 pha đóng–mở–đóng–mở.

Phương pháp (manuscript): skeleton RTMW 52 keypoint × 64 frame → nhánh neural (Transformer) và nhánh tri thức (predicate → event → instance graph so với template) → cross-attention có hướng dẫn bởi template → fusion có gate.

## 3. Công việc theo mốc

| Ngày | Nội dung | Commit tiêu biểu |
|---|---|---|
| 06–09/10 | Xử lý dữ liệu IPN thành 13 lớp, keypoint, tensor cache; template 13 lớp; đánh giá V4 ban đầu | `7c08f32`, `4ab4cef` |
| 16/10 | Workflow symbolic IPN V2 trên Colab | `eac0d4f` |
| 19–20/10 | Neural-only cải tiến, notebook Colab 13 lớp, báo cáo từng lớp | `4ff4733`, `6b13f82` |
| 21/10 | Ablation A/B/C, handbook, playbook template, chọn neural C, làm cứng symbolic | `2a2f3dd`, `f810d55`, `fc2892b`, `f0ab17a`, `e0a1e50` |
| 22/10 | Audit theo cặp lớp, sửa regression role, soft role alignment | `160fdd6`, `1ef544c`, `e817f16` |
| 23/10 | Tái cấu trúc repo, full test DATE-27, notebook IPN-13 | (chưa commit) |

## 4. Thay đổi code so với bản cũ

### 4.1 Neural-only (mốc so sánh: trước commit `4ff4733`)

**Mô hình**

- Thêm vận tốc `(dx, dy)` của từng keypoint: input mỗi frame từ `52×3 = 156` lên `52×5 = 260` chiều. Vận tốc frame đầu bằng 0; nếu một trong hai frame thiếu keypoint thì vận tốc bằng 0 (không tạo vận tốc giả).
- Ba chế độ pooling: `attention` (mặc định mới), `masked_mean` (bỏ qua frame rỗng) và `mean` (giữ để tái tạo kiến trúc cũ). Cờ `--disable-motion-features` và `--pooling mean` tái tạo kiến trúc cũ.
- Kiểm tra cứng shape đầu vào `[B, T, 52, 3]`; clip toàn frame rỗng vẫn cho logits hữu hạn.

**Huấn luyện**

- Jitter tọa độ (0,005) chỉ trên keypoint hợp lệ; random frame masking; keypoint dropout.
- Class-weighted cross entropy (power 0,5) và label smoothing 0,05; gradient clipping.
- Validation macro-F1 mỗi epoch; `best.pt`, scheduler và early stopping theo macro-F1 thay vì accuracy.

**Dữ liệu, cache và đánh giá**

- Manifest sai format, nhãn không phải số hoặc split rỗng thì dừng ngay; không còn tìm tensor theo tên trên toàn cache (tránh nhầm lớp/split); phát hiện cache trùng; `torch.load(weights_only=True)`; tensor phải `[64, 52, 3]` và hữu hạn; kiểm tra train/validation không trùng source.
- Evaluator xuất precision/recall/F1/support từng lớp, confusion matrix, balanced accuracy, prediction từng clip; chặn checkpoint 13 lớp chạy nhầm manifest 27 lớp.

### 4.2 Symbolic IPN V2

| Thay đổi | Mô tả | Mục đích |
|---|---|---|
| Lọc trend nhiễu | `trend_consistency = |dịch chuyển ròng| / tổng biến thiên`, ngưỡng 0,6; áp dụng cho chuyển động tay, ngón, pinch, spread, phối hợp hai tay | Cửa sổ đảo chiều mạnh không còn phát predicate hướng |
| Sequence nhiều pha | Các pha cùng subject, pha sau bắt đầu sau pha trước, không dùng lại occurrence | Chặn match giả ở G07 |
| Abstention | `min_best_score = 0,15`, `min_top2_margin = 0,02`; dưới ngưỡng thì abstain | Không ép về B0A khi hòa điểm; báo coverage và selective accuracy |
| Vai trò `active`/`return` | Run directional đầu là active, run đối hướng sau là return; chuỗi A–B–A thành active–return–active; gần như đồng thời thì ambiguous | Phân biệt pha chính với pha trả về của click/pinch |
| Soft role alignment | Điểm template nhân `floor + (1 − floor) × active_share`, `floor = 0,85` | Role chỉ hạ tối đa 15%, không xóa evidence; template cũ mặc định 1,0 |
| Chẩn đoán từng clip | `val_diagnostics.jsonl`: 13 score, rank, tie, margin, lý do abstain, event khớp | Chỉnh ngưỡng/template chỉ bằng train/validation |
| Audit theo cặp | `scripts/eval_ipn_pair_arbitration.py`: chỉ xét khi neural top-2 là G01/G08, G02/G09, G10/G11; hòa điểm tuyệt đối luôn abstain; mặc định `audit_only` | Đo override có lợi/có hại/net trước khi bật |
| Cache graph an toàn | Provenance ghi config và builder (`phase_v4_action_role_audit`, `directional_reversal_v2`); fingerprint theo định danh tương đối và SHA-256 | Graph cũ không bị dùng nhầm |

Chi tiết và lý do từng thay đổi: `IPN_SYMBOLIC_POST_C_CHANGELOG_VI.md`.

### 4.3 Kiến trúc V4 — ba đầu ra

V4 xuất ba nhóm logits từ một lần forward (`cross_attention_v3.py` và `cross_attention_v4.py`):

- **`neural_logits`:** classifier trên vector pooled của Transformer; chỉ nhìn keypoint.
- **`kg_logits`:** 13 template query đọc hai memory (token keypoint và token instance graph) bằng hai cross-attention riêng có gate theo token; một bộ chấm điểm requirement cho điểm từng lớp. Đây là **learned head**, không phải rule matcher.
- **`logits` (final):** `ConfidenceLogitFusion`: hai logits được chia cho nhiệt độ học được rồi trộn bằng một gate theo lớp, `final = gate × neural + (1 − gate) × kg`. Gate là **trọng số của neural**, khởi tạo 0,8, nhận vào vector pooled của neural, đặc trưng KG của lớp, và độ tin cậy/entropy của hai đầu. Hai đầu dùng chung encoder được train chung nên không độc lập; xem `GIAI_THICH_THAY_DOI_VA_THU_NGHIEM_VI.md` mục 5.5.

Hệ quả: kết quả final là tổ hợp lồi của hai đầu nên thường nằm giữa chúng, nghiêng về đầu có gate lớn hơn.

### 4.4 Quy trình và repo

- Cấu trúc: `results/runs/` (lần chạy mới), `results/final/` (bản chọn theo validation), `results/experiments/` (thí nghiệm lịch sử và ZIP IPN), `docs/paper/` (manuscript); `final/code/` và `final/data/` giữ nguyên để loader không vỡ. Bảng đường dẫn cũ/mới: `LEGACY_ARTIFACT_PATH_MAP.md`.
- Wrapper test fusion mặc định dùng `best.pt` (chọn theo validation). `best_test.pt` chỉ dùng khi truyền `-AllowTestSelected` và chỉ là exploratory.
- `AGENTS.md`: mọi thay đổi code/config/template/notebook phải có entry docs nêu thay đổi, lý do, tương thích và validation.
- Notebook Colab ghim commit, kiểm tra hash, không ghi đè; mỗi run lưu một ZIP gồm metric, hash và toàn bộ lệnh.
- Đã xóa các artifact thừa có thể tái tạo (`.venv-1`, cache); không xóa dữ liệu thô, cache tensor, graph hay checkpoint nguồn.

### 4.5 Cải thiện mới sau audit (2026-10-24, tùy chọn, mặc định tắt)

- **V4 nhận vận tốc, pooling và đặc trưng hình học:** `--neural-motion-features`, `--neural-pooling`, `--neural-hand-features`; nạp được checkpoint C trực tiếp. Kiểm chứng: logits giống hệt neural-only với 12 tổ hợp; checkpoint C thật nạp strict vào V4.
- **`--hand-features` (neural-only):** khoảng cách đầu ngón, tip-to-knuckle và pinch chuẩn hóa theo cỡ bàn tay, cộng thay đổi theo thời gian; biến cử động ngón thành tín hiệu một chiều sạch để đếm click và đo chiều pinch.
- **Augmentation:** `--time-crop-min` (cắt và lấy mẫu lại theo thời gian), `--scale-jitter`, `--shift-std`; không che frame nên không phá nhịp click.
- **`ensemble_neural_only.py`:** trung bình xác suất nhiều seed.
- **`load_checkpoint_compat`:** đọc được checkpoint Colab (đường dẫn Linux) trên Windows.
- **Notebook `IPN13_Neural_Improvement_Sweep_Colab.ipynb`:** 4 biến thể × 3 seed, chọn theo validation, test khóa mặc định.
- Kiểm thử: 153 test pass (30 test mới); 128/128 clip DATE-27 cho dự đoán giống hệt lần chạy đã đóng gói; smoke test huấn luyện/đánh giá/ensemble. **Hiệu quả trên IPN thật chưa đo** (cần Colab).

## 5. Thử nghiệm neural A/B/C trên IPN-13

Thiết kế: cùng Transformer, seed 42, 80 epoch (patience 12), chọn theo validation macro-F1, **chưa chạy test**.

| Run | Cấu hình | Câu hỏi | Val acc | Val macro-F1 |
|---|---|---|---:|---:|
| Cũ | d128, 3 lớp, attention pooling, 50 epoch | Mốc neural cải tiến đầu tiên | 87,10% | 81,09% |
| A | d192, 4 lớp, 8 head, attention pooling; frame mask 0,05, keypoint dropout 0,03 | Mô hình lớn và train lâu hơn có tốt hơn? | 87,61% | 81,57% |
| B | Như A, masked-mean pooling | Attention pooling có bỏ mất nhịp lặp? | 85,91% | 79,96% |
| **C** | Như B, **frame mask 0**, keypoint dropout 0,02 | Augmentation có xóa sự kiện ngắn? | **89,13%** | **83,76%** |

C được chọn làm neural baseline. So với B (chỉ khác augmentation), C hơn +3,2 điểm accuracy và +3,8 điểm macro-F1. **Đây là giả thuyết, chưa phải bằng chứng nhân quả:** một seed, chưa tách frame masking khỏi keypoint dropout, và C overfit (train ≈ 99,9% so với validation 89,1%; best epoch 48/60). Các lớp click cải thiện giữa B và C (G01 0,48 → 0,63; G08 0,62 → 0,70), phù hợp giả thuyết frame masking che mất khoảnh khắc click ngắn.

### 5.1 F1 từng lớp (validation, 589 clip)

| ID | Lớp | Số clip | A | B | C | C: Precision | C: Recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | B0A | 140 | 0,964 | 0,945 | **0,972** | 0,958 | 0,986 |
| 1 | B0B | 141 | **0,982** | 0,972 | 0,979 | 0,986 | 0,972 |
| 2 | G01 | 28 | 0,531 | 0,478 | **0,627** | 0,696 | 0,571 |
| 3 | G02 | 28 | **0,691** | 0,590 | 0,630 | 0,654 | 0,607 |
| 4 | G03 | 28 | 0,893 | 0,909 | **0,945** | 0,963 | 0,929 |
| 5 | G04 | 28 | 0,909 | 0,863 | **0,947** | 0,931 | 0,964 |
| 6 | G05 | 28 | 0,931 | 0,964 | **1,000** | 1,000 | 1,000 |
| 7 | G06 | 28 | 0,909 | **0,945** | 0,915 | 0,871 | 0,964 |
| 8 | G07 | 28 | **0,926** | 0,868 | 0,885 | 0,958 | 0,821 |
| 9 | G08 | 28 | 0,645 | 0,620 | **0,704** | 0,731 | 0,679 |
| 10 | G09 | 28 | **0,667** | 0,655 | 0,644 | 0,613 | 0,679 |
| 11 | G10 | 28 | 0,750 | 0,772 | **0,793** | 0,767 | 0,821 |
| 12 | G11 | 28 | 0,806 | 0,814 | **0,847** | 0,806 | 0,893 |
| | Macro | | 0,816 | 0,800 | 0,838 | | |

C cao nhất ở 8/13 lớp; A thắng B0B, G02, G07, G09; B thắng G06. C chưa đạt tiêu chí đặt trước "mọi lớp F1 ≥ 0,65": G01 (0,627), G02 (0,630), G09 (0,644) còn dưới ngưỡng.

Nhầm lẫn chính của C: G02→G09 7 clip, G09→G02 6, G01→G08 6, G08→G01 6, G07→G10 4, B0B→B0A 3. Phần lớn là cặp click một lần với hai lần, đúng ba cặp mà template symbolic nhắm tới.

Tốc độ (forward-only, GPU Colab, batch 64): 0,18 ms/clip (≈ 5.634 clip/s), không gồm trích keypoint và dựng graph.

## 6. Symbolic thuần luật trên IPN-13 (validation)

| Mốc | Acc thô 13 lớp | Macro-F1 thô | Nhận xét |
|---|---:|---:|---|
| Sau neural C (21/10) | 20,03% | 6,26% | 250/589 clip hòa điểm; 448 clip bị đoán B0A |
| Trend gate + abstention (22/10) | 21,73% | 8,71% | 461/589 bị đoán B0A; coverage 21%, selective acc 31% |
| Role cứng `action_role: active` (artifact 6ef) | 22,24% | 6,09% | 515/589 bị đoán B0A; F1 của G01–G11 bằng 0 → regression |
| Soft role 0,85 (rescore 589 graph) | 23,43% | 9,01% | Coverage 27,8%, selective acc 36,0% |

Kết luận từ audit:

- Accuracy thô tăng không đồng nghĩa tiến bộ: bản role cứng tăng accuracy nhờ đoán B0A nhiều hơn nhưng macro-F1 giảm. Nguyên nhân: 63,2% edge bị gắn `ambiguous` vì sliding window chồng lấn tại điểm đổi chiều. Đã đổi sang phân loại theo thứ tự tâm interval và soft role; rescore trên graph 6ef quay gần đúng mốc trước regression (acc 6,62%, macro-F1 4,69% có abstention, so với mốc 6,62%/4,67%).
- Pairwise audit với neural C (validation accuracy 0,891341, neural margin ≤ 0,20, symbolic 0,15/0,02): artifact 6ef có 2 override được đề xuất, cả hai có hại (net −2); sau soft role còn 1 đề xuất, vẫn có hại. Quét neural margin 0,02–0,50 và symbolic margin 0,02–0,20 không tìm được cấu hình net dương. **Override giữ tắt, locked test không mở.**
- Hiệu quả của builder `phase_v4_action_role_audit` mới chỉ được đo qua rescore; cần một run train/validation mới để xác nhận.

Ý nghĩa: rule matcher thuần luật không thay được neural trên IPN, nên hướng đi cho bài báo IPN là learned `kg_logits` (mục 8).

## 7. DATE-27: kết quả cuối và so sánh baseline

Đánh giá trên đúng `test_1229.txt` (1.229 clip, 27 lớp), checkpoint V4 `cross_attention_v4_template170_retrain_next/best.pt` (epoch 19, chọn theo validation). Test đã được mở trong lúc phát triển, nên đây là mốc tham chiếu validation-selected, không phải locked holdout mới.

| Mô hình/đầu ra | Đúng | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Neural-only độc lập (checkpoint legacy, epoch 23) | 1.171 | 95,28% | 95,13% |
| Neural head trong V4 | 1.163 | 94,63% | 94,29% |
| KG head trong V4 | 1.168 | 95,04% | 94,67% |
| **Final neural + symbolic (V4)** | **1.166** | **94,87%** | **94,52%** |

KG head sửa 9 lỗi của neural head và gây 4 lỗi mới (net +5). Final chỉ hơn neural head 3 clip và thấp hơn KG head 2 clip, phù hợp với việc hai đầu cùng chung encoder và gate trộn khoảng nửa–nửa (mục 4.3); trên validation cả ba đầu ra gần như bằng nhau.

Baseline trong manuscript (Table 2, mean ± SD của 10 entry):

| Phương pháp | Accuracy |
|---|---:|
| SignBart | 88,06 ± 0,76% |
| Sparse-ST-GCN | 92,94 ± 0,40% |
| ProtoGCN | 93,28 ± 0,27% |
| DATE Transformer đã công bố | 94,22% |
| **Ta: Final (một checkpoint)** | **94,87%** |
| **Ta: KG head** | **95,04%** |
| **Ta: neural-only (legacy)** | **95,28%** |
| Proposed Fusion trong manuscript | 96,40 ± 0,17% |

Final cao hơn ProtoGCN 1,59 điểm và DATE Transformer 0,65 điểm; thấp hơn mean Proposed Fusion 1,53 điểm và thấp hơn neural-only 0,41 điểm accuracy, 0,61 điểm macro-F1. Không so Macro-F1 với paper vì paper chỉ công bố accuracy. Baseline RTMW trong paper là mean của 10 entry, còn số của ta là một checkpoint nên không phải kiểm định thống kê tương đương.

Chi tiết và evidence: `DATE27_BASELINE_COMPARISON_2026-10-23_VI.md`, `results/final/date27_v4_validation_selected/`.

## 8. Learned `kg_logits` trên IPN-13: kết quả (run 2026-10-24)

Notebook `IPN13_KG_Logits_Baseline_Comparison_Colab.ipynb` (commit `e817f16`, V4 khởi tạo từ checkpoint neural-only không vận tốc, augmentation 0/0,02 như C, test mở một lần). Protocol: `IPN13_KG_LOGITS_EXPERIMENT_PROTOCOL_VI.md`; phân tích đầy đủ: `AUDIT_IPN13_KET_QUA_VA_CAI_THIEN_2026-10-24_VI.md`.

| Mô hình (test, 1.101 clip) | Accuracy | Macro-F1 |
|---|---:|---:|
| Neural-only C (có vận tốc) | **84,65%** | **77,24%** |
| Neural-only không vận tốc (khởi tạo V4) | 82,56% | 73,65% |
| V4 neural head | 83,02% | 74,40% |
| V4 `kg_logits` | 82,29% | 73,22% |
| V4 final | 82,56% | 73,84% |

F1 từng lớp trên test (C, V4 neural, V4 KG, V4 final): B0A 0,951 / 0,954 / 0,952 / 0,954; B0B 0,973 / 0,972 / 0,964 / 0,966; G01 0,592 / 0,527 / 0,463 / 0,527; G02 0,549 / 0,531 / 0,477 / 0,515; G03 0,804 / 0,667 / 0,674 / 0,667; G04 0,830 / 0,839 / 0,811 / 0,832; G05 0,931 / 0,932 / 0,940 / 0,932; G06 0,845 / 0,855 / 0,847 / 0,847; G07 0,808 / 0,727 / 0,739 / 0,734; G08 0,692 / 0,626 / 0,650 / 0,626; G09 0,660 / 0,635 / 0,672 / 0,617; G10 0,740 / 0,722 / 0,709 / 0,722; G11 0,667 / 0,686 / 0,621 / 0,660.

Đọc kết quả:

- V4 final **bằng đúng** checkpoint khởi tạo của nó (82,56%) và thấp hơn C 2,1 điểm. `kg_logits` sửa 17 lỗi của neural head nhưng gây 25 lỗi mới (net −8). **Không có bằng chứng symbolic giúp trên IPN** trong thiết lập này.
- KG head hơn rõ nhất ở G09 (9 clip được sửa, 1 clip bị hại) và G08 (4 sửa, 2 hại), tức đúng ở các lớp double click mà template nhắm tới; nhưng bị hại ở G01 (1 sửa, 6 hại), G02 (1 và 6) và G11 (0 và 4), nên tổng thể net −8.
- Nguyên nhân chính của khoảng cách với C là V4 bị mất vận tốc (neural không vận tốc thua C 2,1 điểm accuracy, 3,6 điểm macro-F1; G08 recall 36,5% so với 71,2%), không phải do template.
- Gate (trọng số của neural) trên validation từ 0,82 xuống 0,65; validation dao động 80–86% giữa các epoch và huấn luyện dừng ở epoch 17 (patience 5).

Đã triển khai bằng code để khắc phục (chưa chạy): adapter V4 nhận vận tốc nên khởi tạo trực tiếp từ C được; xem mục 4.5 và mục 5 của tài liệu audit.

## 9. Hạn chế và rủi ro

1. **Claim 96,40% của manuscript chưa tái lập:** repo chỉ đóng gói chắc chắn một checkpoint; run 3 (96,18%) trùng `best_test.pt`.
2. **Fusion chưa vượt neural-only trên DATE-27:** 94,87% so với 95,28%; neural-only chỉ còn file `last.pt` (best validation ở epoch 18), nên là baseline legacy.
3. **Test đã bị xem** trong phát triển ở cả DATE-27 và IPN; kết quả test là bằng chứng exploratory, không phải holdout sạch.
4. **A/B/C là một seed;** chưa tách các yếu tố augmentation; chênh lệch nhỏ giữa các lớp 28 clip nằm trong nhiễu.
5. **Learned `kg_logits` trên IPN-13 chưa chứng minh được lợi ích** (mục 8): V4 final bằng checkpoint khởi tạo và thấp hơn neural C; KG net −8 clip trên test. Nên coi đây là kết quả âm cho tới khi V4 được khởi tạo từ checkpoint có vận tốc và chạy lại.
6. **Hai head của V4 dùng chung encoder và được train chung** (không đóng băng theo mặc định; loss gồm `CE(final) + 0,3·CE(neural) + 0,3·CE(kg) + 0,05·BCE(requirement) + 0,01·(gate − 0,8)²`). Neural head của V4 vì vậy không phải mô hình neural-only độc lập, và lợi thế nhỏ của KG head chưa thể gán cho tri thức template khi chưa có ablation template xáo trộn hoặc graph rỗng. Gate là trọng số của neural (khởi tạo 0,8). Chi tiết: `GIAI_THICH_THAY_DOI_VA_THU_NGHIEM_VI.md` mục 5.5.
7. **Domain shift giữa camera:** camera 1CM1 có 3 subject trong test nhưng 1 subject trong train và 0 trong validation; accuracy của C theo camera là 77%, 84%, 86%, 90%. Validation không thể phát hiện điểm yếu này, và khoảng cách validation–test là 4,5 điểm accuracy.
8. **Số mẫu nhỏ:** IPN-13 có 116 clip train mỗi cử chỉ (DATE-27 có khoảng 337 mỗi lớp) và 13 subject trong test nên chênh lệch nhỏ giữa các mô hình nằm trong nhiễu. Test đủ 1.101 clip chính thức; train+validation thiếu 85 clip do loại subject `1CM42_3`; lớp `D0X` không được dùng (13 thay vì 14 lớp).
9. **Test IPN đã bị mở hai lần** (neural-only trước đó và run này). Mọi lựa chọn cải thiện tiếp theo phải dựa trên validation nhiều seed; xem test lần nữa chỉ là exploratory.
10. **Hiệu quả của các cải thiện mới (mục 4.5) chưa đo trên IPN thật.**

## 10. Việc tiếp theo

1. Commit và push code mới, rồi chạy `IPN13_Neural_Improvement_Sweep_Colab.ipynb` (4 biến thể × 3 seed, chọn theo validation) để biết `--hand-features` và augmentation có giúp không.
2. Chạy lại V4 khởi tạo trực tiếp từ C có vận tốc (`--neural-motion-features --neural-pooling masked_mean`), kèm `--freeze-neural-epochs 3` và patience lớn hơn, để trả lời "KG có giúp khi neural đã mạnh hay không".
3. Thêm đầu phụ cho "số lần click" và "một hay hai ngón" (nhãn phân rã) và huấn luyện trên train+validation sau khi chốt cấu hình; xây validation theo subject-grouped K-fold có đủ camera.
4. Xác minh provenance 10 run của manuscript trước khi dùng 96,40%.
5. Chạy ablation template xáo trộn / graph rỗng để biết tri thức template có đóng góp thật hay không.
6. Chạy lại train/validation symbolic với graph `phase_v4_action_role_audit` để đo hiệu quả thật của role v2.

## Phụ lục A. Nguồn trong `docs/`

- `NEURAL_ONLY_EVALUATION_REPORT_2026-10-20_VI.md` — thay đổi neural-only, hợp đồng dữ liệu.
- `HANDBOOK_TOAN_DIEN_IPN_NEUROSYMBOLIC_VI.md` — thiết kế ablation A/B/C, giao thức validation/test.
- `IPN_TEMPLATE_AUTHORING_PLAYBOOK_VI.md` — template 13 lớp, chọn C, cặp nhầm lẫn.
- `IPN_SYMBOLIC_POST_C_CHANGELOG_VI.md`, `IPN_SYMBOLIC_6EF_ANALYSIS_VI.md` — thay đổi và audit symbolic.
- `DATE27_BASELINE_COMPARISON_2026-10-23_VI.md`, `paper/DATE_NeuroSymbolic_Manuscript.pdf` — DATE-27 và baseline.
- `IPN13_KG_LOGITS_EXPERIMENT_PROTOCOL_VI.md` — protocol notebook IPN-13.
- `REPOSITORY_REORGANIZATION_2026-10-23_VI.md`, `LEGACY_ARTIFACT_PATH_MAP.md` — tái cấu trúc repo.

## Phụ lục B. Artifact đã dùng

| Artifact | Nội dung |
|---|---|
| `ipn_symbolic_v2_post_c` (`cb972d1ba1fdd3f9`) | Mốc symbolic sau neural C |
| `035179b821061ab7` | Trend gate + abstention (mốc 0351) |
| `6efa8ad55dde2909` | Role cứng (regression) |
| `d255086f928a0615` | Graph d25 dùng cho soft role |
| `c46e0e469d2cf3b6` | Train/validation artifact cho notebook IPN-13 |
| `neural_ablation_{A,B,C}` | Checkpoint và `val_metrics.json` của A/B/C |

## Validation của tài liệu

Số liệu A/B/C đọc trực tiếp từ `val_metrics.json`, `command.json` và `history.json` trong các ZIP; số symbolic và DATE-27 đối chiếu với các note ở Phụ lục A; bảng baseline đối chiếu với Table 2 và Table 3 của manuscript PDF; số IPN-13 test đọc từ gói `20261024T075203Z-2c6b5c8d30.zip` (metric, dự đoán từng clip, lịch sử huấn luyện, checkpoint, lệnh đã chạy). Mô tả công thức trộn của V4 đã được đối chiếu lại với `cross_attention_v3.py`/`cross_attention_v4.py` sau khi bản đầu của tài liệu này mô tả nhầm lớp V2 cũ (đã sửa). Sau các thay đổi code ngày 2026-10-24: full unit suite `153 passed` (3 + 29 warning PyTorch đã biết), 128/128 clip DATE-27 tái hiện đúng lần chạy đã đóng gói.
