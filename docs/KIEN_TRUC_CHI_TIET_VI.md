# Kiến trúc chi tiết và đối chiếu với bản thảo

Cập nhật: 2026-11-04. Tài liệu này mô tả hệ thống đúng như code đang chạy (`final/code/`), và đối chiếu từng khối với bản thảo `docs/paper/DATE_NeuroSymbolic_Manuscript.pdf` (24 trang, bài toán DATE 27 lớp). Lịch sử thay đổi và kết quả thí nghiệm nằm ở `NHAT_KY_THAY_DOI_VI.md`.

Nhãn bằng chứng dùng trong cả hai tài liệu: **[Đã đo]** có file kết quả; **[Lập luận]** suy luận hợp lý nhưng chưa có thí nghiệm tách riêng; **[Chưa biết]** chưa đo.

## 1. Tổng quan luồng dữ liệu

```
Video clip
  │  (1) Phát hiện người + ước lượng tư thế → JSON 52 điểm (x, y, score)
  ▼
Keypoint JSON ──► (2) chuẩn hóa + lấy mẫu 64 frame ──► Tensor T×52×3 ──► nhánh NEURAL
  │                                                                          │
  └─► (3) predicate → event → instance graph ─────────────► nhánh SYMBOLIC   │
                                                              │               │
              template graph (mỗi lớp một đồ thị, viết tay) ──┘               │
                                  │                                           │
                                  ▼                                           ▼
                (4) dual cross-attention: template query đọc neural memory + graph memory
                                  │                                  neural_logits
                              kg_logits                                       │
                                  └─────────────► (5) fusion có cổng ◄────────┘
                                                           │
                                                      final logits
```

## 2. Khối 1 — Ước lượng tư thế và 52 điểm

- Pipeline gốc (bản thảo mục 3.2): RTMDet-M (ngưỡng điểm và NMS IoU đều 0,3) phát hiện người; RTMW-L ước lượng 133 điểm từ ảnh 288×384; mỗi frame giữ người có điểm trung bình cao nhất.
- Chọn **52 điểm** = 10 điểm thân/mặt + 2×21 điểm bàn tay:

| Nhóm | Điểm | Số |
|---|---|---:|
| Mặt | `nose`, `left_eye`, `right_eye`, `face-8` | 4 |
| Thân | `left/right_shoulder`, `left/right_elbow`, `left/right_wrist` | 6 |
| Tay trái | `left_hand_root` + 20 khớp ngón | 21 |
| Tay phải | `right_hand_root` + 20 khớp ngón | 21 |

  Khớp với bản thảo ("bốn điểm mặt, sáu điểm vai/khuỷu/cổ tay, 21 điểm mỗi tay"). Danh sách tên cố định nằm ở `keypoint_dataset.py` (`KEYPOINT_NAMES`).
- Lưu ý khi video bị lật gương (IPN): tên trái/phải được hoán đổi khi trích xuất để quy ước khớp RTMW (đã kiểm tra trên pilot).
- **Nguồn keypoint thay thế (thí nghiệm mới):** `keypoint_extractor/extract_mediapipe_holistic.py` ghi cùng 52 tên và cùng schema JSON từ MediaPipe Holistic 2D; điểm bàn tay không phát hiện được thì **bỏ** (không điền giả); cổ tay dự phòng bằng cổ tay của pose; `face-8` = trung điểm hai khóe miệng; `score` = 1,0. Mọi bước sau không đổi.

## 3. Khối 2 — Chuẩn hóa và tensor

Công thức (bản thảo Eq. 3), cho điểm `p_{t,k}` ở frame `t`, vai trái/phải `s^L_t, s^R_t`:

```text
o_t = (s^L_t + s^R_t) / 2        d_t = ||s^L_t − s^R_t||        p̂_{t,k} = (p_{t,k} − o_t) / d_t
```

Điều kiện: cả hai vai có điểm ≥ 0,3 và `d_t > 1e-6`, nếu không toàn bộ frame đặt bằng 0; điểm tin cậy cắt về [0, 1]. Lấy mẫu đều về **T = 64** frame (clip dài thì lấy mẫu, clip ngắn lặp frame cuối). Đầu ra `X ∈ R^{64×52×3}`.

Hai điều cần ghi trong bài khi dùng IPN:
- Điểm RTMW không được chuẩn hóa về [0, 1] (trung vị khoảng 4–7), nên sau khi cắt về [0, 1] điểm tin cậy gần như luôn bằng 1; ngưỡng 0,3 của vai vì vậy hầu như không loại frame nào ngoài trường hợp RTMW không phát hiện [Đã đo qua phân bố điểm; chưa thống kê tỉ lệ loại].
- Với MediaPipe, `score` luôn 1,0 cho điểm có mặt.

## 4. Khối 3 — Nhánh neural (Transformer)

`NeuralBaselineAdapter` (`models/cross_attention_v2.py`):

1. Mỗi frame: làm phẳng 52×3 = 156 giá trị → `Linear(156 → d)` → LayerNorm → GELU → Dropout → cộng mã hóa vị trí sin.
2. `TransformerEncoder` pre-norm: `L` lớp, `h` head, feed-forward `4d`.
3. Hai đường song song từ đầu ra `H ∈ R^{64×d}`:
   - **Neural memory:** `M_N = LN(H + Dropout(W H + b))` (bản thảo Eq. 4), dùng cho cross-attention.
   - **Neural logits:** gộp theo thời gian (pooling) → `z_N ∈ R^{C}` qua classifier hai lớp.

| Siêu tham số | Bản thảo (DATE) | Thí nghiệm IPN |
|---|---|---|
| d (độ rộng token) | 128 | **192** |
| Số lớp / số head | 3 / 4 | **4 / 8** |
| Pooling | mean | **masked_mean** (bỏ frame rỗng) |
| Đầu vào mỗi frame | 156 (x, y, score) | **260** (thêm vận tốc dx, dy; =0 nếu một trong hai frame không hợp lệ) |
| Dropout | 0,15 | 0,15 |

Khác biệt IPN so với bản thảo đến từ vòng chọn cấu hình neural trên validation IPN (xem nhật ký, mục A/B/C): **cần ghi rõ trong bài là cấu hình neural của thí nghiệm IPN khác DATE**, vì bản thảo hiện mô tả d128/3 lớp/4 head/mean pooling.

## 5. Khối 4 — Nhánh symbolic

### 5.1 Predicate → event → instance graph (không dùng nhãn)

- **Predicate** là phát biểu về thuộc tính quan sát được trên tọa độ chuẩn hóa: vị trí tay so với vai/mặt, khoảng cách hai tay, hình dạng ngón (khoảng cách gốc–đầu ngón so với độ dài đường đi), chuyển động tay/ngón theo cửa sổ, phối hợp hai tay. Ngưỡng mặc định (`ipn_predicate_v2`): `min_score 0,3`, cửa sổ 5, chuyển động tay 0,12, ngón 0,04, pinch 0,08, `trend_consistency 0,6`. Mọi ngưỡng đều **hình học 2D**.
- **Event** gộp predicate cùng tên/chủ thể/đối tượng, cho phép đứt quãng ngắn; lọc theo thời lượng và độ tin cậy; sự kiện chồng/nối nhau thành event phức hợp theo pha (`phase_v4_action_role_audit`, `max_phase 16`, `min_phase 4`, overlap 0,4/2).
- **Instance graph**: nút Video, Event, BodyPart, các nút khái niệm; cạnh `HAS_EVENT`, `INVOLVES`, `HAS_POSITION`, `HAS_HAND_SHAPE`, `HAS_MOTION`, `HAS_FINGER_MOTION`, `HAS_COORDINATION`, `HAS_JOINT_ANGLE`, `HAS_SWEEP`, `HAS_EVIDENCE`, và quan hệ thời gian `BEFORE`/`OVERLAPS`.

### 5.2 Template graph

Mỗi lớp một đồ thị viết tay (`configs/ipn/gesture_templates.yaml`, 13 lớp; `gesture_templates_14.yaml` cho HandS 14 lớp). Nút có mức yêu cầu `required` / `preferred` / `optional` / `negative` và trọng số có dấu. `negative_evidence` hiện để trống vì bộ khớp phạt trên toàn clip chứ không theo khoảng thời gian (chi tiết ở nhật ký).

### 5.3 Graph encoder dùng chung

`GraphEncoderV2` + `StrictGraphLayer` (attention nhận biết quan hệ, mặt nạ kết nối và padding, cạnh ngược cho trao đổi hai chiều; 2 lớp). Instance graph và template graph đi qua cùng bộ trọng số.
- Instance: các nút được nén bằng **16 truy vấn học được** (`instance_tokens = 16`) thành bộ nhớ độ dài cố định `M_G` (đúng với bản thảo mục 3.4.1).
- Template: giữ **từng nút** làm một truy vấn `Q_c` (một truy vấn cho mỗi nút của lớp `c`).
- V4 thêm **mục tiêu requirement có cấu trúc**: nhãn "yêu cầu có mặt hay không" lấy từ bộ ba `(chủ thể, quan hệ, khái niệm)` của instance graph, để yêu cầu tay trái/phải không bị gộp (`StructuredRequirementGraphEncoder`).

## 6. Khối 5 — Dual cross-attention và chấm điểm requirement

`DualCrossAttentionV4NeuroSymbolicModel` (`models/cross_attention_v4.py`, kế thừa `cross_attention_v3.py`):

1. Template self-attention tinh chỉnh `Q_c → Q̃_c`.
2. Hai cross-attention độc lập (4 head) cùng dùng `Q̃_c` làm query: một trên `M_N`, một trên `M_G`.
3. Cổng theo token (MLP sigmoid trên query tinh chỉnh, hai đặc trưng thu được và hiệu tuyệt đối) trộn hai đặc trưng; residual + norm + feed-forward → token khớp `R_c`; nút padding bị che.
4. Chấm điểm (bản thảo Eq. 5): `p = σ(Linear(R_c))`, `m = tanh(Linear(R_c))`,
   `z^G_c = ℓ(R̄_c) + a·μ_P(p+m) − b·μ_D(p) − q·μ_R(1−p)`,
   với `P, D, R` là các tập nút dương, bị cấm, bắt buộc; `μ` là trung bình có trọng số tuyệt đối; `a, b, q` khởi tạo 1 và học tự do. Đây là độ tương thích mềm, không ép thỏa mãn logic; vì đọc cả neural memory nên `z^G` là **graph-guided** chứ không thuần symbolic.

Tham số mặc định V4 trong code: `d_model 128, layers 3, heads 4, graph_layers 2, decoder_layers 1, instance_tokens 16, dropout 0,15` (trùng bản thảo). Thí nghiệm IPN truyền `--d-model 192 --layers 4 --heads 8`.

### 6.1 Hình dạng tensor

Ký hiệu: `B` batch, `T=64`, `K=52`, `D` độ rộng token, `C` số lớp, `M` số nút tối đa của template, `I=16` token instance sau nén.

```text
keypoints            [B,T,52,3]  → phẳng [B,T,156] → [B,T,D] → Transformer → H [B,T,D]
neural memory M_N    [B,T,D]     → mở rộng [B*C,T,D]
instance graph       [B,N,D]     → LearnedInstanceCompressor → [B,16,D] → mở rộng [B*C,16,D]
template tokens      [C,M,D]     → mở rộng [B*C,M,D]   (query)
token khớp R         [B,C,M,D]   → presence/match [B,C,M] → kg_logits [B,C]
final                [B,C]
```

Mỗi mẫu được so với từng template nên chi phí tăng tuyến tính theo `C`. Với cấu hình DATE (27 lớp, d128) mô hình có **1.680.744 tham số huấn luyện** (khớp "1,681M" của bản thảo); riêng adapter neural 635.675 tham số, bộ từ vựng khái niệm 170 mục. Đây là số của một cấu hình cụ thể; số tham số của cấu hình IPN (d192, 4 lớp, 8 head, 13 lớp) chưa được thống kê và cần đo lại nếu đưa vào bảng.

Cổng theo token nhận `[q, neural_out, graph_out, |neural_out − graph_out|]` (4D chiều) và trộn `g·neural_out + (1−g)·graph_out`; cổng này khác cổng lớp cuối ở mục 7.

### 6.2 Giới hạn đã biết của mục tiêu requirement ở V4

`StructuredRequirementGraphEncoder` lưu yêu cầu theo **nút đích** (`facts[target] = (chủ thể, quan hệ, khái niệm)`). Nếu hai event trái/phải cùng trỏ vào một nút khái niệm dùng chung thì cạnh đọc sau có thể ghi đè cạnh trước, nên V4 **chưa bảo đảm** phân biệt mọi yêu cầu tay trái/tay phải trong template hiện tại. Cách sửa đúng là biểu diễn yêu cầu ở mức cạnh hoặc nhân bản nút khái niệm theo event. Chưa sửa; cần nêu thành giới hạn khi viết bài (hoặc sửa trước khi chạy thí nghiệm chính thức).

## 7. Khối 6 — Fusion theo độ tin cậy

`ConfidenceLogitFusion` (bản thảo Eq. 6):

```text
z_c = g_c · z^N_c / τ_N + (1 − g_c) · z^G_c / τ_G ,   0 < g_c < 1
```

`τ_N, τ_G > 0` học được. `g_c` là cổng sigmoid theo lớp, điều kiện hóa trên vector neural gộp `h̄`, đặc trưng KG của lớp `R̄_c`, xác suất cực đại và entropy của từng nhánh. **`g_c` là trọng số của nhánh NEURAL** (khởi tạo 0,8), `1 − g_c` là của KG. Dự đoán `ŷ = argmax_c z_c`.

Ba đầu ra được báo cáo: `neural` (`z_N`), `kg` (`z_G`), `final` (`z`). Ghi chú diễn giải: encoder neural dùng chung, nên nhánh neural của V4 **không** đồng nhất với mô hình neural-only độc lập (encoder được tinh chỉnh trong lúc huấn luyện V4).

## 8. Huấn luyện V4

| Mục | Giá trị |
|---|---|
| Hàm mất mát (`--logit-branch final`) | `CE(final) + 0,3·CE(neural) + 0,3·CE(kg) + 0,05·BCE(requirement) + 0,01·(gate − 0,8)²` |
| Khởi tạo | Encoder nạp từ checkpoint neural C; 3 epoch đầu đóng băng encoder, sau đó huấn luyện chung (`--freeze-neural-epochs 3`) |
| Tối ưu | AdamW, learning rate encoder 2e-4, phần còn lại 3e-4, gradient clipping 1,0 |
| Batch / epoch | 32 / tối đa 50, `early-stop-patience 12`, `scheduler-patience 4` |
| Tiêu chí chọn checkpoint | `accuracy`, `macro_f1`, hoặc **`final_kg_mean_macro_f1`** = trung bình macro-F1 validation của `final` và `kg` (thêm 2026-11-04, xem nhật ký) |
| Augmentation | jitter 0,005, keypoint dropout 0,02, **không** frame masking (frame masking che mất nhịp click ngắn) |
| Neural C (khởi tạo) | Transformer d192/4 lớp/8 head, masked_mean, vận tốc, label smoothing 0,05, class-weight power 0,5, patience 12, chọn theo macro-F1 |

Bản thảo (Bảng 1) mô tả: AdamW, lr 2e-4 (N) / 3e-4 (G), weight decay 0,01, batch 16, 30 epoch. **Bản thảo không nêu các loss phụ `0,3·CE(neural)`, `0,3·CE(kg)`, `BCE(requirement)` và prior cổng `(gate − 0,8)²`**, trong khi chúng quyết định cách encoder và hai nhánh được học; nên bổ sung vào mục phương pháp.

## 9. Đối chiếu bản thảo ↔ code

| Nội dung bản thảo | Code | Trạng thái |
|---|---|---|
| 52 điểm (4 mặt + 6 thân + 42 tay), vai chuẩn hóa, T=64 | `keypoint_dataset.py`, `pose/normalize_keypoints.py` | Khớp |
| Encoder: 156→128, 3 lớp, 4 head, FFN 512, sin position | `NeuralBaselineAdapter` | Khớp (d128/3/4 là mặc định) |
| Neural memory Eq. 4 và pooling mean | `neural_projector`, `neural_pooling` | Khớp; IPN dùng masked_mean |
| Instance memory 16 truy vấn học được | `instance_tokens = 16`, `LearnedInstanceCompressor` (`cross_attention_v3.py`) | Khớp (hàm `compress_instance_tokens` ở `cross_attention_v2.py` là bản chunk-pooling cũ, không dùng trong V4) |
| Template query theo nút, graph encoder dùng chung | `encode_templates`, `GraphEncoderV2` | Khớp |
| Dual cross-attention 4 head, cổng token, Eq. 5 | `TemplateQueryDecoderLayer`, `RequirementScorer` | Khớp |
| Eq. 6, cổng theo lớp, nhiệt độ học | `ConfidenceLogitFusion` | Khớp |
| Hàm mất mát phụ, prior cổng | `train_cross_attention_v4.py` | **Bản thảo chưa nêu** |
| Siêu tham số huấn luyện (Bảng 1: 30 epoch, batch 16) | lệnh thực tế DATE/IPN | Cần đối chiếu lại với lệnh đã chạy |
| "Báo cáo trung bình 10 lần chạy" | repo chỉ đóng gói 1 checkpoint (94,87%) | **Chưa tái lập được** (xem mục 10) |

## 10. Giới hạn và điều cần cẩn trọng khi viết

1. **Con số 96,40% của bản thảo chưa tái lập được từ repo.** Checkpoint chọn theo validation đạt 94,87% trên 1.229 clip test DATE; một run trong bảng 3 của bản thảo (96,18%) trùng đúng với `best_test.pt` — checkpoint chọn *nhìn vào test* — nên cần xác minh quy trình chọn checkpoint của 10 run trước khi trình bày như ước lượng khách quan.
2. `kg_logits` đọc cả neural memory, nên cải thiện của nhánh KG chưa chứng minh được tri thức template đóng góp. Cần ablation nhân quả (template xáo trộn, đồ thị rỗng) [Chưa làm].
3. Predicate và ngưỡng đều 2D, viết tay; thay đổi nguồn keypoint không làm đổi ngưỡng nhưng làm đổi phân bố sự kiện (RTMW luôn xuất cả hai bàn tay, MediaPipe chỉ xuất tay phát hiện được: số sự kiện mỗi clip ít hơn khoảng 2,4 lần, mật độ trên mỗi tay gần như bằng nhau) [Đã đo].
4. **Đồ thị instance phải được nạp thật.** Dataset trả đồ thị rỗng lặng lẽ khi không tìm thấy file; từng có lỗi đường dẫn khiến V4 chạy không có đồ thị (xem nhật ký 7.0). Mọi lần chạy V4 cần kiểm tra tỉ lệ mẫu có đồ thị không rỗng trước khi diễn giải nhánh KG.
5. Tăng thêm kênh z (3D) chỉ nên đưa vào nhánh neural (đổi `input_dims` 3→4 và vận tốc +3), giữ symbolic 2D; chưa thực hiện.

## 11. Bản đồ mã nguồn

| Thành phần | Tệp |
|---|---|
| Tên 52 điểm, tensor | `final/code/src_neurosymbolic/keypoint_dataset.py` |
| Chuẩn hóa | `.../pose/normalize_keypoints.py` |
| Adapter IPN HandS (14 lớp) | `.../pose/ipn_hands_adapter.py` |
| Predicate, event | `.../predicates/extract_predicates.py`, `.../events/build_events_phase_v2.py` |
| Instance/template graph | `.../kg/build_instance_graph_phase_v2.py`, `.../kg/build_template_graph.py`, `.../kg/graph_matching.py` |
| Mô hình | `.../models/cross_attention_v2.py` (encoder, graph encoder, decoder, scorer), `cross_attention_v3.py`, `cross_attention_v4.py` |
| Huấn luyện / đánh giá V4 | `.../train_cross_attention_v4.py`, `.../test_cross_attention_v2.py` |
| Neural-only (backbone C, LSTM/BiGRU/TCN/MLP-Mixer) | `final/code/neural_only_experiments/` |
| Precompute tensor và đồ thị | `.../precompute_keypoint_tensors.py`, `.../precompute_instance_graphs_phase_v2.py` |
| Trích keypoint | `keypoint_extractor/` (RTMW, MediaPipe Holistic) |
| Template | `configs/ipn/` |
| Ensemble logits | `scripts/ensemble_v4_logits.py` |
