# Handbook toàn diện — Dynamic Hand Gesture Recognition trên IPN Hand

Phiên bản tài liệu: 21/10/2026

Phạm vi chính: pipeline IPN Hand 13 lớp, neural-only hiện tại, symbolic template V2 và hướng tích hợp neuro-symbolic.

Nguồn sự thật: code và artifact của đúng entry point đang chạy. Khi tài liệu cũ, notebook output và code khác nhau, ưu tiên code + metadata trong checkpoint.

---

## 0. Tài liệu này giải quyết điều gì?

Repository này chứa nhiều thế hệ thí nghiệm. Nếu đọc theo tên file mà không biết lịch sử, rất dễ trộn lẫn:

1. mô hình fusion V4 cho bộ cử chỉ nội bộ 27 lớp;
2. baseline neural-only dùng để so sánh nhiều backbone;
3. thí nghiệm IPN Hand 13 lớp đang chạy trên Colab;
4. rule-only symbolic matcher dùng YAML template;
5. hướng kết hợp neural và template vẫn đang được kiểm chứng.

Handbook này giải thích toàn bộ đường đi của một mẫu dữ liệu, từng phép biến đổi và lý do tồn tại của nó. Mỗi khái niệm quan trọng đều trả lời năm câu hỏi:

- Nó nằm ở đâu trong code?
- Input và output có shape gì?
- Nó chạy lúc train, validation hay test?
- Nó định giải quyết vấn đề gì?
- Khi nào nó có thể làm kết quả xấu đi?

### Cách đọc đề xuất

- Muốn hiểu dữ liệu đầu vào: đọc mục 3–6.
- Muốn hiểu neural, pooling, augmentation, train và evaluation: đọc mục 7–13.
- Muốn hiểu template: đọc mục 14; muốn hiểu fusion/XAI: đọc mục 15–16.
- Muốn trình bày/bảo vệ: đọc mục 17–22.
- Muốn debug Colab: đọc mục 11, 18 và 20.

---

## 1. Project đang giải bài toán nào?

### 1.1 Bài toán hiện tại: isolated gesture classification

Input là một clip đã được cắt sao cho phần lớn clip chứa đúng một cử chỉ. Output là một trong 13 nhãn:

| ID | Nhãn | Ý nghĩa |
|---:|---|---|
| 0 | B0A | Trỏ bằng một ngón |
| 1 | B0B | Trỏ bằng hai ngón |
| 2 | G01 | Click một ngón, một lần |
| 3 | G02 | Click hai ngón, một lần |
| 4 | G03 | Đẩy lên |
| 5 | G04 | Đẩy xuống |
| 6 | G05 | Đẩy sang trái |
| 7 | G06 | Đẩy sang phải |
| 8 | G07 | Mở hai lần |
| 9 | G08 | Double click một ngón |
| 10 | G09 | Double click hai ngón |
| 11 | G10 | Zoom vào |
| 12 | G11 | Zoom ra |

Nhãn `D0X` của IPN gốc bị loại khỏi classifier 13 lớp. Vì vậy hệ thống hiện tại **không phải** detector chạy trên video liên tục và tự tìm đoạn bắt đầu/kết thúc cử chỉ.

### 1.2 Ba bài toán không được đánh đồng

| Bài toán | Input | Output | Trạng thái |
|---|---|---|---|
| Isolated classification | Clip đã cắt | 13 lớp | Đang làm |
| Continuous spotting | Video dài chưa cắt | đoạn thời gian + nhãn | Chưa triển khai đầy đủ |
| Open-set/unknown | Video có cử chỉ lạ | known/unknown + nhãn | Chưa triển khai |

Một classifier 13 lớp luôn buộc phải chọn một trong 13 nhãn. Nó không có lớp “không biết”.

### 1.3 Hai protocol cùng tồn tại trong repo

| Protocol | Số lớp | Dữ liệu/đường chạy | Có dùng trực tiếp checkpoint của nhau? |
|---|---:|---|---|
| Bộ nội bộ | 27 | `final/data/Input`, fusion V4 | Không |
| IPN Hand | 13 | `external_data/ipn_processed` hoặc Drive | Không |

Classifier head 13 output không thể đánh giá manifest 27 lớp, và ngược lại. **Neural-only evaluator** hiện kiểm tra output dimension và ID↔name contract để dừng sớm thay vì cho ra con số sai. Evaluator V2–V5 cũ không có mức kiểm tra semantic mapping tương đương; khi dùng nó phải tự bảo đảm checkpoint, manifest và template directory cùng protocol.

### 1.4 Bốn workstream phải phân biệt

| Workstream | Trạng thái | Bằng chứng hiện có | Cách được dùng trong handbook |
|---|---|---|---|
| Packaged V4, 27 lớp | Đã có train/test/demo wrappers | checkpoint/artifact ngoài repo | Kiến trúc fusion học sâu tham chiếu |
| IPN neural-only, 13 lớp | Đang là baseline chính | ba seed validation, test đã được xem một phần | Nhánh cần đạt gate từng lớp |
| IPN V4 13 lớp lịch sử | Đã từng chạy exploratory một seed | test Macro-F1 V4 0,705; same-recipe neural 0,701 | Bằng chứng lịch sử, không phải final holdout |
| IPN rule-only V2 | Có full run cũ và matcher đang sửa tiếp | full-run Macro-F1 rất thấp | Nhánh symbolic cần cải thiện trước fusion |

V4 được parameterize bởi `--num-classes`; nó không bị giới hạn về mặt code ở 27 lớp. Tuy nhiên packaged workflow chính và checkpoint 27 lớp không thể dùng trực tiếp cho IPN 13 lớp. Thí nghiệm IPN V4 lịch sử đã chạy, nhưng chỉ một seed, dùng test đã mở và mức tăng 0,004 Macro-F1 so với đối chứng chưa chứng minh template giúp ổn định.

---

## 2. Bản đồ kiến trúc ở mức cao

```text
Video IPN gốc + annotation
        │
        ▼
Cắt clip theo [t_start, t_end]
        │
        ▼
RTMW-L: frame → keypoint + confidence
        │
        ▼
Chọn 52 keypoint và normalize theo hai vai
        │
        ├──────────── Neural path ─────────────────────────────┐
        │                                                       │
        │  resample/pad 64 frame → tensor [64,52,3]             │
        │  → motion features → projection → temporal encoder    │
        │  → pooling → classifier → 13 logits                  │
        │                                                       │
        └──────────── Symbolic path ────────────────────────────┤
                                                                │
           predicate → event/phase → instance graph             │
           → so khớp 13 template YAML → symbolic score          │
                                                                │
                                ┌───────────────────────────────┘
                                ▼
                  Neural-only hoặc neural + symbolic
                                │
                                ▼
                  nhãn, confidence, evidence, report
```

Hiện tại neural-only là nhánh có kết quả đầy đủ nhất trên IPN. Symbolic IPN V2 đang ở giai đoạn kiểm tra rule, graph và temporal matcher. Mục tiêu template cải thiện 10–15 điểm phần trăm cho lớp khó là **mục tiêu nghiên cứu**, chưa phải kết quả đã chứng minh.

---

## 3. Từ annotation đến clip

### 3.1 Dữ liệu IPN gốc

Đường dẫn thường dùng:

```text
external_data/ipn_full/
├── videos/*.avi
└── annotations/
    ├── Annot_List.txt
    ├── Annot_TrainList.txt
    ├── Annot_TestList.txt
    ├── Video_TrainList.txt
    ├── Video_TestList.txt
    ├── classIdx.txt
    └── metadata.csv
```

Một annotation có dạng logic:

```text
video,label,id,t_start,t_end,frames
```

Frame annotation là 1-based và đoạn `[t_start, t_end]` là inclusive. Số frame kỳ vọng:

```text
frames = t_end - t_start + 1
```

### 3.2 Vì sao phải audit video trước khi cắt?

AVI có thể báo số frame trong header khác số frame OpenCV giải mã tuần tự. Annotation cũng có thể trỏ quá frame cuối thực sự giải mã được. Vì vậy project có:

- `scripts/audit_ipn_data.py`;
- `scripts/verify_ipn_annotation_frames.py`;
- báo cáo audit dưới `external_data/ipn_audit/` khi artifact tồn tại.

“Frame count mismatch” không đồng nghĩa video hoàn toàn hỏng. Nó có nghĩa phải thận trọng khi seek/cắt và phải xác nhận output thực tế.

### 3.3 Split theo subject

`scripts/create_ipn_manifest.py` tạo mapping 0–12, loại `D0X` và chia train/validation theo subject. Protocol hiện dùng:

| Split | Số clip | Số subject |
|---|---:|---:|
| Train | 2.443 | 29 |
| Validation | 589 | 7 |
| Test | 1.101 | 13 |

Subject `1CM42_3` được loại theo cấu hình hiện tại. Train, validation và test phải không giao nhau về subject. Đây là điều quan trọng hơn chỉ kiểm tra tên file không trùng: cùng một người xuất hiện ở train và validation có thể gây subject leakage.

### 3.4 Cắt clip

`scripts/cut_ipn_clips.py`:

1. đổi `t_start` 1-based thành index OpenCV `t_start - 1`;
2. decode tuần tự đến frame đầu để tránh seek lệch keyframe;
3. ghi AVI MJPG;
4. nếu thiếu tail frame thì thử direct seek;
5. mở lại clip và kiểm tra số frame giải mã.

Output:

```text
external_data/ipn_processed/clips/<split>/<label>/<subject>/<clip>.avi
```

File có sẵn được skip trừ khi bật overwrite. “Đã tồn tại” không tự động đồng nghĩa “đã được kiểm tra với code mới”.

---

## 4. Từ frame ảnh đến 52 keypoint

### 4.1 RTMW-L làm gì?

Entrypoint batch là `keypoint_extractor/run_rtmw_splits_dual.py`. MMPose/RTMW-L nhận frame ảnh và trả whole-body keypoint, tối đa 133 điểm theo schema COCO WholeBody.

Project chỉ giữ 52 điểm cần cho cử chỉ:

- một số điểm mặt/thân;
- vai, khuỷu, cổ tay;
- gốc bàn tay;
- các khớp ngón của hai bàn tay.

Mỗi keypoint:

```text
(x_pixel, y_pixel, confidence)
```

Nếu có nhiều người, extractor chọn người có mean keypoint score cao nhất **ở từng frame**. Không có tracker duy trì identity xuyên suốt clip, nên cảnh nhiều người có nguy cơ đổi người giữa các frame.

### 4.2 JSON keypoint

Cấu trúc rút gọn:

```json
{
  "frame_id": 0,
  "instances": [{
    "keypoints_by_name": {
      "left_wrist": {"xy": [123.4, 210.8], "score": 0.95}
    }
  }]
}
```

Frame không phát hiện được người có `instances=[]`. Frame ID của extractor bắt đầu từ 0.

### 4.3 Mirror và trái/phải

Extractor có logic swap tên trái/phải theo cấu hình. Normalize không tự swap. Nếu dữ liệu nguồn và tùy chọn mirror không khớp, các cử chỉ trái/phải có thể bị đảo nhãn ngữ nghĩa dù tensor vẫn có shape hợp lệ.

---

## 5. Normalize: biến pixel thành tọa độ tương đối

Code: `final/code/src_neurosymbolic/pose/normalize_keypoints.py`.

Với hai vai hợp lệ:

```text
origin = (left_shoulder + right_shoulder) / 2
scale  = distance(left_shoulder, right_shoulder)
p_norm = (p_xy - origin) / scale
```

### 5.1 Tác dụng

- Trừ origin: giảm ảnh hưởng người đứng lệch trái/phải trong ảnh.
- Chia shoulder width: giảm ảnh hưởng người đứng gần/xa camera.
- Giữ confidence: model biết điểm nào đáng tin.

### 5.2 Nó không giải quyết được gì?

- rotation lớn;
- perspective mạnh;
- self-occlusion;
- pose estimator gán sai tay;
- hai vai bị mất hoặc sai;
- identity switching khi có nhiều người.

Nếu score của một trong hai vai dưới ngưỡng hoặc khoảng cách vai bằng 0, tọa độ normalized của frame được đưa về 0. Confidence vẫn được giữ theo implementation hiện tại.

Điểm tinh tế: “normalization-invalid” không đồng nghĩa “neural-mask-invalid”. Tensor không lưu cờ `valid` của normalize; neural model tự coi frame hợp lệ nếu **bất kỳ confidence nào > 0**. Vì confidence được giữ, một frame có hai vai không hợp lệ nhưng các điểm khác còn score vẫn có thể đi qua encoder và pooling với `x/y=0`. Đây là khoảng trống implementation cần nhớ khi phân tích frame mask.

---

## 6. Chuẩn hóa thời gian và tensor cache

### 6.1 Vì sao cần đúng 64 frame?

Batch neural cần các mẫu có cùng chiều dài. `keypoints_to_tensor()` trong `final/code/src_neurosymbolic/models/keypoint_dataset.py` biến clip thành:

```text
[T, K, C] = [64, 52, 3]
```

Trong đó:

- `T=64`: số frame;
- `K=52`: số keypoint;
- `C=3`: normalized x, normalized y, confidence.

### 6.2 Ba trường hợp

```text
Số frame = 64  → giữ nguyên
Số frame > 64  → lấy 64 index phân bố đều
Số frame < 64  → lặp frame cuối cho đủ 64
```

Đây là sampling + repetition, không phải nội suy quỹ đạo.

Hệ quả:

- một click rất ngắn có thể bị bỏ nếu rơi giữa hai index được lấy;
- clip ngắn có static tail giả do lặp frame cuối;
- tốc độ động tác sau resampling có thể thay đổi;
- số lần lặp có thể khó học nếu temporal representation quá thô.

### 6.3 Cache `.pt`

`final/code/src_neurosymbolic/precompute_keypoint_tensors.py` lưu tensor dưới:

```text
keypoint_tensor_cache/T64/<split>/<label>/<subject>/<clip>.pt
```

Neural-only loader chỉ đọc cache; nó không normalize JSON lại trong lúc train. Nếu thay:

- keypoint extractor;
- mirror policy;
- normalization;
- thứ tự 52 keypoint;
- `NUM_FRAMES`;

thì phải tạo cache version mới. Nếu không, code mới vẫn train trên tensor cũ.

### 6.4 Manifest neural-only

Mỗi dòng split file có ba trường, parser lấy trường đầu làm source và trường cuối làm integer label:

```text
external_data/.../keypoints/train/G01/...json 0 2
```

`final/code/neural_only_experiments/dataset.py`:

- resolve cache theo split và label;
- ưu tiên layout split-scoped;
- từ chối nhiều cache trùng tên thay vì chọn ngẫu nhiên;
- kiểm tra tensor 3 chiều, đúng `T`, không NaN/Inf;
- trả `keypoints`, `label`, `path`.

---

## 7. Neural-only: đi qua model từng bước

Code trung tâm: `final/code/neural_only_experiments/models.py`.

Với batch size `B`:

```text
Input                        [B,64,52,3]
Thêm velocity                [B,64,52,5]
Flatten theo từng frame      [B,64,260]
Input projection             [B,64,128]
Temporal encoder             [B,64,128]
Pooling                      [B,128]
Classifier                   [B,13]
```

### 7.1 Frame validity

Một frame được xem là hợp lệ nếu có ít nhất một keypoint có confidence > 0:

```text
frame_valid[b,t] = max_k confidence[b,t,k] > 0
```

Mask này hiện được dùng ở pooling. Transformer encoder không nhận padding mask, vì vậy token của frame invalid vẫn đi qua self-attention trước khi bị loại ở masked pooling.

### 7.2 Motion feature: velocity

Mặc định model nối thêm vận tốc `(dx,dy)`:

```text
velocity[t] = xy[t] - xy[t-1]
velocity[0] = 0
```

Nếu keypoint không hợp lệ ở frame hiện tại hoặc trước đó, velocity của điểm đó được đưa về 0.

Vì input gốc có `(x,y,score)` và velocity có `(dx,dy)`, mỗi keypoint có 5 feature:

```text
x, y, confidence, dx, dy
```

52 điểm tạo `52 × 5 = 260` giá trị trên mỗi frame. Motion feature không phải một model phụ; nó là feature engineering nằm ngay trước projection.

### 7.3 Input projection

```text
Linear(260 → 128)
→ LayerNorm(128)
→ GELU
→ Dropout
```

Mục tiêu là biến vector keypoint thô của từng frame thành token `d_model=128` mà temporal encoder có thể xử lý.

### 7.4 Positional encoding

Transformer không tự biết token nào là frame 1 hay frame 64. Project cộng sinusoidal positional encoding:

```text
PE(pos,2i)   = sin(pos / 10000^(2i/d))
PE(pos,2i+1) = cos(pos / 10000^(2i/d))
```

Nó không có parameter được học và không được lưu như weight persistent. Nó cung cấp tín hiệu về vị trí/thứ tự frame.

### 7.5 Self-attention trực giác và công thức

Mỗi token sinh ba vector:

```text
Q = XWq   (query: tôi đang tìm gì?)
K = XWk   (key: tôi có thông tin gì?)
V = XWv   (value: nội dung sẽ truyền đi)
```

Attention:

```text
Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k))V
```

Một frame có thể đọc thông tin từ tất cả frame khác. Điều này phù hợp với cử chỉ động vì frame hiện tại có thể cần so sánh với tư thế đầu hoặc cuối.

`heads=4` nghĩa là có bốn không gian attention song song. Mỗi head có thể học quan hệ khác nhau, nhưng không được phép diễn giải cứng rằng một head chắc chắn là “tay trái” hay “click”.

### 7.6 Transformer block hiện tại

Mỗi layer dùng:

- multi-head self-attention;
- feed-forward width `4 × d_model`;
- GELU;
- dropout;
- pre-norm (`norm_first=True`);
- residual connection do PyTorch encoder layer thực hiện.

Cấu hình Colab chính:

```text
d_model = 128
layers  = 3
heads   = 4
dropout = 0.15
```

Output vẫn là 64 token `[B,64,128]`. Để phân loại cả clip, cần bước pooling.

---

## 8. Pooling: nó ở đâu và tại sao bây giờ mới được nhắc tới?

Pooling nằm trong `NeuralOnlyClassifier.forward()` ở `models.py`. Nó đã luôn tồn tại vì encoder trả một token cho mỗi frame, còn classifier cần một vector cho cả clip.

Nó trở nên nổi bật trong thí nghiệm hiện tại vì các lỗi chính liên quan đến **toàn bộ diễn biến thời gian**: một click so với hai click, zoom vào so với zoom ra.

### 8.1 Mean pooling

```text
z = (h₁ + h₂ + ... + h₆₄) / 64
```

Code giữ mode này để tương thích checkpoint cũ và làm ablation.

Ưu điểm:

- đơn giản;
- không thêm parameter;
- dùng toàn bộ frame.

Nhược điểm:

- tính cả frame invalid;
- sự kiện ngắn bị pha loãng;
- không học được frame nào quan trọng hơn.

### 8.2 Masked-mean pooling

Chỉ lấy trung bình frame hợp lệ:

```text
z = Σ valid[t]·h[t] / max(Σ valid[t], 1)
```

Ưu điểm:

- không để frame rỗng ảnh hưởng trực tiếp vào trung bình;
- dùng toàn chuỗi hợp lệ;
- có thể ổn định hơn khi detection bị mất frame.

Nhược điểm:

- vẫn nén toàn bộ thứ tự thành một trung bình;
- hai chuỗi khác thứ tự có thể cho vector gần giống;
- sự kiện click ngắn vẫn có thể bị pha loãng.

### 8.3 Attention pooling

Mỗi token nhận một score học được:

```text
s_t = W₂ tanh(W₁h_t)
α_t = softmax(s_t)
z   = Σ α_t h_t
```

Frame invalid bị mask trước softmax. Nếu toàn bộ frame invalid, code đưa score về 0 để softmax hữu hạn thay vì NaN.

Ưu điểm:

- học cách tập trung vào frame giàu tín hiệu;
- tránh pha loãng một khoảnh khắc quan trọng;
- thường tốt khi clip có nhiều đoạn đứng yên.

Nhược điểm:

- có thể chỉ tập trung một nhịp click nổi bật;
- double click có thể bị nhìn như single click;
- attention weight không tự động là explanation nhân quả.

### 8.4 Pooling không giống self-attention

```text
Self-attention: frame trao đổi thông tin với frame khác trong encoder.
Pooling:        gộp tất cả token cuối thành một vector cho classifier.
```

Đổi pooling không bỏ Transformer. Nó chỉ đổi cách đọc output Transformer.

### 8.5 Pooling được lưu ở đâu?

CLI có `--pooling {mean,masked_mean,attention}`. Giá trị được lưu trong `checkpoint["args"]`. Evaluator đọc lại để dựng đúng model. Checkpoint cũ thiếu field này fallback về `mean` để giữ tương thích lịch sử.

---

## 9. Các backbone khác

Mọi backbone nhận `[B,64,128]` và trả `[B,64,128]`, sau đó dùng cùng pooling/classifier.

### 9.1 TCN

TCN dùng Conv1d theo thời gian với dilation `1,2,4,...`:

```text
Conv1d kernel 3, dilation d
→ GELU → Dropout
→ Conv1d kernel 1 → Dropout
→ residual add
```

Ưu điểm: tính cục bộ, nhanh, receptive field tăng theo dilation.

Rủi ro: quan hệ rất xa phụ thuộc số layer/receptive field.

### 9.2 BiGRU

GRU đọc chuỗi hai chiều thời gian. Hidden mỗi chiều khoảng `d_model/2`, ghép lại thành `d_model`.

Ưu điểm: inductive bias tuần tự mạnh.

Rủi ro: khó song song hơn Transformer, có thể quên chi tiết ở chuỗi dài.

### 9.3 MLP-Mixer

Xen kẽ:

- token mixing trên trục thời gian;
- channel mixing trên trục feature.

Nó yêu cầu đúng số frame cấu hình. Đây là baseline không dùng attention/recurrent/temporal convolution theo nghĩa truyền thống.

### 9.4 Tại sao run hiện tại dùng Transformer?

Transformer là cấu hình được chọn cho baseline IPN hiện tại, không có nghĩa luôn tốt nhất trong mọi dữ liệu. Muốn tuyên bố tốt hơn TCN/BiGRU/Mixer phải chạy protocol so sánh công bằng cùng split, seed và tuning budget.

---

## 10. Classifier, logits và xác suất

Sau pooling:

```text
LayerNorm(128)
→ Linear(128→128)
→ GELU
→ Dropout
→ Linear(128→13)
```

Output là logits `[B,13]`, chưa phải xác suất. Softmax chỉ cần khi hiển thị probability:

```text
p_c = exp(logit_c) / Σ_j exp(logit_j)
```

Prediction:

```text
ŷ = argmax_c logit_c
```

Softmax 0.9 không đảm bảo xác suất đã được calibration đúng. Model có thể overconfident.

---

## 11. Training: từng cơ chế và tác dụng

Code: `final/code/neural_only_experiments/train_neural_only.py`.

### 11.1 Seed

Seed điều khiển:

- khởi tạo weight;
- thứ tự shuffle;
- augmentation ngẫu nhiên;
- dropout và các phép random của PyTorch.

Ba seed `42, 123, 3407` là ba lần train độc lập cùng cấu hình. Chúng không phải ba kiến trúc.

### 11.2 Augmentation là gì?

Augmentation tạo biến thể trong lúc train; validation/test không bị augment.

#### Coordinate jitter

```text
x' = x + εx
y' = y + εy
ε ~ Normal(0, jitter_std²)
```

Chỉ thêm nhiễu lên keypoint hợp lệ. Nó mô phỏng sai số pose estimator.

#### Frame masking

Với xác suất cấu hình trên mỗi frame, toàn bộ frame được đưa về 0. Nó mô phỏng mất detection và buộc model không phụ thuộc tuyệt đối vào một frame.

Rủi ro: click là sự kiện ngắn. Che đúng frame chứa nhịp thứ hai có thể biến double click thành tín hiệu giống single click.

#### Keypoint dropout

Một số keypoint được đưa về 0. Nó mô phỏng occlusion/mất khớp. Rủi ro là bỏ đúng đầu ngón quan trọng cho click/zoom.

### 11.3 Class weight

Với số mẫu lớp `c` là `n_c`:

```text
w_c = (mean_count / n_c)^power
```

Sau đó weight được normalize để mean bằng 1.

- `power=0`: tắt class weight;
- `power=0.5`: căn bậc hai của inverse frequency;
- `power=1`: inverse frequency đầy đủ.

Class weight giúp nhóm G ít mẫu hơn B0A/B0B. Nó không trực tiếp tách G01 khỏi G08 vì hai lớp này có số mẫu gần bằng nhau.

### 11.4 Cross-entropy và label smoothing

Cross-entropy cơ bản:

```text
L = -log p_y
```

Label smoothing 0.05 không đặt target đúng thành 1 tuyệt đối; một phần nhỏ probability target được phân cho lớp khác. Mục tiêu là giảm overconfidence và regularize.

Quá nhiều smoothing có thể làm giảm khả năng phân biệt các cặp vốn rất giống nhau.

### 11.5 Optimizer AdamW

AdamW dùng learning rate thích nghi theo moment gradient và tách weight decay khỏi gradient update. Default script:

```text
lr = 3e-4
weight_decay = 1e-2
```

### 11.6 Gradient clipping

Sau backward:

```text
clip_grad_norm_(parameters, 1.0)
```

Nếu norm gradient quá lớn, toàn bộ gradient được scale xuống. Nó giảm nguy cơ update bùng nổ, đặc biệt với model tuần tự.

### 11.7 Scheduler

`ReduceLROnPlateau(mode="max")` theo dõi metric validation được chọn. Nếu metric không cải thiện đủ số epoch:

```text
lr_new = lr_old × factor
```

Notebook hiện dùng checkpoint selection theo `macro_f1`, nên scheduler cũng nhìn Macro-F1.

### 11.8 Early stopping

Nếu validation metric không vượt best quá `min_delta` trong `patience` epoch, train dừng. Best checkpoint không nhất thiết là epoch cuối.

### 11.9 Best và last checkpoint

```text
best.pt → epoch có selected validation metric tốt nhất
last.pt → trạng thái epoch vừa chạy cuối cùng
```

Checkpoint chứa:

- model state;
- toàn bộ CLI args;
- label ID↔name;
- epoch và validation metrics;
- best epoch/value;
- early-stopping state tóm tắt.

Nó chưa phải full optimizer-resume checkpoint: optimizer và scheduler state không được lưu trong neural-only script hiện tại.

---

## 12. Ba ablation đã cấu hình

Ba run đều dùng Transformer và seed 42 để sàng lọc rẻ trước khi chạy đủ ba seed. Notebook mặc định để `RUN_ABLATIONS=False`; việc cấu hình tồn tại không tự chứng minh run đã bắt đầu hoặc hoàn thành. Trạng thái thật phải đọc `ABLATION_COMPLETE.json` trên Drive.

| Run | Pooling | Epoch/patience | Frame mask | Keypoint dropout | Câu hỏi |
|---|---|---|---:|---:|---|
| A | attention | 80/12 | 0.05 | 0.03 | 50 epoch có quá ngắn? |
| B | masked mean | 80/12 | 0.05 | 0.03 | Attention pooling có bỏ mất nhịp lặp? |
| C | masked mean | 80/12 | 0.00 | 0.02 | Augmentation có xóa sự kiện ngắn? |

Thiết kế so sánh:

```text
baseline cũ ↔ A : tác dụng của thời gian train
A ↔ B            : tác dụng của pooling
B ↔ C            : tác dụng của augmentation
```

Không được kết luận chỉ từ accuracy tổng. Tiêu chí hiện tại:

```text
Primary:   per-class validation F1 của mọi lớp ≥ 0.65
Secondary: validation Macro-F1 không kém baseline đáng kể
Safety:    accuracy không giảm mạnh và lỗi không chỉ chuyển từ lớp này sang lớp kia
```

Cấu hình thắng seed 42 phải chạy lại seed `42,123,3407` trước khi gọi là ổn định.

---

## 13. Validation, test và trạng thái test hiện tại

### 13.1 Vai trò ba split

```text
Train      → học weight
Validation → chọn kiến trúc/hyperparameter/checkpoint
Test       → ước lượng cuối sau khi quyết định đã khóa
```

Nếu xem test, sửa model rồi test lại cùng tập, test đã trở thành development data. Kết quả vẫn hữu ích để debug nhưng không còn là đánh giá holdout độc lập.

Trong project hiện tại, IPN test **đã được xem** qua các thí nghiệm neural-only, IPN V4 lịch sử và full symbolic run. Vì vậy tên `locked_test` mô tả cơ chế chống ghi đè/chạy nhầm của notebook, không còn có nghĩa split này chưa từng được con người quan sát. Tất cả kết quả trên test hiện tại phải được gọi là exploratory/development evidence. Muốn có tuyên bố cuối khách quan cần holdout mới hoặc protocol đánh giá được thiết kế lại trước khi tiếp tục tune.

### 13.2 Metric

Với từng lớp:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2·precision·recall / (precision + recall)
```

Trong report hiện tại, `accuracy` ở mỗi lớp bằng recall:

```text
per-class accuracy = correct_of_class / support_of_class
```

Macro metric là trung bình không trọng số qua các lớp. Vì vậy lớp 52 mẫu có tiếng nói ngang lớp 265 mẫu.

### 13.3 Vì sao accuracy có thể gây hiểu nhầm?

B0A/B0B nhiều mẫu và dễ hơn. Một model rất tốt ở hai lớp này có thể có accuracy tổng cao dù G01/G02 yếu. Macro-F1 và per-class F1 phù hợp hơn với mục tiêu “mọi lớp đạt ít nhất 0.65”.

### 13.4 Confusion matrix

Convention của file hiện tại:

```text
row    = true class
column = predicted class
```

Ô `(G01,G08)=26` nghĩa là 26 mẫu G01 bị dự đoán thành G08. Đây là ví dụ lấy từ test đã được mở, chỉ dùng để chẩn đoán chứ không phải căn cứ độc lập để chọn cấu hình rồi công bố lại trên cùng test.

Các cặp lỗi mang ý nghĩa cơ chế:

- G01↔G08: một click và double click một ngón;
- G02↔G09: một click và double click hai ngón;
- G10↔G11: chiều zoom;
- G03–G06: chiều chuyển động tay.

### 13.5 Artifact evaluator

`test_neural_only.py` tạo:

- metrics JSON;
- `predictions.csv`;
- confusion matrix;
- per-class support/correct/predicted/precision/recall/F1;
- top-k probability và confidence;
- timing inference có phạm vi xác định.

Notebook locked-test bổ sung:

- `test_per_class_metrics.csv`;
- `test_confusion_matrix.csv`;
- `test_report.md`;
- `provenance.json`;
- `LOCKED_TEST_COMPLETE.json`.

Completion marker được viết cuối cùng để phân biệt output hoàn chỉnh và thư mục bị ngắt giữa chừng.

---

## 14. Symbolic pipeline: từ hình học đến khái niệm

Symbolic path không học trực tiếp từ pixel. Nó biến keypoint thành fact có tên, sau đó thành event/graph và so với template.

```text
normalized keypoint
→ predicate
→ atomic event / phase event
→ instance graph
→ template matching
→ score + evidence
```

### 14.1 Predicate là gì?

Predicate là một phát biểu cục bộ có thời gian, subject và confidence, ví dụ:

```text
hand_moves_down(right_hand, frames 12–18, confidence 0.84)
index_extended(right_hand, frames 5–30, confidence 0.91)
pinch_distance_increasing(right_hand, frames 20–40, confidence 0.76)
```

Code: `final/code/src_neurosymbolic/predicates/extract_predicates.py`.

Nhóm predicate:

- spatial/hand shape;
- hand motion;
- finger motion và pinch/spread;
- two-hand coordination;
- sweep;
- một số joint-angle relation.

Các threshold là heuristic, không được neural training học. Chúng dễ giải thích nhưng nhạy với scale, noise, tốc độ và pose quality.

### 14.2 Event là gì?

Predicate frame/window được gộp thành event có khoảng thời gian:

```text
Event {
  start_frame,
  end_frame,
  confidence,
  subject,
  concepts,
  evidence_keypoints
}
```

Legacy builder: `events/build_events.py`.

Phase-aware V2: `events/build_events_phase_v2.py`.

V2 cố giữ atomic event ID và phase để không đếm một chuyển động dài bị chia nhỏ thành hai click độc lập.

### 14.3 Instance graph

Instance graph mô tả **video đang quan sát**:

- node Video;
- node Event;
- node BodyPart;
- concept node;
- Evidence node;
- cạnh `HAS_EVENT`, `INVOLVES`, `HAS_MOTION`, `BEFORE`, `OVERLAPS`, ...

Code V2: `kg/build_instance_graph_phase_v2.py`.

### 14.4 Template graph

Template mô tả **điều một lớp mong đợi**, không phải video cụ thể. Source IPN:

```text
configs/ipn/gesture_templates.yaml
```

Mỗi requirement có mức:

| Mức | Weight mặc định | Ý nghĩa |
|---|---:|---|
| required | 1.0 | bắt buộc |
| preferred | 0.6 | nên có |
| optional | 0.3 | hỗ trợ |
| negative | -1.0 | xuất hiện thì phạt |

Template V2 còn mô tả occurrence và thứ tự:

- G01/G02: một occurrence;
- G08/G09: ít nhất hai occurrence;
- G07: chuỗi phase đóng/mở lặp;
- G10/G11: pinch distance thay đổi ngược chiều.

### 14.5 Rule-only matching

Matcher biến instance graph thành fact/event candidate, rồi tính:

- positive requirement score;
- required-hit ratio;
- negative penalty;
- occurrence satisfaction;
- temporal order score;
- matched/missing evidence.

Output không chỉ có nhãn mà còn có dấu vết: requirement nào match, thiếu, bị phạt và evidence frame nào hỗ trợ.

### 14.6 Trạng thái thực tế của symbolic IPN

Symbolic V2 chưa đạt mức có thể tuyên bố tăng accuracy:

- artifact graph đầy đủ chưa đồng bộ hoàn chỉnh trong bản local;
- có graph schema cũ;
- smoke cohort 26 mẫu không thay thế full validation;
- predicate có thể mâu thuẫn trong cùng clip;
- active phase/dominant hand còn là vấn đề;
- temporal matcher đang có thay đổi local chưa commit, phải phân biệt với baseline GitHub.

Do đó template +10–15 điểm là target cần chứng minh bằng validation và ablation, không phải giả định được phép cộng vào neural score.

Một full-run snapshot đã tồn tại trong notebook kết quả local `results/experiments/ipn13/notebooks/IPN_Symbolic_V2_Colab_res.ipynb` (file này hiện chưa được track trên GitHub):

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Train | 0,0671 | 0,0612 |
| Validation | 0,0560 | 0,0504 |
| Test | 0,0736 | 0,0656 |

Các số này xác nhận rule-only còn rất xa gate, nhưng phải xem là snapshot lịch sử của attempt/config hash trong notebook. Matcher local hiện có thay đổi chưa commit, nên không được gán kết quả trên cho code đang sửa hoặc dùng test snapshot để tune tiếp.

---

## 15. Hai kiểu “neuro-symbolic” trong project

### 15.1 Learned fusion V4 và lần chạy IPN lịch sử

Code dưới `final/code/src_neurosymbolic/models/cross_attention_v4.py` được dùng trong packaged protocol 27 lớp và cũng có thể parameterize cho 13 lớp:

- neural keypoint tokens;
- instance graph tokens;
- template graph làm class-specific query;
- dual cross-attention;
- structured requirement scorer;
- confidence-aware fusion giữa neural logits và `kg_logits`.

Điểm cần nhớ: `kg_logits` V4 không phải pure graph-only. Head này đã attend vào cả neural memory và graph memory.

Báo cáo lịch sử IPN một seed ghi test Macro-F1:

```text
Neural của người 2:       0,688
Same-recipe neural head:  0,701
V4 final fusion:          0,705
```

Mức chênh `0,004` giữa V4 và đối chứng cùng recipe rất nhỏ; V4 tốt hơn ở một số lớp nhưng kém hơn ở lớp khác. Do test đã được dùng và artifact lịch sử không tái hiện đầy đủ từ trạng thái hiện tại, kết quả này chỉ chứng minh “đã từng triển khai IPN V4”, không chứng minh fusion hiện tại chắc chắn có lợi.

### 15.2 Hướng IPN 13 lớp hiện tại

Quy trình đúng đang là:

1. xây neural-only baseline đáng tin;
2. xây rule-only matcher độc lập;
3. đo từng nhánh;
4. chỉ fusion khi biết template giúp case nào;
5. chạy causal ablation để chứng minh đóng góp.

Một hướng fusion bảo thủ cho IPN:

```text
Neural top-2 không thuộc cặp khó → giữ neural
Neural phân vân G01/G08       → template đếm occurrence
Neural phân vân G02/G09       → template đếm occurrence
Neural phân vân G10/G11       → template kiểm tra trend pinch
```

Template chỉ override khi:

- neural margin thấp;
- symbolic score đủ cao;
- evidence temporal đủ rõ;
- rule không mâu thuẫn.

Cách này giảm nguy cơ symbolic rule yếu phá các lớp neural đã tốt.

---

## 16. Explainability: giải thích được đến đâu?

Project có thể xuất:

- neural top-k probability;
- attention weight theo frame;
- predicate/event timeline;
- template requirement matched/missing;
- evidence keypoint/frame;
- branch prediction và fusion gate trong V4.

Nhưng cần phân biệt:

```text
Trace:        code cho biết đã dùng token/fact nào.
Explanation: mô tả hợp lý về quyết định.
Causal proof: xóa/chỉnh evidence rồi prediction thay đổi đúng như dự đoán.
```

Attention map và matched requirement là evidence trace, chưa tự động là causal proof. Causal XAI cần intervention/ablation:

- xóa graph;
- shuffle temporal edges;
- thay template;
- che frame được coi quan trọng;
- đo thay đổi logit/prediction.

---

## 17. Mục tiêu “mọi lớp ≥ 0.65” nên đo thế nào?

### 17.1 Gate đề xuất

```text
Primary gate:
  mean per-class F1 qua 3 seed ≥ 0.65 cho mọi lớp

Stability gate:
  không seed nào collapse nghiêm trọng ở cặp khó

Aggregate gate:
  Macro-F1 và accuracy không giảm ngoài trade-off đã chấp nhận
```

Nếu chỉ lấy seed tốt nhất, ta có nguy cơ cherry-pick khởi tạo thuận lợi.

### 17.2 Template improvement nên báo cáo gì?

Không chỉ báo “+10%”. Phải ghi rõ:

- tăng 10–15 **điểm phần trăm** hay tăng tương đối;
- trên F1 lớp khó hay accuracy tổng;
- trung bình bao nhiêu seed;
- neural configuration nào;
- threshold/gate được chọn trên split nào.

Target thực tế hơn:

```text
Hard-class F1: +0.10 đến +0.15
Macro-F1:      tăng vừa phải, không làm lớp dễ giảm mạnh
Overall acc:   có thể tăng ít hơn vì B0A/B0B đã gần trần
```

### 17.3 Ablation tối thiểu cho template

| Run | Mục đích |
|---|---|
| Neural-only | baseline |
| Rule-only | template đứng một mình tốt đến đâu |
| Neural + template | hệ thống đề xuất |
| Neural + shuffled template | kiểm tra template semantics thật sự có ích |
| Neural + empty graph | kiểm tra graph đóng góp |
| Oracle pair resolver | upper bound cho ba cặp khó |

Nếu fusion tốt hơn neural nhưng shuffled template cũng tốt tương đương, chưa thể kết luận tri thức template là nguyên nhân.

---

## 18. Colab, Google Drive và vòng đời file

### 18.1 `/content` là tạm

```text
/content/dynamic_hand_gesture_recognition
```

là clone code trên máy runtime Colab. Reset runtime có thể xóa toàn bộ. Notebook clone lại từ GitHub khi cần.

### 18.2 Drive là persistent

```text
/content/drive/MyDrive/ipn-extract-data/
├── train.txt
├── val.txt
├── test.txt
├── keypoint_tensor_cache/T64/
└── outputs/
```

Checkpoint và report phải nằm trên Drive nếu muốn sống qua runtime reset.

### 18.3 Output neural chính

```text
outputs/neural_improved_13cls/
├── transformer_seed42/
├── transformer_seed123/
└── transformer_seed3407/
```

Mỗi seed có:

```text
best.pt
last.pt
history.json
train.log
command.json
locked_test/
```

### 18.4 Output ablation

```text
outputs/neural_ablation_A_attention/transformer_seed42/
outputs/neural_ablation_B_masked_mean/transformer_seed42/
outputs/neural_ablation_C_masked_mean_light_dropout/transformer_seed42/
```

Marker:

```text
ABLATION_TRAIN_COMPLETE.json
ABLATION_COMPLETE.json
```

Nếu run dở, notebook mặc định không ghi đè. Khi cho phép restart, thư mục dở được rename với suffix timestamp thay vì bị xóa.

### 18.5 Không dùng Run all tùy tiện

Notebook chứa cả regular train, ablation và locked test. Sau reconnect nên chạy có chọn lọc:

```text
Mount Drive
→ Config
→ Clone/pull code
→ Resolve data/cache
→ Dataset sanity check
→ đúng cell cần chạy
```

---

## 19. Cách đọc artifact để không hiểu sai

### `history.json`

Một row mỗi epoch:

- train loss/accuracy;
- validation loss/accuracy/Macro-F1;
- LR;
- improved hay không;
- best epoch/value đến thời điểm đó;
- epochs without improvement.

### `command.json`

Lệnh thực sự đã chạy. Đây là nguồn kiểm tra cấu hình tốt hơn việc nhớ lại notebook đã từng được sửa thế nào.

### `best.pt`

Weight + metadata của best validation epoch, không phải “best test”.

### `predictions.csv`

Một row mỗi sample, gồm true/predicted label, confidence và top-k. Dùng để tìm case cụ thể cần xem video/keypoint.

### `provenance.json`

Ghi commit, hash checkpoint, hash manifest test, timestamp, command và artifact path. Nó giúp biết hai report có thực sự cùng dữ liệu/code không.

---

## 20. Failure modes và cách chẩn đoán

| Hiện tượng | Nguyên nhân có thể | Kiểm tra |
|---|---|---|
| Accuracy cao, Macro-F1 thấp | lớp nhiều mẫu lấn át | per-class F1, support |
| G01→G08 nhiều | không giữ số lần click | confusion, pooling, temporal sampling |
| G10↔G11 | không giữ chiều trend | velocity, phase, pinch trend |
| Train tăng, val giảm | overfit | learning curve, augmentation |
| Val dao động giữa seed | optimization không ổn định | mean±std 3 seed |
| Tensor toàn 0 | clip không có detection/keypoint hoặc cache lỗi | tensor quality report |
| `x/y` bằng 0 nhưng confidence còn | shoulder anchors làm normalize invalid | kiểm tra raw shoulder score và frame validity |
| Trái/phải đảo | mirror policy | xem clip + skeleton |
| FileNotFound cache | layout/split/T không khớp | manifest source và `T64` path |
| Label mapping mismatch | checkpoint/manifest khác protocol | checkpoint label_names |
| Colab reconnect không thấy biến | RAM đã reset | chạy lại init cells, đọc Drive |
| Output dir tồn tại nhưng thiếu marker | run bị ngắt | marker/state recovery |
| Template score cao nhưng sai | threshold/fact quá phổ biến | matched evidence, negative rules |
| Fusion kém neural | gate/template chưa đáng tin | branch-wise metric + ablation |

---

## 21. Bản đồ code cần nhớ

### Data/keypoint

| Tác vụ | File |
|---|---|
| Audit IPN | `scripts/audit_ipn_data.py` |
| Tạo manifest | `scripts/create_ipn_manifest.py` |
| Cắt clip | `scripts/cut_ipn_clips.py` |
| Extract RTMW | `keypoint_extractor/run_rtmw_splits_dual.py` |
| Normalize | `final/code/src_neurosymbolic/pose/normalize_keypoints.py` |
| Tensor 64 frame | `final/code/src_neurosymbolic/models/keypoint_dataset.py` |
| Precompute cache | `final/code/src_neurosymbolic/precompute_keypoint_tensors.py` |

### Neural-only

| Tác vụ | File |
|---|---|
| Dataset/cache resolver | `final/code/neural_only_experiments/dataset.py` |
| Model/backbone/pooling | `final/code/neural_only_experiments/models.py` |
| Train/loss/augmentation | `final/code/neural_only_experiments/train_neural_only.py` |
| Evaluation/report | `final/code/neural_only_experiments/test_neural_only.py` |
| Colab | `notebooks/Neural_Only_Improved_Colab.ipynb` |
| Regression tests | `tests/test_neural_only_improvements.py` |

### Symbolic/fusion

| Tác vụ | File |
|---|---|
| Predicate | `final/code/src_neurosymbolic/predicates/extract_predicates.py` |
| Event V2 | `final/code/src_neurosymbolic/events/build_events_phase_v2.py` |
| Instance graph V2 | `final/code/src_neurosymbolic/kg/build_instance_graph_phase_v2.py` |
| Template builder | `final/code/src_neurosymbolic/kg/build_template_graph.py` |
| Matcher | `final/code/src_neurosymbolic/kg/graph_matching.py` |
| IPN templates | `configs/ipn/gesture_templates.yaml` |
| Learned V4 fusion | `final/code/src_neurosymbolic/models/cross_attention_v4.py` |
| Explain demo | `final/code/src_neurosymbolic/explain_demo_prediction.py` |

---

## 22. Những điều code hiện tại chưa làm

- Không phát hiện gesture boundary trong video liên tục.
- Không có unknown/reject class.
- Không đảm bảo probability calibration.
- Không tracking identity nhiều người xuyên frame.
- Không tự bảo đảm subject-disjoint nếu người dùng đưa manifest khác; neural trainer chỉ kiểm tra path overlap train/val.
- Không lưu optimizer/scheduler để resume neural training đúng trạng thái.
- Không chứng minh attention là explanation nhân quả.
- Không chứng minh template tăng 10–15 điểm trên full IPN.
- Không cho phép cộng trực tiếp metric 13 lớp với kết quả fusion 27 lớp.
- Không bảo đảm cùng seed cho kết quả bit-for-bit trên mọi GPU/backend.

---

## 23. Lộ trình kỹ thuật từ trạng thái hiện tại

### Giai đoạn A — Neural gate

1. Hoàn thành A/B/C trên validation seed 42.
2. So per-class F1 và confusion của sáu lớp khó.
3. Nếu chưa đạt, ưu tiên representation giữ temporal count, ví dụ temporal-pyramid pooling, thay vì chỉ tăng epoch.
4. Chạy cấu hình thắng trên ba seed.
5. Gate: mọi lớp đạt mean validation F1 ≥ 0.65.

### Giai đoạn B — Symbolic gate

1. Chốt schema/provenance graph.
2. Regenerate graph full train/val.
3. Fix contradictory predicate, active phase và dominant hand.
4. Rule-only full validation.
5. Gate riêng cho G01/G02/G08/G09/G10/G11.

### Giai đoạn C — Fusion gate

1. Calibrate neural margin và symbolic confidence trên validation.
2. Chỉ override trong cặp khó.
3. So neural-only, rule-only và fused.
4. Chạy shuffled/empty-template ablation.
5. Không dùng test để chọn threshold.

### Giai đoạn D — Final evaluation

Nếu test hiện tại đã được dùng để phân tích/tune, cần một holdout mới hoặc protocol đánh giá mới để có tuyên bố khách quan. Báo cáo phải kèm seed, mean±std, per-class metric, confusion matrix, commit và manifest hash.

---

## 24. Câu hỏi tự kiểm tra

1. Vì sao input là `[B,64,52,3]` nhưng projection có thể nhận 260 feature/frame?
2. Self-attention và attention pooling khác nhau ở đâu?
3. Vì sao class weight không giải quyết trực tiếp G01↔G08?
4. Vì sao frame masking có thể hại double click?
5. Vì sao `best.pt` tốt hơn `last.pt` cho báo cáo?
6. Vì sao accuracy cao không đảm bảo mọi lớp tốt?
7. Vì sao test đã xem không nên tiếp tục dùng để tune?
8. Predicate, event, instance graph và template graph khác nhau thế nào?
9. Vì sao `kg_logits` V4 không được gọi là pure symbolic?
10. Cần ablation nào để chứng minh template semantics thực sự có ích?

Đáp án ngắn:

1. Model nối thêm `(dx,dy)`: `52×5=260`.
2. Self-attention trao đổi thông tin giữa token; pooling gộp token thành vector clip.
3. G01/G08 có frequency gần nhau; lỗi là temporal semantics.
4. Nó có thể che nhịp click thứ hai.
5. `best.pt` được chọn bằng validation metric tốt nhất.
6. Lớp nhiều mẫu có thể chi phối accuracy.
7. Vì quyết định sau khi xem test gây test leakage.
8. Fact cục bộ → đoạn thời gian → graph quan sát → graph kỳ vọng theo lớp.
9. Head đó đọc cả neural và graph memory.
10. Neural/rule/fused + shuffled template + empty graph, chọn trên validation.

---

## 25. Tóm tắt một trang

Project hiện nhận **clip cử chỉ đã cắt**, không phải video liên tục. RTMW-L biến frame thành keypoint; 52 điểm được normalize theo vai và resample/pad thành tensor `[64,52,3]`. Neural model thêm velocity để thành `[64,52,5]`, project mỗi frame lên 128 chiều, dùng Transformer/TCN/BiGRU/MLP-Mixer để học quan hệ thời gian, rồi pooling thành một vector và phân loại 13 lớp.

Pooling không phải khái niệm mới tự sinh ra: nó luôn cần thiết để đổi 64 token thành một vector. Thí nghiệm A/B/C làm nó nổi bật vì lỗi hiện tại nằm ở single/double click và chiều zoom. Attention pooling chọn frame nổi bật; masked mean dùng toàn bộ frame hợp lệ. Augmentation chỉ chạy khi train, giúp chịu noise nhưng có thể xóa sự kiện ngắn.

Model được train bằng weighted, label-smoothed cross-entropy; AdamW; scheduler theo validation; early stopping; checkpoint chọn theo validation Macro-F1. Seed là nguồn random cho một lần train, không phải cấu hình. Mục tiêu neural hiện tại là per-class F1 ≥0.65 qua ba seed.

Symbolic path biến keypoint thành predicate, event, graph rồi so với template YAML. Nó có khả năng biểu diễn số lần click và chiều zoom rõ ràng hơn neural black-box, nhưng full IPN rule-only vẫn chưa đạt gate. Mục tiêu template +10–15 điểm cho lớp khó phải được chứng minh bằng validation và causal ablation. Fusion an toàn nên chỉ dùng template để phân xử khi neural không chắc trong các cặp đã biết, thay vì override mọi prediction.

Cuối cùng, `/content` của Colab là tạm; Drive mới là nơi lưu cache/checkpoint/report. Mọi kết quả đáng tin phải gắn với config, seed, commit, manifest hash và quy tắc chọn checkpoint. Không trộn metric 13 lớp với 27 lớp, không chọn cấu hình bằng test, và không gọi attention/template trace là causal explanation nếu chưa có intervention.
