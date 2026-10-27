# Giải thích các thay đổi và thử nghiệm: làm gì, vì sao, kết quả ra sao

Tài liệu này viết để **đọc hiểu**, không phải để báo cáo. Mỗi thay đổi được giải thích theo cùng một khung: nó là gì, vì sao nghĩ ra, ảnh hưởng thế nào, và **đã có bằng chứng đo được chưa**. Số liệu lấy từ các note trong `docs/` và các file kết quả; bản tóm tắt số liệu nằm ở `BAO_CAO_TIEN_DO_2026-10-24_VI.md`.

## 0. Cách đọc: ba loại bằng chứng

Trong tài liệu này mỗi nhận định được gắn một nhãn, vì "nghĩ rằng nó giúp" và "đã đo thấy nó giúp" là hai việc khác nhau:

- **[Đã đo]**: có thí nghiệm hoặc số liệu cụ thể trong repo.
- **[Lập luận]**: lý do hợp lý từ thiết kế hoặc từ tài liệu, nhưng chưa có thí nghiệm tách riêng yếu tố đó.
- **[Chưa biết]**: chưa đo và cũng chưa có căn cứ chắc.

Điều quan trọng nhất cần nhớ khi trả lời câu hỏi "vì sao tốt hơn": **chỉ ablation A/B/C là thí nghiệm có đối chứng**. Hầu hết thay đổi code còn lại (vận tốc, class weight, label smoothing…) được đưa vào vì lập luận, và repo chưa có thí nghiệm tách riêng từng cái.

## 1. Bức tranh lớn

### 1.1 Bài toán khó ở chỗ nào

IPN-13 có 13 lớp: B0A, B0B (hai tư thế "không có cử chỉ", nhiều clip: mỗi lớp 140 clip validation) và G01–G11 (11 cử chỉ thật, mỗi lớp 28 clip validation). Phần khó là các cặp giống nhau về hình dạng và chỉ khác nhau về **thời gian**:

| Cặp | Khác nhau ở | Vì sao khó |
|---|---|---|
| G01 và G08 | Click một ngón một lần và hai lần | Phải đếm số lần click; click ngắn |
| G02 và G09 | Click hai ngón một lần và hai lần | Như trên, thêm phụ thuộc vào ngón giữa |
| G10 và G11 | Khoảng cách pinch tăng và giảm | Cùng động tác, ngược chiều thời gian |
| G03–G06 | Hướng chuyển động tay | Phụ thuộc hướng và quy ước camera |

Một mô hình nhìn "ảnh tĩnh trung bình" của clip sẽ thấy G01 và G08 gần như giống nhau, vì cả hai đều có ngón trỏ đi xuống. Muốn phân biệt phải hiểu **trình tự và số lần**.

### 1.2 Hai nhánh và ý tưởng neuro-symbolic

- **Nhánh neural:** keypoint → Transformer → xác suất 13 lớp. Giỏi nhận ra mẫu chung nhưng là hộp đen, dễ nhầm các cặp kể trên.
- **Nhánh symbolic:** keypoint → predicate ("ngón trỏ đang đi xuống") → event (khoảng thời gian) → instance graph (đồ thị mô tả clip) → so với **template** (mô tả điều mỗi lớp mong đợi, ví dụ G08 = đúng hai lần ngón trỏ đi xuống). Giải thích được nhưng phụ thuộc luật viết tay và nhạy với nhiễu.

Ý tưởng cả dự án: lỗi của neural tập trung ở các cặp *đếm* và *chiều*, đúng là thứ mà luật dễ biểu diễn, nên để template hỗ trợ neural ở những chỗ đó thay vì thay thế neural.

### 1.3 Quy tắc kỷ luật xuyên suốt

Train học trọng số; **validation** để chọn cấu hình và checkpoint; **test** chỉ để đo cuối. Nếu nhìn test rồi sửa mô hình, test trở thành dữ liệu phát triển và con số không còn là ước lượng khách quan. Cả DATE-27 lẫn IPN đã bị xem test trong lúc phát triển (`HANDBOOK…` mục 13.1), nên số test hiện có chỉ là bằng chứng exploratory. Đây là lý do nhiều thay đổi ở phần 6 (quy trình) tồn tại.

## 2. Thay đổi phía neural

### 2.1 Thêm vận tốc (velocity)

**Là gì.** Mỗi keypoint trước đây có 3 giá trị `(x, y, confidence)`. Giờ thêm `(dx, dy)` = vị trí frame hiện tại trừ frame trước. Mỗi frame có `52 × 5 = 260` giá trị thay vì `52 × 3 = 156`. Frame đầu có vận tốc 0.

**Vì sao nghĩ ra.** Cử chỉ động được định nghĩa bởi *chuyển động*. Với tọa độ thô, mạng phải tự suy ra "ngón này đang đi xuống" bằng cách so sánh hai token frame khác nhau. Đưa vận tốc vào thì thông tin đó nằm sẵn trong từng token. Handbook mô tả đây là feature engineering ngay trước lớp projection (`HANDBOOK…` mục 7.2).

**Chi tiết quan trọng: không tạo vận tốc giả.** Nếu keypoint bị mất ở một frame (confidence 0, tọa độ không đáng tin), hiệu hai vị trí sẽ là một bước nhảy lớn hoàn toàn do mất detection chứ không do tay di chuyển. Vì vậy code đặt vận tốc bằng 0 khi một trong hai frame không hợp lệ.

**Ảnh hưởng.**
- Input projection đổi từ `156 → d_model` sang `260 → d_model`; checkpoint cũ (3 đặc trưng) không nạp được vào model mới và ngược lại. Có cờ `--disable-motion-features` để tái tạo kiến trúc cũ.
- Adapter neural của V4 ban đầu chỉ có projection `52×3`, nên không nạp được checkpoint có vận tốc như C và lần chạy IPN-13 đầu phải dùng checkpoint không vận tốc. Từ 2026-10-24 adapter có tùy chọn `--neural-motion-features` (mặc định tắt) nên nạp được C trực tiếp (mục 5.4).

**Bằng chứng.** [Đã đo, một seed, trên test đã bị xem nên chỉ là exploratory]. Lần chạy IPN-13 (2026-10-24) vô tình tạo ra đúng phép đối chứng cần có: neural C (có vận tốc) và neural cùng siêu tham số nhưng tắt vận tốc, đánh giá trên cùng 1.101 clip test: **84,65% so với 82,56% accuracy, 77,24% so với 73,65% macro-F1**. Hơn nữa phần lớn chênh lệch nằm ở cử chỉ double click: recall G08 là 71,2% so với 36,5%, G09 là 65,4% so với 50,0%. Điều này khớp lý do thêm vận tốc (đếm click cần thông tin chuyển động). Hạn chế: một seed và đã dùng test. **Hiệu chỉnh 2026-10-27:** sweep ba seed cho thấy test accuracy của C dao động 81,0% đến 84,65% giữa các seed (seed 42 là seed tốt nhất), nên chênh lệch 2,1 điểm này nằm trong dao động seed và chưa thể coi là đã chứng minh; cần ba seed không vận tốc để kết luận.

### 2.2 Pooling: gộp 64 token thành một vector cho cả clip

**Vấn đề.** Transformer trả về 64 token (một cho mỗi frame), nhưng classifier cần *một* vector cho cả clip. Cách gộp (pooling) quyết định thông tin nào sống sót.

| Chế độ | Công thức trực giác | Điểm mạnh | Điểm yếu |
|---|---|---|---|
| `mean` (cũ) | Trung bình đều 64 frame | Đơn giản | Tính cả frame rỗng; sự kiện ngắn bị pha loãng |
| `masked_mean` | Trung bình chỉ các frame hợp lệ | Frame rỗng không làm nhiễu | Vẫn nén toàn bộ thứ tự thành một trung bình |
| `attention` | Mỗi frame nhận điểm học được, lấy trung bình có trọng số | Tập trung vào frame giàu tín hiệu | Có thể chỉ chú ý một nhịp click nổi bật, nhìn double click như single click |

**Vì sao thay đổi.** Lỗi chính của neural liên quan đến diễn biến thời gian (một click hay hai), nên cách đọc output Transformer trở thành ứng viên nghi ngờ (`HANDBOOK…` mục 8). Handbook nêu cả hai chiều lập luận: attention *có thể* giúp (tránh pha loãng) nhưng *cũng có thể* hại (chỉ thấy một nhịp). Chính sự không chắc chắn này là lý do chạy ablation A so với B.

**Ảnh hưởng.** Pooling nằm trong `checkpoint["args"]`; evaluator đọc lại để dựng đúng model. Checkpoint cũ thiếu trường này mặc định về `mean`. **Lưu ý V4:** adapter neural của V4 luôn dùng mean pooling khi tạo neural memory, nên với checkpoint `masked_mean` thì hình dạng tương thích nhưng ngữ nghĩa pooling khác. Vì vậy V4 không dùng để thay thế neural-only, và neural-only luôn được đánh giá riêng.

**Bằng chứng.** [Đã đo] một phần: A (attention) so với B (masked_mean) cho thấy attention tốt hơn khi cả hai dùng cùng augmentation. Xem mục 3.

### 2.3 Augmentation: tạo biến thể khi train

Augmentation chỉ áp dụng lúc train, không áp dụng cho validation hay test. Ba loại:

- **Jitter tọa độ (0,005):** cộng nhiễu nhỏ vào keypoint hợp lệ, mô phỏng sai số của bộ trích pose.
- **Frame masking:** với một xác suất, đưa cả một frame về 0, mô phỏng mất detection và ép mô hình không dựa vào một frame duy nhất.
- **Keypoint dropout:** đưa vài keypoint về 0, mô phỏng che khuất.

**Vì sao nghĩ ra.** Dataset nhỏ (mỗi cử chỉ G chỉ khoảng 116 clip train, còn B0A và B0B khoảng 583 clip mỗi lớp), mô hình dễ học thuộc. Augmentation là cách phổ biến để chống overfit.

**Cái giá, và đây là mấu chốt của thí nghiệm C.** Handbook cảnh báo ngay từ khi viết (`HANDBOOK…` mục 11.2): click là sự kiện *ngắn*. Che đúng frame chứa nhịp click thứ hai có thể biến double click thành tín hiệu giống single click, tức là biến G08 thành G01. Augmentation thường hữu ích nhưng ở đây có thể phá chính tín hiệu cần học.

**Bằng chứng.** [Đã đo] (một seed): bỏ frame masking (C) cho kết quả tốt hơn B. Xem mục 3.

### 2.4 Class weight và label smoothing

- **Class-weighted cross entropy** (`power = 0,5`): lớp ít mẫu bị phạt nặng hơn khi sai. Trọng số là căn bậc hai của nghịch đảo tần suất. B0A/B0B có hơn 140 clip còn mỗi G chỉ 28, nên nếu không cân bằng, mô hình có thể "lười" học các G. Handbook nhấn mạnh class weight **không** tự tách G01 khỏi G08 vì hai lớp này có số mẫu gần bằng nhau; nó chỉ giúp nhóm G so với nhóm B.
- **Label smoothing 0,05:** mục tiêu không còn là 1,0 tuyệt đối cho lớp đúng mà để lại một phần rất nhỏ cho lớp khác, giảm tự tin quá mức và regularize. Nếu quá mạnh có thể làm mờ khác biệt giữa các cặp vốn rất giống nhau.

**Bằng chứng.** [Lập luận]; chưa có ablation tách riêng.

### 2.5 Chọn checkpoint theo macro-F1 thay vì accuracy

**Ví dụ vì sao.** B0A và B0B chiếm 281/589 = 47,7% validation. Một mô hình đoán đúng hoàn toàn hai lớp này và sai hết 11 cử chỉ vẫn đạt accuracy 47,7% nhưng macro-F1 chỉ khoảng 15%. Accuracy bị hai lớp dễ chi phối. Macro-F1 là trung bình F1 của 13 lớp, mỗi lớp có tiếng nói bằng nhau, nên phản ánh đúng mục tiêu "mọi cử chỉ đều phải tốt".

**Ảnh hưởng.** `best.pt`, scheduler giảm learning rate và early stopping đều theo macro-F1. Vì checkpoint "tốt nhất" có thể là epoch giữa chừng, `best.pt` khác `last.pt`. Với DATE-27, neural-only chỉ còn file `last.pt` nên không chắc là checkpoint chọn theo validation, đó là lý do phải gọi nó là baseline legacy.

### 2.6 Kiểm tra dữ liệu: "dừng ngay" thay vì "đoán"

Trước đây dataset có thể tìm tensor theo *tên file* trên toàn bộ cache. Nếu hai lớp hay hai split có file trùng tên, nó lấy nhầm âm thầm, dẫn đến nhãn và dữ liệu không khớp hoặc rò rỉ train/validation mà không báo lỗi. Giờ: manifest sai format, split rỗng, cache trùng tên, tensor sai shape hay chứa NaN đều dừng ngay và báo rõ; train và validation không được trùng source; checkpoint 13 lớp không được chạy trên manifest 27 lớp.

Đây không làm tăng accuracy; nó làm cho **mọi con số về sau đáng tin**. Một kết quả cao do dữ liệu bị lấy nhầm còn tệ hơn một kết quả thấp.

## 3. Ba thử nghiệm A, B, C

### 3.1 Vì sao có thí nghiệm này

Sau khi cải tiến neural, validation đạt 87,10% accuracy, 81,09% macro-F1 (bản "cũ" trong bảng, d128, 3 lớp, 50 epoch). Mục tiêu đặt trước (`HANDBOOK…` mục 12 và 17): **mọi lớp có F1 validation từ 0,65**. Các lớp click vẫn dưới ngưỡng, nên cần biết *nguyên nhân* trước khi nhờ symbolic can thiệp. Playbook viết rõ: nếu neural vẫn có lớp dưới 0,65, phải xác định đó là thiếu tín hiệu, mất cân bằng, segmentation hay pooling; "template không nên che một neural baseline chưa ổn định".

Ba run chạy trên Colab, cùng Transformer, cùng seed 42, 80 epoch, patience 12, chỉ train trên train split và chọn theo validation, **không gọi test**.

### 3.2 Thiết kế: mỗi lần đổi một yếu tố

```text
cũ ↔ A : kích thước mô hình và thời gian train
A  ↔ B : cách pooling
B  ↔ C : mức augmentation
```

| Run | Pooling | Frame masking | Keypoint dropout | Câu hỏi |
|---|---|---:|---:|---|
| A | attention | 0,05 | 0,03 | Mô hình lớn hơn (d192, 4 lớp, 8 head) và train lâu hơn có tốt hơn? Attention có chọn được frame quan trọng? |
| B | masked_mean | 0,05 | 0,03 | Pooling đơn giản có ổn định hơn? Attention có bỏ mất nhịp lặp? |
| C | masked_mean | 0,00 | 0,02 | Giảm augmentation có giữ được tín hiệu động tác nhỏ? |

Nguyên tắc: nếu đổi ba thứ cùng lúc thì khi kết quả tốt lên sẽ không biết nhờ cái nào. Đổi một thứ mỗi bước (B giữ nguyên mọi thứ của A trừ pooling; C giữ nguyên mọi thứ của B trừ augmentation) cho phép gán sự thay đổi cho đúng yếu tố, với điều kiện là các bước độc lập, một điều kiện không hoàn toàn đúng ở đây (xem 3.5).

### 3.3 Kết quả [Đã đo, validation, một seed]

| Run | Accuracy | Macro-F1 | F1 thấp nhất của một lớp | Best epoch |
|---|---:|---:|---:|---:|
| Cũ | 87,10% | 81,09% | | 49 |
| A | 87,61% | 81,57% | 0,531 | 34 |
| B | 85,91% | 79,96% | 0,478 | 17 |
| **C** | **89,13%** | **83,76%** | **0,628** | 48 |

Cách đọc từng bước:

1. **Cũ so với A (+0,5 điểm accuracy):** mô hình lớn hơn và train lâu hơn cải thiện rất ít. Nghĩa là sức chứa của mô hình **không phải** nút thắt chính.
2. **A so với B (A hơn 1,7 điểm accuracy, 1,6 điểm macro-F1):** khi dùng augmentation như nhau, attention pooling tốt hơn masked_mean. Trực giác "chọn frame quan trọng" có vẻ đúng ở điều kiện này.
3. **B so với C (C hơn 3,2 điểm accuracy, 3,8 điểm macro-F1):** chỉ đổi augmentation (bỏ frame masking, giảm keypoint dropout) mà cải thiện lớn nhất cả chuỗi. F1 của các lớp click tăng: G01 từ 0,48 lên 0,63, G08 từ 0,62 lên 0,70. Đây đúng là hiện tượng mà cảnh báo ở mục 2.3 dự đoán.
4. **C là cấu hình thắng** theo tiêu chí đặt trước: macro-F1 cao nhất, rồi F1 thấp nhất của một lớp. F1 thấp nhất nhảy từ 0,53 (A) lên 0,63 (C), tức lớp tệ nhất đã tốt lên rõ.

### 3.4 Nó giúp cải thiện gì

- Neural baseline IPN từ 87,6% lên 89,1% accuracy và từ 81,6% lên 83,8% macro-F1 (so với A).
- Chọn ra cấu hình neural để làm nền cho symbolic và cho V4.
- Chỉ ra **hướng nghi ngờ**: augmentation có thể phá tín hiệu ngắn, nên đáng thử các augmentation "an toàn" hơn với sự kiện ngắn.
- Định hướng cho phần symbolic: ba cặp lỗi còn lại của C (G01↔G08, G02↔G09, G10↔G11; xem số lỗi ở `IPN_TEMPLATE_AUTHORING_PLAYBOOK_VI.md`) là phạm vi mà template được phép can thiệp.

### 3.5 Vì sao cần dè dặt khi diễn giải

1. **Một seed.** Mỗi lớp cử chỉ chỉ có 28 clip validation; một clip sai làm F1 lớp đó lệch khoảng 3,5 điểm. Chênh lệch giữa các run ở từng lớp riêng lẻ nằm trong nhiễu của seed. Gate mà handbook đặt ra là trung bình qua 3 seed.
2. **Hai yếu tố trong bước C:** C vừa bỏ frame masking (0,05 → 0) vừa giảm keypoint dropout (0,03 → 0,02). Chưa biết yếu tố nào có tác dụng.
3. **Thiết kế không đủ ô:** attention thắng masked_mean ở điều kiện augmentation mạnh (A so với B), nhưng C (masked_mean, ít augmentation) thắng cả A. Câu hỏi còn bỏ ngỏ là **attention kết hợp với bỏ frame masking** sẽ ra sao. Có thể attention vốn tốt nhưng bị frame masking kìm hãm, và C thắng chỉ vì bỏ masking chứ không phải vì masked_mean. Đây là thí nghiệm bị thiếu rõ nhất (đề xuất ở mục 7).
4. **Early stopping nhiễu.** B dừng ở epoch 29 (best 17); C chạy 60 epoch (best 48). Với validation nhỏ, chuỗi epoch dao động, nên "best epoch" một phần là may rủi.
5. **Overfit.** C đạt train gần 99,9% trong khi validation 89,1%, khoảng cách lớn. Giả thuyết "ít augmentation hơn thì tốt hơn" vừa mâu thuẫn với trực giác thông thường (augmentation chống overfit) vừa phù hợp với riêng bài toán sự kiện ngắn; cần seed bổ sung để chắc.
6. **C chưa đạt mục tiêu đã đặt**: G01 (0,627), G02 (0,630), G09 (0,644) còn dưới 0,65.

## 4. Thay đổi phía symbolic

### 4.1 Vì sao symbolic cần sửa nhiều

Sau khi chạy symbolic V2 trên 589 clip validation, kết quả rất thấp: accuracy 20,03%, macro-F1 6,26%. Quan sát quan trọng nhất (`IPN_SYMBOLIC_POST_C_CHANGELOG_VI.md`):

- 250/589 clip **hòa điểm**, top-2 margin trung bình chỉ 0,012.
- 448/589 clip bị dự đoán thành B0A do quy tắc "hòa điểm thì chọn lớp đầu".
- Trong 11 cử chỉ G01–G11 (308 clip), chỉ đúng 9 clip.
- Các predicate đối nghịch (ví dụ pinch tăng và pinch giảm) cùng xuất hiện trên gần như mọi clip, nghĩa là predicate quá nhiễu.

Các thay đổi dưới đây đều là phản ứng với một triệu chứng cụ thể như vậy.

### 4.2 Lọc trend nhiễu (`trend_consistency`)

**Vấn đề.** Predicate hướng ("tay đi sang phải") chỉ kiểm tra điểm đầu và điểm cuối cửa sổ đủ xa nhau, bỏ qua việc ở giữa tay có thể đi tới đi lui.

**Cách làm.** `trend_consistency = |dịch chuyển ròng| / tổng biến thiên`. Ví dụ: tay đi phải 10, trái 9, phải 10 → dịch chuyển ròng 11, tổng biến thiên 29, tỉ lệ 0,38. Chuyển động thẳng cho tỉ lệ 1,0. Ngưỡng 0,6 loại những cửa sổ đảo chiều mạnh khỏi predicate hướng.

**Ảnh hưởng.** Ít predicate đối nghịch xuất hiện đồng thời. Giá trị ngưỡng được ghi vào config và provenance nên graph dựng bằng ngưỡng cũ không bị dùng nhầm.

**Bằng chứng.** [Đã đo] một phần: accuracy thô 20,03% → 21,73%, nhưng cải thiện nhỏ; ngưỡng 0,6 là lựa chọn ban đầu chưa được tối ưu.

### 4.3 Abstention: "không biết" cũng là một câu trả lời

**Vấn đề.** Khi hai template gần như bằng điểm, ép chọn một lớp (và mặc định về B0A) tạo ra 448 dự đoán B0A phần lớn là ngẫu nhiên, làm số liệu trông như symbolic đang "quyết định" trong khi nó không biết.

**Cách làm.** Chỉ chấp nhận dự đoán khi điểm tốt nhất từ 0,15 **và** khoảng cách top-1 và top-2 từ 0,02. Nếu không, đánh dấu abstain. Báo cáo bổ sung **coverage** (tỉ lệ clip được chấp nhận) và **selective accuracy** (độ chính xác chỉ trên clip được chấp nhận).

**Ảnh hưởng.** Hai con số này kể chuyện thật hơn accuracy: ví dụ coverage 21% và selective accuracy 31% cho biết symbolic chỉ dám trả lời 1/5 số clip và trong đó đúng chưa đến 1/3. Hai ngưỡng 0,15/0,02 là cấu hình an toàn ban đầu, chưa tối ưu.

### 4.4 Sequence nhiều pha

Cử chỉ như G07 gồm 4 pha (đóng, mở, đóng, mở). Matcher cũ có thể "ghép" một pha dùng lại cùng một chút bằng chứng cho nhiều bước, hoặc ghép các pha chồng lấn. Giờ yêu cầu các pha cùng một subject (một bàn tay), pha sau bắt đầu sau khi pha trước kết thúc và không dùng lại cùng occurrence. Mục đích: không cho G07 khớp giả.

### 4.5 Vai trò `active` và `return`

**Vấn đề.** Click gồm hai chiều chuyển động: nhấn xuống (hành động chính) rồi nhấc lên (trả về). Cả hai đều sinh predicate hướng, nên G01 (một click) và G08 (hai click) dễ bị lẫn: một click có thể bị đếm thành hai event hướng ngược nhau.

**Cách làm.** Run directional đầu là `active`, run đối hướng kế tiếp là `return`. Chuỗi A–B–A thành active–return–active, nên double click không mất lần nhấn thứ hai. Hai run đối hướng gần như đồng thời thì đánh dấu `ambiguous`.

**Bài học từ lần làm sai.** Bản đầu đặt `action_role: active` làm điều kiện **bắt buộc** trong template và gắn `ambiguous` cho cả hai run mỗi khi khoảng thời gian chồng nhau. Nhưng sliding window luôn chồng nhau tại điểm đổi chiều, nên 63,2% edge bị gắn ambiguous và template mất gần hết bằng chứng đúng. Kết quả [Đã đo]: macro-F1 giảm từ 8,7% xuống 6,1% và F1 của cả G01–G11 bằng 0, trong khi accuracy thô lại tăng nhẹ (22,24%) vì mô hình đoán B0A nhiều hơn. Nếu chỉ nhìn accuracy sẽ tưởng là cải thiện. Phát hiện được là nhờ xem macro-F1 và F1 từng lớp.

**Sửa.** Role không còn là điều kiện cứng; phân loại dựa trên thứ tự tâm interval (chỉ gần như đồng thời mới ambiguous); role chỉ dùng làm diagnostic và làm tín hiệu mềm (4.6). Rescore trên chính graph bị regression cho kết quả gần như quay lại mốc trước (accuracy 6,62% có abstention, trùng mốc 6,62%), xác nhận đúng nguyên nhân.

### 4.6 Soft role alignment

Thay vì "đúng role mới được tính", điểm template được nhân với một hệ số:

```text
hệ số = floor + (1 − floor) × tỉ lệ active      (floor = 0,85)
```

Ví dụ: tỉ lệ active 0 → hệ số 0,85; 0,5 → 0,925; 1 → 1,0. Role chỉ hạ điểm tối đa 15%, không xóa bằng chứng. Template cũ không khai báo policy mặc định floor 1,0, nên không đổi điểm số cũ (tương thích ngược). Giá trị 0,85 chọn bằng sweep trên validation (0,80–0,90): nó đồng hạng tốt nhất ở raw accuracy và tốt nhất về macro-F1 có ngưỡng và selective accuracy, **chưa phải hằng số được xác nhận trên test**.

**Kết quả** [Đã đo, rescore 589 graph validation]: accuracy thô 21,73% → 23,43%, macro-F1 8,68% → 9,01%, coverage 20,7% → 27,8%, selective accuracy 32,0% → 36,0%.

### 4.7 Chẩn đoán từng clip và audit theo cặp

**Chẩn đoán.** Mỗi clip validation xuất 13 điểm template, thứ hạng, hòa điểm, margin, lý do abstain và event khớp. Nhờ đó chỉnh template có căn cứ (xem clip nào sai vì sao) mà không cần động vào test.

**Audit theo cặp: phần dễ hiểu nhầm nhất.** Neural C đã đạt 89,13%, tức chỉ sai khoảng 64/589 clip. Nếu cho symbolic (đúng khoảng 23%) ghi đè neural tùy ý, nó sẽ phá nhiều dự đoán đúng hơn là sửa dự đoán sai. Vì vậy cơ chế được thiết kế rất thận trọng:

1. Chỉ xét khi **neural top-2** là một trong ba cặp G01/G08, G02/G09, G10/G11.
2. Chỉ xét khi neural không chắc (margin thấp) và symbolic có bằng chứng rõ trong **đúng cặp đó**.
3. Hòa điểm tuyệt đối luôn abstain.
4. Mặc định `audit_only`: *không đổi* dự đoán, chỉ đếm xem nếu đổi thì có bao nhiêu **lợi**, **hại**, và **net**.

**Kết quả** [Đã đo, validation]: ở artifact 6ef, 2 override được đề xuất, cả hai có hại (net −2); sau soft role còn 1 đề xuất, vẫn có hại. Quét neural margin 0,02–0,50 và symbolic margin 0,02–0,20 không tìm được cấu hình nào net dương. Kết luận đã chốt: **chưa bật override**. Đây là kết quả "tiêu cực" nhưng có giá trị: nó ngăn việc bật một cơ chế sẽ làm kết quả tệ đi.

### 4.8 Template: những gì được sửa và vì sao

Sau thí nghiệm A, template được sửa (commit `678040a`). Một điểm đáng nhớ trong playbook: **`negative_evidence` được để trống**, vì matcher kiểm tra bằng chứng phủ định trên *toàn clip* chứ không chỉ trong khoảng thời gian khớp; nếu bật, một dấu hiệu ngược chiều xuất hiện ở đâu đó trong clip sẽ phạt oan lớp đúng. Chỉ bật lại khi matcher giới hạn phủ định vào đúng interval.

Nguyên tắc sửa template (playbook mục 5), mỗi lần chỉ đổi **một giả thuyết**: lấy lỗi từ validation, xem clip thật, viết giả thuyết quan sát được, kiểm tra extractor có sinh được dấu hiệu đó, sửa, chạy lại toàn bộ validation, chỉ giữ nếu lớp mục tiêu cải thiện mà không hại lớp khác.

### 4.9 Kết luận chung về symbolic

Symbolic thuần luật chưa là classifier dùng được trên IPN (khoảng 23% accuracy thô, 9% macro-F1). Vai trò thực tế của nó hiện là **công cụ chẩn đoán và an toàn** (audit, abstention), cùng với hướng **learned `kg_logits`** (mục 5). Giá trị khoa học của vòng này nằm ở việc *chứng minh được* rằng bật fusion theo luật sẽ có hại, và phát hiện kịp lỗi hồi quy do role cứng.

## 5. Kiến trúc V4: neural, kg_logits và final

> **Đính chính (2026-10-24).** Bản đầu của mục này mô tả công thức trộn `neural + gate × (kg − neural.detach())` và gate khởi tạo 0,05. Đó là của lớp `TemplateQueryNeuroSymbolicModel` (bản V2 cũ, `cross_attention_v2.py`), **không phải V4**. V4 thật (`DualCrossAttentionV4NeuroSymbolicModel`, `cross_attention_v3.py` và `cross_attention_v4.py`) dùng `ConfidenceLogitFusion` như dưới đây. Mọi nhận định về "gate là hệ số của KG" và "tên `--initial-neural-gate` không khớp hành vi" đều sai và đã được gỡ bỏ.

### 5.1 Ba đầu ra khác nhau thế nào

Một lần forward cho ba nhóm logits:

| Đầu ra | Cách tính | Thấy gì |
|---|---|---|
| `neural_logits` | Classifier trên vector pooled của Transformer | Chỉ keypoint |
| `kg_logits` | 13 template query đọc **hai** memory bằng hai cross-attention riêng (neural memory từ token keypoint; graph memory từ instance graph), có gate theo token để chọn nguồn; rồi bộ chấm điểm requirement cho điểm từng lớp | Keypoint, graph và template |
| `logits` (final) | Trộn có điều kiện độ tin cậy của hai đầu trên | Không có thông tin riêng ngoài hai đầu và vài thống kê về chúng |

Công thức trộn của V4 (`ConfidenceLogitFusion`):

```text
neural_scaled = neural_logits / T_neural        # T học được
kg_scaled     = kg_logits     / T_kg
gate (mỗi lớp) = sigmoid( MLP[ vector pooled của neural,
                               đặc trưng KG của lớp đó,
                               độ tin cậy và entropy của neural và của KG ] )
final = gate × neural_scaled + (1 − gate) × kg_scaled
```

**Gate là trọng số của NEURAL.** Nó khởi tạo ở **0,8** (bias `log(0,8/0,2)`), và loss phụ `0,01 × (gate − 0,8)²` (`--initial-neural-gate`, mặc định 0,8) giữ nó gần mức đó: tên tham số khớp với hành vi. Hợp lý về thiết kế: lúc bắt đầu nhánh KG chưa học gì nên tin neural nhiều. Gate tăng lên 1 thì final là neural, giảm về 0 thì final là KG.

`gate_mean` trong `history.json` là gate trung bình theo mẫu và lớp, nghĩa là **trọng số trung bình dành cho neural**.

### 5.2 Vì sao thiết kế như vậy

- **Template làm "câu hỏi" (query):** thay vì để mạng tự học "lớp nào là gì", mỗi lớp có một template graph mô tả điều nó mong đợi. Template query đọc memory của clip để trả lời "bằng chứng của clip có khớp mô tả này không?". Đây là cách đưa tri thức có cấu trúc vào mạng học được.
- **Hai memory, hai cross-attention:** neural memory (token keypoint) giữ chi tiết hình học; graph memory (event) cho biết "có hai lần click". Gate theo token quyết định mỗi nút template lấy thông tin từ đâu.
- **Gate dựa trên độ tin cậy:** mạng biết khi nào neural chắc (entropy thấp) để tin neural, và khi nào KG chắc hơn.
- **`kg_logits` không phải symbolic thuần:** vì nó attend cả neural memory, một sự cải thiện của KG chưa chứng minh được tri thức template đóng góp. Cần ablation với template xáo trộn hoặc graph rỗng để chứng minh (mục 7).

### 5.3 Kết quả liên quan [Đã đo]

- **DATE-27** (1.229 clip test): neural head 94,63%, KG head 95,04%, final 94,87%. KG head sửa 9 lỗi, gây 4 lỗi mới (net +5). Final nằm giữa hai đầu, đúng với việc nó là tổ hợp lồi của hai logits, và thấp hơn neural-only độc lập 95,28%. Nghĩa là trên DATE-27, fusion **chưa** chứng minh lợi thế so với neural thuần.
- **IPN-13** (1.101 clip test, run 2026-10-24, V4 khởi tạo từ checkpoint không vận tốc): neural head 83,02%, KG head 82,29%, final 82,56% (macro-F1 74,40 / 73,22 / 73,84). Trên test, KG head sửa 17 lỗi nhưng gây 25 lỗi mới (net −8), nên KG **chưa giúp** trên IPN. Gate (trọng số của neural) trên validation là 0,82 ở epoch 1 và 0,65 ở epoch 17, tức final vẫn nghiêng về neural, và kết quả final bằng đúng kết quả của checkpoint khởi tạo (82,56%).
- **IPN lịch sử** (test macro-F1, một seed, đã dùng test): neural 0,688; neural cùng recipe 0,701; V4 final 0,705. Chênh 0,004 rất nhỏ.

### 5.4 Vấn đề vận tốc trong V4 (đã xử lý bằng code, 2026-10-24)

Trước đây adapter neural của V4 chỉ nhận 3 đặc trưng mỗi điểm, không nạp được checkpoint C (có vận tốc). Cách tạm là train thêm một checkpoint không vận tốc để khởi tạo V4. Lần chạy IPN-13 cho thấy cái giá của cách này [Đã đo, một seed]: neural không vận tốc đạt 82,56% accuracy / 73,65% macro-F1, còn C có vận tốc đạt 84,65% / 77,24% trên cùng test (+2,1 và +3,6 điểm); đặc biệt recall của G08 là 36,5% so với 71,2%. V4 khởi tạo từ bản không vận tốc nên bị chặn trên bởi mức này. (Lưu ý 2026-10-27: C ở đây là seed 42, seed tốt nhất trong ba seed; so với trung bình ba seed của C trên test (82,6%) thì V4 final (82,56%) không thấp hơn. Chênh lệch 2,1 điểm nằm trong dao động giữa các seed.)

Adapter neural của V4 nay có thêm các tùy chọn `--neural-motion-features`, `--neural-pooling` và `--neural-hand-features` (mặc định tắt, nên checkpoint cũ vẫn tái tạo đúng: 128/128 clip DATE-27 cho dự đoán giống hệt lần chạy đã đóng gói). Kiểm tra bằng checkpoint C thật: nạp vào V4 không lỗi và neural head của V4 cho đúng logits của C lúc nạp.

### 5.5 Neural có "đi vào" final hai lần không, và encoder được cập nhật thế nào

Câu hỏi hay gặp: `kg_logits` đã đọc token của neural, final lại trộn thêm `neural_logits`, vậy có phải neural được tính hai lần? Kết luận sau khi đọc code (`cross_attention_v3.py`, `cross_attention_v4.py`, `train_cross_attention_v4.py`):

**Về thông tin: đúng, hai đầu không độc lập.** `kg_logits` đọc token keypoint của encoder neural (qua `neural_projector`, không `detach`), và gate nhận cả vector pooled của neural. Encoder là **một module dùng chung** (`model.neural`) cho cả hai đầu.

**Về số học: không cộng hai lần.** Final là tổ hợp lồi của hai logits đã chia nhiệt độ, với trọng số gate theo lớp.

**Encoder được cập nhật bởi những loss nào.** Ở chế độ `--logit-branch final` (chế độ của lần chạy DATE-27 và của notebook IPN-13), tổng loss là:

```text
L = CE(final) + 0,3·CE(neural) + 0,3·CE(kg) + 0,05·BCE(requirement) + 0,01·(gate − 0,8)²
```

Gradient chảy vào encoder từ nhiều đường: `CE(neural)` trực tiếp qua classifier; `CE(final)` qua phần neural của tổ hợp lồi; `CE(kg)` và `BCE(requirement)` ngược qua decoder, tới memory, tới token của encoder; và gate (phụ thuộc vector pooled). Kết quả: encoder **bị kéo bởi cả loss neural lẫn loss template**.

**Đóng băng encoder chỉ xảy ra khi bạn yêu cầu.** `--freeze-neural-epochs` mặc định `0` và `--freeze-neural-for-kg` mặc định tắt. Với `final`, encoder được đóng băng khi `epoch ≤ freeze_neural_epochs`, tức với mặc định 0 thì **chưa bao giờ đóng băng**. Optimizer dùng hai nhóm learning rate: encoder 2e-4, phần còn lại 3e-4. Gradient clipping chung (norm tổng ≤ 1,0).

**Đã kiểm tra từ checkpoint của hai lần chạy:**

| Mục | DATE-27 (đã đóng gói) | IPN-13 (run 2026-10-24) |
|---|---|---|
| `logit_branch` | `final` | `final` |
| Đóng băng encoder | không | không |
| Khởi tạo neural | không có (train từ đầu) | checkpoint neural-only không vận tốc |
| Frame masking, keypoint dropout | 0,05 và 0,03 | 0 và 0,02 (cờ bạn thêm) |
| Early stop / scheduler patience | 5 / 2 | 5 / 2 |
| Epoch dừng, epoch được chọn | 24, 19 | 17, 12 |
| Validation neural / KG / final (epoch chọn) | 95,55 / 95,55 / 95,50 | xem history (dao động 80–86) |
| Gate (trọng số neural) trên validation | 0,64 (ep 1), 0,26 (ep 5), 0,33 (ep 10), 0,50 (ep 15), 0,55 (ep 19) | 0,82 (ep 1), 0,73 (ep 12), 0,65 (ep 17) |

Hệ quả cần nhớ:

- **Neural head của V4 không phải mô hình neural-only độc lập.** Nó cùng encoder với KG head và được train cùng, nên so "neural head" với "neural-only" không phải so "có hay không có KG".
- **Ba đầu ra gần bằng nhau** là dấu hiệu hai đầu học những thứ rất giống nhau do chung encoder, nên fusion có ít chỗ để bổ sung cho nhau.
- **Validation của V4 trên IPN dao động mạnh giữa các epoch** (khoảng 80% đến 86%), trong khi patience là 5 và learning rate bị giảm một nửa từ epoch 4; V4 dừng ở epoch 17. Đây là nhiễu chọn checkpoint chứ không phải tín hiệu.
- Điều mình **không** kiểm tra được: gate trên test, và gate theo từng lớp (history chỉ lưu trung bình).

**Về augmentation của V4 trên IPN (đã giải quyết).** Notebook ban đầu không truyền hai cờ augmentation nên V4 sẽ train với frame masking 0,05; bạn đã thêm `--frame-mask-probability 0 --keypoint-dropout-probability 0.02` trước khi chạy, và checkpoint xác nhận điều đó. Kết quả V4 vì thế đã dùng cùng augmentation với C.

## 6. Thay đổi quy trình và repo: vì sao

| Thay đổi | Vấn đề giải quyết |
|---|---|
| Tách `runs/`, `final/`, `experiments/` | Kết quả cũ, kết quả hợp lệ và thí nghiệm lẫn lộn, khó biết số nào báo cáo được |
| Wrapper test dùng `best.pt` mặc định; `best_test.pt` cần `-AllowTestSelected` | Chọn checkpoint theo test làm số bị thổi phồng; đặt hàng rào để không báo nhầm |
| `AGENTS.md`: mọi thay đổi có entry docs | Sau vài tuần không ai nhớ vì sao đổi; docs ghi lý do, tương thích và validation |
| Notebook ghim commit, kiểm hash, không ghi đè, mỗi run một ZIP | Tái lập được; không vô tình đè kết quả cũ |
| Không stage artifact/notebook kết quả | Repo không phình và không lẫn dữ liệu sinh ra |

Một ví dụ cụ thể cho thấy hàng rào có ích: `best_test.pt` đạt 96,18% trên DATE-27, cao hơn mọi checkpoint chọn theo validation (94,87%), nhưng vì được chọn *nhìn* vào test nên không thể báo cáo như kết quả khách quan. Và con số 96,18% này trùng với run 3 trong bảng của manuscript, nên cần kiểm tra cách chọn checkpoint của 10 run đó.

## 7. Những ý tưởng đã nghĩ ra để thử, và trạng thái

| Ý tưởng | Lý do | Trạng thái và kết quả |
|---|---|---|
| Thêm vận tốc | Chuyển động là bản chất cử chỉ động | Đã đo (1 seed, test): +2,1 điểm accuracy, +3,6 điểm macro-F1; G08 recall 36,5% lên 71,2% |
| Attention pooling | Tập trung vào frame giàu tín hiệu | Đã đo (A so với B): tốt hơn masked_mean khi augmentation mạnh |
| Mô hình lớn hơn, train lâu hơn | Sức chứa có phải nút thắt? | Đã đo (cũ so với A): chỉ +0,5 điểm accuracy; không phải nút thắt chính |
| Bỏ frame masking (C) | Click ngắn bị che mất | Đã đo: +3,2 điểm accuracy so với B; chưa tách khỏi keypoint dropout |
| Class weight, label smoothing | Mất cân bằng lớp; chống tự tin quá mức | Đã làm; chưa có ablation riêng |
| Template chỉ can thiệp ba cặp khó | Tránh symbolic yếu phá lớp neural đã tốt | Đã làm; override chưa có lợi nên tắt |
| Role cứng `action_role: active` | Phân biệt pha nhấn và pha nhả | **Thất bại**: F1 G01–G11 về 0; đã gỡ, thay bằng soft role |
| Soft role 0,85 | Giữ lợi ích role mà không xóa bằng chứng | Đã đo: coverage 20,7% → 27,8%, macro-F1 8,7% → 9,0% (rescore) |
| Abstention | Không ép dự đoán khi không biết | Đã làm; coverage và selective accuracy |
| `negative_evidence` để trống | Matcher phạt trên toàn clip, phạt oan | Đã làm; chờ matcher giới hạn theo interval |
| Train V4 learned `kg_logits` trên IPN | Template làm query học được thay vì luật cứng | **Notebook sẵn sàng; chưa có kết quả** |
| Train neural không vận tốc để khởi tạo V4 | V4 không nhận vận tốc | Đã đưa vào notebook; chưa chạy |

Các thí nghiệm nên làm tiếp (một số do tài liệu nêu, một số là đề xuất của tài liệu này):

1. **Ba seed cho C** (42, 123, 3407) để biết chênh lệch có vượt nhiễu không. *Do handbook nêu.*
2. **Ablation vận tốc với ba seed:** đã có một cặp so sánh (mục 2.1); cần ba seed để chắc. *Đề xuất.*
3. **Attention pooling kết hợp bỏ frame masking** (ô còn thiếu của bảng A/B/C), và **tách frame masking khỏi keypoint dropout**. Kết quả cho biết C thắng nhờ yếu tố nào. *Đề xuất, rút ra từ bảng A/B/C.*
4. **Ablation template** (handbook mục 17.3): neural + template xáo trộn, neural + graph rỗng, oracle giải cặp khó (cận trên). Nếu template xáo trộn cũng tốt ngang template thật thì chưa thể nói tri thức template là nguyên nhân. *Do handbook nêu.*
5. **Kiểm tra bias trên DATE-27 cũ:** chạy neural cải tiến và neural kiểu cũ (`--disable-motion-features --pooling mean --class-weight-power 0` …) trên cùng bộ 27 lớp. Nếu bản cải tiến cũng tốt hơn ở đó thì là cải thiện chung của pipeline; nếu chỉ tốt hơn trên IPN thì phải ablation từng thành phần. *Do báo cáo neural-only nêu.*
6. **Gate cho G07** (phase không chồng lấn, không tái dùng bằng chứng, khoảng cách thời gian hợp lý) trước khi cho phép override. *Do playbook nêu.*
7. **Chuyển giao 13 → 27 lớp** (nạp backbone, bỏ classifier, fine-tune) và báo riêng với training từ đầu. *Do báo cáo neural-only nêu.*
8. **Đưa vận tốc vào V4** để khỏi phải train checkpoint khởi tạo riêng. *Đã làm bằng code ngày 2026-10-24 (mục 5.4); chờ chạy trên Colab.*

## 8. Từ điển ngắn

- **Logits:** điểm thô cho mỗi lớp, chưa phải xác suất; softmax đổi thành xác suất.
- **Precision (P):** trong số clip *được đoán* là lớp X, bao nhiêu đúng. **Recall (R):** trong số clip *thực sự* là lớp X, bao nhiêu được đoán đúng. **F1:** trung bình điều hòa của P và R. Lớp có R cao do đoán quá nhiều vẫn có P thấp, nên cần xem cả hai.
- **Macro-F1:** trung bình F1 của 13 lớp, mỗi lớp trọng số bằng nhau; không bị B0A/B0B chi phối như accuracy.
- **Pooling:** gộp các token theo thời gian thành một vector cho cả clip.
- **Memory (trong cross-attention):** tập token mà template query được phép "đọc". V4 có hai nguồn: token keypoint (neural) và token event (graph).
- **Template:** mô tả điều một lớp mong đợi (bằng chứng bắt buộc, nên có, phạt, số lần xuất hiện, thứ tự). Không phải video cụ thể.
- **Instance graph:** đồ thị mô tả điều quan sát được trong một video.
- **Abstain / coverage / selective accuracy:** từ chối trả lời / tỉ lệ clip được trả lời / độ chính xác chỉ trên số được trả lời.
- **Override:** việc để symbolic ghi đè dự đoán neural; ở đây tắt mặc định.
- **Validation-selected / test-selected:** checkpoint chọn bằng validation (hợp lệ để báo cáo) / chọn bằng test (chỉ exploratory).
- **Provenance:** hồ sơ nguồn gốc của một kết quả: commit, hash dữ liệu, lệnh đã chạy.

## 9. Một số câu hỏi có thể gặp và hướng trả lời trung thực

**Vì sao C tốt hơn?** C chỉ khác B ở augmentation: bỏ frame masking và giảm keypoint dropout, và hơn B khoảng 3,2 điểm accuracy. Giả thuyết là frame masking che mất khoảnh khắc click ngắn, và các lớp click quả thật cải thiện nhiều nhất. Nhưng đây là một seed, chưa tách hai yếu tố, nên là giả thuyết có bằng chứng gián tiếp chứ chưa phải kết luận nhân quả.

**Vận tốc giúp bao nhiêu?** Một lần chạy cho thấy +2,1 điểm accuracy và +3,6 điểm macro-F1 trên test IPN (neural C so với bản không vận tốc cùng siêu tham số), tập trung ở double click (G08 recall 36,5% lên 71,2%). Mới một seed và test đã bị xem; sweep ba seed ngày 2026-10-27 cho thấy test accuracy của C dao động 81,0% đến 84,65% giữa các seed, nên chênh lệch này nằm trong dao động seed và cần ba seed để kết luận.

**Vì sao fusion trên DATE-27 không vượt neural-only?** Final là tổ hợp lồi của hai logits nên nằm giữa chúng. Gate (trọng số của neural) trung bình khoảng 0,55 ở epoch được chọn, tức neural và KG có trọng số gần ngang nhau, và trên validation ba đầu ra gần như bằng nhau (khoảng 95,5%). Hai đầu cũng không độc lập: chung một encoder được train chung từ đầu (mục 5.5), nên trộn chúng cho rất ít lợi ích bổ sung. Cũng cần lưu ý neural-only độc lập chỉ còn `last.pt`, và neural head trong V4 là một head được train chung, không phải cùng một mô hình với neural-only.

**Symbolic chỉ đạt khoảng 23%, vậy nó có ích gì?** Làm classifier độc lập thì chưa. Giá trị hiện tại: (1) chẩn đoán lỗi có cấu trúc của neural, (2) audit chứng minh override theo luật sẽ có hại, tránh bật nhầm, (3) cung cấp template và graph cho learned `kg_logits`.

**Số 96,40% của paper có tái lập được không?** Chưa. Repo chỉ đóng gói được một checkpoint (94,87%), và run 3 của paper (96,18%) trùng đúng checkpoint chọn theo test. Cần xác minh provenance của 10 run.

**Kết quả IPN đã chạy test chưa?** Chưa cho learned `kg_logits`. Các số IPN hiện có là validation. Test đã bị xem trong các thí nghiệm lịch sử nên số test dù có cũng chỉ là exploratory.

## Nguồn

`HANDBOOK_TOAN_DIEN_IPN_NEUROSYMBOLIC_VI.md` (mục 6–17), `NEURAL_ONLY_EVALUATION_REPORT_2026-10-20_VI.md`, `IPN_TEMPLATE_AUTHORING_PLAYBOOK_VI.md`, `IPN_SYMBOLIC_POST_C_CHANGELOG_VI.md`, `IPN_SYMBOLIC_6EF_ANALYSIS_VI.md`, `DATE27_BASELINE_COMPARISON_2026-10-23_VI.md`, `IPN13_KG_LOGITS_EXPERIMENT_PROTOCOL_VI.md`; số A/B/C đọc từ `val_metrics.json`, `command.json`, `history.json` trong các ZIP ablation; code: `final/code/neural_only_experiments/models.py`, `final/code/src_neurosymbolic/models/cross_attention_v2.py`.

## Validation của tài liệu

Số liệu A/B/C và số lỗi theo cặp đối chiếu với file gốc; công thức vận tốc, pooling và gate đối chiếu với `models.py` và `cross_attention_v2.py`; các phần [Lập luận] và đề xuất được gắn nhãn riêng. Tài liệu chỉ mô tả, không đổi code hay dữ liệu.
