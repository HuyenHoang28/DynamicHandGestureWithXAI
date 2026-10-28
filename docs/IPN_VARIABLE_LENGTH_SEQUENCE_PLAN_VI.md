# Kế hoạch chuỗi frame biến độ dài cho IPN

Ngày lập: 2026-10-28. Trạng thái: bước 1–3 hoàn tất; implementation bước 4 hoàn tất và chờ smoke report trên cache Drive đầy đủ; bước 5 chưa bắt đầu. Người dùng xác nhận notebook bước 3 đã chạy xong trên bộ IPN ở Google Drive.

## Vấn đề và quyết định ban đầu

Pipeline hiện ép mọi clip về 64 frame: clip ngắn lặp frame cuối, clip dài lấy mẫu đều trên toàn khoảng thời gian. Cách này giữ kích thước tensor cố định nhưng có thể pha loãng chuyển động ở clip ngắn và bỏ sót pha click ngắn ở clip dài. Mục tiêu là bảo toàn chuỗi gốc trong một giới hạn hợp lý, dùng zero-padding chỉ để ghép batch và mask padding xuyên suốt mô hình.

Thay đổi này chưa làm mất khả năng tái lập pipeline cũ. Cache `T64`, checkpoint và chế độ fixed-length hiện tại phải được giữ nguyên trong lúc chạy đối chứng.

## Sáu bước thực hiện

1. **Thống kê dữ liệu gốc.** Đo phân bố độ dài toàn bộ clip và riêng G01/G02/G08/G09; đếm các khoảng `<64`, `64–127`, `128–255`, `>=256`; xác minh dữ liệu có ground truth vị trí từng click hay không.
2. **Chọn `max_frames`.** Dựa trên percentile, chi phí attention và coverage của các lớp click; không mặc định chọn 64.
3. **Giữ nguyên chuỗi trong giới hạn.** Clip không vượt `max_frames` giữ toàn bộ frame, không repeat-last và không downsample; vận tốc được tính trên các frame thật.
4. **Batch động có mask.** Zero-pad đến clip dài nhất trong batch, truyền padding mask vào Transformer và pooling, đồng thời dùng bucket theo độ dài nếu cần giảm VRAM.
5. **Xử lý clip vượt giới hạn.** Ưu tiên sliding window có overlap và gộp dự đoán; không lấy mẫu thưa toàn clip nếu điều đó có thể bỏ mất một click.
6. **Ablation nhiều seed.** So sánh fixed-64 hiện tại với variable-length ở các giới hạn ứng viên; báo cáo macro-F1, các confusion G01↔G08 và G02↔G09, kết quả theo nhóm độ dài, tốc độ và VRAM.

## Bước 1 — triển khai audit độ dài

Script `scripts/audit_ipn_sequence_lengths.py` đọc trường `frames` từ ba manifest đã chia train/validation/test. Đây là chiều dài clip được cắt từ annotation gốc, không phải chiều dài tensor cache `T64`. Script:

- kiểm tra cột bắt buộc, frame dương và `clip_id` không trùng;
- xuất thống kê chung, theo split và riêng bốn lớp click;
- xuất từng clip cùng bucket độ dài để phục vụ phân tích lỗi sau này;
- không tuyên bố peak do detector suy ra là ground truth của vị trí click.

Đầu ra được ghi vào `external_data/ipn_sequence_length_audit/` và không được version hóa như artifact thí nghiệm.

### Giới hạn dữ liệu

Annotation IPN trong workspace chỉ cho biết biên clip và nhãn gesture, không đánh dấu frame của click thứ nhất/thứ hai. Vì vậy bước 1 có thể đo chính xác độ dài nhưng chưa thể đo chính xác vị trí tương đối của click thứ hai. Detector temporal hiện tại đã được ghi nhận là đếm dao động chưa đáng tin cậy; kết quả detector, nếu phân tích sau này, phải mang nhãn `ước lượng`, không phải ground truth.

### Kết quả đo ngày 2026-10-28

Audit chạy trên đủ ba manifest gồm 2.443 train, 589 validation và 1.101 test, tổng cộng 4.133 clip. Không dùng tensor `T64` để suy ra độ dài.

| Phạm vi | Số clip | Min | Median | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toàn bộ | 4.133 | 9 | 113 | 268 | 304 | 383,68 | 650 |
| Train | 2.443 | 9 | 115 | 264,8 | 303 | 384,58 | 623 |
| Validation | 589 | 13 | 100 | 267 | 308 | 395,08 | 624 |
| Test | 1.101 | 10 | 110 | 273 | 305 | 361 | 650 |

Phân bố toàn bộ theo các khoảng đã định trước:

| Khoảng | Số clip | Tỷ lệ |
|---|---:|---:|
| `<64` | 1.285 | 31,09% |
| `64–127` | 912 | 22,07% |
| `128–255` | 1.405 | 34,00% |
| `>=256` | 531 | 12,85% |

Riêng các lớp click:

| Lớp | Số clip | Min | Median | P95 | P99 | Max | `<64` | `64–127` | `128–255` | `>=256` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G01 | 196 | 13 | 48,5 | 113,25 | 143,85 | 167 | 142 | 47 | 7 | 0 |
| G02 | 196 | 18 | 48,5 | 114,25 | 158,35 | 525 | 132 | 59 | 4 | 1 |
| G08 | 196 | 27 | 60 | 119,25 | 145,05 | 205 | 109 | 79 | 8 | 0 |
| G09 | 196 | 25 | 62 | 130,5 | 152 | 238 | 103 | 80 | 13 | 0 |

Các số liệu cho thấy 64 vừa tạo padding lặp cho phần lớn clip click ngắn, vừa downsample một phần đáng kể clip double-click. Giới hạn 256 bao phủ 783/784 clip click; ngoại lệ duy nhất là một clip G02 dài 525 frame. Tuy nhiên 256 chỉ bao phủ khoảng 87% toàn bộ tập, nên quyết định bước 2 phải cân bằng coverage chung, chi phí attention và sliding window cho phần đuôi dài; chưa chốt `max_frames` chỉ từ các lớp click.

### Tương thích và an toàn

Bước 1 chỉ đọc manifest và tạo báo cáo; không sửa cache, model, checkpoint, split hay runtime behavior. Test được viết để bao phủ biên các bucket, percentile, coverage theo giới hạn, phát hiện `clip_id` trùng và contract về ground truth click.

### Validation đã thực hiện

- Chạy audit thành công trên 4.133 dòng của ba manifest và sinh `sequence_length_audit.json` cùng `sequence_lengths.csv`.
- Review độc lập đã đối chiếu: mọi `frames` đều bằng `t_end - t_start + 1`, split trong dòng khớp file nguồn, đủ 784 clip click, số trong tài liệu khớp JSON và semantics biên 256 là inclusive.
- `py_compile` thành công cho script và file test; các direct assertions về bucket, percentile, coverage và contract ground truth click đã chạy thành công.
- Chưa chạy được pytest vì `.venv` hiện không cài package `pytest` (`No module named pytest`). Do đó không tuyên bố test suite đã pass; cần chạy `python -m pytest tests/test_audit_ipn_sequence_lengths.py -q` sau khi cài dependencies phát triển.

## Bước 2 — chọn `max_frames`

### Phạm vi và phương pháp

Quyết định kiến trúc chỉ dùng 3.032 clip train + validation; test không tham gia lựa chọn. Script `scripts/analyze_ipn_max_frames.py` đánh giá 64, 128, 256 và 384. Với clip dài hơn giới hạn, phép tính giả định sliding window overlap 50%, stride bằng `limit // 2`, đồng thời thêm window cuối neo tại `N - limit` để không bỏ đuôi clip.

Các proxy chi phí chỉ dùng để so sánh thiết kế:

- `total windows`: số lần encoder phải chạy nếu mỗi window là một sample;
- `token proxy`: tổng số timestep đưa qua encoder;
- `attention proxy`: tổng `T²` của các window, chưa gồm hệ số layer/head/batch;
- `peak attention proxy`: `T²` của window dài nhất, phản ánh rủi ro VRAM trên mỗi sample tốt hơn tổng proxy.

### Kết quả train + validation

| Giới hạn | Coverage toàn bộ | Clip cần window | Tổng window | Coverage click | Tổng attention so với 64 | Peak so với 64 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 31,60% | 2.074 | 11.749 | 359/576 (62,33%) | 1,00× | 1× |
| 128 | 53,56% | 1.408 | 5.932 | 547/576 (94,97%) | 1,69× | 4× |
| 256 | 87,90% | 367 | 3.436 | 575/576 (99,83%) | 2,15× | 16× |
| 384 | 98,88% | 34 | 3.068 | 575/576 (99,83%) | 1,92× | 36× |

Coverage click là tổng G01/G02/G08/G09 trên train + validation. Cả 256 và 384 chỉ còn một clip G02 dài 525 frame cần sliding window; G01, G08 và G09 đều được giữ nguyên hoàn toàn ở 256.

Tổng attention của 384 thấp hơn 256 trong proxy này vì 384 cần ít window overlap hơn. Tuy nhiên một window 384 tạo ma trận attention lớn gấp 2,25 lần window 256 và 36 lần chuỗi 64. Đây là peak theo sample; peak batch thực tế còn phụ thuộc bucket, batch size, số layer/head, mixed precision và implementation PyTorch, nên chưa được coi là benchmark VRAM.

### Quyết định kiến trúc

**Chọn `max_frames = 256` cho lần triển khai variable-length đầu tiên.** Clip `N <= 256` giữ nguyên toàn bộ. Clip `N > 256` dùng sliding window dài 256, stride 128 và bắt buộc có window cuối kết thúc tại frame cuối.

Lý do:

1. Giữ nguyên 575/576 clip click dùng để phát triển và validation, đạt mục tiêu bảo toàn tín hiệu đếm click mà không cần tăng lên 384.
2. Giữ nguyên 87,90% toàn bộ train + validation; số clip cần window giảm từ 2.074 ở giới hạn 64 xuống 367.
3. Peak attention chỉ bằng 44,44% của phương án 384 (`256² / 384²`), an toàn hơn để bắt đầu trên GPU T4 đang được notebook sử dụng.
4. 384 không tăng coverage click so với 256. Lợi ích của nó chủ yếu nằm ở các lớp dài khác và phải được cân bằng bằng benchmark VRAM/thời gian thực tế.

Trade-off được chấp nhận:

- 367 clip train + validation vẫn cần chính sách window và gộp prediction ở bước 5.
- Chi phí attention peak lý thuyết vẫn gấp 16 lần chuỗi 64; batch size 64 hiện tại không được giả định là còn phù hợp.
- Một clip G02 dài 525 frame không được đưa nguyên khối; overlap 50% phải bảo đảm click gần biên xuất hiện đầy đủ trong ít nhất một window nếu độ dài sự kiện không vượt overlap 128 frame.

Điều kiện xem xét lại 384:

- benchmark T4 cho thấy 384 chạy ổn với batch hiệu dụng và thời gian chấp nhận được; hoặc
- ablation ở bước 6 cho thấy các lớp gesture dài bị hại đáng kể bởi window 256; hoặc
- phân tích frame-level chứng minh một gesture cần context liên tục dài hơn 256 frame.

### Compatibility và validation bước 2

Bước 2 chỉ thêm analyzer và quyết định thiết kế; chưa sửa model, cache hoặc runtime. Analyzer khóa đầu vào lựa chọn kiến trúc ở split train/validation và từ chối test; giới hạn frame phải từ 2 trở lên để stride luôn hợp lệ. Báo cáo máy được ghi tại `external_data/ipn_sequence_length_audit/max_frames_analysis.json` và không version hóa. `py_compile`, direct assertions về biên window/đuôi clip, provenance split và chạy analyzer trên dữ liệu thật đều thành công. Review độc lập xác nhận bảng, click sums và coverage window khớp JSON. Pytest vẫn chưa chạy do môi trường thiếu package `pytest`.

## Bước 3 — giữ nguyên chuỗi và tính vận tốc trên frame thật

### Thay đổi đã thực hiện

Module `final/code/src_neurosymbolic/variable_keypoint_cache.py` tạo cache schema v1 tách biệt hoàn toàn khỏi `T64`:

```text
keypoint_tensor_cache/
├── T64/          # legacy, không thay đổi
└── variable_v1/  # schema mới
```

Mỗi payload `variable_v1` chứa:

| Trường | Ý nghĩa |
|---|---|
| `keypoints` | Tensor float `[N, 52, 3]`, giữ đúng thứ tự và toàn bộ N frame thật |
| `velocity` | Tensor float `[N, 52, 2]`, tính trên hai frame thật kề nhau |
| `length` | N frame trong tensor |
| `source_frames` | Số frame nguồn; ở schema v1 phải bằng N |
| `schema_version` | Bằng 1 |
| `exceeds_max_frames` | Metadata cho biết `N > 256`; không cắt hoặc window ở bước 3 |
| `source_size`, `source_mtime_ns`, `source_sha256` | Fingerprint JSON nguồn để phát hiện cache stale |

Preprocessing mới không repeat-last, không zero-pad, không truncate, không nội suy và không lấy mẫu thưa. Clip 5, 64 hay 300 frame lần lượt tạo tensor dài đúng 5, 64 và 300. Clip rỗng được biểu diễn rõ bằng `[0, 52, 3]`, `length=0`; bước Dataset sau này phải quyết định loại clip rỗng trước khi đưa vào model.

Frame có `valid=false` sau chuẩn hóa vai được đặt confidence về 0 trong cache, kể cả khi detector gốc có score dương; tọa độ chuẩn hóa zero của frame lỗi vì vậy không thể sinh vận tốc giả. Vận tốc frame đầu luôn bằng 0. Với `t > 0`:

```text
velocity[t] = xy[t] - xy[t-1]
```

nhưng chỉ khi confidence của cùng keypoint ở cả `t-1` và `t` đều lớn hơn 0; nếu một trong hai không hợp lệ thì vận tốc bằng 0. Nhờ tính trước padding/window, không xuất hiện bước nhảy giả từ frame thật sang padding.

Cache được kiểm tra schema, shape, dtype float, số hữu hạn và tính nhất quán của length trước khi ghi/đọc. Ghi file dùng file tạm cùng thư mục rồi `replace`, giảm nguy cơ để lại cache đích nửa chừng. Loader bắt buộc `torch.load(..., weights_only=True)` và fail closed trên PyTorch quá cũ thay vì fallback sang pickle không hạn chế. Cache lưu kích thước, `mtime_ns` và SHA-256 của JSON nguồn; lần chạy sau chỉ skip khi schema, fingerprint nguồn và `exceeds_max_frames` theo threshold CLI hiện tại đều khớp, còn cache cũ/stale hoặc metadata threshold cũ được dựng lại.

### Kết quả materialization cục bộ

Workspace hiện chỉ chứa 40 keypoint JSON dưới `external_data/ipn_processed/keypoints`, dù manifest mô tả 4.133 clip. Lần chạy trên đủ ba manifest cho kết quả:

| Split | Ghi mới | Đã có | Thiếu/thất bại |
|---|---:|---:|---:|
| Train | 40 | 0 | 2.403 |
| Validation | 0 | 0 | 589 |
| Test | 0 | 0 | 1.101 |

Vì vậy implementation bước 3 đã hoàn tất nhưng cache đầy đủ chưa thể materialize tại máy này. CLI đã được sửa để trả mã lỗi nếu có bất kỳ source nào thất bại, tránh báo thành công giả. Khi bộ keypoint JSON đầy đủ được mount/copy vào workspace, chạy lại cùng lệnh sẽ kiểm tra schema cùng fingerprint của 40 cache rồi tạo lại cache stale và tạo phần còn thiếu. Đường dẫn source và `keypoint_root` đều được canonicalize trước khi lấy relative path, nên CLI hỗ trợ nhất quán cả root tương đối lẫn tuyệt đối và từ chối source nằm ngoài root.

### Tương thích và an toàn

- Không sửa `keypoints_to_tensor` hoặc cache `T64`; checkpoint và thí nghiệm fixed-64 cũ giữ nguyên hành vi.
- Chưa nối `variable_v1` vào Dataset/DataLoader/model; runtime huấn luyện hiện tại chưa đổi.
- `max_frames=256` ở bước này chỉ là metadata. Window dài 256/stride 128 thuộc bước 5.
- Không version hóa tensor cache sinh ra.

### Validation bước 3

- 8 unit test bằng `unittest` pass: giữ N ở 5/64/300; confidence-gated velocity kể cả frame normalization lỗi; empty round-trip; fingerprint freshness; root tuyệt đối; cache lỗi bị từ chối; hành vi repeat/downsample legacy không đổi.
- `py_compile` và `git diff --check` pass.
- Chạy precompute thật đã tạo 40 cache từ 40 JSON có sẵn và phơi bày trung thực 4.093 source còn thiếu.
- Pytest tổng thể chưa chạy vì `.venv` không có package `pytest`; riêng test bước 3 dùng `unittest` nên đã chạy đầy đủ.

### Notebook tạo cache đầy đủ trên Google Drive

`notebooks/IPN_Variable_Length_Cache_Colab.ipynb` là quy trình Colab cho bộ keypoint IPN đã extract trên Drive. Notebook mount Drive, checkout/pull rõ ràng branch `HuyenHoang` (không phụ thuộc default branch), yêu cầu mọi path trong manifest có JSON tương ứng và báo cáo nhưng bỏ qua JSON ngoài manifest, vô hiệu hóa report PASS cũ trước khi chạy, tạo `variable_v1`, rồi kiểm tra chính xác tập path manifest ↔ cache cùng schema + SHA-256 fingerprint. `CACHE_BUILD_REPORT.json` được thay thế atomically trên Drive khi audit hoàn tất. Notebook chạy CPU; không cần GPU vì chỉ chuẩn hóa JSON, tính hiệu frame và ghi tensor. Nó không sửa hoặc ghi đè `T64`.

Validation notebook tại repository: file qua `json.tool`, đủ nbformat 4 và toàn bộ 5 code cell compile thành công; review độc lập đã kiểm tra convention clone/Drive, CLI và cách ly `T64`. Chưa chạy end-to-end trên Colab vì bộ 4.133 JSON chỉ có trên Google Drive của dự án; trạng thái `PASS` cuối cùng phải lấy từ `CACHE_BUILD_REPORT.json` do chính notebook tạo trên Drive.

## Bước 4 — batch động và padding mask

### Data contract mới

`final/code/neural_only_experiments/variable_dataset.py` đọc cache `variable_v1`. Mỗi sample trả `keypoints [N,52,3]`, `velocity [N,52,2]`, `length`, nhãn và source path. Dataset từ chối cache rỗng, nhãn âm, shape/schema sai và clip `N>256`; trường hợp cuối thông báo rõ phải qua windowing bước 5, không âm thầm cắt.

`collate_variable_batch` pad zero đến clip dài nhất **trong batch**, không pad cố định tới 256:

```text
keypoints    [B, T_batch_max, 52, 3]
velocity     [B, T_batch_max, 52, 2]
padding_mask [B, T_batch_max]  # True chỉ ở padding
lengths      [B]
```

Frame thật được xác định bằng `length/padding_mask`, không suy ra từ confidence; vì vậy một frame thật mất toàn bộ detection vẫn không bị nhầm là padding. Keypoint và velocity ở padding luôn bằng 0.

### Model contract mới

`NeuralOnlyClassifier.forward` và `NeuralBaselineAdapter.forward` nhận tùy chọn `padding_mask` và `velocity`. Với chế độ mới:

- velocity lấy trực tiếp từ cache bước 3, không tính hiệu qua padding;
- Transformer nhận `src_key_padding_mask`, nên frame padding không làm key/value attention;
- `mean`, `masked_mean` và attention pooling đều loại padding;
- adapter neuro-symbolic trả mask cùng token; các model fusion V2/V3/V4 truyền mask này vào cross-attention memory;
- sequence vượt positional capacity và sample toàn padding bị từ chối rõ ràng.

API legacy `model(keypoints)` và cache `T64` không đổi. State-dict không thêm parameter trainable, positional encoding vẫn là buffer non-persistent, nên checkpoint cũ giữ tương thích. Chế độ padding-mask ở bước 4 chỉ hỗ trợ Transformer; TCN, BiGRU và MLP-Mixer fail rõ thay vì chạy mask sai. Hỗ trợ đúng cho các backbone này cần thiết kế riêng và không nằm trong baseline Transformer đã chọn.

### Ranh giới với bước 5

Bước 4 không xử lý clip dài hơn 256. Dataset cố ý từ chối chúng để không tạo batch quá positional capacity. Bước 5 sẽ biến mỗi clip dài thành các window 256/stride 128 và định nghĩa gộp prediction theo clip; chỉ sau đó mới có thể train/evaluate đủ 4.133 clip.

### Validation bước 4

- 5 unit test Dataset/collate pass: path Drive/relative, batch 5/8 frame, zero padding, mask chính xác, không repeat-last, clip rỗng/dài bị từ chối và legacy API tách biệt.
- 5 unit test model pass: logits bất biến trước giá trị padding lớn, velocity explicit khớp legacy trên frame thật, mask validation, positional guard, unsupported backbone guard và mask từ adapter neuro-symbolic.
- Smoke test trên cache thật cục bộ: ba clip 223/250/203 frame tạo batch `[3,250,52,3]`, logits `[3,13]` hữu hạn.
- Các test cũ phụ thuộc pytest chưa chạy vì `.venv` thiếu package `pytest`; 10 test bước 4 dùng `unittest` đều pass.
- Notebook `notebooks/IPN_Variable_Length_Step4_Smoke_Colab.ipynb` dùng cache đầy đủ trên Drive để audit tỷ lệ `<=256`/`>256`, chạy DataLoader + Transformer trên clip hợp lệ, kiểm tra padding invariance và lưu `STEP4_SMOKE_REPORT.json`. Notebook không huấn luyện và không bỏ qua clip dài trong metric; chúng được báo cáo là chờ bước 5.
