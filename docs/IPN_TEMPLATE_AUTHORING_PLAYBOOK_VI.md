# Playbook viết và đánh giá template IPN 13 lớp

Tài liệu này là quy trình làm việc cho giai đoạn symbolic song song với các thí nghiệm neural A/B/C. Mục tiêu là dùng template để sửa các lỗi có cấu trúc của neural, không thay thế neural bằng một bộ luật cứng và không điều chỉnh theo test set.

## 1. Điểm xuất phát và tiêu chí chốt neural

Ba cấu hình đã hoàn tất trên cùng validation split và seed 42:

| Tên | Pooling | Frame masking | Keypoint dropout | Câu hỏi cần trả lời |
|---|---|---:|---:|---|
| A_attention | Attention pooling | 0.05 | 0.03 | Attention có giúp chọn frame quan trọng không? |
| B_masked_mean | Masked mean | 0.05 | 0.03 | Pooling đơn giản có ổn định hơn không? |
| C_masked_mean_light_dropout | Masked mean | 0.00 | 0.02 | Giảm augmentation có giữ được tín hiệu động tác nhỏ không? |

| Run | Accuracy | Macro-F1 | Minimum class F1 | Best epoch |
|---|---:|---:|---:|---:|
| A_attention | 0.8761 | 0.8157 | 0.5306 | 34 |
| B_masked_mean_matched_A | 0.8591 | 0.7996 | 0.4783 | 17 |
| **C_masked_mean_light_dropout** | **0.8913** | **0.8376** | **0.6275** | 48 |

Theo tiêu chí định trước, **C là neural baseline được chọn**. Trên C, ba lớp còn dưới F1 0.65 là G01 (0.6275), G02 (0.6296) và G09 (0.6441). Nhầm lẫn có cấu trúc chính là G01→G08: 6, G08→G01: 6; G02→G09: 7, G09→G02: 6; G10→G11: 2, G11→G10: 0. Đây là cơ sở validation cho revision template sau C; không dùng kết quả test để viết luật.

Chỉ dùng validation để chọn cấu hình. Thứ tự xếp hạng là:

1. Macro-F1 cao hơn.
2. Chỉ khi Macro-F1 bằng nhau theo giá trị lưu trong artifact, minimum per-class F1 cao hơn. Nếu muốn coi hai kết quả là “gần bằng nhau”, phải định nghĩa tolerance trước thí nghiệm thay vì đổi quy tắc sau khi thấy kết quả.
3. Kiểm tra confusion matrix, đặc biệt G01/G08, G02/G09, G10/G11 và G07.
4. Mốc mong muốn trước khi sang symbolic: mọi lớp có F1 validation ít nhất 0.65.

Không chọn bằng accuracy đơn thuần vì B0A/B0B có support lớn hơn các lớp gesture. Không chạy locked test để quyết định cấu hình hoặc sửa template.

Sau khi ba run hoàn tất, chạy trên Colab:

```bash
!python /content/dynamic_hand_gesture_recognition/scripts/summarize_neural_ablations.py \
  --output-root /content/drive/MyDrive/ipn-extract-data/outputs
```

Kết quả được lưu tại `outputs/ablation_comparison_seed42/`:

- `ablation_summary.csv`: xếp hạng tổng quát và lớp yếu nhất;
- `ablation_per_class.csv`: precision, recall, F1 từng lớp;
- `ablation_pair_confusions.csv`: số nhầm theo các cặp trọng tâm;
- `ablation_report.md`: bản báo cáo đọc nhanh.

## 2. Template được phép làm gì

Template nên đóng vai trò trọng tài có điều kiện:

```text
Neural tự tin                         -> giữ dự đoán neural
Neural phân vân trong một cặp đã biết -> kiểm tra evidence symbolic
Evidence rõ, đủ và không mâu thuẫn    -> cho phép đổi nhãn
Evidence thiếu hoặc mâu thuẫn          -> giữ dự đoán neural
```

Không tái sử dụng cứng ngưỡng top-1 0.70 và khoảng cách top-1/top-2 0.10 của bản nháp: trên C, cặp ngưỡng này chỉ bao phủ 3/64 lỗi validation. Confidence gate và margin gate phải được tune lại trên validation của C, đồng thời báo cáo precision của override.

Template không nên can thiệp mọi lớp. Phạm vi ưu tiên theo lỗi còn lại của C là:

- G02 và G09: click một lần so với hai lần bằng hai ngón;
- G01 và G08: click một lần so với hai lần bằng một ngón;
- G10 và G11: khoảng cách pinch tăng so với giảm;
- G07 chỉ tiếp tục được kiểm chứng rule-only; không ưu tiên override vì neural C đã đạt F1 0.8846 và matcher sequence còn hạn chế.

## 3. Template hiện tại và việc cần kiểm chứng

| Lớp | Dấu hiệu chính hiện có | Nhầm cần xử lý | Rủi ro hiện tại |
|---|---|---|---|
| B0A | `index_extended`, ưu tiên đứng yên trong cùng episode | G01, G08, B0B | Chưa mã hóa rõ “không có click” |
| B0B | `index_middle_extended`, ưu tiên đứng yên trong cùng episode | G02, G09, B0A | Chưa mã hóa rõ “không có click” |
| G01 | Một episode index đi xuống, max 1 | G08, B0A | Pha đi lên mới là preferred; max occurrence là phạt mềm |
| G02 | Index và middle cùng đi xuống trong một episode, max 1 | G09, B0B | Shape và motion có thể bị tách thành hai phase |
| G03 | Tay đi lên, ưu tiên nhanh trong cùng episode | G04, G05, G06 | Chuyển động chéo có thể sinh nhiều predicate hướng |
| G04 | Tay đi xuống, ưu tiên nhanh trong cùng episode | G03, G05, G06 | Chưa có kiểm tra hướng trội/net trajectory |
| G05 | Image-right, ưu tiên nhanh, tức trái của người diễn | G06 và hướng chéo | Phụ thuộc quy ước camera/mirroring |
| G06 | Image-left, ưu tiên nhanh, tức phải của người diễn | G05 và hướng chéo | Phụ thuộc quy ước camera/mirroring |
| G07 | 4 phase closed–open–closed–open, ưu tiên spread trend | G10, G11 | Matcher có thể nhận một subsequence ngẫu nhiên |
| G08 | Đúng 2 episode index click | G01, B0A | Hai click gần nhau có thể bị merge thành một event |
| G09 | Đúng 2 episode two-finger click | G02, B0B | Phụ thuộc tracking ngón giữa và tách event |
| G10 | Pinch distance tăng, ưu tiên finger spread cùng tăng | G11, G07 | Chưa yêu cầu rõ trạng thái đầu pinch và cuối mở |
| G11 | Pinch distance giảm, ưu tiên finger spread cùng giảm | G10, G07 | Nhiễu có thể tạo cả chiều tăng và giảm |

Các giới hạn của matcher phải nhớ khi viết template:

- Negative evidence vẫn được matcher kiểm tra trên toàn clip, không chỉ active interval. Vì vậy revision sau thí nghiệm A để trống `negative_evidence`; chỉ bật lại khi matcher giới hạn được negative vào matched interval.
- Occurrence dựa trên merge và dedup event; hai click sát nhau có thể bị gộp.
- Shape là tín hiệu theo frame, motion cần một cửa sổ dài hơn; hai dấu hiệu đúng có thể nằm ở hai phase kề nhau.
- Sequence matching hiện mang tính greedy và chưa tối ưu trực tiếp toàn bộ ràng buộc `BEFORE`.
- Từ revision `phase_v4_action_role_audit`, matcher vẫn gắn vai trò `active`, `return`, `ambiguous` theo từng subject/family nhưng template classifier dùng `action_role: any`. Role chỉ là diagnostic/gate cho pairwise arbitration cho tới khi validation chứng minh precision đủ tốt. Run đối hướng có tâm interval theo thứ tự vẫn được xem là action rồi return dù biên sliding window chồng nhau; chỉ run gần như đồng thời mới `ambiguous`.
- `kg_logits` của mô hình V4 không phải điểm symbolic thuần vì nhánh đó có attention tới neural memory. Override hậu xử lý phải dùng evidence từ template matcher.

## 4. Quy trình authoring cho từng cặp

### 4.1 G01 so với G08

Một click hợp lệ phải có cùng subject, `index_extended` và `index_finger_moves_down` trong cùng episode. Dùng atomic ID hoặc interval để chứng minh hai episode là khác nhau.

- đúng 1 episode rõ ràng: evidence cho G01;
- ít nhất 2 episode tách biệt: evidence cho G08;
- hai episode bị merge, thiếu pha, hoặc bằng chứng mâu thuẫn: không override.

Đừng dùng chỉ `index_finger_oscillates` để kết luận double click; đây chỉ nên là preferred evidence.

### 4.2 G02 so với G09

Mỗi episode phải đồng thời có `index_middle_extended`, `index_finger_moves_down` và `middle_finger_moves_down`.

- đúng 1 episode đầy đủ: evidence cho G02;
- ít nhất 2 episode đầy đủ, tách biệt: evidence cho G09;
- middle keypoint thiếu hoặc hai ngón không đồng pha: không override.

### 4.3 G10 so với G11

Chỉ override khi tồn tại một chiều pinch trội trong active interval:

- khoảng cách thumb–index tăng, tốt hơn nếu spread cũng tăng: G10;
- khoảng cách thumb–index giảm, tốt hơn nếu spread cũng giảm: G11;
- cả hai chiều cùng mạnh, net change nhỏ hoặc nhiễu lớn: không override.

Negative evidence chiều ngược không nên lấy từ toàn clip; cần giới hạn vào matched interval trước khi dùng mạnh.

Matcher xuất `role_intervals` cho mọi candidate đầy đủ required directional evidence, kể cả template `action_role: any`; `active_intervals` là alias tương thích của nhánh `active`. Arbitration G10/G11 phải từ chối nếu cả hai chiều đều có active interval hoặc nếu không có active interval rõ ràng; không được suy diễn từ evidence `return`.

### 4.4 G07

Yêu cầu bốn phase khác nhau theo đúng thứ tự `closed → open → closed → open`. Không tái sử dụng cùng atomic evidence cho nhiều phase. Chỉ override khi temporal score cao, không có fast-motion conflict và khoảng cách giữa các phase hợp lý.

Đây là **gate cần triển khai thêm**, chưa phải bảo đảm của matcher hiện tại. Matcher hiện chỉ duyệt các event đã sắp thứ tự; nó chưa tự loại phase chồng lấn, chưa cấm tái sử dụng atomic ID giữa các phase và chưa kiểm tra khoảng cách thời gian hợp lý. Không bật override G07 trước khi ba điều kiện này có code và test tương ứng.

## 5. Chu trình sửa template an toàn

Mỗi lần chỉ thay đổi một giả thuyết:

1. Lấy confusion và các mẫu lỗi từ validation của cấu hình neural đã chốt.
2. Xem clip/keypoint trace của true positive, false positive và false negative trong đúng cặp.
3. Viết giả thuyết quan sát được, ví dụ “G08 phải có hai episode down với interval không giao nhau”.
4. Kiểm tra predicate extractor hiện có thực sự sinh được dấu hiệu đó. Nếu không, sửa extractor/matcher trước; không viết predicate tưởng tượng vào YAML.
5. Sửa một template hoặc một cơ chế matcher.
6. Chạy unit test, sau đó đánh giá lại toàn bộ validation.
7. Chỉ giữ thay đổi nếu cải thiện F1 của lớp mục tiêu mà không làm giảm rõ rệt Macro-F1 hoặc lớp đối ứng.
8. Ghi lại phiên bản template, cấu hình, threshold và artifact nguồn.

Không được xem lỗi locked test rồi quay lại sửa template. Locked test chỉ chạy một lần sau khi neural, template, fusion policy và threshold đều đã khóa bằng validation.

## 6. Báo cáo bắt buộc cho mỗi vòng template

Ngoài accuracy và Macro-F1, mỗi vòng phải có:

- precision, recall, F1 và support của từng lớp;
- confusion matrix trước/sau;
- số mẫu đủ điều kiện arbitration;
- số override neural → symbolic;
- precision của các override;
- số override đúng, sai và không quyết định được theo từng cặp;
- log cho từng override: neural top-2, margin, symbolic scores, required-hit ratio, temporal score, occurrence count, negative hits, matched event IDs và lý do.

Script `scripts/eval_ipn_pair_arbitration.py` thực thi đúng contract này cho ba cặp G01/G08, G02/G09 và G10/G11. Mặc định script chỉ audit. Cờ `--enable-override` chỉ dùng để mô phỏng trên validation và bắt buộc truyền `--max-neural-margin`; notebook chính không bật override.

Một template có thể tăng recall nhưng gây nhiều override sai. Vì vậy mục tiêu “template giúp 10–15%” phải được định nghĩa rõ là tăng bao nhiêu điểm F1 trên lớp yếu hoặc giảm bao nhiêu lỗi trong cặp, không mặc định là tăng 10–15 điểm accuracy toàn bộ hệ thống.

## 7. Gate trước khi mở locked test

Chỉ mở locked test khi thỏa toàn bộ:

- cấu hình neural được chọn bằng validation và đã cố định;
- template và matcher có version/commit cụ thể;
- threshold arbitration được chọn trên validation;
- không còn thay đổi dựa trên test;
- báo cáo per-class và confusion trước/sau đã được lưu;
- mọi override đều có log giải thích được;
- unit/integration tests của predicate, template graph và matcher đều pass.

Nếu neural vẫn có lớp dưới 0.65 sau A/B/C, trước hết phải xác định đó là thiếu tín hiệu, mất cân bằng, segmentation hay pooling. Template chỉ hợp lý khi lỗi có cấu trúc và predicate quan sát được; nó không nên che một neural baseline chưa ổn định.
