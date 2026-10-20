# Báo cáo đánh giá Neural-only — 20/10/2026

## 1. Phạm vi và kết luận

Báo cáo này chỉ đánh giá nhánh neural-only. Phần symbolic/template được tạm hoãn.

Run hiện có dùng Transformer với seed 42 trên bộ IPN 13 lớp. Checkpoint được chọn theo validation macro-F1. Kết quả tốt nhất ở epoch 49:

| Chỉ số | Giá trị |
|:--|--:|
| Train accuracy | 0,9574 |
| Validation accuracy | 0,8710 |
| Validation macro-F1 | 0,8109 |
| Validation loss | 0,4990 |
| Số mẫu train | 2.443 |
| Số mẫu validation | 589 |

Kết luận hiện tại: quá trình train thành công và checkpoint `best.pt` tốt hơn `last.pt`, nhưng chưa phải kết quả nghiệm thu cuối vì chưa có đánh giá locked test, confusion matrix và metric từng cử chỉ.

## 2. Kết quả từng cử chỉ

Artifact hiện tại chỉ có `history.json`, `train.log`, `command.json`, `best.pt` và `last.pt`. `history.json` chỉ lưu metric tổng hợp theo epoch, không lưu prediction hoặc confusion matrix. Vì vậy không thể suy ngược một cách hợp lệ accuracy/F1 từng cử chỉ từ artifact hiện tại.

Evaluator đã được bổ sung để lần test tiếp theo tự xuất `label_name`, `correct`, `predicted`, per-class `accuracy`/recall, precision, F1, confusion matrix, macro precision/recall/F1 và balanced accuracy. Nó cũng dừng sớm nếu manifest có ID ngoài output range của checkpoint, thiếu metadata tên lớp, dùng mapping nhãn khác hoặc tự mâu thuẫn giữa ID và tên lớp. Một evaluation subset có ít lớp hơn vẫn được chấp nhận nếu contract của các lớp xuất hiện là đúng.

Các lớp và support cần xuất hiện trong báo cáo locked test là:

| ID | Cử chỉ | Test support | Accuracy/Recall | Precision | F1 |
|--:|:--|--:|:--:|:--:|:--:|
| 0 | B0A | 265 | Chờ test | Chờ test | Chờ test |
| 1 | B0B | 264 | Chờ test | Chờ test | Chờ test |
| 2 | G01 | 52 | Chờ test | Chờ test | Chờ test |
| 3 | G02 | 52 | Chờ test | Chờ test | Chờ test |
| 4 | G03 | 52 | Chờ test | Chờ test | Chờ test |
| 5 | G04 | 52 | Chờ test | Chờ test | Chờ test |
| 6 | G05 | 52 | Chờ test | Chờ test | Chờ test |
| 7 | G06 | 52 | Chờ test | Chờ test | Chờ test |
| 8 | G07 | 52 | Chờ test | Chờ test | Chờ test |
| 9 | G08 | 52 | Chờ test | Chờ test | Chờ test |
| 10 | G09 | 52 | Chờ test | Chờ test | Chờ test |
| 11 | G10 | 52 | Chờ test | Chờ test | Chờ test |
| 12 | G11 | 52 | Chờ test | Chờ test | Chờ test |

Trong bài toán single-label classification, “độ chính xác của một cử chỉ” nên được báo cáo là recall của lớp đó:

```text
per-class accuracy = số clip của lớp được đoán đúng / tổng số clip thật của lớp
                   = recall của lớp
```

Không nên chỉ báo per-class accuracy. Cần giữ thêm precision và F1 vì một mô hình có thể recall cao cho một lớp do dự đoán lớp đó quá nhiều.

## 3. Thay đổi so với neural-only ban đầu trên GitHub

Mốc so sánh là code trước commit `4ff4733` (`feat: improve neural training and add Colab workflow`). Các thay đổi chính:

### Mô hình

- Thêm vận tốc `(dx, dy)` của từng keypoint vào input, từ 3 đặc trưng thành 5 đặc trưng mỗi điểm.
- Không tạo vận tốc giả khi một trong hai frame bị thiếu keypoint.
- Thay mean pooling cố định bằng attention pooling mặc định.
- Bổ sung `masked_mean` và vẫn giữ `mean` để tái tạo kiến trúc cũ.
- Kiểm tra cứng input shape `[B, T, 52, 3]`.
- Xử lý clip toàn frame rỗng để logits không thành NaN/Inf.

### Huấn luyện

- Thêm jitter tọa độ chỉ trên keypoint hợp lệ.
- Thêm random frame masking và keypoint dropout.
- Thêm class-weighted cross entropy để giảm ảnh hưởng mất cân bằng lớp.
- Thêm label smoothing.
- Gradient clipping tiếp tục được sử dụng.
- Tính validation macro-F1 mỗi epoch.
- Chọn `best.pt`, điều khiển scheduler và early stopping theo metric cấu hình; mặc định là macro-F1 thay vì accuracy.
- Lưu `best_metric_value`, `best_accuracy`, `max_val_accuracy`, metric được chọn và history chi tiết.
- Kiểm tra train/validation không dùng trùng source.

### Dataset và cache

- Split sai format, label không phải số hoặc split rỗng sẽ fail-fast.
- Không còn tìm tensor theo filename trên toàn cache vì có thể lấy nhầm class/split.
- Phát hiện cache trùng/ambiguous thay vì chọn file đầu tiên.
- Dùng `torch.load(..., weights_only=True)` cho tensor cache.
- Kiểm tra tensor đúng số chiều, đúng 64 frame và không chứa NaN/Inf.

### Đánh giá và notebook

- Test script xuất macro-F1, precision/recall/F1/support từng lớp và prediction từng clip.
- Top-k không lỗi khi số lớp nhỏ hơn 5.
- Có notebook Colab riêng, đọc dữ liệu từ Drive, train trên GPU và khóa test bằng `RUN_LOCKED_TEST=False`.
- Notebook IPN đặt `NUM_CLASSES=13`, kiểm tra label ID 0–12 và chọn checkpoint theo validation macro-F1.
- Log subprocess được stream ra notebook và được lưu vào `train.log`.
- Bổ sung trong lần audit này: report per-class có tên cử chỉ, số đúng, số được dự đoán, per-class accuracy/recall, precision, F1, confusion matrix và balanced accuracy.
- Bổ sung trong lần audit này: chặn checkpoint 13 lớp chạy nhầm manifest 27 lớp hoặc manifest dùng cùng ID cho tên cử chỉ khác.

## 4. Code có bị gắn cứng với IPN không?

Phần backbone neural không chứa template hoặc luật riêng cho IPN. `num_classes`, split files, cache root, backbone và các hyperparameter đều có thể cấu hình. Vì vậy cùng code có thể retrain trên bộ 27 nhãn cũ.

Tuy nhiên code vẫn có các hợp đồng dữ liệu cần giữ đúng:

- tensor đầu vào mặc định là 64 frame, 52 keypoint, 3 giá trị `(x, y, confidence)`;
- label ID phải liên tục từ `0` đến `num_classes - 1`;
- đường dẫn trong manifest phải cho phép suy ra split và tên lớp;
- cache phải dùng cùng normalization và thứ tự 52 keypoint;
- train phải có ít nhất một mẫu cho mọi class ID.

Do đó code là dataset-configurable, không phải dataset-independent tuyệt đối.

## 5. Có thể dùng checkpoint IPN 13 lớp cho bộ 27 nhãn cũ không?

Không thể test trực tiếp. `best.pt` hiện tại có classifier 13 output và đã học mapping B0A, B0B, G01–G11. Bộ 27 nhãn có output space và phân phối dữ liệu khác. Evaluator hiện sẽ phát hiện contract 13-vs-27 không tương thích và dừng trước inference.

Có hai cách hợp lệ:

1. **So sánh công bằng:** khởi tạo model mới, đặt `--num-classes 27`, rồi train lại toàn bộ trên train split của bộ cũ.
2. **Transfer learning:** nạp phần backbone tương thích, bỏ classifier 13 lớp, tạo classifier 27 lớp và fine-tune trên bộ cũ. Kết quả transfer phải được báo riêng, không trộn với training from scratch.

Không có cơ sở để hứa bộ 27 lớp sẽ đạt đúng 87,10%. Nó có thể cao hơn hoặc thấp hơn vì số lớp, độ khó, số subject, chất lượng keypoint và split khác nhau.

Bộ 27 lớp local đã được kiểm tra và tương thích với input contract hiện tại:

| Split | Số mẫu | Số lớp | ID |
|:--|--:|--:|:--|
| Train | 9.097 | 27 | 0–26 |
| Validation | 1.956 | 27 | 0–26 |
| Test | 1.229 | 27 | 0–26 |

Cache có 12.285 tensor, mỗi tensor smoke-check có shape `[64, 52, 3]`. Không có source overlap hoặc resolved-cache overlap giữa ba split. Như vậy dữ liệu cũ đã sẵn sàng về mặt cấu trúc để retrain; điều còn thiếu là chạy training/evaluation đầy đủ.

## 6. Thí nghiệm kiểm tra bias được khuyến nghị

Chạy hai cấu hình trên đúng bộ 27 lớp cũ, cùng split và các seed `42, 123, 3407`:

### Baseline tương thích code cũ

```powershell
python final/code/neural_only_experiments/train_neural_only.py `
  --backbone transformer `
  --cache-root final/data/Input/keypoint_tensor_cache `
  --train-file final/data/Input/train.txt `
  --val-file final/data/Input/val.txt `
  --output-dir <OUTPUT_OLD_STYLE> `
  --num-classes 27 `
  --num-frames 64 `
  --epochs 50 `
  --disable-motion-features `
  --pooling mean `
  --class-weight-power 0 `
  --label-smoothing 0 `
  --jitter-std 0 `
  --frame-mask-probability 0 `
  --keypoint-dropout-probability 0 `
  --select-metric macro_f1 `
  --seed 42
```

### Code cải tiến

```powershell
python final/code/neural_only_experiments/train_neural_only.py `
  --backbone transformer `
  --cache-root final/data/Input/keypoint_tensor_cache `
  --train-file final/data/Input/train.txt `
  --val-file final/data/Input/val.txt `
  --output-dir <OUTPUT_IMPROVED> `
  --num-classes 27 `
  --num-frames 64 `
  --epochs 50 `
  --d-model 128 `
  --layers 3 `
  --heads 4 `
  --dropout 0.15 `
  --label-smoothing 0.05 `
  --class-weight-power 0.5 `
  --jitter-std 0.005 `
  --frame-mask-probability 0.05 `
  --keypoint-dropout-probability 0.03 `
  --select-metric macro_f1 `
  --seed 42
```

Lặp lại cho ba seed. Chỉ sau khi chốt cấu hình bằng validation mới chạy test. Báo cáo cần gồm mean ± standard deviation của accuracy và macro-F1, metric từng lớp, confusion matrix và prediction từng clip.

Nếu code cải tiến tăng ổn định trên cả IPN 13 lớp và bộ cũ 27 lớp, đó là bằng chứng nó cải thiện neural pipeline nói chung. Nếu chỉ tăng trên IPN, cần ablation để xác định nguyên nhân là class weighting, augmentation, motion features hay attention pooling.

## 7. Việc còn thiếu trước khi chốt báo cáo

- Chạy locked test để sinh `locked_test/test_metrics.json`, `predictions.csv`, bảng từng lớp, confusion matrix, Markdown report và provenance hash.
- Điền bảng metric 13 cử chỉ ở mục 2.
- Thêm confusion matrix.
- Chạy tối thiểu ba seed hoặc ghi rõ kết quả hiện tại chỉ là single-seed.
- Lưu commit hash cùng artifact.
- Sửa checkpoint serialization để không pickle trực tiếp các đối tượng `Path`, giúp checkpoint tạo trên Colab load được bằng Python 3.11 local.

## 8. Validation của code báo cáo

Test tập trung cho neural-only hiện đạt `23 passed`. Có ba warning của PyTorch về nested tensor khi Transformer dùng `norm_first=True`; đây không phải test failure.
