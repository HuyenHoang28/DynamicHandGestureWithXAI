# Baseline gần đây trên IPN Hand (2024–2027)

Ngày tổng hợp: 2026-10-26

Thư mục này lưu 6 bài báo gần đây có **thực nghiệm định lượng trực tiếp trên IPN Hand**. Các bài chỉ nhắc/cite dataset nhưng không chạy đánh giá đã bị loại.

## Kết luận nhanh để chọn baseline

Nếu bài toán của project là nhận dạng gesture từ 21 keypoint/skeleton trên split gốc của IPN Hand, các mốc đối chiếu hữu ích nhất là:

| Mức đối chiếu | Model | Kết quả nên dùng |
|---|---|---:|
| Baseline graph có giám sát | ST-SGCN (2024) | 89.00% isolated accuracy; paper không ghi rõ split |
| Baseline self-supervised | SkelMAE + ST-GCN (2024) | 92.80% isolated accuracy |
| Baseline online mạnh | AG-MAE (2025) | 73.93% Levenshtein accuracy; 19.4 ms gồm trích keypoint |
| Baseline offline mạnh | AG-MAE (2025) | 93.70% isolated accuracy |
| Baseline RNN rất nhẹ | MediaPipe + RNN (2025) | 86.38 ± 2.03% trên split 37/13; 58,614 tham số |

IPN HandS dùng annotation đã tinh chỉnh và 14 lớp, vì vậy mốc 91.2% của bài này **không được xếp trực tiếp** cùng bảng 13 lớp của IPN Hand gốc.

## Sáu bài đã tải và kết quả trích xuất

| Năm | Bài báo / PDF cục bộ | Model và đầu vào | Protocol IPN | Kết quả chính | Vị trí trong PDF |
|---:|---|---|---|---|---|
| ICIC 2026 / LNCS 2027 | [Adaptive Sparse Spatio-Temporal Graph Convolution with Cross-Attention for Hand Gesture Recognition](./2026_ASCA_STGCN.pdf) | ASCA-STGCN; skeleton; adaptive sparse graph + hai nhánh spatial/temporal + cross-attention | Isolated classification; paper nói dùng official split khi có nhưng không nêu lại subject IDs | **89.43% accuracy**; ST-SGCN 89.00%; 10.2M tham số; 11.7 ms/sequence được đo trên Briareo, không phải IPN | Table 1, trang in 510 (PDF 9); complexity ở Table 3, PDF 10 |
| 2025 | [Anatomically Guided Spatio-Temporal Masked Auto-Encoder for Online Hand Gesture Recognition](./2025_AG_MAE.pdf) | AG-MAE + ST-GCN; 21 keypoint 3D từ MediaPipe | Provided train/test split; báo cáo cả isolated/offline và continuous/online | **93.70% isolated accuracy**; **73.93% Levenshtein accuracy**; **19.4 ms** gồm khoảng 19.33 ms trích keypoint | Tables 4 và 6, PDF 9 |
| 2025 | [IPN HandS: Efficient Annotation Tool and Dataset for Skeleton-Based Hand Gesture Recognition](./2025_IPN_HandS.pdf) | LSTM 3 tầng, 256 hidden units/tầng; 42-D tọa độ 2D | IPN HandS đã sửa temporal boundary; 14 lớp; 37 train/13 test subjects; bỏ No-Gesture; trung bình 3 lần chạy | **91.2 ± 1.4% accuracy**, **78.0 ± 1.2% average recall**; MP Hands 87.5/62.7; MP Holistic 89.2/66.9 | Experimental setup PDF 14; Table 6, PDF 15 |
| 2025 | [Hand Gesture Recognition Using MediaPipe Landmarks and Deep Learning Networks](./2025_MediaPipe_Landmarks_RNN.pdf) | RNN nhẹ; 21 landmark 2D MediaPipe; 125 timesteps; chuẩn hóa theo wrist frame đầu | Chính: 10-fold subject-wise CV trên 50 subjects; tác giả chạy thêm split gốc 37/13 | **84.66 ± 1.09%** (10-fold); **86.38 ± 2.03%** (split 37/13); **58,614 tham số** | Table 1 và đoạn so sánh split gốc, PDF 5 |
| 2024 | [Skeleton-Based Self-Supervised Feature Extraction for Improved Dynamic Hand Gesture Recognition](./2024_SkelMAE_STGCN.pdf) | SkelMAE topology-aware + ST-GCN; 21 keypoint 3D MediaPipe | Isolated classification; tuân theo evaluation protocol của dataset | **92.80% accuracy**; ST-GCN 65.1%; ResNeXt-101 RGB-Flow 86.3%; Dist-Time 87.5% | Table VIII, PDF 8 |
| 2024 | [Spatio-Temporal Sparse Graph Convolution Network for Hand Gesture Recognition](./2024_ST_SGCN.pdf) | ST-SGCN; skeleton 3D; sparse directed spatial-temporal interactions | Isolated classification trên IPN Hand; **paper không nêu rõ train/test split** | **89.00% accuracy**; mô hình 10.4M tham số; số 12.5 ms/sequence và 0.1 GB VRAM được đo trên Briareo, không phải IPN | Table I/đoạn comparison, PDF 5; efficiency Table III, PDF 5 |

## Bảng đối chiếu theo bài toán và mức độ tương thích protocol

### A. IPN Hand gốc — isolated gesture classification

| Model | Năm | Modality | Accuracy | Ghi chú protocol |
|---|---:|---|---:|---|
| AG-MAE | 2025 | 3D keypoints | **93.70%** | Provided split; offline result |
| SkelMAE + ST-GCN | 2024 | 3D keypoints | **92.80%** | Theo protocol dataset |
| ASCA-STGCN | ICIC 2026 / LNCS 2027 | Skeleton | **89.43%** | Paper nói dùng official split khi có nhưng không nêu subject IDs; không báo multi-seed |
| ST-SGCN | 2024 | 3D skeleton | **89.00%** | Train/test split không được nêu trong paper; dùng như mốc tham khảo, không coi là so sánh tuyệt đối |
| MediaPipe + RNN | 2025 | 2D landmarks | **86.38 ± 2.03%** | Kết quả chạy lại trên split 37/13 |
| ResNeXt-101 | 2021 | RGB + optical flow | **86.32%** | Mốc gốc được các bài mới dùng để đối chiếu |

Đây không phải leaderboard tuyệt đối. Không dùng con số 84.66 ± 1.09% của MediaPipe + RNN để xếp chung nếu cần so sánh tuyệt đối, vì đó là trung bình 10-fold subject-wise CV chứ không phải một split 37/13 cố định. Tương tự, cần xác minh split từ code/supplement của ST-SGCN trước khi tuyên bố chênh lệch trực tiếp với các kết quả dùng split gốc.

### B. IPN Hand gốc — continuous/online recognition

| Model | Năm | Modality | Levenshtein accuracy | Inference |
|---|---:|---|---:|---:|
| AG-MAE | 2025 | 3D keypoints | **73.93%** | 19.4 ms, gồm keypoint extraction |
| TMMF | 2021 | RGB + optical flow | 68.12% | Không báo trong bảng AG-MAE |
| TSN-TSM | 2021 | RGB + segmentation | 65.27% | 15.2 ms |
| ResNeXt-101 | 2021 | RGB + optical flow | 42.47% | 53.7 ms |

Đây là bảng phù hợp nếu pipeline chạy trực tiếp trên video liên tục và phải vừa tìm ranh giới vừa nhận dạng gesture. Không so Levenshtein accuracy với isolated classification accuracy.

### C. IPN HandS — refined 14-class skeleton annotations

| Annotation dùng để train/test LSTM | Accuracy | Average recall |
|---|---:|---:|
| IPN HandS refined | **91.2 ± 1.4%** | **78.0 ± 1.2%** |
| MediaPipe Holistic | 89.2 ± 0.6% | 66.9 ± 0.7% |
| MediaPipe Hands | 87.5 ± 1.0% | 62.7 ± 2.0% |

Average recall nên được báo cáo cùng accuracy vì tập refined vẫn mất cân bằng mạnh; accuracy đơn lẻ có thể bị hai lớp pointing chi phối.

## Khuyến nghị cho baseline của project

1. Báo cáo riêng hai bài toán: **isolated accuracy/macro recall** và **continuous Levenshtein accuracy**.
2. Với pipeline 21 landmark, dùng SkelMAE 92.80% và AG-MAE 93.70% làm mốc chính; dùng ST-SGCN 89.00% như mốc tham khảo cho đến khi xác minh được split từ code/supplement.
3. Dùng MediaPipe + RNN 86.38 ± 2.03% làm baseline nhẹ, gần với hệ thống triển khai thực tế.
4. Nếu dùng annotation IPN HandS 14 lớp, chỉ so với bảng IPN HandS; không đặt cạnh kết quả 13 lớp mà không ghi rõ protocol.
5. Khi báo inference time, tách thời gian trích MediaPipe khỏi thời gian model nếu có thể. AG-MAE cho thấy gần như toàn bộ 19.4 ms nằm ở bước trích keypoint.

## Nguồn trực tuyến

- ASCA-STGCN: [Springer](https://link.springer.com/chapter/10.1007/978-981-92-3520-9_40)
- AG-MAE: [DOI IEEE 3DV 2025](https://doi.org/10.1109/3DV66043.2025.00133), [HAL](https://hal.science/hal-05454159v1)
- IPN HandS: [MDPI](https://www.mdpi.com/2076-3417/15/11/6321)
- MediaPipe landmarks + RNN: [SCITEPRESS](https://www.scitepress.org/Papers/2025/130535/130535.pdf)
- SkelMAE: [DOI IEEE FG 2024](https://doi.org/10.1109/FG59268.2024.10581975), [HAL](https://hal.science/hal-04551730v4)
- ST-SGCN: [DOI IEEE FG 2024](https://doi.org/10.1109/FG59268.2024.10581903), [HAL](https://hal.science/hal-04574611v3)
- Dataset và baseline gốc: [IPN Hand](https://gibranbenitez.github.io/IPN_Hand/), [official results](https://gibranbenitez.github.io/IPN_Hand/Results.html)

## Nhật ký thay đổi và kiểm chứng

- Thay đổi: thêm 6 PDF và tài liệu tổng hợp baseline IPN Hand 2024–2027 (ASCA thuộc ICIC 2026 nhưng bản LNCS ghi năm 2027).
- Lý do: cung cấp bộ baseline có thể trích dẫn và đối chiếu cho thí nghiệm của project.
- Tương thích/an toàn: không thay đổi code, cấu hình hoặc runtime; không thêm notebook/result sinh tự động. Các PDF là bản truy cập công khai từ nhà xuất bản hoặc kho tác giả.
- Kiểm chứng: xác nhận cả 6 tệp có header `%PDF`, đọc được bằng `pypdf`, và đối chiếu các số liệu với bảng/đoạn tương ứng trong từng PDF. Các metric/protocol khác nhau đã được tách riêng để tránh so sánh sai.
