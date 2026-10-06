# Người 1 — kiểm tra dữ liệu IPN trước khi cắt clip

## Kết quả kiểm tra ngày 06/10/2026

Đã đọc tuần tự toàn bộ **800.488 frame của 200 video** bằng OpenCV. Kết quả kiểm tra: **chưa đạt để cắt hàng loạt theo annotation**.

| Hạng mục | Kết quả |
|---|---:|
| Archive `videos01.tgz`–`videos05.tgz` hiện có | 5/5 |
| Video mở và đọc được frame | 200/200 |
| Video train/test theo danh sách tải về | 148/52 |
| Nhóm subject train/test suy ra từ tên file | 37/13, không trùng nhóm |
| Tổng annotation | 5.649 |
| Đoạn thuộc 13 lớp cử chỉ, bỏ D0X | 4.218 |
| Video khớp cả số frame danh sách và header | 89 |
| Video có số frame giải mã khác danh sách/metadata | 107 |
| Video chỉ lệch header, vẫn khớp danh sách/metadata | 4 |
| Đoạn annotation kết thúc sau độ dài giải mã | 89 đoạn trong 89 video |
| Video có annotation kết thúc sớm hơn metadata đúng 1 frame | 14 |

Trong 89 đoạn vượt độ dài giải mã, có **7 đoạn cử chỉ và 82 đoạn D0X**. Các con số lệch frame/header và thiếu phần đuôi có thể cùng xảy ra trên một video, không cộng chúng thành tổng số video lỗi.

Không thiếu/trùng video trong đối chiếu danh sách; annotation train + test khớp danh sách tổng; mapping nhãn và độ dài từng đoạn hợp lệ. Cả 8 file annotation giải nén khớp SHA256 với các entry trong ZIP annotation đã tải. Bằng chứng nằm ở `external_data/ipn_audit/annotation_archive_checksums.json`; kiểm tra này không xác thực ZIP với checksum chuẩn từ tác giả.

Ví dụ `1CM1_1_R_#217`: metadata ghi 3.855 frame, giải mã được 3.983, header ghi 3.987. Ngược lại, `1CM1_1_R_#219` có metadata 3.922 nhưng giải mã và annotation kết thúc ở 3.921. Vì các dạng lệch khác nhau, chưa có căn cứ áp dụng một phép cộng/trừ frame chung.

Bước tiếp theo là đối chiếu video/annotation nguồn và kiểm tra đồng bộ thời gian trên các trường hợp lệch; chưa tự sửa annotation hay cắt toàn bộ. Sai lệch số frame chưa chứng minh mọi video bị hỏng hoặc mọi nhãn bị sai. Chưa thực hiện chia validation, cắt clip hay trích keypoint.

## Tạo danh sách để xem bằng mắt

Sau khi đã có `external_data/ipn_audit/videos.csv`, chạy:

```powershell
.\.venv\Scripts\python.exe scripts\build_ipn_frame_review.py
```

Script tạo `external_data/ipn_audit/FRAME_REVIEW.csv` và `FRAME_REVIEW.md`. Đây là danh sách kiểm tra cục bộ, không commit lên Git. Kết quả hiện tại được ưu tiên như sau:

- `P0-GESTURE`: 7 video có annotation của cử chỉ (`B0A`–`G11`) vượt frame cuối giải mã. Kiểm tra trước.
- `P0-D0X`: 82 video chỉ có đoạn nền vượt frame cuối. Có thể kiểm tra sau vì thí nghiệm 13 lớp bỏ `D0X`.
- `P1`: 18 video có số frame giải mã khác metadata nhưng annotation chưa vượt frame cuối.
- `P2`: 4 video chỉ lệch số frame header; dùng số frame giải mã để đối chiếu.
- `OK`: 89 video làm mẫu đối chứng.

Bảy video P0-GESTURE hiện cần kiểm tra trước là: `1CM42_31_R_#132` (B0B), `1CV12_21_R_#112` (B0A), `1CV12_23_R_#117` (G01), `1CV12_23_R_#118` (G06), `1CV12_23_R_#119` (G09), `4CM11_29_R_#57` (G08), `4CM11_29_R_#60` (G05).

Đây là hai bước khác nhau: script tạo danh sách và tính chênh lệch tự động; người 1 vẫn phải mở video, xem vùng gần `t_start`/`t_end`, rồi điền `review_status` trong CSV (`confirmed`, `annotation_ok` hoặc `needs_investigation`). Có thể mở video bằng VLC; với 30 FPS, frame gần đúng tương ứng `giây = frame / 30`. Không sửa `Annot_List.txt` trong lúc kiểm tra.

Script được rà soát độc lập; đã kiểm tra giải mã video thử 7 frame, phát hiện file không đọc được, phát hiện lệch header riêng lẻ và tránh dùng báo cáo thành công cũ khi đầu vào bị thiếu.

## Chạy lại kiểm tra

Chạy từ thư mục gốc project:

```powershell
.\.venv\Scripts\python.exe scripts\audit_ipn_data.py
```

Script chỉ đọc video/annotation nguồn, không cắt clip, không tạo validation và không sửa nhãn. OpenCV phải có trong môi trường Python. Mặc định đọc toàn bộ frame của từng video với 4 worker; có thể giảm tải bằng `--workers 2`.

Kết quả được lưu trong `external_data/ipn_audit/` (được Git ignore):

- `REPORT.md`: tổng hợp, số đoạn từng lớp và danh sách vấn đề.
- `videos.csv`: kết quả từng video, split, nhóm subject, số frame metadata, header và giải mã thực tế.
- `audit.json`: kết quả có cấu trúc để xử lý tiếp.

Mã thoát `0` là đạt các kiểm tra; `1` là phát hiện vấn đề dữ liệu. Không coi mã thoát `1` là lý do tự sửa nhãn hay bỏ qua video.

Các kiểm tra gồm tên/số lượng video; danh sách train/test; metadata; mapping nhãn; annotation trùng, sai split, sai khoảng frame; khoảng trống/chồng lấn; số frame giải mã; và sự hiện diện của năm archive `videos01.tgz`–`videos05.tgz`. Archive chỉ được kiểm tra tồn tại, chưa kiểm tra toàn bộ nội dung nén.

## Cách đọc kết quả

- `expected_frames`: số frame trong `Video_TrainList.txt`/`Video_TestList.txt`, được đối chiếu với metadata.
- `header_frames`: số frame OpenCV đọc từ header video.
- `decoded_frames`: số frame thực sự đọc được tuần tự.
- `annotation_match`: số frame thực tế khớp danh sách video.
- `header_match`: số frame thực tế khớp header.
- `last_annotated_frame`: frame kết thúc lớn nhất trong annotation của video.
- `annotations_past_decoded_end`: số đoạn annotation kết thúc sau frame cuối đọc được; có nguy cơ cắt thiếu đoạn.
- `valid`: video mở được và số frame thực tế, danh sách, header khớp nhau. Không chứng minh nội dung cử chỉ/nhãn chính xác bằng mắt.

Lệch header riêng lẻ chưa chứng minh video hỏng. Khi số frame thực tế khác annotation, cần kiểm tra đồng bộ thời gian trước khi dùng chỉ số frame để cắt. Không tự cộng/trừ frame hay sửa `t_end` chỉ để các con số khớp nhau.

Subject được suy ra bằng phần tên trước `_R_#` hoặc `_L_#`, ví dụ `1CM1_1`. Đây là quy ước nhóm từ tên file, không phải xác minh danh tính người bằng hình ảnh. Script ghi rõ giới hạn này và coi nguồn gốc các file annotation tải về là giả định; chưa xác thực bằng checksum chuẩn từ tác giả.

Kiểm tra tự động không thay thế xem clip bằng mắt. Riêng người 1 vẫn cần xem mẫu B0A, B0B, G01, G02 ở bước tiếp theo.

Để cập nhật định dạng báo cáo từ kết quả đã giải mã mà không đọc lại video:

```powershell
.\.venv\Scripts\python.exe scripts\audit_ipn_data.py --refresh-report
```

Chỉ dùng refresh khi dữ liệu nguồn và annotation chưa thay đổi. Nếu đã thay dữ liệu, chạy lại kiểm tra đầy đủ. Khi chạy lại đầy đủ, báo cáo cũ được đánh dấu chưa hoàn tất và CSV lưu dần các video đã xử lý.
