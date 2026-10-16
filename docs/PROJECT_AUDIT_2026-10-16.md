# Project audit — 2026-10-16

## Kết luận

Code path chính của mô hình 27 lớp chạy end-to-end với artifact cục bộ. Nhánh IPN symbolic V2 đã có schema, matcher, validator và regression test, nhưng chưa thể qua gate P4/P5 vì dữ liệu cục bộ không đủ để sinh lại toàn bộ graph V2.

## Việc đã hoàn thiện trong đợt audit

- Khóa pytest vào thư mục `tests/` và đổi tên hai CLI function để không bị pytest collect nhầm.
- Sửa mapping graph giữa precompute và evaluator, gồm layout filename theo subject; chặn traversal chứa `..`.
- Không cho motion/coordination nối qua cửa sổ mất keypoint; thêm conflict group cho trạng thái góc khuỷu tay.
- Bắt buộc event schema V2 có ít nhất một concept `required`; `max_occurrences` là ràng buộc cứng; hỗ trợ fallback atomic IDs của graph cũ.
- Sửa `rerun_keypoints.ps1`: resolve đúng absolute output path và chỉ bật `--no-cache` khi được yêu cầu.
- Keypoint runner trả exit code lỗi khi còn video thất bại; wrapper train/test/demo cũng fail-fast theo exit code Python.
- `create_ipn_manifest.py` dùng `--output-dir` thật sự cho clip/keypoint path và mặc định loại subject `1CM42_3` theo protocol đã khóa.
- Đưa benchmark extractor và manifest model ra khỏi `.gitignore` để clone sạch có đủ file mà wrapper gọi.
- Thêm `requirements-dev.txt`, `pytest.ini`, hướng dẫn test và tham số override checkpoint/RTMW paths.
- Thêm smoke evaluator tái tạo predicate/event/instance graph V2 trực tiếp từ 26 mẫu IPN train/val đã đóng gói; hoàn tất visual review 13 lớp và sửa G05/G06 theo hệ tọa độ ảnh.

## Bằng chứng kiểm chứng

- `python -m pytest -q`: **40 passed**.
- `python -m compileall -q final/code scripts keypoint_extractor`: pass.
- Parse toàn bộ 27 file PowerShell: pass.
- `python scripts/verify_repo.py`: pass trên artifact cục bộ.
- `python scripts/eval_ipn_symbolic_examples_v2.py`: chạy đủ 26 mẫu smoke IPN, không thiếu lớp.
  - train accuracy `0,0769`, macro-F1 `0,0256`;
  - validation accuracy `0`, macro-F1 `0`;
  - đây là smoke cohort một mẫu/lớp/split, không thay thế full validation 589 clip.
- Báo cáo smoke cục bộ: `final/outputs/ipn_symbolic_v2_smoke/report.json` (generated artifact, ignored); report lưu hash template, từng normalized input, toàn cohort và các module symbolic để kiểm tra provenance.

## Phần còn bị chặn bởi dữ liệu/môi trường

### P0 — IPN symbolic V2 chưa có cohort hợp lệ

Validator hiện báo:

- thiếu `1101/4133` instance graph, toàn bộ test split;
- `3032` graph còn lại là schema cũ, thiếu `schema_version=2`;
- `external_data/ipn_processed/keypoints` chỉ có 40 JSON cục bộ, không đủ để regenerate 4.133 graph.

Do đó chưa được báo metric P4 rule-only hoặc chạy P5 causal ablation. Smoke 26 mẫu đã chạy và cho thấy predicate vẫn phát sinh tín hiệu trái chiều trên cùng clip; sửa YAML đơn thuần chưa đủ để qua gate. Cần lấy đủ raw keypoint/cache đúng split và hash, chạy lại precompute V2, rồi yêu cầu validator pass 100% trước evaluation.

### P1 — Môi trường Python chưa khóa hoàn toàn

`.venv` hiện có hai lỗi `pip check`: thiếu `chumpy` cho MMPose và `opencv-python 5.0.0.93` yêu cầu NumPy 2 trong khi môi trường dùng NumPy 1.26.4. Chưa tự động thay package vì việc chọn Torch/CUDA/MMCV/MMPose version phải theo máy đích. Cần tạo lock/constraints riêng cho CPU và CUDA sau khi xác nhận một fresh install.

### P2 — Khoảng trống kiểm thử và CI

Coverage audit khoảng 54%; dataset loading, normalization, model forward/train, RTMW extraction và UI/video pipeline còn ít test. Repo chưa có CI workflow. Nên thêm smoke test model-forward với fixture nhỏ và workflow chạy `pytest`, `compileall`, `pip check` trên môi trường đã khóa.

## Đánh giá nguyên nhân

- **Repo/data friction — confidence cao:** artifact IPN thiếu test graph và graph hiện có là schema cũ; validator chứng minh trực tiếp.
- **Verification gap — confidence cao:** trước audit pytest collect nhầm CLI và wrapper PowerShell có thể không truyền lỗi Python ra ngoài.
- **Protocol drift — confidence cao:** manifest exclusion/path và artifact hash không đồng nhất với protocol/documentation.
- **Task complexity — confidence cao:** P4/P5 cần regenerate dữ liệu và chạy thí nghiệm nhiều seed; đây không phải phần có thể hoàn tất trung thực chỉ bằng sửa source.
