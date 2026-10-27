# Tái cấu trúc repository ngày 2026-10-23

## Thay đổi

- Giữ code chạy chính trong `final/code/`, input/checkpoint trong `final/data/`, raw IPN trong `external_data/` để không phá loader hiện tại.
- Chuyển output cũ, benchmark, notebook có output và ZIP thí nghiệm IPN sang `results/experiments/`.
- Lưu run mới trong `results/runs/`; đóng gói checkpoint chọn theo validation và evidence chính trong `results/final/date27_v4_validation_selected/`.
- Chuyển manuscript sang `docs/paper/DATE_NeuroSymbolic_Manuscript.pdf`.
- Notebook Colab báo cáo chính nằm tại `notebooks/DATE27_NeuroSymbolic_Baseline_Comparison_Colab.ipynb`.
- Wrapper test fusion mặc định đổi từ `best_test.pt` sang `best.pt`. Muốn dùng checkpoint test-selected phải truyền rõ `-AllowTestSelected`.

## Lý do

Cây cũ trộn source code, output lịch sử, artifact IPN và evidence báo cáo. Cấu trúc mới tách run tái lập, final hợp lệ và thí nghiệm exploratory, đồng thời ngăn vô tình báo cáo checkpoint được chọn theo test.

## Tương thích và an toàn

- Không xóa raw dataset, tensor cache, graph, môi trường Python hoặc checkpoint nguồn.
- Các artifact chỉ được move, không bị ghi đè; `scripts/organize_local_artifacts.ps1` kiểm tra mọi source/destination nằm trong workspace trước khi move và có thể chạy lại an toàn.
- Đường dẫn output mặc định thay đổi sang `results/`; lệnh truyền `--output`/`-Output` cũ vẫn dùng được.
- `results/` chứa file sinh ra và checkpoint lớn nên được Git ignore; README và tài liệu provenance vẫn được version hóa.
- Tài liệu lịch sử có thể còn ghi đường dẫn cũ; ánh xạ đầy đủ nằm trong `docs/LEGACY_ARTIFACT_PATH_MAP.md`.

## Validation

- `scripts/verify_repo.py`: `OK` trước khi tái cấu trúc.
- Full unit suite: `123 passed`, 3 warning PyTorch đã biết.
- Full DATE evaluation trên 1.229 clip hoàn tất bằng `best.pt`; kết quả và hash được lưu trong final bundle.
- Notebook Colab parse JSON hợp lệ, code cell qua kiểm tra cú pháp và không chứa output nhúng.

## Audit bổ sung và dọn dẹp cuối (2026-10-23)

### Thay đổi

- Xóa `.venv-1/` (venv MSYS2 Python 3.14 thừa, 12 MB, không script/tài liệu nào tham chiếu) và `.pytest_cache/`, các `__pycache__` ngoài môi trường. `.venv/` (môi trường đang dùng) và `node_modules/` (cần cho `npm run report`) được giữ lại.
- Chuyển `docs/research/edge_candidates_cpu_smoke.json` (output smoke sinh tự động) sang `results/experiments/legacy/`. `docs/research/` chỉ còn tài liệu audit; manuscript nằm ở `docs/paper/`. Không có file nào tham chiếu đường dẫn cũ.
- Đối chiếu bảng baseline trong `DATE27_BASELINE_COMPARISON_2026-10-23_VI.md` với manuscript PDF (Table 2/3): SignBart 88,06 ± 0,76, Sparse-ST-GCN 92,94 ± 0,40, ProtoGCN 93,28 ± 0,27, Transformer 94,22, Proposed 96,40 ± 0,17 đều khớp.

### Lưu ý khi báo cáo

- Run 3 của cột Proposed trong Table 3 là `96,18%`, trùng đúng accuracy của `best_test.pt` (chọn theo test). Chưa đủ để kết luận, nhưng cần xác minh provenance 10 run của manuscript trước khi dùng `96,40%` làm claim tái lập; số tái lập hợp lệ trong repo là `94,87%` (Final) / `95,04%` (KG head).
- Notebook IPN-13 ghim commit `e817f16` (đã có trên `origin/HuyenHoang`); các CLI flag mà notebook dùng (`--neural-checkpoint`, `--select-metric`, `--eval-branch`, `--splits`, các ngưỡng predicate...) đã được đối chiếu với source ở đúng commit đó. Code `final/code` không bị sửa sau commit này nên notebook không phụ thuộc thay đổi local chưa commit.

### Tương thích và an toàn

Chỉ xóa artifact có thể tái tạo và di chuyển một file smoke; không đổi code, dữ liệu, checkpoint hay runtime.

### Validation

`scripts/verify_repo.py`: `OK`; full unit suite sau dọn dẹp: `123 passed`, 3 warning PyTorch đã biết.
