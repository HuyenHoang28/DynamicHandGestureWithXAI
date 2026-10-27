# Legacy artifact path map

Các báo cáo lịch sử được giữ nguyên câu chữ để không làm sai provenance tại thời điểm viết. Khi mở artifact local sau lần dọn ngày 2026-10-23, dùng ánh xạ sau:

| Đường dẫn lịch sử | Đường dẫn hiện tại |
|---|---|
| `final/outputs/<path>` | `results/experiments/legacy/final_outputs/<path>` |
| `final/Experiment/reports/<path>` | `results/experiments/legacy/final_experiment_reports/<path>` |
| `benchmark_data/<path>` | `results/experiments/date27/keypoint_benchmark_data/<path>` |
| `notebooks/IPN_Symbolic_V2_Colab_res.ipynb` | `results/experiments/ipn13/notebooks/IPN_Symbolic_V2_Colab_res.ipynb` |
| ZIP kết quả IPN nằm trực tiếp dưới `external_data/` | `results/experiments/ipn13/artifact_zips/` |

Raw dataset và input đang chạy không đổi:

- `external_data/ipn_full`, `external_data/ipn_processed`, `external_data/ipn_archives`;
- `final/data/Input`;
- `final/data/Output/checkpoints`.

Output mới phải ghi vào `results/runs/`. Final bundle dùng để báo cáo nằm trong `results/final/date27_v4_validation_selected/`.
