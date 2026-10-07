"""Summarize existing local IPN demo artifacts; no model training or inference."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final/outputs/ipn_smoke_20261005"
REPORTS = ROOT / "docs/reports/ipn"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    manifest = read(ROOT / "external_data/ipn_smoke_20261005/manifest.json")
    batch = read(OUT / "batch_summary.json")
    timings = {r["clip_id"]: r for r in batch["clips"]}
    rows = []
    tiles = []
    for item in manifest:
        folder = OUT / item["clip_id"]
        pred = read(folder / "prediction.json")
        norm = read(folder / "normalized_keypoints.json")["frames"]
        raw = read(next((folder / "keypoint_extraction/keypoints").rglob("*.json")))
        if len(norm) != item["frames"] or len(raw) != item["frames"]:
            raise RuntimeError(f"Frame count mismatch: {item['clip_id']}")
        if timings[item["clip_id"]]["status"] not in ("ok", "reused"):
            raise RuntimeError(f"Failed clip: {item['clip_id']}")
        valid = sum(bool(frame["valid"]) for frame in norm)
        detected = sum(bool(frame.get("instances")) for frame in raw)
        evidence = read(folder / "model_evidence.json")
        row = {**item, "detected_frames": detected, "valid_frames": valid,
               "processed_frames": len(norm), "valid_ratio": valid / len(norm),
               "detection_ratio": detected / len(raw),
               "neural_gate": evidence["final_class_neural_gate"],
               "elapsed_s": timings[item["clip_id"]]["elapsed_s"]}
        for branch in ("final", "neural", "kg"):
            row[branch + "_label"] = pred[branch]["prediction_label"]
            row[branch + "_confidence"] = pred[branch]["confidence"]
        row["top3"] = "; ".join(f"{p['label_name']}:{p['probability']:.4f}" for p in pred["final"]["top_k"][:3])
        rows.append(row)

        cap = cv2.VideoCapture(str(ROOT / item["clip_path"]))
        middle = item["frames"] // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read thumbnail: {item['clip_id']}")
        if middle < len(raw) and raw[middle].get("instances"):
            points = raw[middle]["instances"][0].get("keypoints_by_name", {})
            for name, point in points.items():
                xy = point.get("xy")
                if xy and float(point.get("score", 0)) >= 0.3:
                    color = (0, 255, 255) if "shoulder" in name else (0, 255, 0)
                    cv2.circle(frame, (round(xy[0]), round(xy[1])), 3, color, -1)
        tile = np.full((280, 320, 3), 245, dtype=np.uint8)
        tile[40:280] = cv2.resize(frame, (320, 240))
        cv2.putText(tile, f"{item['label']} -> {row['final_label']}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1)
        cv2.putText(tile, f"p={row['final_confidence']:.3f}; valid={row['valid_ratio']:.0%}", (5, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        tiles.append(tile)

    total = sum(r["processed_frames"] for r in rows)
    counts = Counter(r["final_label"] for r in rows)
    controls = [r for r in rows if r["label"] == "D0X"]
    checkpoint = ROOT / "final/data/Output/checkpoints/cross_attention_v4_template170_retrain_next/best_test.pt"
    metrics = dict(clips=len(rows), source_videos=len({r["video"] for r in rows}), frames=total,
                   valid_frames=sum(r["valid_frames"] for r in rows),
                   detected_frames=sum(r["detected_frames"] for r in rows),
                   final_label_counts=dict(counts),
                   branch_disagreements=sum(r["neural_label"] != r["kg_label"] for r in rows),
                   confidence_ge_08=sum(r["final_confidence"] >= .8 for r in rows),
                   d0x_predicted_no_gesture=sum(r["final_label"] == "no_gesture" for r in controls),
                   batch_elapsed_s=batch["elapsed_s"],
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report_data.json").write_text(json.dumps(dict(metrics=metrics, clips=rows), indent=2), encoding="utf-8")
    with (OUT / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(OUT / "keypoint_contact_sheet.jpg"), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]))

    lines = [
        "# Báo cáo kiểm thử IPN Hand — checkpoint V4 hiện tại",
        "Ngày chạy: 05/10/2026. Thiết bị: CPU; PyTorch không phát hiện CUDA.",
        "## Kết quả",
        f"Hoàn tất {len(rows)}/{len(manifest)} clip, gồm 13 nhãn cử chỉ và 3 đoạn D0X, từ {metrics['source_videos']} video trong danh sách test IPN. Tổng cộng {total} frame.",
        f"Phát hiện người: {metrics['detected_frames']}/{total} frame ({metrics['detected_frames']/total:.1%}). Chuẩn hóa hợp lệ theo hai vai: {metrics['valid_frames']}/{total} frame ({metrics['valid_frames']/total:.1%}).",
        f"Hai nhánh neural và KG dự đoán khác nhau ở {metrics['branch_disagreements']}/{len(rows)} clip. Có {metrics['confidence_ge_08']} clip có confidence đầu ra ít nhất 0,8.",
        f"Thời gian xử lý batch trong tiến trình: {metrics['batch_elapsed_s']:.1f} giây ({metrics['batch_elapsed_s']/60:.1f} phút). Bao gồm nạp RTMW ở clip đầu, trích xuất keypoint, nạp V4 theo clip, suy luận và ghi các artifact; không bao gồm giải nén/cắt clip hay thời gian import ban đầu.",
        f"Trong 3 đoạn D0X, {metrics['d0x_predicted_no_gesture']} đoạn được dự đoán no_gesture. Đây chỉ là đối chiếu định tính giữa hai khái niệm gần nhau, chưa xác nhận hai định nghĩa nhãn tương đương.",
        "Phân bố dự đoán: " + ", ".join(f"{label}: {count}" for label, count in counts.most_common()) + ".",
        "## Kết quả từng clip",
        "Confidence là giá trị đầu ra của model, không phải độ chính xác hoặc xác suất đúng đã được kiểm định trên IPN.",
        "| IPN | Video | Frame | Final | Confidence | Neural | KG | Vai hợp lệ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['label']} | {r['video']} | {r['start_frame']}–{r['end_frame']} | {r['final_label']} | {r['final_confidence']:.3f} | {r['neural_label']} | {r['kg_label']} | {r['valid_ratio']:.1%} |")
    lines += [
        "## Cách thực hiện và phạm vi",
        "Nguồn: các ZIP video và annotation mà người dùng đã đặt trong external_data. Chọn annotation từ Annot_TestList.txt; nhãn lấy nguyên mã IPN.",
        "Chọn cố định 4 video, luân phiên theo lớp; lấy đoạn ngắn nhất của lớp trong video đã chọn. Chọn thêm một đoạn D0X ít nhất 15 frame trên mỗi video trong 3 video đầu. Lựa chọn dựa trên nhãn và độ dài trước khi chạy, không dựa trên kết quả dự đoán. Mẫu thiên về đoạn ngắn, không đại diện cho toàn bộ IPN.",
        "Tách đủ frame t_start đến t_end, đánh số từ 1 và bao gồm cả hai đầu. Clip được mã hóa XVID, giữ FPS và độ phân giải; kiểm tra giải mã đủ số frame. Không cắt bớt độ dài cử chỉ.",
        "Dùng run_demo của demo_video_v4.py: RTMW → normalize → predicates → events → instance graph → V4 với 27 template hiện tại. Giữ mirror_swap=True như pipeline có sẵn. Dùng cùng tiến trình để tái sử dụng RTMW; mỗi clip vẫn nạp V4 theo hàm demo hiện tại.",
        "Checkpoint: final/data/Output/checkpoints/cross_attention_v4_template170_retrain_next/best_test.pt. Đây là checkpoint được chọn theo test của bộ dữ liệu gốc, nên không dùng kết quả này làm đánh giá benchmark độc lập.",
        "SHA256: " + metrics["checkpoint_sha256"],
        "## Diễn giải và giới hạn",
        "Pipeline hiện tại có thể tiếp nhận và xử lý các clip IPN đã thử. Mức độ đúng của cử chỉ IPN chưa được xác định: model chỉ xuất 27 nhãn của project, trong khi bộ mẫu dùng 13 nhãn cử chỉ IPN và D0X. Không tính accuracy, macro-F1 hoặc confusion matrix giữa hai bộ nhãn khác nhau.",
        "Chỉ số vai hợp lệ đo điều kiện kỹ thuật của normalize (hai vai score ít nhất 0,3 và khoảng cách khác 0), không đo sai số tọa độ keypoint. RTMW có thể xuất score lớn hơn 1; normalize kẹp về [0,1]. Vì vậy score không được diễn giải như xác suất định vị đúng.",
        "keypoint_contact_sheet.jpg hiển thị frame giữa từng clip, chấm vàng là vai và chấm xanh là các keypoint khác có score ít nhất 0,3. Một ảnh giữa clip không chứng minh độ chính xác của toàn bộ chuỗi chuyển động.",
        "Một mẫu mỗi nhãn và 3 đối chứng chưa đủ để kết luận khả năng tổng quát hóa, độ ổn định hoặc độ chính xác ngoài miền. Kết quả phụ thuộc cách cắt clip, người biểu diễn và bối cảnh.",
        "## Bước tiếp theo",
        "1. Với mục tiêu thử khả năng chạy trên video IPN: xem contact sheet và các clip gốc, đặc biệt vị trí bàn tay/vai. Phép thử kỹ thuật trên mẫu này đã hoàn tất; chưa cần train chỉ để chạy video.",
        "2. Muốn đánh giá 27 cử chỉ đang có trên bối cảnh mới: thu thêm video có đúng 27 nhãn đó, gán nhãn độc lập, rồi chạy checkpoint cố định để tính accuracy, macro-F1 và confusion matrix.",
        "3. Muốn báo cáo accuracy trên 13 cử chỉ IPN: giữ khung kiến trúc, cấu hình đầu ra 13 lớp và train/fine-tune trên train IPN; nhánh V4 cần template tương ứng IPN. Tách validation từ train theo người, giữ test độc lập. Bắt đầu neural-only để có baseline, sau đó đánh giá V4. Nếu muốn xử lý D0X, phải quy định rõ bài toán nền/không cử chỉ trước khi train.",
        "4. Tính thời gian CPU ở đây chỉ để tham khảo vận hành; lần đầu bao gồm khởi tạo và mỗi clip có I/O, symbolic matching, ghi bằng chứng. Không coi đó là FPS thời gian thực của riêng mạng V4.",
        "## Chạy lại",
        ".\\.venv\\Scripts\\python.exe scripts/prepare_ipn_smoke.py",
        ".\\.venv\\Scripts\\python.exe scripts/run_ipn_smoke.py --manifest external_data/ipn_smoke_20261005/manifest.json --output-dir final/outputs/ipn_smoke_20261005 --device cpu",
        ".\\.venv\\Scripts\\python.exe scripts/report_ipn_smoke.py",
        "Dữ liệu, script kiểm thử và kết quả được Git ignore. Không thay đổi kiến trúc, trọng số hoặc template của project.",
    ]
    formatted = []
    for i, line in enumerate(lines):
        separator = "\n" if i and line.startswith("|") and lines[i-1].startswith("|") else "\n\n"
        formatted.append((separator if i else "") + line)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "REPORT_VI.md").write_text("".join(formatted), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
