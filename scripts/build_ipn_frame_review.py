"""Build a human-review queue from audit_ipn_data.py's videos.csv."""
import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'external_data/ipn_audit')
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    source = audit / 'videos.csv'
    if not source.exists():
        parser.error(f'Missing {source}; run audit_ipn_data.py first')
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 200:
        parser.error(f'Expected 200 completed video rows, found {len(rows)}')
    annotations = {}
    annotation_file = ROOT / 'external_data/ipn_full/annotations/Annot_List.txt'
    with annotation_file.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            annotations.setdefault(row['video'], []).append(row)
    output = []
    for row in rows:
        decoded = int(row['decoded_frames'])
        expected = int(row['expected_frames'])
        header = int(row['header_frames'])
        tail = int(row['annotations_past_decoded_end'])
        tail_rows = [a for a in annotations.get(row['video'], []) if int(a['t_end']) > decoded]
        tail_labels = sorted({a['label'] for a in tail_rows})
        tail_ranges = ';'.join(f"{a['label']}:{a['t_start']}-{a['t_end']}" for a in tail_rows)
        gesture_tail = any(label != 'D0X' for label in tail_labels)
        if tail and gesture_tail:
            priority = 'P0-GESTURE: gesture annotation extends beyond decoded video'
        elif tail:
            priority = 'P0-D0X: background annotation extends beyond decoded video'
        elif decoded != expected:
            priority = 'P1: decoded count differs from annotation/metadata'
        elif header != decoded:
            priority = 'P2: header-only mismatch'
        else:
            priority = 'OK: no count mismatch'
        row.update(priority=priority, tail_labels=';'.join(tail_labels), tail_ranges=tail_ranges,
                   decoded_minus_expected=decoded - expected,
                   header_minus_decoded=header - decoded,
                   review_status='pending')
        output.append(row)
    output.sort(key=lambda r: ({'P0-GESTURE': 0, 'P0-D0X': 1, 'P1': 2, 'P2': 3, 'OK': 4}[r['priority'].split(':', 1)[0]], r['video']))
    fields = list(output[0])
    with (audit / 'FRAME_REVIEW.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    counts = Counter(r['priority'].split(':', 1)[0] for r in output)
    lines = ['# IPN frame review queue', '', 'Mở `FRAME_REVIEW.csv`, kiểm tra theo thứ tự priority và điền `review_status` bằng `confirmed`, `annotation_ok` hoặc `needs_investigation`.', '',
             '| Nhóm | Số video | Cách kiểm tra |', '|---|---:|---|',
             f'| P0-GESTURE — cử chỉ vượt frame cuối | {counts["P0-GESTURE"]} | Ưu tiên cao nhất; mở đúng đoạn cử chỉ và kiểm tra frame cuối |',
             f'| P0-D0X — nền vượt frame cuối | {counts["P0-D0X"]} | Có thể kiểm tra sau vì thí nghiệm 13 lớp bỏ D0X |',
             f'| P1 — decoded khác metadata | {counts["P1"]} | Đối chiếu frame cuối và một mốc annotation gần cuối |',
             f'| P2 — chỉ lệch header | {counts["P2"]} | Thường chỉ là header AVI; kiểm tra decoded count và annotation |',
             f'| OK | {counts["OK"]} | Có thể dùng làm mẫu đối chứng |', '',
             '## Quy trình kiểm tra từng video', '',
             '1. Mở video bằng VLC hoặc trình phát có hiển thị số frame/thời gian.',
             '2. Đổi frame sang thời gian gần đúng bằng `giây = frame / 30`; video IPN khoảng 30 FPS.',
             '3. Đọc `Annot_List.txt` để lấy `label`, `t_start`, `t_end` của video.',
             '4. Kiểm tra vùng gần `t_start` và `t_end`, tập trung đoạn cuối nếu P0.',
             '5. Ghi kết luận vào `review_status` và cột ghi chú nếu cần. Không sửa annotation gốc.', '',
             'Các giá trị `decoded_minus_expected` và `header_minus_decoded` là số frame chênh lệch; chúng không tự chứng minh video hỏng.']
    (audit / 'FRAME_REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {audit / "FRAME_REVIEW.csv"} and {audit / "FRAME_REVIEW.md"}')
    print(dict(counts))


if __name__ == '__main__':
    main()
