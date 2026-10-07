"""Create a small human-review CSV for gesture annotation boundaries."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'external_data/ipn_audit')
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    with (audit / 'videos.csv').open(encoding='utf-8-sig', newline='') as f:
        videos = {r['video']: r for r in csv.DictReader(f)}
    with (ROOT / 'external_data/ipn_full/annotations/Annot_List.txt').open(encoding='utf-8-sig', newline='') as f:
        annotations = list(csv.DictReader(f))
    seek = {}
    seek_file = audit / 'ANNOTATION_SEEK_REVIEW.csv'
    if seek_file.exists():
        with seek_file.open(encoding='utf-8-sig', newline='') as f:
            seek = {r['video']: r for r in csv.DictReader(f)}
    rows = []
    for video, summary in videos.items():
        decoded = int(summary['decoded_frames'])
        candidates = [r for r in annotations if r['video'] == video and r['label'] != 'D0X' and int(r['t_end']) > decoded]
        for ann in candidates:
            rows.append({
                'video': video,
                'label': ann['label'],
                'label_id': ann['id'],
                't_start': ann['t_start'],
                't_end': ann['t_end'],
                'annotation_frames': ann['frames'],
                'decoded_frames': summary['decoded_frames'],
                'header_frames': summary['header_frames'],
                'expected_frames': summary['expected_frames'],
                'fps': summary['fps'],
                'direct_seek': seek.get(video, {}).get('last_annotation_seek', 'not_checked'),
                'visual_review': '',
                'notes': '',
            })
    rows.sort(key=lambda r: (r['label'], r['video']))
    if len(rows) != 7:
        parser.error(f'Expected 7 gesture boundary rows, found {len(rows)}')
    out = audit / 'GESTURE_REVIEW.csv'
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {out} with {len(rows)} rows')


if __name__ == '__main__':
    main()
