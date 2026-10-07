"""Verify annotated boundary frames by direct OpenCV seeking (1-based IPN -> 0-based OpenCV)."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def seek_frame(path, one_based_frame):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        target = int(one_based_frame) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        return bool(ok and frame is not None and frame.size), position
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'external_data/ipn_audit')
    parser.add_argument('--videos-root', type=Path, default=ROOT / 'external_data/ipn_full/videos')
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    videos_root = args.videos_root.resolve()
    review_file = audit / 'FRAME_REVIEW.csv'
    annotation_file = ROOT / 'external_data/ipn_full/annotations/Annot_List.txt'
    if not review_file.exists():
        parser.error('Run build_ipn_frame_review.py first')
    with review_file.open(encoding='utf-8-sig', newline='') as f:
        review = {r['video']: r for r in csv.DictReader(f)}
    annotations = {}
    with annotation_file.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            annotations.setdefault(row['video'], []).append(row)
    output = []
    for video, summary in review.items():
        path = Path(summary['path'])
        if not path.exists():
            path = videos_root / f'{video}.avi'
        rows = annotations.get(video, [])
        last = max((int(r['t_end']) for r in rows), default=0)
        ok, position = seek_frame(path, last) if last else (False, 0)
        tail = [r for r in rows if int(r['t_end']) > int(summary['decoded_frames'])]
        tail_checks = []
        for row in tail:
            tail_ok, tail_position = seek_frame(path, int(row['t_end']))
            tail_checks.append(f"{row['label']}:{row['t_start']}-{row['t_end']}={'OK' if tail_ok else 'FAIL'}@{tail_position}")
        result = dict(video=video, priority=summary['priority'], decoded_frames=summary['decoded_frames'],
                      last_annotated_frame=last, last_annotation_seek='OK' if ok else 'FAIL',
                      seek_position=position, tail_seek_results=';'.join(tail_checks),
                      interpretation=('annotation boundary directly readable' if ok else 'annotation boundary not directly readable'))
        output.append(result)
        print(f"{video}: end {last} -> {result['last_annotation_seek']} (position {position})", flush=True)
    fields = list(output[0])
    with (audit / 'ANNOTATION_SEEK_REVIEW.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    bad = [r for r in output if r['last_annotation_seek'] == 'FAIL']
    tail_bad = [r for r in output if 'FAIL' in r['tail_seek_results']]
    (audit / 'ANNOTATION_SEEK_REVIEW.md').write_text(
        '# IPN annotation boundary seek review\n\n'
        f'- Videos checked: {len(output)}\n'
        f'- Last annotation frame directly readable: {len(output) - len(bad)}\n'
        f'- Last annotation frame not directly readable: {len(bad)}\n'
        f'- Videos with a flagged tail boundary that failed direct seek: {len(tail_bad)}\n\n'
        'A successful seek means OpenCV could read the 1-based annotation frame using zero-based index `frame - 1`. '
        'It does not verify that the visual label is correct; inspect flagged gesture clips by eye.\n',
        encoding='utf-8')
    print(f'Wrote {audit / "ANNOTATION_SEEK_REVIEW.csv"}; failed last-frame seeks: {len(bad)}; failed flagged-tail seeks: {len(tail_bad)}')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
