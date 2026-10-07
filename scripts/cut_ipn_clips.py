"""Cut inclusive IPN annotation intervals into reproducible isolated clips."""
import argparse
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def cut(row, videos_root, overwrite=False):
    source = videos_root / f"{row['source_video']}.avi"
    destination = ROOT / row['clip_path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return dict(row, status='exists', decoded_frames='')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open source: {source}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Do not use CAP_PROP_POS_FRAMES for the source seek: AVI keyframe seeking
    # can land on a nearby frame. Sequentially decode until the 1-based IPN
    # start frame so annotation boundaries remain exact.
    target = int(row['t_start']) - 1
    skipped = 0
    while skipped < target:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f'Cannot seek sequentially to frame {row["t_start"]}: {source}')
        skipped += 1
    # MJPG is more robust for independently cut AVI clips than XVID/MPEG-4:
    # a malformed source packet must not make the output tail undecodable.
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot create clip: {destination}')
    count = 0
    missing_indices = []
    try:
        for _ in range(int(row['frames'])):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            count += 1
        missing_indices = list(range(target + count, target + int(row['frames'])))
        if missing_indices:
            # Some IPN AVI files report one extra tail frame that sequential
            # MPEG-4 decoding cannot advance to, while direct frame seeking can
            # still read it. Recover only those missing indices and verify the
            # decoder's reported position.
            fallback = cv2.VideoCapture(str(source))
            try:
                for index in missing_indices:
                    fallback.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = fallback.read()
                    position = int(fallback.get(cv2.CAP_PROP_POS_FRAMES))
                    if not ok or frame is None or position < index + 1:
                        break
                    writer.write(frame)
                    count += 1
            finally:
                fallback.release()
    finally:
        writer.release()
        cap.release()
    check = cv2.VideoCapture(str(destination))
    decoded = 0
    while True:
        ok, frame = check.read()
        if not ok:
            break
        decoded += 1
    check.release()
    if count != int(row['frames']) or decoded != int(row['frames']):
        raise RuntimeError(f'Frame mismatch {row["clip_id"]}: wrote={count}, decoded={decoded}, expected={row["frames"]}')
    return dict(row, status='created', decoded_frames=str(decoded))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT / 'external_data/ipn_processed/manifest.csv')
    parser.add_argument('--videos-root', type=Path, default=ROOT / 'external_data/ipn_full/videos')
    parser.add_argument('--one-per-label', action='store_true', help='Cut one deterministic train clip per label')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r['label'], r['split'], r['source_video'], int(r['t_start'])))
    if args.one_per_label:
        chosen = []
        for label in sorted({r['label'] for r in rows}):
            chosen.append(next(r for r in rows if r['label'] == label and r['split'] == 'train'))
        rows = chosen
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error('No manifest rows selected')
    results = []
    for index, row in enumerate(rows, 1):
        result = cut(row, args.videos_root.resolve(), args.overwrite)
        results.append(result)
        print(f'[{index}/{len(rows)}] {result["status"]}: {row["clip_id"]}', flush=True)
    report = args.manifest.parent / ('sample_cut_report.csv' if args.one_per_label or args.limit else 'cut_report.csv')
    with report.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['clip_id', 'split', 'label', 'source_video', 't_start', 't_end', 'frames', 'decoded_frames', 'clip_path', 'status'])
        writer.writeheader()
        writer.writerows({key: result.get(key, '') for key in writer.fieldnames} for result in results)
    print(f'Wrote {report}')


if __name__ == '__main__':
    main()
