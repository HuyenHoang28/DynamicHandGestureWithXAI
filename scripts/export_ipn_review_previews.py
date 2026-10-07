"""Export numbered image sheets and short clips for rows in GESTURE_REVIEW.csv."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'external_data/ipn_audit')
    parser.add_argument('--videos-root', type=Path, default=ROOT / 'external_data/ipn_full/videos')
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    out = audit / 'review_previews'
    out.mkdir(parents=True, exist_ok=True)
    with (audit / 'GESTURE_REVIEW.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        video = row['video']
        path = Path(f"{row['video']}.avi")
        if not path.exists():
            path = args.videos_root.resolve() / f'{video}.avi'
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f'SKIP {video}: cannot open {path}')
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start = int(row['t_start']) - 1
        end = int(row['t_end']) - 1
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sorted(set(max(0, min(total - 1, i)) for i in np.linspace(max(0, start - 15), min(total - 1, end + 15), 12).astype(int)))
        frames = []
        for index in indices:
            frame = read_frame(cap, index)
            if frame is None:
                continue
            frame = frame.copy()
            color = (0, 220, 0) if start <= index <= end else (0, 165, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 5)
            cv2.putText(frame, f'frame {index + 1} | {row["label"]} | {"INSIDE" if start <= index <= end else "OUTSIDE"}',
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            frames.append(frame)
        if frames:
            h, w = frames[0].shape[:2]
            blank = np.zeros_like(frames[0])
            while len(frames) < 12:
                frames.append(blank.copy())
            sheet = np.vstack([np.hstack(frames[:6]), np.hstack(frames[6:12])])
            cv2.imwrite(str(out / f'{video}_{row["label"]}_overview.jpg'), sheet)
        clip_path = out / f'{video}_{row["label"]}_clip.avi'
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*'XVID'), fps, (int(cap.get(3)), int(cap.get(4))))
        for index in range(max(0, start - 15), min(total, end + 16)):
            frame = read_frame(cap, index)
            if frame is not None:
                writer.write(frame)
        writer.release()
        cap.release()
        print(f'Wrote {video}: overview JPG and review AVI')
    print(f'Output folder: {out}')


if __name__ == '__main__':
    main()
