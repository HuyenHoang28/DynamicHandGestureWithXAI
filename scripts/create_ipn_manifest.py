"""Create reproducible subject-disjoint IPN 13-class manifests."""
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['B0A', 'B0B'] + [f'G{i:02}' for i in range(1, 12)]
LABEL_IDS = {label: i for i, label in enumerate(LABELS)}


def subject_of(video):
    match = re.fullmatch(r'(.+_\d+)_[RL]_#\d+', video)
    if not match:
        raise ValueError(f'Cannot infer subject from video name: {video}')
    return match.group(1)


def read_video_list(path):
    with path.open(encoding='utf-8-sig') as f:
        return {line.split()[0]: int(line.split()[1]) for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'external_data/ipn_processed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-subjects', type=int, default=7)
    parser.add_argument(
        '--exclude-subject',
        action='append',
        default=[],
        metavar='SUBJECT',
        help='Subject ID to omit from all generated manifests (repeatable).',
    )
    args = parser.parse_args()
    # Sort and deduplicate so the recorded metadata is stable even when the
    # same subject is supplied more than once or in a different order.
    excluded_subjects = {subject.strip() for subject in args.exclude_subject if subject.strip()}
    ann_dir = ROOT / 'external_data/ipn_full/annotations'
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_videos = read_video_list(ann_dir / 'Video_TrainList.txt')
    test_videos = read_video_list(ann_dir / 'Video_TestList.txt')
    train_subjects_all = {subject_of(video) for video in train_videos}
    test_subjects_all = {subject_of(video) for video in test_videos}
    available_subjects = sorted(train_subjects_all - excluded_subjects)
    if not 1 <= args.val_subjects < len(available_subjects):
        parser.error('--val-subjects must be between 1 and number of train subjects - 1')
    # Keep the original seed-to-subject ordering stable: shuffle the complete
    # train subject list first, then remove exclusions before selecting val.
    # This means excluding a train-only subject outside the original val set
    # does not silently change the validation subjects.
    shuffled = sorted(train_subjects_all)
    random.Random(args.seed).shuffle(shuffled)
    shuffled = [subject for subject in shuffled if subject not in excluded_subjects]
    val_subjects = sorted(shuffled[:args.val_subjects])
    split_by_video = {
        **{
            v: 'train'
            for v in train_videos
            if subject_of(v) not in excluded_subjects
        },
        **{
            v: 'test'
            for v in test_videos
            if subject_of(v) not in excluded_subjects
        },
    }
    split_by_video.update({video: 'val' for video in train_videos if subject_of(video) in val_subjects})
    with (ann_dir / 'Annot_List.txt').open(encoding='utf-8-sig', newline='') as f:
        annotations = list(csv.DictReader(f))
    fields = ['clip_id', 'source_video', 'subject', 'split', 'label', 'label_id', 't_start', 't_end', 'frames', 'clip_path', 'keypoint_path']
    rows = []
    for ann in annotations:
        label = ann['label']
        video = ann['video']
        if label not in LABEL_IDS:
            continue
        subject = subject_of(video)
        if subject in excluded_subjects:
            continue
        split = split_by_video[video]
        start, end = int(ann['t_start']), int(ann['t_end'])
        clip_id = f'{video.replace("#", "")}_{label}_f{start:06d}_f{end:06d}'
        rows.append(dict(clip_id=clip_id, source_video=video, subject=subject, split=split,
                         label=label, label_id=LABEL_IDS[label], t_start=start, t_end=end,
                         frames=int(ann['frames']),
                         clip_path=f'external_data/ipn_processed/clips/{split}/{label}/{subject}/{clip_id}.avi',
                         keypoint_path=f'external_data/ipn_processed/keypoints/{split}/{label}/{subject}/{clip_id}.json'))
    rows.sort(key=lambda r: (['train', 'val', 'test'].index(r['split']), r['label'], r['source_video'], r['t_start']))
    with (out / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for split in ('train', 'val', 'test'):
        split_rows = [r for r in rows if r['split'] == split]
        with (out / f'{split}_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_rows)
        with (out / f'{split}.txt').open('w', encoding='utf-8') as f:
            for r in split_rows:
                f.write(f"{r['keypoint_path']} 0 {r['label_id']}\n")
    summary = dict(seed=args.seed, excluded_subjects=sorted(excluded_subjects),
                   validation_subjects=val_subjects,
                   train_subjects=sorted(train_subjects_all - set(val_subjects) - excluded_subjects),
                   test_subjects=sorted(test_subjects_all - excluded_subjects),
                   video_counts={s: len({r['source_video'] for r in rows if r['split'] == s}) for s in ('train', 'val', 'test')},
                   clip_counts={s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')},
                   label_counts={s: dict(sorted(Counter(r['label'] for r in rows if r['split'] == s).items())) for s in ('train', 'val', 'test')})
    (out / 'split_subjects.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
