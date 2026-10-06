"""Read-only IPN source audit. Decode every frame; write reports separately."""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['D0X', 'B0A', 'B0B'] + [f'G{i:02}' for i in range(1, 12)]


def write_reports(report, results, out):
    """Also supports refreshing presentation from a completed, saved decode run."""
    generated_prefixes = ('Video decode/frame mismatch:', 'Unreadable video:',
                          'Decoded/annotation mismatch:', 'Decoded/header mismatch:', 'Missing source archive:')
    report['issues'] = [issue for issue in report['issues'] if not issue.startswith(generated_prefixes)]
    intervals = defaultdict(list)
    with (Path(report['source']) / 'annotations/Annot_List.txt').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            intervals[row['video']].append(int(row['t_end']))
    for r in results:
        r['annotation_match'] = r['decoded_frames'] == r['expected_frames']
        r['header_match'] = r['decoded_frames'] == r['header_frames']
        r['last_annotated_frame'] = max(intervals[r['video']], default=0)
        r['annotations_past_decoded_end'] = sum(end > r['decoded_frames'] for end in intervals[r['video']])
        if not r['opened'] or r['decoded_frames'] == 0:
            report['issues'].append(f'Unreadable video: {r["video"]}')
        if not r['annotation_match']:
            report['issues'].append(f'Decoded/annotation mismatch: {r["video"]}: decoded={r["decoded_frames"]}, expected={r["expected_frames"]}')
        if not r['header_match']:
            report['issues'].append(f'Decoded/header mismatch: {r["video"]}: decoded={r["decoded_frames"]}, header={r["header_frames"]}')
    report['provenance'] = 'Local downloaded annotation files; not authenticated against canonical checksums.'
    report['annotation_frame_mismatches'] = sum(not r['annotation_match'] for r in results)
    report['header_frame_mismatches'] = sum(not r['header_match'] for r in results)
    report['unreadable_videos'] = sum(not r['opened'] or r['decoded_frames'] == 0 for r in results)
    report['annotations_past_decoded_end'] = sum(r['annotations_past_decoded_end'] for r in results)
    archive_dir = Path(report['source']).parent / 'ipn_archives'
    report['archives'] = {f'videos{i:02}.tgz': (archive_dir / f'videos{i:02}.tgz').is_file() for i in range(1, 6)}
    report['archive_note'] = 'Presence only; compressed contents/checksums not verified.'
    for name, present in report['archives'].items():
        issue = f'Missing source archive: {name}'
        if not present and issue not in report['issues']:
            report['issues'].append(issue)
    report['status'] = 'FAIL' if report['issues'] else 'PASS'
    if results:
        with (out / 'videos.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    (out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# IPN data audit', '', f'Status: **{report["status"]}**', '',
             f'- Videos matching both annotation and header counts: {report["valid_videos"]}/{report["videos"]}',
             f'- Unreadable/zero-frame videos: {report["unreadable_videos"]}',
             f'- Decoded count differs from annotation: {report["annotation_frame_mismatches"]}',
             f'- Decoded count differs from header: {report["header_frame_mismatches"]} (may overlap the previous category)',
             f'- Annotation segments ending past decoded video length: {report["annotations_past_decoded_end"]}',
             f'- Total decoded frames: {report["decoded_frames"]}',
             f'- Train/test videos: {report["split_videos"]["train"]}/{report["split_videos"]["test"]}',
             f'- Train/test subject groups: {len(report["subjects"]["train"])}/{len(report["subjects"]["test"])}',
             f'- Archive files present: {sum(report["archives"].values())}/5; compressed contents not verified.',
             '- Local annotation provenance is assumed; no canonical checksum verification.',
             '- Subject grouping is inferred from the filename prefix before _R_# or _L_#; identity has not been independently verified.',
             '- Full decoding checks readability/counts, not visual correctness of labels or absence of decoder concealment.',
             '- A header-only mismatch is not proof of corrupt video. Annotation mismatches require investigation before clipping.',
             '', '| Label | Train | Test | Total |', '|---|---:|---:|---:|']
    for label in LABELS:
        c = report['annotation_counts']
        lines.append(f'| {label} | {c["train"].get(label, 0)} | {c["test"].get(label, 0)} | {c["all"].get(label, 0)} |')
    lines += ['', '## Issues', ''] + (['- ' + issue for issue in report['issues']] or ['None.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def decode(path):
    cap = cv2.VideoCapture(str(path))
    result = dict(video=path.stem, path=str(path), opened=cap.isOpened(),
                  header_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                  fps=cap.get(cv2.CAP_PROP_FPS), decoded_frames=0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                break
            result['decoded_frames'] += 1
    finally:
        cap.release()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'external_data/ipn_full')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'external_data/ipn_audit')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--refresh-report', action='store_true', help='Refresh completed JSON/CSV report formatting without decoding again; source annotations must be unchanged')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    source = args.data_root.resolve()
    out = args.output_dir.resolve()
    if out == source or source in out.parents:
        parser.error('Report directory must be outside the source dataset')
    if args.refresh_report:
        report = json.loads((out / 'audit.json').read_text(encoding='utf-8'))
        if report.get('status') not in ('PASS', 'FAIL') or 'decoded_frames' not in report:
            parser.error('Refresh requires a completed full decode audit')
        with (out / 'videos.csv').open(encoding='utf-8-sig', newline='') as f:
            results = list(csv.DictReader(f))
        for row in results:
            for field in ('header_frames', 'decoded_frames', 'expected_frames'):
                row[field] = int(row[field])
            for field in ('opened', 'valid'):
                row[field] = row[field] == 'True'
            row['fps'] = float(row['fps'])
        if len(results) != report['videos']:
            parser.error('Saved CSV is incomplete')
        if sum(r['decoded_frames'] for r in results) != report['decoded_frames'] or len({r['video'] for r in results}) != len(results):
            parser.error('Saved CSV has duplicates or inconsistent frame totals')
        write_reports(report, results, out)
        print(f'Refreshed {len(results)} saved video results; no video decoded.')
        return 1 if report['issues'] else 0
    out.mkdir(parents=True, exist_ok=True)
    # Invalidate previous reports before starting so interrupted reruns cannot look successful.
    (out / 'audit.json').write_text(json.dumps({'status': 'RUNNING'}), encoding='utf-8')
    (out / 'REPORT.md').write_text('# IPN data audit\n\nStatus: **RUNNING / incomplete**\n', encoding='utf-8')
    (out / 'videos.csv').write_text('', encoding='utf-8')
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    ann = source / 'annotations'
    try:
        splits = {}
        for split in ('train', 'test'):
            rows = [line.split() for line in (ann / f'Video_{split.title()}List.txt').read_text().splitlines() if line.strip()]
            splits[split] = {r[0]: int(r[1]) for r in rows}
            check(len(rows) == len(splits[split]), f'Duplicate video in {split} list')
        check(len(splits['train']) == 148, 'Expected 148 train videos')
        check(len(splits['test']) == 52, 'Expected 52 test videos')
        check(not (splits['train'].keys() & splits['test'].keys()), 'Train/test video overlap')
        expected = {**splits['train'], **splits['test']}
        with (ann / 'metadata.csv').open(encoding='utf-8-sig', newline='') as f:
            metadata_rows = list(csv.DictReader(f))
        metadata = {r['Video Name']: r for r in metadata_rows}
        check(len(metadata_rows) == len(metadata), 'Duplicate metadata video')
        check(metadata.keys() == expected.keys(), 'Metadata video set differs from split lists')
        with (ann / 'classIdx.txt').open(encoding='utf-8-sig', newline='') as f:
            class_rows = list(csv.DictReader(f))
        mapping = {r['label']: int(r['id']) for r in class_rows}
        check(len(class_rows) == 14 and mapping == dict(zip(LABELS, range(1, 15))), 'Unexpected class mapping')
        annotations = {}
        for split, name in [('all', 'Annot_List.txt'), ('train', 'Annot_TrainList.txt'), ('test', 'Annot_TestList.txt')]:
            with (ann / name).open(encoding='utf-8-sig', newline='') as f:
                rows = [r for r in csv.reader(f) if r]
            if rows and rows[0][0] == 'video':
                rows = rows[1:]
            annotations[split] = [tuple([r[0], r[1]] + [int(x) for x in r[2:]]) for r in rows]
            check(all(len(r) == 6 for r in annotations[split]), f'Invalid column count in {name}')
        check(Counter(annotations['all']) == Counter(annotations['train']) + Counter(annotations['test']), 'Combined split annotations differ from Annot_List')
        check(len(annotations['all']) == 5649, 'Expected 5649 total annotations')
        subjects = defaultdict(set)
        video_subject = {}
        for split, videos in splits.items():
            for video, frames in videos.items():
                match = re.fullmatch(r'(.+_\d+)_[RL]_#\d+', video)
                check(match is not None, f'Unrecognized subject naming: {video}')
                subject = match.group(1) if match else 'UNKNOWN'
                video_subject[video] = subject
                subjects[split].add(subject)
                m = metadata.get(video, {})
                check(m.get('Set') == split and int(m.get('Frames', -1)) == frames, f'Metadata mismatch: {video}')
        check(len(subjects['train']) == 37 and len(subjects['test']) == 13, 'Expected 37/13 subject groups')
        check(not subjects['train'] & subjects['test'], 'Train/test subject-group overlap')
        for split in ('train', 'test'):
            check({r[0] for r in annotations[split]} == set(splits[split]), f'Annotation video set mismatch: {split}')
            check(len(set(annotations[split])) == len(annotations[split]), f'Duplicate annotation: {split}')
            for video, label, label_id, start, end, frames in annotations[split]:
                check(video in splits[split], f'Annotation in wrong split: {video}')
                check(mapping.get(label) == label_id, f'Invalid label/ID: {video} {label} {label_id}')
                check(1 <= start <= end <= expected.get(video, 0) and end-start+1 == frames,
                      f'Invalid frame interval: {video} {start}:{end} length={frames}')
        intervals = defaultdict(list)
        for video, label, label_id, start, end, frames in annotations['all']:
            intervals[video].append((start, end))
        for video, segments in intervals.items():
            previous = 0
            for start, end in sorted(segments):
                check(start == previous + 1, f'Annotation gap/overlap: {video} after {previous}, next {start}')
                previous = end
            check(previous == expected.get(video), f'Annotation does not cover video end: {video}')
    except (OSError, ValueError, KeyError, IndexError) as exc:
        issues.append(f'Cannot parse source metadata: {exc}')
        (out / 'audit.json').write_text(json.dumps({'status': 'FAIL', 'issues': issues}, indent=2), encoding='utf-8')
        (out / 'REPORT.md').write_text('# IPN data audit\n\nStatus: **FAIL**\n\n' + '\n'.join(issues), encoding='utf-8')
        print(issues[-1], flush=True)
        return 1

    paths = sorted(p for p in (source / 'videos').rglob('*') if p.is_file() and p.suffix.lower() in ('.avi', '.mp4'))
    counts = Counter(p.stem for p in paths)
    check(set(counts) == set(expected), f'Video set mismatch: missing={sorted(set(expected)-set(counts))}, extra={sorted(set(counts)-set(expected))}')
    check(all(n == 1 for n in counts.values()), 'Duplicate video filenames')
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(decode, paths), 1):
            video = result['video']
            result.update(subject=video_subject.get(video, ''), split='train' if video in splits['train'] else 'test' if video in splits['test'] else 'unknown', expected_frames=expected.get(video, 0))
            result['valid'] = result['opened'] and result['decoded_frames'] > 0 and result['decoded_frames'] == result['expected_frames'] == result['header_frames']
            check(result['valid'], f'Video decode/frame mismatch: {video}: expected={result["expected_frames"]}, decoded={result["decoded_frames"]}, header={result["header_frames"]}')
            results.append(result)
            # Preserve completed decode evidence if a later video interrupts the run.
            with (out / 'videos.csv').open('a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(result))
                if i == 1:
                    writer.writeheader()
                writer.writerow(result)
            print(f'[{i}/{len(paths)}] {video}: {result["decoded_frames"]} frames, valid={result["valid"]}', flush=True)
    report = dict(status='PASS' if not issues else 'FAIL', timestamp=datetime.now(timezone.utc).isoformat(),
                  source=str(source), decode_mode='Full sequential OpenCV read of every video',
                  subject_rule='Filename prefix before _R_# or _L_#; inferred grouping, not independently verified identity',
                  videos=len(paths), valid_videos=sum(r['valid'] for r in results),
                  decoded_frames=sum(r['decoded_frames'] for r in results),
                  split_videos={s: len(v) for s, v in splits.items()},
                  subjects={s: sorted(v) for s, v in subjects.items()},
                  annotation_counts={s: dict(sorted(Counter(r[1] for r in rows).items())) for s, rows in annotations.items()},
                  issues=issues)
    write_reports(report, results, out)
    print(f'{report["status"]}: {len(report["issues"])} issues. Report: {out}', flush=True)
    return 1 if report['issues'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
