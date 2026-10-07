"""Prepare a small, reproducible IPN smoke-test subset from local ZIP archives."""
import csv
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external_data"
ARCHIVES = DATA / "ipn_archives"
DEST = DATA / "ipn_smoke_20261005"
VIDEOS = ["1CM1_1_R_#218", "1CM42_15_R_#198", "1CV12_12_R_#90", "4CM11_13_R_#30"]


def main():
    with zipfile.ZipFile(ARCHIVES / "annotations-20261005T055143Z-1-001.zip") as archive:
        rows = list(csv.reader(io.StringIO(archive.read("annotations/Annot_TestList.txt").decode())))
    selected = []
    for index, label in enumerate(["B0A", "B0B"] + [f"G{k:02}" for k in range(1, 12)]):
        candidates = [r for r in rows if r[0] == VIDEOS[index % 4] and r[1] == label]
        selected.append(min(candidates, key=lambda r: int(r[5])))
    for video in VIDEOS[:3]:
        candidates = [r for r in rows if r[0] == video and r[1] == "D0X" and int(r[5]) >= 15]
        selected.append(min(candidates, key=lambda r: int(r[5])))

    raw_dir = DEST / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    needed = {r[0] + ".avi" for r in selected if not (raw_dir / (r[0] + ".avi")).exists()}
    for zip_path in sorted(ARCHIVES.glob("videos-*.zip")):
        if not needed:
            break
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if not needed:
                    break
                if not name.endswith(".tgz"):
                    continue
                print(f"Scanning {zip_path.name}: {name}", flush=True)
                with archive.open(name) as nested, tarfile.open(fileobj=nested, mode="r|gz") as tar:
                    for member in tar:
                        basename = Path(member.name).name
                        if basename not in needed or not member.isfile():
                            continue
                        with tar.extractfile(member) as src, (raw_dir / basename).open("xb") as dst:
                            shutil.copyfileobj(src, dst)
                        needed.remove(basename)
                        print(f"Extracted {basename}", flush=True)
                        if not needed:
                            break
    if needed:
        raise FileNotFoundError(f"Missing videos: {sorted(needed)}")

    clip_dir = DEST / "clips"
    clip_dir.mkdir(exist_ok=True)
    manifest = []
    for video, label, label_id, start, end, length in selected:
        start, end, length = int(start), int(end), int(length)
        assert end - start + 1 == length
        clip_id = f"{label}_{video.replace('#', '')}_{start}_{end}"
        destination = clip_dir / f"{clip_id}.avi"
        cap = cv2.VideoCapture(str(raw_dir / f"{video}.avi"))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create {destination}")
        count = 0
        try:
            for _ in range(length):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                count += 1
        finally:
            writer.release()
            cap.release()
        if count != length:
            raise RuntimeError(f"Incomplete clip {clip_id}: {count}/{length}")
        check = cv2.VideoCapture(str(destination))
        decoded = 0
        while check.read()[0]:
            decoded += 1
        check.release()
        if decoded != length:
            raise RuntimeError(f"Clip decode mismatch: {clip_id} {decoded}/{length}")
        manifest.append(dict(clip_id=clip_id, clip_path=destination.relative_to(ROOT).as_posix(),
                             video=video, label=label, label_id=int(label_id), start_frame=start,
                             end_frame=end, frames=length, fps=fps, split="test"))
        print(f"Prepared {clip_id}: {count} frames", flush=True)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {DEST / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
