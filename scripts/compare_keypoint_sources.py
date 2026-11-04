"""Compare two 52-keypoint JSON trees (e.g. RTMW vs MediaPipe Holistic) on the clips they share.

Reports, per side: frame-count agreement, hand coverage (fraction of frames with all 21 points),
body-name agreement, and a left/right convention check (median error of each name against the same
name and against its mirrored counterpart in the reference), in units of the reference shoulder width.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

HAND_POINTS = 21


def frames_of(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for frame in data:
        instances = frame.get("instances") or []
        out.append(instances[0].get("keypoints_by_name", {}) if instances else {})
    return out


def xy(kp: dict, name: str):
    item = kp.get(name)
    return item["xy"] if item and item.get("xy") else None


def mirrored(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def hand_complete(kp: dict, side: str) -> bool:
    return sum(1 for n in kp if n.startswith(f"{side}_") and n.split("_", 1)[1] not in ("eye", "shoulder", "elbow", "wrist")) >= HAND_POINTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Root with <split>/<label>/<subject>/<clip>.json (RTMW).")
    parser.add_argument("--candidate", type=Path, required=True, help="Root with the same layout (MediaPipe).")
    parser.add_argument("--names", nargs="+", default=["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "nose"])
    args = parser.parse_args(argv)

    same: dict[str, list[float]] = {n: [] for n in args.names}
    flip: dict[str, list[float]] = {n: [] for n in args.names}
    cover = {"ref_left": [], "ref_right": [], "cand_left": [], "cand_right": []}
    clips = frame_mismatch = 0
    for cand in sorted(args.candidate.rglob("*.json")):
        ref = args.reference / cand.relative_to(args.candidate)
        if not ref.is_file():
            continue
        clips += 1
        rf, cf = frames_of(ref), frames_of(cand)
        if len(rf) != len(cf):
            frame_mismatch += 1
        for r, c in zip(rf, cf):
            ls, rs = xy(r, "left_shoulder"), xy(r, "right_shoulder")
            if ls is None or rs is None:
                continue
            width = math.dist(ls, rs)
            if width < 20:
                continue
            for n in args.names:
                a, b, m = xy(r, n), xy(c, n), xy(c, mirrored(n))
                if a is not None and b is not None:
                    same[n].append(math.dist(a, b) / width)
                if a is not None and m is not None:
                    flip[n].append(math.dist(a, m) / width)
            for side in ("left", "right"):
                cover[f"ref_{side}"].append(hand_complete(r, side))
                cover[f"cand_{side}"].append(hand_complete(c, side))
    print(f"clips compared={clips} frame-count mismatches={frame_mismatch}")
    print(f"{'name':16}{'median err same name':>24}{'median err mirrored name':>26}   (units of reference shoulder width)")
    for n in args.names:
        if same[n] and flip[n]:
            print(f"{n:16}{st.median(same[n]):24.3f}{st.median(flip[n]):26.3f}")
    for k, v in cover.items():
        if v:
            print(f"hand coverage {k}: {sum(v) / len(v):.3f} over {len(v)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
