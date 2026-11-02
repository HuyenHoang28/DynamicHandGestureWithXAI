import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "final" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src_neurosymbolic.models.keypoint_dataset import KEYPOINT_NAMES  # noqa: E402
from src_neurosymbolic.pose.ipn_hands_adapter import (  # noqa: E402
    CLASS_MAP,
    frame_to_canonical,
    hand_names,
    parse_temporal_annotations,
    prepare_dataset,
    video_subject,
)


def _hand(side: str, x_offset: float) -> dict:
    return {
        "handedness": side,
        "landmarks": {
            str(index): [x_offset + index / 100.0, 0.2 + index / 200.0, -0.01 * index]
            for index in range(21)
        },
    }


def _frame() -> dict:
    return {
        "pose_landmarks": {
            "0": [0.5, 0.1, 0.0],
            "2": [0.45, 0.08, 0.0],
            "5": [0.55, 0.08, 0.0],
            "9": [0.47, 0.16, 0.0],
            "10": [0.53, 0.16, 0.0],
            "11": [0.4, 0.4, 0.0],
            "12": [0.6, 0.4, 0.0],
            "13": [0.35, 0.6, 0.0],
            "14": [0.65, 0.6, 0.0],
        },
        "hand_landmarks": [_hand("Left", 0.1), _hand("Right", 0.6)],
    }


def test_hand_mapping_covers_21_unique_project_names_per_side() -> None:
    left = hand_names("left")
    right = hand_names("right")
    assert set(left) == {str(index) for index in range(21)}
    assert set(right) == {str(index) for index in range(21)}
    assert len(set(left.values())) == len(set(right.values())) == 21
    assert not (set(left.values()) & set(right.values()))


def test_frame_adapter_populates_exact_canonical_52_layout() -> None:
    converted = frame_to_canonical(_frame(), frame_id=7)
    points = converted["instances"][0]["keypoints_by_name"]
    assert converted["frame_id"] == 7
    assert set(points) == set(KEYPOINT_NAMES)
    assert points["left_hand_root"]["xy"] == pytest.approx([64.0, 96.0])
    assert points["right_hand_root"]["xy"] == pytest.approx([384.0, 96.0])
    assert points["left_wrist"] == points["left_hand_root"]
    assert points["right_wrist"] == points["right_hand_root"]
    assert points["face-8"]["xy"] == pytest.approx([320.0, 76.8])
    assert {item["score"] for item in points.values()} == {1.0}


def test_missing_hands_remain_missing_instead_of_creating_zero_confidence_points() -> None:
    frame = _frame()
    frame["hand_landmarks"] = []
    converted = frame_to_canonical(frame, frame_id=0)
    points = converted["instances"][0]["keypoints_by_name"]
    assert "left_hand_root" not in points
    assert "right_hand_root" not in points
    assert "left_shoulder" in points and "right_shoulder" in points


def test_duplicate_or_unknown_handedness_is_rejected() -> None:
    frame = _frame()
    frame["hand_landmarks"] = [_hand("Left", 0.1), _hand("Left", 0.2)]
    with pytest.raises(ValueError, match="Duplicate left"):
        frame_to_canonical(frame, 0)
    frame["hand_landmarks"] = [_hand("Unknown", 0.1)]
    with pytest.raises(ValueError, match="Unsupported handedness"):
        frame_to_canonical(frame, 0)


def test_taxonomy_is_the_historical_ipn13_contract() -> None:
    assert [CLASS_MAP[index][0] for index in sorted(CLASS_MAP)] == [
        "B0A", "B0B", "G01", "G02", "G03", "G04", "G05",
        "G06", "G07", "G08", "G09", "G10", "G11",
    ]
    assert [CLASS_MAP[index][1] for index in sorted(CLASS_MAP)] == list(range(13))


def test_annotation_and_subject_parsing(tmp_path: Path) -> None:
    path = tmp_path / "annotations.txt"
    path.write_text("0, 1, 4\n3, 5, 9\n14, 10, 12\n", encoding="utf-8")
    assert parse_temporal_annotations(path) == [(0, 1, 4), (3, 5, 9), (14, 10, 12)]
    assert video_subject("1CM42_18_R_#177") == "1CM42_18"
    with pytest.raises(ValueError):
        video_subject("bad-name")


def test_end_to_end_adapter_writes_project_manifest_and_excludes_nog_grb(tmp_path: Path) -> None:
    hands = tmp_path / "hands"
    output = tmp_path / "prepared"
    (hands / "landmarks_gt").mkdir(parents=True)
    (hands / "temporal_annotations").mkdir()
    (hands / "splits").mkdir()
    video = "1CM1_4_R_#229"
    frames = {f"{video}_{index:06d}.jpg": _frame() for index in range(1, 7)}
    (hands / "landmarks_gt" / f"{video}.json").write_text(json.dumps(frames), encoding="utf-8")
    (hands / "temporal_annotations" / f"{video}.txt").write_text(
        "0, 1, 2\n3, 3, 4\n14, 5, 6\n", encoding="utf-8"
    )
    (hands / "splits" / "train_list.txt").write_text(video + "\n", encoding="utf-8")
    (hands / "splits" / "test_list.txt").write_text("", encoding="utf-8")
    protocol = tmp_path / "subjects.json"
    protocol.write_text(
        json.dumps({
            "train_subjects": ["1CM1_4"],
            "validation_subjects": ["VAL"],
            "test_subjects": ["TEST"],
            "excluded_subjects": [],
        }),
        encoding="utf-8",
    )

    report = prepare_dataset(hands, output, protocol)
    assert report["input_fingerprint"]["data_contract_version"]
    assert len(report["input_fingerprint"]["sha256"]) == 64
    assert (output / "conversion_status.json").read_text(encoding="utf-8") == '{\n  "status": "complete"\n}'
    assert report["counts"]["clips"] == {"train": 1}
    assert report["counts"]["skipped"] == {"GRB": 1, "NOG": 1}
    split_line = (output / "train.txt").read_text(encoding="utf-8").strip()
    assert "/train/G01/1CM1_4/" in split_line.replace("\\", "/")
    assert split_line.endswith(" 0 2")
    manifest = (output / "manifest.csv").read_text(encoding="utf-8")
    assert ",G01,2,3,3,4,2," in manifest
    clip_path = next((output / "keypoints").rglob("*.json"))
    clip = json.loads(clip_path.read_text(encoding="utf-8"))
    assert [frame["frame_id"] for frame in clip] == [0, 1]


def _make_fixture(tmp_path: Path, *, frames: int = 4, annotations: str = "3, 1, 4\n") -> tuple[Path, Path, Path, str]:
    hands = tmp_path / "hands"
    output = tmp_path / "prepared"
    (hands / "landmarks_gt").mkdir(parents=True)
    (hands / "temporal_annotations").mkdir()
    (hands / "splits").mkdir()
    video = "1CM1_4_R_#229"
    frame_data = {f"{video}_{index:06d}.jpg": _frame() for index in range(1, frames + 1)}
    (hands / "landmarks_gt" / f"{video}.json").write_text(json.dumps(frame_data), encoding="utf-8")
    (hands / "temporal_annotations" / f"{video}.txt").write_text(annotations, encoding="utf-8")
    (hands / "splits" / "train_list.txt").write_text(video + "\n", encoding="utf-8")
    (hands / "splits" / "test_list.txt").write_text("", encoding="utf-8")
    protocol = tmp_path / "subjects.json"
    protocol.write_text(json.dumps({
        "train_subjects": ["1CM1_4"], "validation_subjects": ["VAL"],
        "test_subjects": ["TEST"], "excluded_subjects": [],
    }), encoding="utf-8")
    return hands, output, protocol, video


@pytest.mark.parametrize("missing", ["gt", "annotation"])
def test_preflight_rejects_missing_official_gt_or_annotation(tmp_path: Path, missing: str) -> None:
    hands, output, protocol, video = _make_fixture(tmp_path)
    if missing == "gt":
        (hands / "landmarks_gt" / f"{video}.json").unlink()
        message = "split/GT video mismatch"
    else:
        (hands / "temporal_annotations" / f"{video}.txt").unlink()
        message = "split/annotation video mismatch"
    with pytest.raises(ValueError, match=message):
        prepare_dataset(hands, output, protocol)
    assert not list(output.rglob("*.json")) or all(path.name == "conversion_status.json" for path in output.rglob("*.json"))


@pytest.mark.parametrize(
    ("annotations", "message"),
    [("3, 1, 2\n3, 4, 4\n", "gap"), ("3, 1, 3\n3, 3, 4\n", "overlap")],
)
def test_preflight_rejects_interval_gap_or_overlap(tmp_path: Path, annotations: str, message: str) -> None:
    hands, output, protocol, _ = _make_fixture(tmp_path, annotations=annotations)
    with pytest.raises(ValueError, match=message):
        prepare_dataset(hands, output, protocol)
    assert not list((output / "keypoints").rglob("*.json")) if (output / "keypoints").exists() else True


def test_preflight_rejects_truncated_temporal_coverage(tmp_path: Path) -> None:
    hands, output, protocol, _ = _make_fixture(tmp_path, annotations="3, 1, 3\n")
    with pytest.raises(ValueError, match="GT has 4 frames"):
        prepare_dataset(hands, output, protocol)


def test_overwrite_refuses_existing_derived_cache(tmp_path: Path) -> None:
    hands, output, protocol, _ = _make_fixture(tmp_path)
    (output / "keypoint_tensor_cache").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="new output root"):
        prepare_dataset(hands, output, protocol, overwrite=True)
    assert not (output / "conversion_status.json").exists()


def test_include_grab_adds_grb_as_label_13_and_keeps_default_13_classes(tmp_path: Path) -> None:
    hands, output, protocol, video = _make_fixture(tmp_path, frames=6, annotations="0, 1, 2\n3, 3, 4\n14, 5, 6\n")
    report = prepare_dataset(hands, output, protocol, include_grab=True)
    assert report["counts"]["clips"] == {"train": 2}
    assert report["counts"]["skipped"] == {"NOG": 1}
    assert report["taxonomy"]["include_grab"] is True
    assert report["taxonomy"]["included"]["14"] == {"label": "GRB", "label_id": 13}
    lines = sorted((output / "train.txt").read_text(encoding="utf-8").strip().splitlines())
    assert {line.rsplit(" ", 1)[1] for line in lines} == {"13", "2"}
    assert any("/GRB/" in line.replace("\\", "/") for line in lines)

    default_output = tmp_path / "prepared_default"
    default = prepare_dataset(hands, default_output, protocol)
    assert default["counts"]["skipped"] == {"GRB": 1, "NOG": 1}
    assert default["taxonomy"]["include_grab"] is False
    assert default["input_fingerprint"]["sha256"] != report["input_fingerprint"]["sha256"]
