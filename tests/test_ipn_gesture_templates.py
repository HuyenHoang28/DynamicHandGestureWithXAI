import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.kg.build_template_graph import parse_template_yaml


TEMPLATES = ROOT / "configs" / "ipn" / "gesture_templates.yaml"
EXPECTED_LABELS = {"B0A", "B0B", *(f"G{number:02d}" for number in range(1, 12))}


def _templates() -> dict:
    return parse_template_yaml(TEMPLATES)


def _required_event_concepts(template: dict, field: str) -> list[set[str]]:
    return [set(event["requirements"].get(field, {}).get("required", [])) for event in template["expected_events"]]


def test_ipn_template_labels_are_exactly_the_locked_manifest() -> None:
    parsed = _templates()

    assert set(parsed["gestures"]) == EXPECTED_LABELS


def test_ipn_directional_and_pinch_semantics_are_canonical() -> None:
    gestures = _templates()["gestures"]

    assert _required_event_concepts(gestures["G03"], "motions") == [{"hand_moves_up"}]
    assert _required_event_concepts(gestures["G04"], "motions") == [{"hand_moves_down"}]
    assert _required_event_concepts(gestures["G05"], "motions") == [{"hand_moves_right"}]
    assert _required_event_concepts(gestures["G06"], "motions") == [{"hand_moves_left"}]
    assert _required_event_concepts(gestures["G10"], "finger_motions") == [{"pinch_distance_increasing"}]
    assert _required_event_concepts(gestures["G11"], "finger_motions") == [{"pinch_distance_decreasing"}]


def test_ipn_click_occurrence_constraints_are_canonical() -> None:
    gestures = _templates()["gestures"]

    for label in ("G01", "G02"):
        event = gestures[label]["expected_events"][0]
        assert event["max_occurrences"] == 1
    for label in ("G08", "G09"):
        event = gestures[label]["expected_events"][0]
        assert event["min_occurrences"] == 2


def test_ipn_g07_requires_closed_open_closed_open_sequence() -> None:
    gestures = _templates()["gestures"]

    assert _required_event_concepts(gestures["G07"], "hand_shapes") == [
        {"hand_closed"},
        {"hand_open"},
        {"hand_closed"},
        {"hand_open"},
    ]
