import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.kg.build_template_graph import parse_template_yaml


TEMPLATES = ROOT / "configs" / "ipn" / "gesture_templates.yaml"
EXPECTED_LABELS = {"B0A", "B0B", *(f"G{number:02d}" for number in range(1, 12))}


def _templates() -> dict:
    return parse_template_yaml(TEMPLATES)


def _event_concepts(template: dict, field: str, level: str = "required") -> list[set[str]]:
    return [
        set(event["requirements"].get(field, {}).get(level, []))
        for event in template["expected_events"]
    ]


def _required_event_concepts(template: dict, field: str) -> list[set[str]]:
    return _event_concepts(template, field, "required")


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
        assert event["max_occurrences"] == 2


def test_ipn_g07_requires_closed_open_closed_open_sequence() -> None:
    gestures = _templates()["gestures"]

    assert _required_event_concepts(gestures["G07"], "hand_shapes") == [
        {"hand_closed"},
        {"hand_open"},
        {"hand_closed"},
        {"hand_open"},
    ]


def test_post_experiment_a_templates_do_not_use_clip_global_negative_evidence() -> None:
    gestures = _templates()["gestures"]

    assert all(template["negative_evidence"] == {} for template in gestures.values())


def test_baseline_templates_keep_stationary_as_non_required_evidence() -> None:
    gestures = _templates()["gestures"]

    assert _required_event_concepts(gestures["B0A"], "hand_shapes") == [{"index_extended"}]
    assert _required_event_concepts(gestures["B0A"], "motions") == [set()]
    assert _event_concepts(gestures["B0A"], "motions", "preferred") == [{"hand_stationary"}]
    assert _required_event_concepts(gestures["B0B"], "hand_shapes") == [{"index_middle_extended"}]
    assert _required_event_concepts(gestures["B0B"], "motions") == [set()]
    assert _event_concepts(gestures["B0B"], "motions", "preferred") == [{"hand_stationary"}]


def test_temporal_supporting_signals_remain_preferred_instead_of_hard_gates() -> None:
    gestures = _templates()["gestures"]

    assert _event_concepts(gestures["G07"], "finger_motions", "preferred") == [
        set(),
        {"fingers_spread_increasing"},
        {"fingers_spread_decreasing"},
        {"fingers_spread_increasing"},
    ]
    assert _event_concepts(gestures["G10"], "finger_motions", "preferred") == [
        {"fingers_spread_increasing"}
    ]
    assert _event_concepts(gestures["G11"], "finger_motions", "preferred") == [
        {"fingers_spread_decreasing"}
    ]
