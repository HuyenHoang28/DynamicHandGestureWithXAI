import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.xai_v4_analysis import (  # noqa: E402
    auroc,
    delete_events,
    event_fact_map,
    fact_matches,
    supporting_events,
)


def graph() -> dict:
    return {
        "nodes": [
            {"id": "event:e1", "type": "Event", "properties": {"subject": "right_hand"}},
            {"id": "event:e2", "type": "Event", "properties": {"subject": "left_hand"}},
            {"id": "concept:up", "type": "Concept", "properties": {"name": "hand_moves_up"}},
            {"id": "concept:fist", "type": "Concept", "properties": {"name": "fist"}},
            {"id": "bodypart:right_hand", "type": "BodyPart", "properties": {"name": "right_hand"}},
        ],
        "edges": [
            {"source": "event:e1", "relation": "INVOLVES", "target": "bodypart:right_hand"},
            {"source": "event:e1", "relation": "HAS_MOTION", "target": "concept:up"},
            {"source": "event:e2", "relation": "HAS_HAND_SHAPE", "target": "concept:fist"},
        ],
    }


def test_event_fact_map_collects_subject_relation_concept():
    facts = event_fact_map(graph())
    assert ("right_hand", "HAS_MOTION", "hand_moves_up") in facts["event:e1"]
    assert ("left_hand", "HAS_HAND_SHAPE", "fist") in facts["event:e2"]


def test_fact_matches_supports_any_subject_wildcard():
    fact = ("right_hand", "HAS_MOTION", "hand_moves_up")
    assert fact_matches(("any", "HAS_MOTION", "hand_moves_up"), fact)
    assert fact_matches(fact, fact)
    assert not fact_matches(("left_hand", "HAS_MOTION", "hand_moves_up"), fact)


def test_supporting_events_selects_only_events_with_required_facts():
    assert supporting_events(graph(), [("any", "HAS_MOTION", "hand_moves_up")]) == ["event:e1"]
    assert supporting_events(graph(), [("any", "HAS_SWEEP", "none")]) == []


def test_delete_events_removes_nodes_and_touching_edges_without_mutating_input():
    original = graph()
    reduced = delete_events(original, {"event:e1"})
    assert {node["id"] for node in reduced["nodes"]} == {
        "event:e2", "concept:up", "concept:fist", "bodypart:right_hand",
    }
    assert all("event:e1" not in (edge["source"], edge["target"]) for edge in reduced["edges"])
    assert len(original["edges"]) == 3 and len(original["nodes"]) == 5


def test_auroc_handles_ties_and_missing_class():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert auroc([0.4, 0.6], [True, True]) is None
