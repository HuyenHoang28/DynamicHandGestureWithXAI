import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.kg.build_template_graph import build_template_graph, parse_template_yaml
from src_neurosymbolic.events.build_events_phase_v2 import build_events_phase_v2
from src_neurosymbolic.kg.build_instance_graph_phase_v2 import build_instance_graph_phase_v2
from src_neurosymbolic.kg.graph_matching import score_template


WEIGHTS = {"required": 1.0, "preferred": 0.6, "optional": 0.3, "negative": -1.0}


def event_template(concept: str, minimum: int = 1, maximum: int | None = None) -> dict:
    return {
        "subject": "any",
        "min_occurrences": minimum,
        "max_occurrences": maximum,
        "requirements": {"finger_motions": {"required": [concept]}},
    }


def instance_graph(concepts: list[str]) -> dict:
    nodes = [{"id": "video:test", "type": "Video", "properties": {}}]
    edges = []
    for index, concept in enumerate(concepts, start=1):
        event_id = f"event:{index}"
        concept_id = f"finger:{index}:{concept}"
        nodes.extend(
            [
                {
                    "id": event_id,
                    "type": "Event",
                    "properties": {
                        "subject": "right_hand",
                        "start_frame": index * 10,
                        "end_frame": index * 10 + 2,
                        "duration": 3,
                        "confidence": 1.0,
                    },
                },
                {"id": concept_id, "type": "FingerMotion", "properties": {"name": concept}},
            ]
        )
        edges.append({"source": event_id, "relation": "HAS_FINGER_MOTION", "target": concept_id, "properties": {"confidence": 1.0}})
    return {"graph_type": "instance_graph", "nodes": nodes, "edges": edges}


class SymbolicTemplateV2Tests(unittest.TestCase):
    def test_yaml_weights_are_applied_and_validated(self) -> None:
        yaml_text = """
version: 2
template_policy:
  matching:
    required_weight: 2.5
    preferred_weight: 0.7
    optional_weight: 0.2
    negative_weight: -1.5
gestures:
  G01:
    expected_events:
      - subject: any
        finger_motions:
          required: [index_finger_moves_down]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "templates.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            parsed = parse_template_yaml(path)
        self.assertEqual(parsed["template_policy"]["matching"]["required_weight"], 2.5)

    def test_legacy_v1_policy_is_normalized_and_emitted_as_v1(self) -> None:
        yaml_text = """
version: 1
template_policy:
  matching:
  required: 1.25
gestures:
  G01:
    expected_events:
      - subject: any
        finger_motions:
          required: [click]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "templates.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            parsed = parse_template_yaml(path)
        weights = {
            level: parsed["template_policy"]["matching"][f"{level}_weight"]
            for level in WEIGHTS
        }
        graph = build_template_graph("G01", 2, parsed["gestures"]["G01"], weights, schema_version=parsed["version"])
        self.assertEqual(graph["schema_version"], 1)
        self.assertEqual(weights["required"], 1.25)

    def test_negative_evidence_requires_a_list(self) -> None:
        yaml_text = """
version: 2
gestures:
  G01:
    expected_events:
      - finger_motions:
          required: [click]
    negative_evidence:
      finger_motions: pinch_distance_increasing
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "templates.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_template_yaml(path)

    def test_v2_event_requires_at_least_one_required_concept(self) -> None:
        yaml_text = """
version: 2
gestures:
  G01:
    expected_events:
      - finger_motions:
          preferred: [click]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "templates.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one required"):
                parse_template_yaml(path)

    def test_repeated_concepts_use_event_specific_nodes_and_before_edge(self) -> None:
        template = {
            "description": "two pulses",
            "notes": [],
            "expected_events": [event_template("pulse"), event_template("pulse")],
            "negative_evidence": {},
        }
        graph = build_template_graph("G08", 9, template, WEIGHTS)
        pulse_nodes = [node for node in graph["nodes"] if node.get("properties", {}).get("name") == "pulse"]
        self.assertEqual(len(pulse_nodes), 2)
        self.assertNotEqual(pulse_nodes[0]["id"], pulse_nodes[1]["id"])
        self.assertTrue(any(edge["relation"] == "BEFORE" for edge in graph["edges"]))

    def test_occurrence_constraints_separate_single_and_double_click(self) -> None:
        single = build_template_graph(
            "G01",
            2,
            {"description": "", "notes": [], "expected_events": [event_template("click", 1, 1)], "negative_evidence": {}},
            WEIGHTS,
        )
        double = build_template_graph(
            "G08",
            9,
            {"description": "", "notes": [], "expected_events": [event_template("click", 2)], "negative_evidence": {}},
            WEIGHTS,
        )
        one_click = instance_graph(["click"])
        two_clicks = instance_graph(["click", "click"])
        self.assertGreater(score_template(one_click, single)["score"], score_template(one_click, double)["score"])
        self.assertGreater(score_template(two_clicks, double)["score"], score_template(two_clicks, single)["score"])
        self.assertEqual(score_template(two_clicks, single)["temporal_score"], 0.0)

        legacy_double = dict(double)
        legacy_double.pop("schema_version")
        self.assertGreater(
            score_template(one_click, legacy_double)["score"],
            score_template(one_click, double)["score"],
        )

    def test_ordered_event_sequence_rewards_correct_order(self) -> None:
        sequence = build_template_graph(
            "G07",
            8,
            {
                "description": "",
                "notes": [],
                "expected_events": [event_template(name) for name in ("closed", "open", "closed", "open")],
                "negative_evidence": {},
            },
            WEIGHTS,
        )
        correct = score_template(instance_graph(["closed", "open", "closed", "open"]), sequence)
        wrong = score_template(instance_graph(["open", "closed", "open", "closed"]), sequence)
        self.assertEqual(correct["temporal_score"], 1.0)
        self.assertGreater(correct["score"], wrong["score"])

    def test_max_occurrence_violation_zeros_multi_event_template(self) -> None:
        template = build_template_graph(
            "GXX",
            99,
            {
                "description": "",
                "notes": [],
                "expected_events": [event_template("click", 1, 1), event_template("open")],
                "negative_evidence": {},
            },
            WEIGHTS,
        )
        result = score_template(instance_graph(["click", "click", "open"]), template)
        self.assertTrue(result["event_matches"][0]["observed_occurrences"] > 1)
        self.assertEqual(result["temporal_score"], 0.0)
        self.assertEqual(result["score"], 0.0)

    def test_phase_splits_of_one_atomic_event_count_once(self) -> None:
        double = build_template_graph(
            "G08",
            9,
            {"description": "", "notes": [], "expected_events": [event_template("click", 2)], "negative_evidence": {}},
            WEIGHTS,
        )
        split_phase_graph = instance_graph(["click", "click"])
        for edge in split_phase_graph["edges"]:
            edge.setdefault("properties", {})["atomic_event_ids"] = ["atomic_click_1"]
        result = score_template(split_phase_graph, double)
        self.assertEqual(result["event_matches"][0]["observed_occurrences"], 1)
        self.assertEqual(result["temporal_score"], 0.5)

    def test_long_atomic_predicate_split_by_phase_builder_counts_once(self) -> None:
        concept = "index_finger_moves_down"
        double = build_template_graph(
            "G08",
            9,
            {"description": "", "notes": [], "expected_events": [event_template(concept, 2)], "negative_evidence": {}},
            WEIGHTS,
        )
        events = build_events_phase_v2(
            {
                "video_id": "long-click",
                "predicates": [
                    {
                        "name": concept,
                        "subject": "right_hand",
                        "object": None,
                        "start_frame": 0,
                        "end_frame": 40,
                        "confidence": 1.0,
                        "evidence_keypoints": ["right_forefinger4"],
                    }
                ],
            },
            max_phase_duration=16,
            min_phase_duration=1,
        )
        self.assertGreater(len(events["events"]), 1)
        result = score_template(build_instance_graph_phase_v2(events), double)
        self.assertEqual(result["event_matches"][0]["observed_occurrences"], 1)
        self.assertEqual(result["temporal_score"], 0.5)

    def test_occurrences_from_different_hands_do_not_form_double_click(self) -> None:
        double = build_template_graph(
            "G08",
            9,
            {"description": "", "notes": [], "expected_events": [event_template("click", 2)], "negative_evidence": {}},
            WEIGHTS,
        )
        graph = instance_graph(["click", "click"])
        event_nodes = [node for node in graph["nodes"] if node["type"] == "Event"]
        event_nodes[1]["properties"]["subject"] = "left_hand"
        result = score_template(graph, double)
        self.assertEqual(result["event_matches"][0]["observed_occurrences"], 1)
        self.assertEqual(result["temporal_score"], 0.5)

    def test_event_level_atomic_ids_are_used_as_legacy_fallback(self) -> None:
        double = build_template_graph(
            "G08",
            9,
            {"description": "", "notes": [], "expected_events": [event_template("click", 2)], "negative_evidence": {}},
            WEIGHTS,
        )
        graph = instance_graph(["click", "click"])
        for node in graph["nodes"]:
            if node["type"] == "Event":
                node["properties"]["atomic_event_ids"] = ["atomic_click_1"]
        result = score_template(graph, double)
        self.assertEqual(result["event_matches"][0]["observed_occurrences"], 1)
        self.assertEqual(result["temporal_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
