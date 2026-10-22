import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.kg.build_template_graph import build_template_graph, parse_template_yaml
from src_neurosymbolic.events.build_events_phase_v2 import build_events_phase_v2, classify_action_roles
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


def add_event_fact(
    graph: dict,
    event_id: str,
    node_type: str,
    relation: str,
    concept: str,
) -> None:
    concept_id = f"{node_type.lower()}:{event_id}:{concept}"
    graph["nodes"].append(
        {"id": concept_id, "type": node_type, "properties": {"name": concept}}
    )
    graph["edges"].append(
        {
            "source": event_id,
            "relation": relation,
            "target": concept_id,
            "properties": {"confidence": 1.0},
        }
    )


class SymbolicTemplateV2Tests(unittest.TestCase):
    def test_directional_triplet_marks_two_active_runs_and_preserves_atomic_ids(self) -> None:
        runs = [
            {"atomic_event_id": "atomic_event_000001", "subject": "right_hand", "predicate_name": "hand_moves_down", "start_frame": 0, "end_frame": 4},
            {"atomic_event_id": "atomic_event_000002", "subject": "right_hand", "predicate_name": "hand_moves_up", "start_frame": 6, "end_frame": 10},
            {"atomic_event_id": "atomic_event_000003", "subject": "right_hand", "predicate_name": "hand_moves_down", "start_frame": 12, "end_frame": 16},
        ]
        classify_action_roles(runs)
        self.assertEqual([run["action_role"] for run in runs], ["active", "return", "active"])
        self.assertEqual([run["atomic_event_id"] for run in runs], [
            "atomic_event_000001", "atomic_event_000002", "atomic_event_000003"
        ])

    def test_isolated_directional_run_is_active_and_overlap_is_ambiguous(self) -> None:
        isolated = [{"atomic_event_id": "one", "subject": "right_hand", "predicate_name": "pinch_distance_decreasing", "start_frame": 0, "end_frame": 4}]
        classify_action_roles(isolated)
        self.assertEqual(isolated[0]["action_role"], "active")
        overlapping = [
            {"atomic_event_id": "up", "subject": "right_hand", "predicate_name": "hand_moves_up", "start_frame": 0, "end_frame": 5},
            {"atomic_event_id": "down", "subject": "right_hand", "predicate_name": "hand_moves_down", "start_frame": 5, "end_frame": 9},
        ]
        classify_action_roles(overlapping)
        self.assertEqual([item["action_role"] for item in overlapping], ["ambiguous", "ambiguous"])

    def test_pinch_return_does_not_match_active_template_but_legacy_any_does(self) -> None:
        pinch_runs = [
            {"atomic_event_id": "pinch_1", "subject": "right_hand", "predicate_name": "pinch_distance_increasing", "start_frame": 0, "end_frame": 4},
            {"atomic_event_id": "pinch_2", "subject": "right_hand", "predicate_name": "pinch_distance_decreasing", "start_frame": 6, "end_frame": 10},
        ]
        classify_action_roles(pinch_runs)
        self.assertEqual([run["action_role"] for run in pinch_runs], ["active", "return"])
        active_template = build_template_graph(
            "G10", 10,
            {"description": "", "notes": [], "expected_events": [
                {**event_template("pinch_distance_decreasing"), "action_role": "active"}
            ], "negative_evidence": {}},
            WEIGHTS,
        )
        legacy_template = build_template_graph(
            "legacy", 10,
            {"description": "", "notes": [], "expected_events": [event_template("pinch_distance_decreasing")], "negative_evidence": {}},
            WEIGHTS,
        )
        returning_instance = instance_graph(["pinch_distance_decreasing"])
        for edge in returning_instance["edges"]:
            edge["properties"]["action_roles"] = ["return"]
        self.assertEqual(score_template(returning_instance, active_template)["score"], 0.0)
        self.assertGreater(score_template(returning_instance, legacy_template)["score"], 0.0)

    def test_instance_graph_does_not_claim_action_role_builder_when_metadata_missing(self) -> None:
        graph = build_instance_graph_phase_v2({"video_id": "legacy", "events": []})
        self.assertEqual(graph["summary"]["event_builder"], "unknown")
        video = next(node for node in graph["nodes"] if node["type"] == "Video")
        self.assertEqual(video["properties"]["event_builder"], "unknown")

    def test_template_parser_rejects_unknown_action_role_and_defaults_legacy_to_any(self) -> None:
        def parse(role: str | None) -> dict:
            role_line = f"        action_role: {role}\n" if role is not None else ""
            yaml_text = (
                "version: 2\ngestures:\n  G01:\n    expected_events:\n"
                "      - subject: any\n" + role_line
                + "        finger_motions:\n          required: [click]\n"
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "templates.yaml"
                path.write_text(yaml_text, encoding="utf-8")
                return parse_template_yaml(path)

        self.assertEqual(parse(None)["gestures"]["G01"]["expected_events"][0]["action_role"], "any")
        with self.assertRaisesRegex(ValueError, "action_role"):
            parse("sometimes")

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
        # Excess evidence is a bounded penalty, so a single-click template still
        # retains useful evidence while the truly separated double favors G08.
        self.assertGreater(score_template(two_clicks, single)["temporal_score"], 0.0)
        self.assertLess(score_template(two_clicks, single)["temporal_score"], 1.0)

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

    def test_ordered_sequence_rejects_overlapping_phase_reuse(self) -> None:
        sequence = build_template_graph(
            "G07",
            8,
            {
                "description": "",
                "notes": [],
                "expected_events": [
                    event_template(name)
                    for name in ("closed", "open", "closed", "open")
                ],
                "negative_evidence": {},
            },
            WEIGHTS,
        )
        graph = instance_graph(["closed", "open", "closed", "open"])
        event_nodes = [node for node in graph["nodes"] if node["type"] == "Event"]
        event_nodes[0]["properties"]["end_frame"] = 25

        result = score_template(graph, sequence)

        self.assertLess(result["temporal_score"], 1.0)
        for previous, current in zip(
            result["sequence_matches"], result["sequence_matches"][1:]
        ):
            self.assertLess(previous["end_frame"], current["start_frame"])

    def test_ordered_sequence_does_not_reuse_atomic_evidence(self) -> None:
        sequence = build_template_graph(
            "G07",
            8,
            {
                "description": "",
                "notes": [],
                "expected_events": [event_template("closed"), event_template("open")],
                "negative_evidence": {},
            },
            WEIGHTS,
        )
        # Different concepts avoid same-template episode deduplication. The
        # intervals do not overlap, so only the shared occurrence ID can stop
        # the second phase from reusing the first phase's physical evidence.
        graph = instance_graph(["closed", "open"])
        for edge in graph["edges"]:
            edge.setdefault("properties", {})["atomic_event_ids"] = ["same-action"]

        result = score_template(graph, sequence)

        self.assertEqual(result["matched_events"], 1)
        self.assertEqual(result["temporal_score"], 0.5)

    def test_ordered_sequence_backtracks_when_earliest_candidate_blocks_completion(self) -> None:
        sequence = build_template_graph(
            "G07",
            8,
            {
                "description": "",
                "notes": [],
                "expected_events": [event_template("closed"), event_template("open")],
                "negative_evidence": {},
            },
            WEIGHTS,
        )
        graph = instance_graph(["closed", "closed", "open"])
        for index, edge in enumerate(graph["edges"]):
            edge.setdefault("properties", {})["atomic_event_ids"] = [
                "shared-action" if index in (0, 2) else "later-closed"
            ]

        result = score_template(graph, sequence)

        self.assertEqual(result["temporal_score"], 1.0)
        self.assertEqual(result["matched_events"], 2)
        self.assertEqual(result["sequence_matches"][0]["instance_events"], ["event:2"])

    def test_max_occurrence_violation_uses_soft_penalty_multi_event_template(self) -> None:
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
        self.assertTrue(result["event_matches"][0]["max_occurrence_penalty"] < 1.0)
        self.assertGreater(result["temporal_score"], 0.0)
        self.assertGreater(result["score"], 0.0)

    def test_v2_required_concepts_split_across_events_are_not_conjunctive(self) -> None:
        template = build_template_graph(
            "GXX",
            99,
            {
                "description": "",
                "notes": [],
                "expected_events": [
                    {
                        "subject": "any",
                        "requirements": {
                            "hand_shapes": {"required": ["open"]},
                            "finger_motions": {"required": ["click", "swipe"]},
                        },
                    }
                ],
                "negative_evidence": {},
            },
            WEIGHTS,
        )
        graph = instance_graph(["click", "swipe"])
        add_event_fact(graph, "event:1", "HandShape", "HAS_HAND_SHAPE", "open")
        result = score_template(graph, template)
        self.assertLess(result["required_hit"], result["required_total"])
        self.assertEqual(result["missing_required_count"], 1)
        self.assertLess(result["positive_score"], result["positive_total"])
        self.assertEqual(result["score"], 0.0)

    def test_v2_required_concepts_split_across_hands_are_not_conjunctive(self) -> None:
        template = build_template_graph(
            "GXX",
            99,
            {
                "description": "",
                "notes": [],
                "expected_events": [
                    {
                        "subject": "any",
                        "requirements": {
                            "hand_shapes": {"required": ["open"]},
                            "finger_motions": {"required": ["click"]},
                        },
                    }
                ],
                "negative_evidence": {},
            },
            WEIGHTS,
        )
        graph = instance_graph(["unrelated", "click"])
        next(node for node in graph["nodes"] if node["id"] == "event:2")["properties"]["subject"] = "left_hand"
        add_event_fact(graph, "event:1", "HandShape", "HAS_HAND_SHAPE", "open")
        result = score_template(graph, template)
        self.assertEqual(result["required_hit"], 1)
        self.assertEqual(result["missing_required_count"], 1)
        self.assertEqual(result["score"], 0.0)

    def test_overlapping_or_adjacent_fragments_count_once_for_click_occurrence(self) -> None:
        double = build_template_graph(
            "G08",
            9,
            {"description": "", "notes": [], "expected_events": [event_template("click", 2)], "negative_evidence": {}},
            WEIGHTS,
        )
        for start_frame in (12, 13):
            graph = instance_graph(["click", "click"])
            event = next(node for node in graph["nodes"] if node["id"] == "event:2")
            event["properties"]["start_frame"] = start_frame
            event["properties"]["end_frame"] = start_frame + 2
            result = score_template(graph, double)
            self.assertEqual(result["event_matches"][0]["observed_occurrences"], 1)
            self.assertEqual(result["temporal_score"], 0.5)

    def test_legacy_v1_adjacent_events_keep_event_id_occurrence_semantics(self) -> None:
        legacy_double = build_template_graph(
            "G08",
            9,
            {
                "description": "",
                "notes": [],
                "expected_events": [event_template("click", 2)],
                "negative_evidence": {},
            },
            WEIGHTS,
            schema_version=1,
        )
        graph = instance_graph(["click", "click"])
        second = next(node for node in graph["nodes"] if node["id"] == "event:2")
        second["properties"]["start_frame"] = 13
        second["properties"]["end_frame"] = 15

        result = score_template(graph, legacy_double)

        self.assertEqual(result["event_matches"][0]["observed_occurrences"], 2)
        self.assertEqual(result["temporal_score"], 1.0)

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

    def test_persistent_shape_does_not_merge_two_distinct_click_occurrences(self) -> None:
        def click_template(minimum: int, maximum: int) -> dict:
            return build_template_graph(
                "GXX",
                99,
                {
                    "description": "",
                    "notes": [],
                    "expected_events": [
                        {
                            "subject": "any",
                            "min_occurrences": minimum,
                            "max_occurrences": maximum,
                            "requirements": {
                                "hand_shapes": {"required": ["index_extended"]},
                                "finger_motions": {
                                    "required": ["index_finger_moves_down"]
                                },
                            },
                        }
                    ],
                    "negative_evidence": {},
                },
                WEIGHTS,
            )

        events = build_events_phase_v2(
            {
                "video_id": "two-clicks-with-stable-shape",
                "predicates": [
                    {
                        "name": "index_extended",
                        "subject": "right_hand",
                        "object": None,
                        "start_frame": 0,
                        "end_frame": 30,
                        "confidence": 1.0,
                        "evidence_keypoints": ["right_forefinger4"],
                    },
                    {
                        "name": "index_finger_moves_down",
                        "subject": "right_hand",
                        "object": None,
                        "start_frame": 5,
                        "end_frame": 8,
                        "confidence": 1.0,
                        "evidence_keypoints": ["right_forefinger4"],
                    },
                    {
                        "name": "index_finger_moves_down",
                        "subject": "right_hand",
                        "object": None,
                        "start_frame": 20,
                        "end_frame": 23,
                        "confidence": 1.0,
                        "evidence_keypoints": ["right_forefinger4"],
                    },
                ],
            },
            max_phase_duration=16,
            min_phase_duration=1,
        )
        graph = build_instance_graph_phase_v2(events)
        single_result = score_template(graph, click_template(1, 1))
        double_result = score_template(graph, click_template(2, 2))

        self.assertEqual(double_result["event_matches"][0]["observed_occurrences"], 2)
        self.assertGreater(double_result["score"], single_result["score"])

    def test_two_finger_clicks_use_motion_ids_instead_of_persistent_shape_id(self) -> None:
        template = build_template_graph(
            "G09",
            10,
            {
                "description": "",
                "notes": [],
                "expected_events": [
                    {
                        "subject": "any",
                        "min_occurrences": 2,
                        "max_occurrences": 2,
                        "requirements": {
                            "hand_shapes": {"required": ["index_middle_extended"]},
                            "finger_motions": {
                                "required": [
                                    "index_finger_moves_down",
                                    "middle_finger_moves_down",
                                ]
                            },
                        },
                    }
                ],
                "negative_evidence": {},
            },
            WEIGHTS,
        )
        predicates = [
            {
                "name": "index_middle_extended",
                "subject": "right_hand",
                "object": None,
                "start_frame": 0,
                "end_frame": 30,
                "confidence": 1.0,
                "evidence_keypoints": ["right_forefinger4", "right_middle_finger4"],
            }
        ]
        for start, end in ((5, 8), (20, 23)):
            for concept, keypoint in (
                ("index_finger_moves_down", "right_forefinger4"),
                ("middle_finger_moves_down", "right_middle_finger4"),
            ):
                predicates.append(
                    {
                        "name": concept,
                        "subject": "right_hand",
                        "object": None,
                        "start_frame": start,
                        "end_frame": end,
                        "confidence": 1.0,
                        "evidence_keypoints": [keypoint],
                    }
                )

        events = build_events_phase_v2(
            {"video_id": "two-two-finger-clicks", "predicates": predicates},
            max_phase_duration=16,
            min_phase_duration=1,
        )
        result = score_template(build_instance_graph_phase_v2(events), template)

        self.assertEqual(result["event_matches"][0]["observed_occurrences"], 2)
        self.assertEqual(result["required_hit"], result["required_total"])

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
