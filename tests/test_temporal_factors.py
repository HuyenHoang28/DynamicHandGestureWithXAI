import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.predicates.temporal_factors import (  # noqa: E402
    ClickCycle,
    ScalarSample,
    detect_click_cycles,
    detect_pinch_transition,
    synchronize_two_finger_cycles,
)


def samples(values, *, valid=None, confidence=None):
    valid = valid or [True] * len(values)
    confidence = confidence or [1.0] * len(values)
    return [
        ScalarSample(value, is_valid, conf, frame_id)
        for frame_id, (value, is_valid, conf) in enumerate(zip(values, valid, confidence))
    ]


class TemporalFactorTests(unittest.TestCase):
    def test_one_and_two_completed_click_cycles(self):
        one = detect_click_cycles(samples([0.0, 0.8, 0.9, 0.2, 0.1]), active_threshold=0.7, inactive_threshold=0.3)
        two = detect_click_cycles(samples([0.0, 0.8, 0.2, 0.1, 0.85, 0.2]), active_threshold=0.7, inactive_threshold=0.3)
        self.assertEqual(len(one), 1)
        self.assertEqual(len(two), 2)
        self.assertEqual((one[0].start_frame_id, one[0].end_frame_id), (1, 3))
        self.assertEqual(two[0].cycle_id, (1, 2))
        self.assertEqual(two[1].cycle_id, (4, 5))

    def test_close_clicks_remain_distinct_after_reset_and_incomplete_is_ignored(self):
        cycles = detect_click_cycles(
            samples([0.0, 0.8, 0.2, 0.1, 0.85, 0.2, 0.9]),
            active_threshold=0.7,
            inactive_threshold=0.3,
        )
        self.assertEqual(len(cycles), 2)
        self.assertNotEqual(cycles[0].cycle_id, cycles[1].cycle_id)
        self.assertEqual((cycles[0].start_frame_id, cycles[0].end_frame_id), (1, 2))
        self.assertEqual((cycles[1].start_frame_id, cycles[1].end_frame_id), (4, 5))

    def test_click_requires_open_baseline_and_does_not_bridge_missing_span(self):
        starts_closed = detect_click_cycles(
            samples([0.9, 0.8, 0.2]),
            active_threshold=0.7,
            inactive_threshold=0.3,
        )
        occluded = detect_click_cycles(
            [
                ScalarSample(0.1, True, 1.0, 0),
                ScalarSample(0.9, True, 1.0, 1),
                ScalarSample(0.2, True, 1.0, 10),
            ],
            active_threshold=0.7,
            inactive_threshold=0.3,
            maximum_gap_frames=2,
        )
        self.assertEqual(starts_closed, ())
        self.assertEqual(occluded, ())

    def test_two_finger_synchrony_pairs_only_nearby_overlapping_cycles(self):
        left = detect_click_cycles(samples([0, 1, 0, 0, 1, 0]), active_threshold=0.7, inactive_threshold=0.3)
        synced_right = detect_click_cycles(samples([0, 0, 1, 0, 0, 0]), active_threshold=0.7, inactive_threshold=0.3)
        unsynced_right = detect_click_cycles(samples([0, 0, 0, 0, 1, 0]), active_threshold=0.7, inactive_threshold=0.3)
        synced = synchronize_two_finger_cycles(left, synced_right, max_start_delta_frames=1, minimum_overlap_ratio=0.5)
        unsynced = synchronize_two_finger_cycles(left[:1], unsynced_right, max_start_delta_frames=1, minimum_overlap_ratio=0.5)
        self.assertTrue(synced.synchronized)
        self.assertEqual(len(synced.matches), 1)
        self.assertFalse(unsynced.synchronized)

    def test_pinch_opening_and_closing(self):
        opening = detect_pinch_transition(samples([0.2, 0.3, 0.55, 0.8]))
        closing = detect_pinch_transition(samples([0.9, 0.7, 0.4, 0.2]))
        self.assertEqual(opening.direction, "opening")
        self.assertEqual(closing.direction, "closing")
        self.assertGreater(opening.quality, 0.0)
        self.assertGreater(closing.confidence, 0.0)

    def test_action_then_return_uses_dominant_earliest_excursion(self):
        transition = detect_pinch_transition(samples([0.2, 0.3, 0.7, 0.9, 0.7, 0.5, 0.35]))
        self.assertEqual(transition.direction, "opening")
        self.assertEqual((transition.start_frame_id, transition.end_frame_id), (0, 3))

    def test_ambiguous_flat_and_low_valid_data_are_unknown(self):
        ambiguous = detect_pinch_transition(samples([0.5, 0.51, 0.5, 0.51]), minimum_excursion=0.05)
        sparse = detect_pinch_transition(
            samples([0.1, 0.9, 0.2, 0.8], valid=[True, False, False, False]),
            minimum_valid_ratio=0.75,
        )
        self.assertEqual(ambiguous.direction, "unknown")
        self.assertEqual(sparse.direction, "unknown")
        self.assertEqual(sparse.quality, 0.0)

    def test_equal_opposite_pinch_runs_and_occlusion_bridge_are_unknown(self):
        equal_return = detect_pinch_transition(samples([0.2, 0.5, 0.8, 0.5, 0.2]))
        occluded = detect_pinch_transition(
            [
                ScalarSample(0.2, True, 1.0, 0),
                ScalarSample(0.3, True, 1.0, 1),
                ScalarSample(0.9, True, 1.0, 10),
                ScalarSample(1.0, True, 1.0, 11),
            ],
            maximum_gap_frames=2,
            minimum_valid_ratio=1.0,
            minimum_excursion=0.3,
        )
        self.assertEqual(equal_return.direction, "unknown")
        self.assertEqual(occluded.direction, "unknown")

    def test_zero_overlap_never_synchronizes_even_when_threshold_is_zero(self):
        left = ClickCycle((0, 1), 0, 1, 1.0, 1.0)
        right = ClickCycle((2, 3), 2, 3, 1.0, 1.0)
        result = synchronize_two_finger_cycles(
            [left],
            [right],
            max_start_delta_frames=2,
            minimum_overlap_ratio=0.0,
        )
        self.assertFalse(result.synchronized)

    def test_rejects_non_increasing_frame_ids_and_bad_hysteresis(self):
        with self.assertRaises(ValueError):
            detect_click_cycles(
                [ScalarSample(0.0, True, 1.0, 1), ScalarSample(1.0, True, 1.0, 1)],
                active_threshold=0.7,
                inactive_threshold=0.3,
            )
        with self.assertRaises(ValueError):
            detect_click_cycles([], active_threshold=0.3, inactive_threshold=0.7)


if __name__ == "__main__":
    unittest.main()
