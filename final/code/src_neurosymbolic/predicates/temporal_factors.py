"""Interpretable temporal factors computed from scalar frame samples.

This module intentionally has no array or scientific-computing dependencies. A
sample can represent any scalar signal (for example, fingertip distance or a
closed/open score); callers define its units and orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal


@dataclass(frozen=True)
class ScalarSample:
    """One scalar observation and its provenance/quality."""

    value: float
    valid: bool
    confidence: float
    frame_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, int):
            raise TypeError("frame_id must be an int")
        if not isfinite(float(self.value)):
            raise ValueError("value must be finite")
        if not isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")


@dataclass(frozen=True)
class ClickCycle:
    """A completed high-state/low-state cycle with inclusive frame bounds."""

    cycle_id: tuple[int, int]
    start_frame_id: int
    end_frame_id: int
    peak_value: float
    confidence: float


@dataclass(frozen=True)
class SynchronizedCyclePair:
    left: ClickCycle
    right: ClickCycle
    start_delta_frames: int
    overlap_ratio: float


@dataclass(frozen=True)
class SynchronyResult:
    synchronized: bool
    matches: tuple[SynchronizedCyclePair, ...]


@dataclass(frozen=True)
class PinchTransition:
    direction: Literal["opening", "closing", "unknown"]
    confidence: float
    quality: float
    start_frame_id: int | None
    end_frame_id: int | None
    excursion: float


def _ordered_samples(samples: Iterable[ScalarSample]) -> list[ScalarSample]:
    result = list(samples)
    if any(a.frame_id >= b.frame_id for a, b in zip(result, result[1:])):
        raise ValueError("samples must have strictly increasing frame_id values")
    return result


def detect_click_cycles(
    samples: Iterable[ScalarSample],
    *,
    active_threshold: float,
    inactive_threshold: float,
    minimum_confidence: float = 0.0,
    maximum_gap_frames: int = 2,
) -> tuple[ClickCycle, ...]:
    """Return completed hysteretic cycles (active when value >= active threshold).

    The lower threshold must be strictly below the upper threshold. Invalid or
    below-confidence samples are skipped without changing state. A cycle is
    emitted only after an observed inactive baseline, a qualifying active
    observation, and a return at/below the inactive threshold. Missing spans
    longer than ``maximum_gap_frames`` disarm/reset the detector. Unfinished
    cycles are ignored. No temporal merging is applied, so a reset between
    clicks keeps them distinct even when their frame intervals are close.
    """

    if not isfinite(active_threshold) or not isfinite(inactive_threshold):
        raise ValueError("thresholds must be finite")
    if active_threshold <= inactive_threshold:
        raise ValueError("active_threshold must be greater than inactive_threshold")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between 0 and 1")
    if not isinstance(maximum_gap_frames, int) or maximum_gap_frames < 0:
        raise ValueError("maximum_gap_frames must be a non-negative int")

    cycles: list[ClickCycle] = []
    active_start: ScalarSample | None = None
    armed = False
    last_eligible_frame: int | None = None
    peak = float("-inf")
    confidences: list[float] = []
    for sample in _ordered_samples(samples):
        if not sample.valid or sample.confidence < minimum_confidence:
            continue
        if (
            last_eligible_frame is not None
            and sample.frame_id - last_eligible_frame - 1 > maximum_gap_frames
        ):
            active_start = None
            armed = False
            peak = float("-inf")
            confidences = []
        last_eligible_frame = sample.frame_id
        if active_start is None:
            if sample.value <= inactive_threshold:
                armed = True
                continue
            if armed and sample.value >= active_threshold:
                active_start = sample
                peak = sample.value
                confidences = [sample.confidence]
            continue

        peak = max(peak, sample.value)
        confidences.append(sample.confidence)
        if sample.value <= inactive_threshold:
            cycle_confidence = sum(confidences) / len(confidences)
            cycles.append(
                ClickCycle(
                    cycle_id=(active_start.frame_id, sample.frame_id),
                    start_frame_id=active_start.frame_id,
                    end_frame_id=sample.frame_id,
                    peak_value=peak,
                    confidence=cycle_confidence,
                )
            )
            active_start = None
            armed = True
            peak = float("-inf")
            confidences = []
    return tuple(cycles)


def synchronize_two_finger_cycles(
    left_cycles: Iterable[ClickCycle],
    right_cycles: Iterable[ClickCycle],
    *,
    max_start_delta_frames: int,
    minimum_overlap_ratio: float = 0.0,
) -> SynchronyResult:
    """Greedily pair one-to-one cycles that start near each other and overlap.

    Overlap is intersection length divided by the longer inclusive cycle length.
    Pairs are selected by smallest start-time delta, then earliest left/right
    frame IDs. This deterministic rule prevents a single cycle being reused.
    """

    if not isinstance(max_start_delta_frames, int) or max_start_delta_frames < 0:
        raise ValueError("max_start_delta_frames must be a non-negative int")
    if not 0.0 <= minimum_overlap_ratio <= 1.0:
        raise ValueError("minimum_overlap_ratio must be between 0 and 1")
    left, right = tuple(left_cycles), tuple(right_cycles)
    candidates: list[tuple[int, int, int, int, SynchronizedCyclePair]] = []
    for li, lcycle in enumerate(left):
        for ri, rcycle in enumerate(right):
            delta = abs(lcycle.start_frame_id - rcycle.start_frame_id)
            start = max(lcycle.start_frame_id, rcycle.start_frame_id)
            end = min(lcycle.end_frame_id, rcycle.end_frame_id)
            intersection = max(0, end - start + 1)
            llength = lcycle.end_frame_id - lcycle.start_frame_id + 1
            rlength = rcycle.end_frame_id - rcycle.start_frame_id + 1
            overlap = intersection / max(llength, rlength)
            if intersection > 0 and delta <= max_start_delta_frames and overlap >= minimum_overlap_ratio:
                candidates.append(
                    (delta, lcycle.start_frame_id, rcycle.start_frame_id, li * max(1, len(right)) + ri,
                     SynchronizedCyclePair(lcycle, rcycle, delta, overlap))
                )
    candidates.sort(key=lambda item: item[:4])
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[SynchronizedCyclePair] = []
    for _, _, _, encoded, pair in candidates:
        li, ri = divmod(encoded, max(1, len(right)))
        if li not in used_left and ri not in used_right:
            used_left.add(li)
            used_right.add(ri)
            matches.append(pair)
    return SynchronyResult(bool(matches), tuple(matches))


def detect_pinch_transition(
    samples: Iterable[ScalarSample],
    *,
    minimum_valid_ratio: float = 0.6,
    minimum_confidence: float = 0.25,
    minimum_excursion: float = 0.05,
    maximum_gap_frames: int = 2,
    minimum_dominance_ratio: float = 1.1,
) -> PinchTransition:
    """Classify the dominant monotonic excursion as opening or closing.

    Opening means increasing scalar values; closing means decreasing values.
    Consecutive same-direction, sufficiently contiguous steps form one
    excursion. The largest excursion wins only when it dominates the strongest
    opposite run by ``minimum_dominance_ratio``. Sparse/low-confidence,
    ambiguous, occlusion-bridged, or effectively flat input returns ``unknown``.
    """

    if not 0.0 <= minimum_valid_ratio <= 1.0:
        raise ValueError("minimum_valid_ratio must be between 0 and 1")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between 0 and 1")
    if not isfinite(minimum_excursion) or minimum_excursion < 0.0:
        raise ValueError("minimum_excursion must be finite and non-negative")
    if not isinstance(maximum_gap_frames, int) or maximum_gap_frames < 0:
        raise ValueError("maximum_gap_frames must be a non-negative int")
    if not isfinite(minimum_dominance_ratio) or minimum_dominance_ratio < 1.0:
        raise ValueError("minimum_dominance_ratio must be finite and at least 1")
    ordered = _ordered_samples(samples)
    eligible = [s for s in ordered if s.valid and s.confidence >= minimum_confidence]
    valid_ratio = len(eligible) / len(ordered) if ordered else 0.0
    mean_confidence = sum(s.confidence for s in eligible) / len(eligible) if eligible else 0.0
    unknown = PinchTransition("unknown", 0.0, 0.0, None, None, 0.0)
    if len(eligible) < 2 or valid_ratio < minimum_valid_ratio:
        return unknown

    runs: list[tuple[float, int, int, int]] = []  # magnitude, direction, start index, end index
    run_start = 0
    run_direction = 0
    run_magnitude = 0.0
    for index in range(1, len(eligible)):
        if eligible[index].frame_id - eligible[index - 1].frame_id - 1 > maximum_gap_frames:
            if run_direction:
                runs.append((run_magnitude, run_direction, run_start, index - 1))
            run_direction = 0
            run_start = index
            run_magnitude = 0.0
            continue
        delta = eligible[index].value - eligible[index - 1].value
        direction = 1 if delta > 0 else -1 if delta < 0 else 0
        if direction == 0:
            continue
        if run_direction == 0:
            run_direction = direction
            run_start = index - 1
            run_magnitude = abs(delta)
        elif direction == run_direction:
            run_magnitude += abs(delta)
        else:
            runs.append((run_magnitude, run_direction, run_start, index - 1))
            run_direction = direction
            run_start = index - 1
            run_magnitude = abs(delta)
    if run_direction:
        runs.append((run_magnitude, run_direction, run_start, len(eligible) - 1))
    if not runs:
        return unknown

    ordered_runs = sorted(runs, key=lambda item: (-item[0], item[2], item[3]))
    dominant = ordered_runs[0]
    opposite = next(
        (candidate for candidate in ordered_runs[1:] if candidate[1] != dominant[1]),
        None,
    )
    if (
        opposite is not None
        and opposite[0] > 0.0
        and dominant[0] / opposite[0] < minimum_dominance_ratio
    ):
        return unknown
    magnitude, direction, start_index, end_index = dominant
    if magnitude < minimum_excursion:
        return unknown
    quality = min(1.0, valid_ratio * mean_confidence * (magnitude / (magnitude + minimum_excursion)))
    if quality <= 0.0:
        return unknown
    return PinchTransition(
        "opening" if direction > 0 else "closing",
        confidence=quality,
        quality=quality,
        start_frame_id=eligible[start_index].frame_id,
        end_frame_id=eligible[end_index].frame_id,
        excursion=magnitude,
    )
