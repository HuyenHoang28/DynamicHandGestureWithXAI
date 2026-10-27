"""Sparse pair-residual correction for the three hard IPN-13 pairs.

Implements the contract in ``docs/IPN_TEMPORAL_FACTORS_PAIR_RESIDUAL_ARCHITECTURE_VI.md``
section 6 as a deterministic post-hoc layer on top of a neural classifier:

* it only acts when the neural top-2 classes are exactly one configured pair,
  the neural margin is small, and a temporal factor gives a decisive,
  sufficiently confident answer for that pair;
* it then moves exactly two logits, ``logit[a] += alpha * s`` and
  ``logit[b] -= alpha * s``; every other logit is untouched;
* with ``alpha == 0`` it is the identity, so it reproduces the neural baseline;
* missing, ``unknown`` or low-quality evidence abstains.

``alpha``, the margin and the quality floor are plain numbers to be chosen on
train/validation only (see ``scripts/eval_pair_residual.py``). A learned pair head
trained on factor labels is a possible later step and is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

LABEL_IDS = {"B0A": 0, "B0B": 1, **{f"G{i:02}": i + 1 for i in range(1, 12)}}
NUM_CLASSES = len(LABEL_IDS)

# family -> (class favoured by "one click"/"opening", class favoured by "two clicks"/"closing")
PAIR_FAMILIES: dict[str, tuple[int, int]] = {
    "index": (LABEL_IDS["G01"], LABEL_IDS["G08"]),
    "two_finger": (LABEL_IDS["G02"], LABEL_IDS["G09"]),
    "pinch": (LABEL_IDS["G10"], LABEL_IDS["G11"]),
}


@dataclass(frozen=True)
class FactorDecision:
    """A temporal factor's answer for one pair: a class id or ``None`` for unknown, plus its quality in [0, 1]."""

    guess: int | None
    quality: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")


def apply_pair_residual(
    logits: torch.Tensor,
    decisions: Sequence[Mapping[str, FactorDecision]],
    *,
    alpha: float,
    max_neural_margin: float,
    min_quality: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(new_logits, applied, pair_index)`` for ``logits`` shaped ``[N, 13]``.

    ``applied`` marks the clips whose logits changed and ``pair_index`` their
    family index in ``PAIR_FAMILIES`` order (``-1`` where nothing changed).
    """

    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must be [N, {NUM_CLASSES}], got {tuple(logits.shape)}")
    if len(decisions) != logits.shape[0]:
        raise ValueError("need one decision mapping per clip")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not 0.0 <= max_neural_margin <= 1.0 or not 0.0 <= min_quality <= 1.0:
        raise ValueError("max_neural_margin and min_quality must be between 0 and 1")

    output = logits.clone()
    applied = torch.zeros(logits.shape[0], dtype=torch.bool)
    pair_index = torch.full((logits.shape[0],), -1, dtype=torch.long)
    if alpha == 0:
        return output, applied, pair_index

    top = logits.softmax(dim=-1).topk(2, dim=-1)
    for clip in range(logits.shape[0]):
        first, second = (int(value) for value in top.indices[clip])
        if float(top.values[clip, 0] - top.values[clip, 1]) > max_neural_margin:
            continue
        for index, (family, (a, b)) in enumerate(PAIR_FAMILIES.items()):
            if {first, second} != {a, b}:
                continue
            decision = decisions[clip].get(family)
            if decision is None or decision.guess not in (a, b) or decision.quality < min_quality:
                break
            shift = alpha * decision.quality * (1.0 if decision.guess == a else -1.0)
            output[clip, a] += shift
            output[clip, b] -= shift
            applied[clip] = True
            pair_index[clip] = index
            break
    return output, applied, pair_index


def correction_effect(before: Sequence[int], after: Sequence[int], labels: Sequence[int]) -> dict[str, int]:
    """Count predictions the residual fixed, broke, or changed without changing correctness."""

    if not len(before) == len(after) == len(labels):
        raise ValueError("before, after and labels must have the same length")
    beneficial = harmful = neutral = 0
    for old, new, truth in zip(before, after, labels):
        if old == new:
            continue
        if new == truth:
            beneficial += 1
        elif old == truth:
            harmful += 1
        else:
            neutral += 1
    return {"beneficial": beneficial, "harmful": harmful, "neutral": neutral, "net": beneficial - harmful}
