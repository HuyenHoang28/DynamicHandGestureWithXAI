"""Scale-invariant hand geometry computed from the 52-keypoint tensor.

Coordinates in the cached tensors are normalised by shoulder width, so a finger
flexing for a click is a tiny change in that frame. Measuring each finger
against the hand's own size turns the same motion into a clean 1-D signal
(tip-to-knuckle distance) whose peaks can be counted, and turns a pinch into a
single distance. This is the same information the symbolic predicates use, but
as a differentiable feature.

The keypoint order is the project-wide order of ``KEYPOINT_NAMES``. A copy of
this module lives in ``src_neurosymbolic/models/hand_geometry.py``; a unit test
keeps the two numerically identical.
"""

import torch

# Index of each hand's joints in the 52-keypoint order.
LEFT_HAND = {"root": 10, "thumb4": 14, "fore1": 15, "fore4": 18, "mid1": 19, "mid4": 22, "ring4": 26, "pinky4": 30}
RIGHT_HAND = {"root": 31, "thumb4": 35, "fore1": 36, "fore4": 39, "mid1": 40, "mid4": 43, "ring4": 47, "pinky4": 51}

FEATURES_PER_HAND = 10
HAND_GEOMETRY_DIMS = 2 * FEATURES_PER_HAND
_RATIO_LIMIT = 5.0


def _dist(xy: torch.Tensor, a: int, b: int) -> torch.Tensor:
    return (xy[:, :, a] - xy[:, :, b]).norm(dim=-1)


def _one_hand(xy: torch.Tensor, ok: torch.Tensor, idx: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    size = _dist(xy, idx["root"], idx["mid1"])
    base = ok[:, :, idx["root"]] & ok[:, :, idx["mid1"]] & (size > 1e-3)
    size = size.clamp_min(1e-3)
    features = []
    valid = []
    for tip in ("thumb4", "fore4", "mid4", "ring4", "pinky4"):
        features.append(_dist(xy, idx[tip], idx["root"]) / size)
        valid.append(base & ok[:, :, idx[tip]])
    for tip, knuckle in (("fore4", "fore1"), ("mid4", "mid1")):
        features.append(_dist(xy, idx[tip], idx[knuckle]) / size)
        valid.append(base & ok[:, :, idx[tip]] & ok[:, :, idx[knuckle]])
    features.append(_dist(xy, idx["thumb4"], idx["fore4"]) / size)
    valid.append(base & ok[:, :, idx["thumb4"]] & ok[:, :, idx["fore4"]])
    for tip, knuckle in (("fore4", "fore1"), ("mid4", "mid1")):
        features.append((xy[:, :, idx[tip], 1] - xy[:, :, idx[knuckle], 1]) / size)
        valid.append(base & ok[:, :, idx[tip]] & ok[:, :, idx[knuckle]])
    return torch.stack(features, dim=-1), torch.stack(valid, dim=-1)


def hand_geometry_features(keypoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(features, valid)`` shaped ``[B, T, 20]`` for ``[B, T, 52, 3]`` input.

    Missing joints (confidence <= 0) give a zero feature and ``valid=False``
    rather than a made-up value, mirroring how velocity is masked.
    """
    xy = keypoints[..., :2]
    ok = keypoints[..., 2] > 0
    left, left_valid = _one_hand(xy, ok, LEFT_HAND)
    right, right_valid = _one_hand(xy, ok, RIGHT_HAND)
    features = torch.cat([left, right], dim=-1).clamp(-_RATIO_LIMIT, _RATIO_LIMIT)
    valid = torch.cat([left_valid, right_valid], dim=-1)
    return features * valid.to(features.dtype), valid


def hand_geometry_inputs(keypoints: torch.Tensor) -> torch.Tensor:
    """Per-frame input block: the geometry plus its frame-to-frame change, ``[B, T, 40]``."""
    features, valid = hand_geometry_features(keypoints)
    change = torch.zeros_like(features)
    both = valid[:, 1:] & valid[:, :-1]
    change[:, 1:] = (features[:, 1:] - features[:, :-1]) * both.to(features.dtype)
    return torch.cat([features, change], dim=-1)
