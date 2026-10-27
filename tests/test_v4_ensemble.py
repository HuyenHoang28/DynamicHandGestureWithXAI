"""Ensembling V4 checkpoints from their saved logits."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ensemble_v4_logits", ROOT / "scripts" / "ensemble_v4_logits.py")
ensemble = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = ensemble
spec.loader.exec_module(ensemble)

PATHS = ["a.json", "b.json", "c.json", "d.json"]
LABELS = [0, 1, 2, 3]


def member(final: torch.Tensor, neural: torch.Tensor | None = None, kg: torch.Tensor | None = None) -> dict:
    return {"paths": PATHS, "labels": LABELS, "logits": {"final": final, "neural": neural if neural is not None else final, "kg": kg if kg is not None else final}}


def one_hot(classes: list[int], strength: float = 6.0, width: int = 13) -> torch.Tensor:
    out = torch.zeros(len(classes), width)
    out[torch.arange(len(classes)), torch.tensor(classes)] = strength
    return out


def test_average_is_the_mean_of_softmax_for_the_chosen_branch() -> None:
    a = member(final=one_hot([0, 1, 2, 3]), kg=one_hot([5, 5, 5, 5]))
    b = member(final=one_hot([0, 1, 2, 4]), kg=one_hot([6, 6, 6, 6]))
    mean = ensemble.average_branch_probabilities([a, b], "final")
    expected = (a["logits"]["final"].softmax(-1) + b["logits"]["final"].softmax(-1)) / 2
    assert torch.allclose(mean, expected)
    kg = ensemble.average_branch_probabilities([a, b], "kg")
    assert torch.allclose(kg, (a["logits"]["kg"].softmax(-1) + b["logits"]["kg"].softmax(-1)) / 2)
    assert not torch.allclose(kg, mean)  # a different branch gives different probabilities


def test_ensemble_can_fix_a_clip_a_single_member_gets_wrong() -> None:
    a = member(final=one_hot([0, 1, 2, 3]))
    wrong = member(final=one_hot([0, 1, 9, 3], strength=3.0))
    right = member(final=one_hot([0, 1, 2, 3], strength=5.0))
    mean = ensemble.average_branch_probabilities([wrong, right, a], "final")
    assert mean.argmax(-1).tolist() == LABELS


def test_rejects_mismatched_clips_unknown_branch_and_bad_weights() -> None:
    a = member(final=one_hot([0, 1, 2, 3]))
    b = {**member(final=one_hot([0, 1, 2, 3])), "paths": PATHS[::-1]}
    with pytest.raises(ValueError):
        ensemble.average_branch_probabilities([a, b], "final")
    with pytest.raises(ValueError):
        ensemble.average_branch_probabilities([a], "gate")
    with pytest.raises(ValueError):
        ensemble.average_branch_probabilities([a], "final", [-1.0])
    with pytest.raises(ValueError):
        ensemble.average_branch_probabilities([], "final")


def test_cli_writes_metrics_and_member_scores(tmp_path: Path) -> None:
    good = member(final=one_hot([0, 1, 2, 3]))
    half = member(final=one_hot([0, 1, 7, 7]))
    torch.save(good, tmp_path / "a.pt")
    torch.save(half, tmp_path / "b.pt")
    out = tmp_path / "metrics.json"
    assert ensemble.main(["--logits", str(tmp_path / "a.pt"), str(tmp_path / "b.pt"), "--branch", "final", "--output", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["total"] == 4 and report["branch"] == "final"
    assert [m["accuracy"] for m in report["members"]] == [1.0, 0.5]
    assert report["per_class"][0]["label_name"] == "B0A"  # 13-class logits get the IPN-13 names
