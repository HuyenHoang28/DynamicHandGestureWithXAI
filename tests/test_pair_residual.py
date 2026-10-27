"""Pair-residual contract: sparse, zero-init, abstains on weak evidence, train/val only."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.pair_residual import (  # noqa: E402
    LABEL_IDS,
    NUM_CLASSES,
    PAIR_FAMILIES,
    FactorDecision,
    apply_pair_residual,
    correction_effect,
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load("eval_pair_residual")
extractor = _load("extract_pair_factors")

G01, G08 = PAIR_FAMILIES["index"]
G02, G09 = PAIR_FAMILIES["two_finger"]
G10, G11 = PAIR_FAMILIES["pinch"]


def logits_with(first: int, second: int, gap: float = 0.2) -> torch.Tensor:
    row = torch.full((NUM_CLASSES,), -6.0)
    row[first] = 2.0
    row[second] = 2.0 - gap
    return row.unsqueeze(0)


def decide(family: str, guess: int | None, quality: float = 0.9):
    return [{family: FactorDecision(guess, quality)}]


def run(logits, decisions, **kwargs):
    params = dict(alpha=2.0, max_neural_margin=0.5, min_quality=0.3)
    params.update(kwargs)
    return apply_pair_residual(logits, decisions, **params)


def test_zero_alpha_is_exactly_the_neural_baseline() -> None:
    logits = logits_with(G01, G08)
    out, applied, index = run(logits, decide("index", G08), alpha=0.0)
    assert torch.equal(out, logits) and not applied.any() and int(index[0]) == -1


def test_only_the_two_pair_logits_move_and_in_opposite_directions() -> None:
    logits = logits_with(G01, G08)
    out, applied, index = run(logits, decide("index", G08, 0.8))
    assert bool(applied[0]) and int(index[0]) == 0
    moved = (out != logits).nonzero()[:, 1].tolist()
    assert sorted(moved) == sorted([G01, G08])
    assert out[0, G08] - logits[0, G08] == pytest.approx(2.0 * 0.8)
    assert out[0, G01] - logits[0, G01] == pytest.approx(-2.0 * 0.8)
    assert out.argmax().item() == G08  # evidence for two clicks flips a G01-leaning clip


def test_guess_for_the_first_class_pushes_toward_it() -> None:
    out, applied, _ = run(logits_with(G08, G01), decide("index", G01))
    assert bool(applied[0]) and out.argmax().item() == G01


@pytest.mark.parametrize(
    "logits, decisions, kwargs",
    [
        (logits_with(G01, G09), decide("index", G08), {}),  # top-2 is not one configured pair
        (logits_with(G01, G08, gap=1.5), decide("index", G08), {"max_neural_margin": 0.1}),  # neural is confident
        (logits_with(G01, G08), decide("index", None), {}),  # unknown evidence abstains
        (logits_with(G01, G08), decide("index", G08, 0.1), {"min_quality": 0.3}),  # low quality abstains
        (logits_with(G01, G08), decide("two_finger", G09), {}),  # factor for a different pair
        (logits_with(G01, G08), decide("index", G10), {}),  # guess outside the pair
        (logits_with(G01, G08), [{}], {}),  # no evidence at all
    ],
)
def test_abstains_and_leaves_logits_untouched(logits, decisions, kwargs) -> None:
    out, applied, index = run(logits, decisions, **kwargs)
    assert torch.equal(out, logits) and not applied.any() and int(index[0]) == -1


def test_each_pair_family_is_routed_to_its_own_classes() -> None:
    for position, (family, (a, b)) in enumerate(PAIR_FAMILIES.items()):
        out, applied, index = run(logits_with(a, b), decide(family, b))
        assert bool(applied[0]) and int(index[0]) == position
        assert out.argmax().item() == b


def test_batch_mixes_applied_and_unchanged_clips() -> None:
    logits = torch.cat([logits_with(G01, G08), logits_with(G02, G09), logits_with(G10, G11)])
    decisions = [{"index": FactorDecision(G08, 0.9)}, {"two_finger": FactorDecision(None, 0.0)}, {"pinch": FactorDecision(G11, 0.9)}]
    out, applied, index = run(logits, decisions)
    assert applied.tolist() == [True, False, True] and index.tolist() == [0, -1, 2]
    assert torch.equal(out[1], logits[1])


def test_input_validation() -> None:
    logits = logits_with(G01, G08)
    with pytest.raises(ValueError):
        apply_pair_residual(logits, [{}], alpha=-1, max_neural_margin=0.5, min_quality=0.3)
    with pytest.raises(ValueError):
        apply_pair_residual(logits, [], alpha=1, max_neural_margin=0.5, min_quality=0.3)
    with pytest.raises(ValueError):
        apply_pair_residual(torch.zeros(1, 5), [{}], alpha=1, max_neural_margin=0.5, min_quality=0.3)
    with pytest.raises(ValueError):
        FactorDecision(G01, 1.5)


def test_correction_effect_counts() -> None:
    assert correction_effect([1, 2, 3, 4, 5], [1, 9, 3, 7, 5], [9, 9, 3, 4, 5]) == {
        "beneficial": 1, "harmful": 1, "neutral": 0, "net": 0,
    }
    assert correction_effect([1, 2], [3, 2], [9, 2]) == {"beneficial": 0, "harmful": 0, "neutral": 1, "net": 0}
    with pytest.raises(ValueError):
        correction_effect([1], [1, 2], [1])


# --- label-free factor extraction ---------------------------------------------


def predicate(name: str, confidence: float, subject: str = "right_hand") -> dict:
    return {"name": name, "confidence": confidence, "subject": subject}


def test_family_decisions_use_no_ground_truth() -> None:
    predicates = [
        predicate("index_click_cycle", 0.8), predicate("index_click_cycle", 0.6),
        predicate("two_finger_click_cycle", 0.7),
        predicate("pinch_opening", 0.9),
    ]
    decisions = extractor.family_decisions(predicates, {"pinch_dominance_ratio": 1.1})
    assert decisions["index"] == ("G08", pytest.approx(0.7))  # two cycles -> double click
    assert decisions["two_finger"] == ("G02", pytest.approx(0.7))  # one cycle -> single click
    assert decisions["pinch"] == ("G10", pytest.approx(0.9))  # opening


def test_family_decisions_abstain_on_missing_or_ambiguous_evidence() -> None:
    assert extractor.family_decisions([], {})["index"] == ("unknown", 0.0)
    ambiguous = [predicate("pinch_opening", 0.9), predicate("pinch_closing", 0.85)]
    assert extractor.family_decisions(ambiguous, {"pinch_dominance_ratio": 1.2})["pinch"] == ("unknown", 0.0)
    triple = [predicate("index_click_cycle", 0.8)] * 3
    assert extractor.family_decisions(triple, {})["index"][0] == "unknown"  # more than two cycles abstains


# --- evaluator ------------------------------------------------------------------


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prediction_row(clip: str, truth: str, top: list[tuple[str, float]]) -> dict:
    return {
        "path": f"keypoints/val/{truth}/S1/{clip}.json", "true_id": LABEL_IDS[truth], "true_label": truth,
        "pred_id": LABEL_IDS[top[0][0]], "pred_label": top[0][0], "confidence": top[0][1],
        "top_labels": " ".join(name for name, _ in top), "top_probs": " ".join(str(p) for _, p in top),
    }


def factor_row(clip: str, split: str = "val", index=("unknown", 0.0)) -> dict:
    row = {"split": split, "source": f"keypoints/val/G08/S1/{clip}.json", "subject": "S1", "true_label": "G08"}
    for family in PAIR_FAMILIES:
        guess, quality = index if family == "index" else ("unknown", 0.0)
        row[f"{family}_guess"], row[f"{family}_quality"] = guess, quality
    return row


PRED_FIELDS = ["path", "true_id", "true_label", "pred_id", "pred_label", "confidence", "top_labels", "top_probs"]


def test_evaluator_reports_beneficial_and_harmful_corrections(tmp_path: Path) -> None:
    predictions = [
        prediction_row("fix", "G08", [("G01", 0.45), ("G08", 0.40), ("B0A", 0.05)]),  # neural wrong, factor says double
        prediction_row("harm", "G01", [("G01", 0.45), ("G08", 0.40), ("B0A", 0.05)]),  # neural right, factor wrong
        prediction_row("keep", "G08", [("G08", 0.9), ("G01", 0.05), ("B0A", 0.02)]),  # confident, untouched
    ]
    write_csv(tmp_path / "p.csv", PRED_FIELDS, predictions)
    write_csv(tmp_path / "f.csv", list(factor_row("x").keys()), [
        factor_row("fix", index=("G08", 0.9)), factor_row("harm", index=("G08", 0.9)), factor_row("keep", index=("G01", 0.9)),
    ])
    out = tmp_path / "report.json"
    assert evaluator.main(["--predictions", str(tmp_path / "p.csv"), "--factors", str(tmp_path / "f.csv"),
                           "--alphas", "2.0", "--margins", "0.2", "--qualities", "0.3", "--output", str(out)]) == 0
    report = json.loads(out.read_text())
    best = report["best_by_net"]
    assert best["beneficial"] == 1 and best["harmful"] == 1 and best["net"] == 0 and best["applied"] == 2
    assert report["baseline"]["applied"] == 0 and report["baseline"]["net"] == 0
    assert report["baseline"]["accuracy_before"] == pytest.approx(2 / 3)


def test_evaluator_refuses_test_split_factor_rows(tmp_path: Path) -> None:
    write_csv(tmp_path / "p.csv", PRED_FIELDS, [prediction_row("a", "G08", [("G08", 0.9), ("G01", 0.05)])])
    write_csv(tmp_path / "f.csv", list(factor_row("a").keys()), [factor_row("a", split="test")])
    with pytest.raises(ValueError, match="only train/val"):
        evaluator.main(["--predictions", str(tmp_path / "p.csv"), "--factors", str(tmp_path / "f.csv"), "--output", str(tmp_path / "r.json")])


def test_extractor_cli_has_no_test_split_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        extractor.main(["--split", "test", "--split-file", "x", "--keypoint-root", "x", "--config", "x", "--output", "x"])
    assert "invalid choice" in capsys.readouterr().err
