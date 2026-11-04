import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final" / "code"))

from src_neurosymbolic.train_cross_attention_v4 import selection_score  # noqa: E402

METRICS = {"accuracy": 0.87, "macro_f1": 0.80, "final_macro_f1": 0.80, "kg_macro_f1": 0.40}


def test_existing_metrics_are_unchanged():
    assert selection_score(METRICS, "accuracy") == 0.87
    assert selection_score(METRICS, "macro_f1") == 0.80


def test_final_kg_mean_penalises_an_untrained_kg_branch():
    early = selection_score(METRICS, "final_kg_mean_macro_f1")
    later = selection_score({**METRICS, "final_macro_f1": 0.79, "kg_macro_f1": 0.77}, "final_kg_mean_macro_f1")
    assert abs(early - 0.60) < 1e-9
    assert later > early
