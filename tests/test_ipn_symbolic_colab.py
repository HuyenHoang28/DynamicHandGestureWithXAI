"""Static safety contract for the focused symbolic V2 Colab notebook."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "IPN_Symbolic_V2_Colab.ipynb"


def _load_notebook_sources() -> tuple[dict, list[str], str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    return notebook, code, "\n".join(code) + "\n" + markdown


def test_ipn_symbolic_v2_colab_contract() -> None:
    notebook, code, text = _load_notebook_sources()
    assert notebook["nbformat"] == 4
    assert isinstance(notebook.get("cells"), list)

    for source in code:
        ast.parse(source, filename=str(NOTEBOOK))

    for reference in (
        "build_template_graph.py",
        "precompute_instance_graphs_phase_v2.py",
        "eval_ipn_template_matching.py",
        "validate_symbolic_inputs.py",
    ):
        assert reference in text

    assert "ACTIVE_SPLITS" in text
    assert re.search(r"\bRUN_TEST\s*=\s*False\b", text)
    assert "set(BUILD_SPLITS) == {'train', 'val'}" in text
    assert "GRAPH_BATCH_SIZE = 250" in text
    for split, count in (("train", 2443), ("val", 589), ("test", 1101)):
        assert re.search(rf"['\"]{split}['\"]\s*:\s*{count}\b", text)

    # Split integrity and provenance are hard gates, not just documentation.
    assert "subject_from_source" in text
    assert "Subject leakage" in text
    assert "1CM42_3" in text
    assert "manifest_sha256" in text
    assert "template_sha256" in text
    assert "run_config.json" in text
    assert "original_manifests" in text

    # Resume checkpoints are durable, and train/val artifacts cannot be confused
    # with a later full run that includes test.
    assert "RUN_MODE = 'full_with_test' if RUN_TEST else 'train_val'" in text
    assert "ARTIFACTS_BASE = DRIVE_DATA_ROOT / 'runs' / RUN_NAME / RUN_MODE" in text
    assert "ARTIFACTS = ARTIFACTS_BASE / attempt_id" in text
    assert "LOCAL_ROOT = Path('/content') / f'{RUN_NAME}_{RUN_MODE}_{attempt_id}'" in text
    assert text.index("attempt_id =") < text.index("LOCAL_GRAPH_ROOT = LOCAL_DATA / 'instance_graphs'", text.index("attempt_id ="))
    assert "uuid.uuid4()" in text
    assert "candidate_complete" in text
    assert "compatibility_keys" in text
    assert "graph_progress.json" in text
    assert "Checkpointed" in text
    assert "completed_run.json" in text
    assert "--trend-consistency', '0.6'" in text
    assert "--min-best-score', '0.15'" in text
    assert "--min-top2-margin', '0.02'" in text
    assert "val_diagnostics.jsonl" in text
    assert "selective_accuracy" in text

    # Final artifact sync must explicitly exclude copied raw inputs and temp batches.
    assert re.search(r"['\"]--exclude['\"]\s*,\s*['\"]data/keypoints/['\"]", text)
    assert re.search(r"['\"]--exclude['\"]\s*,\s*['\"]data/batches/['\"]", text)
    assert "--no-default-counts" not in text

    lowered = text.lower()
    assert "best.pt" not in lowered
    assert not re.search(r"\bbuild_events\s*\(", text)
    for forbidden in ("torch.optim", "model.fit(", "yolo train", "train.py"):
        assert forbidden.lower() not in lowered


def test_completion_marker_is_published_atomically_after_payload_sync() -> None:
    _, code, _ = _load_notebook_sources()
    source = next(cell for cell in code if "completion_tmp.replace" in cell)
    tree = ast.parse(source, filename=str(NOTEBOOK))

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    rsync = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "run"
        and "'rsync'" in ast.get_source_segment(source, node)
    )
    marker_copy = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy2"
        and "completed_run.json" in ast.get_source_segment(source, node)
    )
    marker_replace = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "replace"
    )
    rsync_source = ast.get_source_segment(source, rsync)
    assert "--exclude', 'completed_run.json" in rsync_source
    assert rsync.lineno < marker_copy.lineno < marker_replace.lineno
    assert "validation.json" in source
    assert "payload_sha256" in source
