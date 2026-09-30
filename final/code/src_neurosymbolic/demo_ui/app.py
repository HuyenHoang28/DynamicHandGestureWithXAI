import argparse
import html
import json
import math
import sys
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src_neurosymbolic.demo_ui.skeleton_renderer import (
    render_keypoint_video,
    transcode_browser_video,
)
from src_neurosymbolic.demo_video_v4 import (
    initialize_rtmw,
    load_input,
    render_markdown,
    requirement_evidence,
    select_device,
    write_json,
)
from src_neurosymbolic.events.build_events import build_events
from src_neurosymbolic.kg.build_instance_graph import build_instance_graph
from src_neurosymbolic.kg.graph_matching import (
    load_template_graphs,
    match_instance_to_templates,
)
from src_neurosymbolic.models.cross_attention_v2 import (
    NeuralBaselineAdapter,
    load_checkpoint_compat,
)
from src_neurosymbolic.models.cross_attention_fusion import NeuralKeypointTemplateCrossAttention
from src_neurosymbolic.models.cross_attention_v4 import DualCrossAttentionV4NeuroSymbolicModel
from src_neurosymbolic.models.keypoint_dataset import keypoints_to_tensor
from src_neurosymbolic.predicates.extract_predicates import extract_predicates


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = FINAL_ROOT.parent
DEFAULT_CHECKPOINT = (
    FINAL_ROOT
    / "checkpoints"
    / "final_structured_req_epoch19"
    / "last.pt"
)
PROJECT_CHECKPOINTS = {
    "Kich ban 1 - Neural-only": FINAL_ROOT
    / "checkpoints"
    / "neural_only_transformer_30epoch"
    / "last.pt",
    "Kich ban 2 - V4 KG-only": FINAL_ROOT
    / "checkpoints"
    / "scenario2_v4_kg_only"
    / "last.pt",
    "Kich ban 3 - Neuro-symbolic fusion": DEFAULT_CHECKPOINT,
    "Kich ban 3 - Template170 best-test": FINAL_ROOT
    / "outputs"
    / "cross_attention_v4_template170_retrain_next"
    / "best_test.pt",
}
DEFAULT_TEMPLATE_DIR = WORKSPACE_ROOT / "data" / "template_graphs"
TEMPLATE170_DIR = FINAL_ROOT / "data" / "template_graphs"
DEFAULT_POSE_WEIGHTS = (
    WORKSPACE_ROOT
    / "checkpoints"
    / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
)


def display_path(path: Path) -> str:
    for root in (PACKAGE_ROOT, FINAL_ROOT, WORKSPACE_ROOT):
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def checkpoint_choices() -> list[tuple[str, str]]:
    return [(label, display_path(path)) for label, path in PROJECT_CHECKPOINTS.items()]


def template_dir_for_checkpoint(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    template170_checkpoint = PROJECT_CHECKPOINTS["Kich ban 3 - Template170 best-test"].resolve()
    if checkpoint == template170_checkpoint:
        return TEMPLATE170_DIR
    return DEFAULT_TEMPLATE_DIR


def resolve_user_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()

    for root in (WORKSPACE_ROOT, FINAL_ROOT, PACKAGE_ROOT):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (WORKSPACE_ROOT / path).resolve()


def pipeline_status_html(state: str, title: str, detail: str) -> str:
    icons = {
        "waiting": "UPLOAD",
        "extracting": "RTMW",
        "ready": "READY",
        "predicting": "V4",
        "complete": "DONE",
        "error": "ERROR",
    }
    return f"""
    <div class="pipeline-status state-{state}">
      <div class="status-icon">{icons[state]}</div>
      <div class="status-copy">
        <div class="status-title">{title}</div>
        <div class="status-detail">{detail}</div>
        <div class="status-progress"><span></span></div>
      </div>
    </div>
    """


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predicate_rows(data: dict) -> list[list]:
    return [
        [
            row.get("subject"),
            row.get("name"),
            row.get("start_frame"),
            row.get("end_frame"),
            round(float(row.get("confidence") or 0.0), 4),
        ]
        for row in data.get("predicates", [])
    ]


def event_rows(data: dict) -> list[list]:
    rows = []
    for event in data.get("events", []):
        descriptions = []
        for field in (
            "motions",
            "positions",
            "hand_shapes",
            "finger_motions",
            "coordinations",
            "sweeps",
            "joint_angles",
        ):
            descriptions.extend(event.get(field, []))
        rows.append(
            [
                event.get("event_id"),
                event.get("subject"),
                event.get("start_frame"),
                event.get("end_frame"),
                event.get("duration"),
                ", ".join(descriptions),
                round(float(event.get("confidence") or 0.0), 4),
            ]
        )
    return rows


def requirement_rows(data: dict) -> list[list]:
    return [
        [
            "PASS" if row.get("satisfied") else "MISS",
            row.get("level"),
            " / ".join(row.get("fact") or [row.get("concept", "")]),
            round(float(row.get("weight") or 0.0), 3),
            round(float(row.get("model_presence") or 0.0), 4),
            bool(row.get("observed_in_instance_graph")),
        ]
        for row in data.get("requirements", [])
    ]


def prediction_rows(data: dict) -> list[list]:
    return [
        [
            branch.upper(),
            data[branch]["prediction_label"],
            round(float(data[branch]["confidence"]), 4),
        ]
        for branch in ("final", "neural", "kg")
    ]


def final_top_k_rows(data: dict) -> list[list]:
    return [
        [
            index,
            row["label_name"],
            round(float(row["probability"]), 4),
        ]
        for index, row in enumerate(data["final"]["top_k"], start=1)
    ]


def attention_rows(data: dict) -> list[list]:
    rows = []
    for row in data.get("attention", {}).get("important_frames", []):
        rows.append(["neural_frame", row["frame_index"], round(float(row["attention"]), 6)])
    for row in data.get("attention", {}).get("important_graph_tokens", []):
        rows.append(["graph_token", row["token_index"], round(float(row["attention"]), 6)])
    return rows


def graph_rows(data: dict) -> list[list]:
    nodes = {node["id"]: node for node in data.get("nodes", [])}

    def name(node_id: str) -> str:
        node = nodes.get(node_id, {})
        props = node.get("properties", {})
        return str(
            props.get("name")
            or props.get("event_id")
            or props.get("video_id")
            or node_id
        )

    return [
        [
            name(edge.get("source", "")),
            edge.get("relation"),
            name(edge.get("target", "")),
        ]
        for edge in data.get("edges", [])
    ]


EVENT_FIELDS = (
    ("motions", "Motion", "timeline-motion"),
    ("positions", "Position", "timeline-position"),
    ("hand_shapes", "Hand shape", "timeline-shape"),
    ("finger_motions", "Finger motion", "timeline-finger"),
    ("coordinations", "Coordination", "timeline-coordination"),
    ("sweeps", "Sweep", "timeline-sweep"),
    ("joint_angles", "Joint angle", "timeline-angle"),
)

RELATION_FIELDS = {
    "HAS_MOTION": "motions",
    "HAS_POSITION": "positions",
    "HAS_HAND_SHAPE": "hand_shapes",
    "HAS_FINGER_MOTION": "finger_motions",
    "HAS_COORDINATION": "coordinations",
    "HAS_SWEEP": "sweeps",
    "HAS_JOINT_ANGLE": "joint_angles",
}


def timeline_items(events: dict) -> tuple[list[dict], int]:
    items = []
    max_frame = 0
    for event in events.get("events", []):
        start = int(event.get("start_frame") or 0)
        end = int(event.get("end_frame") or start)
        max_frame = max(max_frame, end)
        subject = str(event.get("subject") or "unknown")
        confidence = float(event.get("confidence") or 0.0)
        for field, type_name, css_class in EVENT_FIELDS:
            for concept in event.get(field, []):
                items.append(
                    {
                        "subject": subject,
                        "concept": str(concept),
                        "type": type_name,
                        "css_class": css_class,
                        "start": start,
                        "end": end,
                        "duration": end - start + 1,
                        "confidence": confidence,
                    }
                )
    items.sort(key=lambda item: (item["start"], item["end"], item["subject"], item["concept"]))
    return items, max_frame + 1


def timeline_html(events: dict) -> str:
    items, total_frames = timeline_items(events)
    if not items:
        return "<div class='timeline-empty'>No temporal events were extracted.</div>"

    rows = []
    for item in items[:80]:
        left = 100.0 * item["start"] / max(1, total_frames)
        width = 100.0 * item["duration"] / max(1, total_frames)
        rows.append(
            f"""
            <div class="timeline-row">
              <div class="timeline-label">
                <strong>{html.escape(item['subject'])}</strong>
                <span>{html.escape(item['type'])}: {html.escape(item['concept'])}</span>
              </div>
              <div class="timeline-track">
                <div class="timeline-bar {item['css_class']}"
                     style="left:{left:.3f}%;width:{max(width, 1.2):.3f}%"
                     title="frames {item['start']}-{item['end']}"></div>
              </div>
              <div class="timeline-frames">{item['start']}-{item['end']}</div>
            </div>
            """
        )
    return f"""
    <div class="timeline-card">
      <div class="timeline-heading">
        <div><strong>Temporal evidence</strong><span>{total_frames} frames</span></div>
        <div class="timeline-legend">
          <span class="legend-motion">Motion</span>
          <span class="legend-position">Position</span>
          <span class="legend-shape">Shape</span>
          <span class="legend-coordination">Coordination</span>
        </div>
      </div>
      <div class="timeline-axis"><span>0</span><span>{total_frames // 2}</span><span>{total_frames - 1}</span></div>
      {''.join(rows)}
    </div>
    """


def requirement_intervals(requirement: dict, events: dict) -> list[tuple[int, int]]:
    fact = requirement.get("fact") or []
    if len(fact) != 3:
        return []
    subject, relation, concept = fact
    field = RELATION_FIELDS.get(relation)
    if field is None:
        return []
    intervals = []
    for event in events.get("events", []):
        event_subject = event.get("subject")
        if subject != "any" and event_subject != subject:
            continue
        if concept in event.get(field, []):
            intervals.append(
                (
                    int(event.get("start_frame") or 0),
                    int(event.get("end_frame") or 0),
                )
            )
    return intervals


def decision_explanation_html(
    predictions: dict,
    evidence: dict,
    symbolic: dict,
    events: dict,
) -> str:
    final = predictions["final"]
    neural = predictions["neural"]
    kg = predictions["kg"]
    gate = float(evidence["final_class_neural_gate"])
    symbolic_label = symbolic.get("prediction") or "none"
    symbolic_score = 0.0
    if symbolic.get("top_k"):
        symbolic_score = float(symbolic["top_k"][0].get("score") or 0.0)

    requirement_lines = []
    for row in evidence.get("requirements", []):
        if not row.get("satisfied") or row.get("level") == "negative":
            continue
        intervals = requirement_intervals(row, events)
        frame_text = ", ".join(f"{start}-{end}" for start, end in intervals[:3])
        timing_text = f"frames {frame_text}" if intervals else "cross-attended learned evidence"
        fact = row.get("fact") or [row.get("concept", "")]
        requirement_lines.append(
            "<li><strong>{}</strong><span>{}</span></li>".format(
                html.escape(" / ".join(str(value) for value in fact)),
                html.escape(timing_text),
            )
        )
        if len(requirement_lines) >= 8:
            break

    agreement = neural["prediction_label"] == kg["prediction_label"]
    agreement_text = (
        f"Both learned branches agree on {final['prediction_label']}."
        if agreement
        else "The learned branches disagree; the final fusion gate resolves the decision."
    )
    symbolic_note = (
        "The rule-only matcher agrees with the learned model."
        if symbolic_label == final["prediction_label"]
        else (
            f"The rule-only matcher prefers {symbolic_label}; it is shown as an audit signal "
            "and does not override the learned final logits."
        )
    )
    return f"""
    <div class="decision-flow">
      <div class="decision-step neural-step">
        <div class="step-number">1</div>
        <div><strong>Neural sequence branch</strong>
          <p>Reads the complete keypoint sequence in temporal order.</p>
          <b>{html.escape(neural['prediction_label'])} &middot; {neural['confidence']:.2%}</b>
        </div>
      </div>
      <div class="decision-arrow">&#8595;</div>
      <div class="decision-step kg-step">
        <div class="step-number">2</div>
        <div><strong>Learned KG cross-attention branch</strong>
          <p>Template queries attend to neural frames and instance-graph tokens.</p>
          <b>{html.escape(kg['prediction_label'])} &middot; {kg['confidence']:.2%}</b>
          <ul class="requirement-timeline">{''.join(requirement_lines)}</ul>
        </div>
      </div>
      <div class="decision-arrow">&#8595;</div>
      <div class="decision-step symbolic-step">
        <div class="step-number">3</div>
        <div><strong>Rule-only symbolic audit</strong>
          <p>{html.escape(symbolic_note)}</p>
          <b>{html.escape(symbolic_label)} &middot; {symbolic_score:.2%}</b>
        </div>
      </div>
      <div class="decision-arrow">&#8595;</div>
      <div class="decision-step final-step">
        <div class="step-number">4</div>
        <div><strong>Final fusion</strong>
          <p>{html.escape(agreement_text)} Gate for this class: {gate:.1%} neural / {1.0 - gate:.1%} KG.</p>
          <b>{html.escape(final['prediction_label'])} &middot; {final['confidence']:.2%}</b>
        </div>
      </div>
    </div>
    """


def build_pipeline_html(active_stage: str = "video") -> str:
    stages = [
        ("video", "01", "Video"),
        ("pose", "02", "RTMW Pose"),
        ("predicates", "03", "Predicates"),
        ("graph", "04", "Instance Graph"),
        ("transformer", "05", "Graph Transformer"),
        ("attention", "06", "Dual Cross-Attention"),
        ("fusion", "07", "Fusion Gate"),
        ("prediction", "08", "Prediction"),
    ]
    keys = [item[0] for item in stages]
    active_index = keys.index(active_stage) if active_stage in keys else 0
    cards = []
    for index, (key, number, label) in enumerate(stages):
        state = "complete" if index < active_index else "current" if index == active_index else "pending"
        cards.append(
            f"""
            <div class="pipeline-node pipeline-{state}" data-stage="{key}">
              <span>{number}</span><strong>{html.escape(label)}</strong>
            </div>
            """
        )
        if index < len(stages) - 1:
            cards.append('<div class="pipeline-arrow">&#8594;</div>')
    return f'<div class="pipeline-overview">{"".join(cards)}</div>'


def build_prediction_summary_html(
    predictions: dict,
    evidence: dict,
    report: dict,
) -> str:
    final = predictions["final"]
    neural = predictions["neural"]
    kg = predictions["kg"]
    gate = float(evidence["final_class_neural_gate"])
    validation = evidence.get("checkpoint_validation") or {}
    val_accuracy = float(validation.get("accuracy") or 0.0)
    labels = evidence.get("branch_labels") or {
        "neural": "Neural branch",
        "kg": "Learned KG branch",
        "fusion": "Fusion ratio",
        "fusion_value": None,
    }

    def metric(label: str, value: str, css_class: str) -> str:
        return (
            f'<div class="metric-card {css_class}"><span>{html.escape(label)}</span>'
            f"<strong>{html.escape(value)}</strong></div>"
        )

    return f"""
    <div class="prediction-dashboard">
      <div class="final-prediction-card">
        <span class="eyebrow">Final Gesture Prediction</span>
        <h2>{html.escape(final['prediction_label'])}</h2>
        <div class="confidence-line">
          <strong>{final['confidence']:.2%}</strong><span>confidence</span>
        </div>
        <div class="confidence-track">
          <div style="width:{100.0 * final['confidence']:.2f}%"></div>
        </div>
      </div>
      <div class="metric-grid">
        {metric(labels["neural"], labels.get("neural_value") or f"{neural['confidence']:.2%}", "metric-neural")}
        {metric(labels["kg"], f"{kg['confidence']:.2%}" if labels["kg"] != "N/A" else "Not used", "metric-kg")}
        {metric(labels["fusion"], labels.get("fusion_value") or f"{gate:.1%} / {1.0 - gate:.1%}", "metric-fusion")}
      </div>
    </div>
    """


def build_predicate_chips_html(predicates: dict, limit: int = 32) -> str:
    rows = sorted(
        predicates.get("predicates", []),
        key=lambda row: (
            int(row.get("start_frame") or 0),
            str(row.get("subject") or ""),
            str(row.get("name") or ""),
        ),
    )
    chips = []
    for row in rows[:limit]:
        subject = str(row.get("subject") or "unknown")
        name = str(row.get("name") or "predicate")
        start = int(row.get("start_frame") or 0)
        end = int(row.get("end_frame") or start)
        confidence = float(row.get("confidence") or 0.0)
        chips.append(
            f"""
            <div class="predicate-chip">
              <span class="chip-check">&#10003;</span>
              <div><strong>{html.escape(subject.replace('_', ' '))}</strong>
              <span>{html.escape(name.replace('_', ' '))}</span></div>
              <small>f{start}-{end} · {confidence:.0%}</small>
            </div>
            """
        )
    return (
        f'<div class="chip-grid">{"".join(chips)}</div>'
        if chips
        else '<div class="empty-card">No predicates extracted.</div>'
    )


def build_requirements_cards_html(evidence: dict, events: dict) -> str:
    cards = []
    for row in evidence.get("requirements", []):
        satisfied = bool(row.get("satisfied"))
        level = str(row.get("level") or "optional")
        fact = row.get("fact") or [row.get("concept", "")]
        intervals = requirement_intervals(row, events)
        frame_text = ", ".join(f"{start}-{end}" for start, end in intervals[:3]) or "not localized"
        state = "requirement-pass" if satisfied else "requirement-miss"
        cards.append(
            f"""
            <div class="requirement-card {state}">
              <div class="requirement-head">
                <span>{"PASS" if satisfied else "MISS"}</span>
                <small>{html.escape(level)}</small>
              </div>
              <strong>{html.escape(" / ".join(str(value) for value in fact))}</strong>
              <div class="requirement-meta">
                <span>Weight {float(row.get('weight') or 0.0):.2f}</span>
                <span>Presence {float(row.get('model_presence') or 0.0):.1%}</span>
                <span>Frames {html.escape(frame_text)}</span>
              </div>
            </div>
            """
        )
    return (
        f'<div class="requirements-grid">{"".join(cards)}</div>'
        if cards
        else '<div class="empty-card">No requirement evidence available.</div>'
    )


def _plot_layout(title: str, height: int = 360) -> dict:
    return {
        "title": {"text": title, "x": 0.02, "font": {"size": 16, "color": "#111827"}},
        "height": height,
        "margin": {"l": 35, "r": 25, "t": 55, "b": 35},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "Inter, Arial, sans-serif", "color": "#111827"},
        "hoverlabel": {"bgcolor": "#ffffff", "font_color": "#111827"},
    }


def build_instance_graph_plot(graph: dict) -> go.Figure:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        return go.Figure(layout=_plot_layout("Instance Graph"))

    node_by_id = {node["id"]: node for node in nodes}
    event_nodes = [
        node for node in nodes
        if str(node.get("type") or "") == "Event"
    ]
    event_nodes = sorted(
        event_nodes,
        key=lambda node: (
            -float(node.get("properties", {}).get("confidence") or 0.0),
            -float(node.get("properties", {}).get("duration") or 0.0),
            int(node.get("properties", {}).get("start_frame") or 0),
        ),
    )[:5]
    selected_event_ids = {node["id"] for node in event_nodes}
    selected_node_ids = {
        node["id"]
        for node in nodes
        if str(node.get("type") or "") == "Video"
    } | selected_event_ids
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        relation = str(edge.get("relation") or "")
        if source in selected_event_ids or target in selected_event_ids:
            selected_node_ids.add(source)
            selected_node_ids.add(target)
        if relation == "HAS_EVENT" and target in selected_event_ids:
            selected_node_ids.add(source)
            selected_node_ids.add(target)
    nodes = [node for node in nodes if node.get("id") in selected_node_ids]
    edges = [
        edge for edge in edges
        if edge.get("source") in selected_node_ids
        and edge.get("target") in selected_node_ids
        and (
            edge.get("source") in selected_event_ids
            or edge.get("target") in selected_event_ids
            or str(edge.get("relation") or "") in {"HAS_EVENT", "BEFORE", "OVERLAPS"}
        )
    ]

    relation_to_column = {
        "HAS_EVENT": "Event",
        "INVOLVES": "BodyPart",
        "HAS_MOTION": "Motion",
        "HAS_POSITION": "Position",
        "HAS_HAND_SHAPE": "HandShape",
        "HAS_FINGER_MOTION": "FingerMotion",
        "HAS_COORDINATION": "Coordination",
        "HAS_SWEEP": "Sweep",
        "HAS_JOINT_ANGLE": "JointAngle",
        "HAS_EVIDENCE": "Evidence",
    }
    type_order = [
        "Video",
        "Event",
        "BodyPart",
        "Motion",
        "Position",
        "HandShape",
        "FingerMotion",
        "Coordination",
        "Sweep",
        "JointAngle",
        "Evidence",
    ]
    x_by_type = {
        "Video": 0.0,
        "Event": 1.2,
        "BodyPart": 2.35,
        "Motion": 3.55,
        "Position": 3.55,
        "HandShape": 3.55,
        "FingerMotion": 3.55,
        "Coordination": 3.55,
        "Sweep": 3.55,
        "JointAngle": 3.55,
        "Evidence": 4.65,
    }

    grouped: dict[str, list[dict]] = {node_type: [] for node_type in type_order}
    for node in nodes:
        grouped.setdefault(str(node.get("type") or "Other"), []).append(node)

    positions = {}
    for node_type, group in grouped.items():
        if not group:
            continue
        if node_type == "Event":
            group = sorted(
                group,
                key=lambda node: (
                    node.get("properties", {}).get("start_frame", 0),
                    node.get("id", ""),
                ),
            )
        else:
            group = sorted(group, key=lambda node: node.get("id", ""))
        span = max(len(group) - 1, 1)
        for index, node in enumerate(group):
            y = 0.0 if len(group) == 1 else 1.0 - 2.0 * index / span
            if node_type in {"Motion", "Position", "HandShape", "FingerMotion", "Coordination", "Sweep", "JointAngle"}:
                y = y * 0.92
            positions[node["id"]] = (x_by_type.get(node_type, 4.9), y)

    edge_traces: dict[str, tuple[list, list]] = {}
    for key in ("event", "body", "concept", "temporal", "evidence", "other"):
        edge_traces[key] = ([], [])
    for edge in edges:
        source = positions.get(edge.get("source"))
        target = positions.get(edge.get("target"))
        if source is None or target is None:
            continue
        relation = str(edge.get("relation") or "")
        if relation == "HAS_EVENT":
            key = "event"
        elif relation == "INVOLVES":
            key = "body"
        elif relation in {"BEFORE", "OVERLAPS"}:
            key = "temporal"
        elif relation == "HAS_EVIDENCE":
            key = "evidence"
        elif relation in relation_to_column:
            key = "concept"
        else:
            key = "other"
        xs, ys = edge_traces[key]
        xs.extend([source[0], target[0], None])
        ys.extend([source[1], target[1], None])

    type_colors = {
        "Video": "#7c3aed",
        "Event": "#1d4ed8",
        "BodyPart": "#0891b2",
        "Motion": "#dc2626",
        "Position": "#059669",
        "HandShape": "#d97706",
        "FingerMotion": "#9333ea",
        "Coordination": "#0f766e",
        "Sweep": "#be123c",
        "JointAngle": "#4f46e5",
        "Evidence": "#64748b",
    }
    figure = go.Figure()
    edge_styles = {
        "event": ("Video to event", "#94a3b8", 1.8, None),
        "body": ("Event to body part", "#67e8f9", 1.4, None),
        "concept": ("Event to semantic concept", "#86efac", 1.4, None),
        "temporal": ("Temporal relation", "#f59e0b", 1.1, "dot"),
        "evidence": ("Event to evidence", "#cbd5e1", 0.9, "dot"),
        "other": ("Other relation", "#e2e8f0", 0.8, None),
    }
    for key, (name, color, width, dash) in edge_styles.items():
        xs, ys = edge_traces[key]
        if not xs:
            continue
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=name,
                line={"width": width, "color": color, **({"dash": dash} if dash else {})},
                hoverinfo="skip",
            )
        )

    for node_type in type_order:
        group = [node for node in grouped.get(node_type, []) if node.get("id") in positions]
        if not group:
            continue
        node_x, node_y, node_text, node_hover, node_size = [], [], [], [], []
        for node in group:
            x, y = positions[node["id"]]
            props = node.get("properties", {})
            label = props.get("name") or props.get("event_id") or props.get("video_id") or node["id"]
            display = str(label).replace("_", " ")
            node_x.append(x)
            node_y.append(y)
            node_text.append(display if node_type in {"Video", "Event", "BodyPart"} else "")
            node_hover.append(
                "<br>".join(
                    [
                        f"<b>{html.escape(str(node_type))}</b>",
                        html.escape(display),
                        *(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in props.items() if k not in {"name", "event_id", "video_id"}),
                    ]
                )
            )
            node_size.append(28 if node_type == "Video" else 18 if node_type == "Event" else 13)
        figure.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                name=node_type,
                text=node_text,
                textposition="middle right",
                hovertext=node_hover,
                hoverinfo="text",
                marker={
                    "size": node_size,
                    "color": type_colors.get(node_type, "#475569"),
                    "line": {"width": 1.5, "color": "#ffffff"},
                },
            )
        )

    column_labels = [
        ("Video", 0.0),
        ("Events", 1.2),
        ("Body parts", 2.35),
        ("Semantic concepts", 3.55),
        ("Evidence", 4.65),
    ]
    for text, x in column_labels:
        figure.add_annotation(
            x=x,
            y=1.16,
            text=f"<b>{text}</b>",
            showarrow=False,
            font={"size": 12, "color": "#334155"},
            align="center",
        )

    figure.update_layout(
        **_plot_layout(f"Instance Graph: Top {len(selected_event_ids)} Events", 560),
        xaxis={"visible": False, "range": [-0.35, 5.05]},
        yaxis={"visible": False, "range": [-1.18, 1.24]},
        legend={"orientation": "h", "y": -0.08, "x": 0, "font": {"size": 10}},
        dragmode="pan",
    )
    return figure


def build_fusion_gate_plot(evidence: dict) -> go.Figure:
    neural = 100.0 * float(evidence.get("final_class_neural_gate") or 0.0)
    kg = 100.0 - neural
    figure = go.Figure()
    figure.add_bar(
        x=[neural],
        y=["Final class"],
        orientation="h",
        name="Neural",
        marker_color="#7c3aed",
        text=[f"Neural {neural:.1f}%"],
        textposition="inside",
    )
    figure.add_bar(
        x=[kg],
        y=["Final class"],
        orientation="h",
        name="Learned KG",
        marker_color="#0369a1",
        text=[f"KG {kg:.1f}%"],
        textposition="inside",
    )
    figure.update_layout(
        **_plot_layout("Fusion Gate Contribution", 230),
        barmode="stack",
        xaxis={"range": [0, 100], "ticksuffix": "%", "title": None},
        yaxis={"title": None},
        legend={"orientation": "h", "y": 1.22, "x": 0},
    )
    return figure


def build_topk_probability_plot(predictions: dict) -> go.Figure:
    rows = list(reversed(predictions["final"]["top_k"]))
    labels = [str(row["label_name"]) for row in rows]
    values = [100.0 * float(row["probability"]) for row in rows]
    colors = ["#1d4ed8" if index == len(rows) - 1 else "#94a3b8" for index in range(len(rows))]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{value:.2f}%" for value in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.3f}%<extra></extra>",
        )
    )
    figure.update_layout(
        **_plot_layout("Top-K Predictions", 310),
        xaxis={"range": [0, max(100.0, max(values, default=0.0) * 1.15)], "ticksuffix": "%"},
        yaxis={"title": None},
        showlegend=False,
    )
    return figure


def build_timeline_plot(events: dict) -> go.Figure:
    items, total_frames = timeline_items(events)
    selected = items[:60]
    figure = go.Figure()
    colors = {
        "Motion": "#7c3aed",
        "Position": "#0369a1",
        "Hand shape": "#047857",
        "Finger motion": "#be185d",
        "Coordination": "#b45309",
        "Sweep": "#0f766e",
        "Joint angle": "#475569",
    }
    legend_types = set()
    for item in reversed(selected):
        label = f"{item['subject'].replace('_', ' ')} · {item['concept'].replace('_', ' ')}"
        show_legend = item["type"] not in legend_types
        legend_types.add(item["type"])
        figure.add_trace(
            go.Bar(
                x=[item["duration"]],
                y=[label],
                base=[item["start"]],
                orientation="h",
                marker_color=colors.get(item["type"], "#64748b"),
                name=item["type"],
                legendgroup=item["type"],
                showlegend=show_legend,
                hovertemplate=(
                    f"{html.escape(item['type'])}<br>frames {item['start']}-{item['end']}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **_plot_layout("Temporal Evidence Timeline", max(360, 24 * len(selected) + 100)),
        barmode="overlay",
        xaxis={"range": [0, max(1, total_frames - 1)], "title": "Frame"},
        yaxis={"title": None, "automargin": True},
        legend={"orientation": "h", "y": 1.04, "x": 0},
    )
    return figure


def build_attention_plot(evidence: dict) -> go.Figure:
    attention = evidence.get("attention", {})
    frames = attention.get("important_frames", [])
    graph_tokens = attention.get("important_graph_tokens", [])
    frame_labels = [f"F{row['frame_index']}" for row in frames]
    graph_labels = [f"G{row['token_index']}" for row in graph_tokens]
    labels = frame_labels + graph_labels
    neural_values = [float(row["attention"]) for row in frames] + [None] * len(graph_tokens)
    graph_values = [None] * len(frames) + [float(row["attention"]) for row in graph_tokens]
    figure = go.Figure(
        go.Heatmap(
            z=[neural_values, graph_values],
            x=labels,
            y=["Neural frames", "Graph tokens"],
            colorscale=[
                [0.0, "#eff6ff"],
                [0.5, "#60a5fa"],
                [1.0, "#1d4ed8"],
            ],
            colorbar={"title": "Attention"},
            hovertemplate="%{y} %{x}: %{z:.6f}<extra></extra>",
            xgap=3,
            ygap=3,
        )
    )
    figure.update_layout(
        **_plot_layout("Important Attention Targets", 320),
        xaxis={"title": "Top attended frame/token", "tickangle": -25},
        yaxis={"title": None},
    )
    return figure


def status_markdown(
    predictions: dict,
    evidence: dict,
    report: dict,
) -> str:
    final = predictions["final"]
    neural = predictions["neural"]
    kg = predictions["kg"]
    gate = float(evidence["final_class_neural_gate"])
    validation = evidence.get("checkpoint_validation") or {}
    return "\n".join(
        [
            f"## Final: `{final['prediction_label']}` ({final['confidence']:.2%})",
            "",
            f"- Neural: `{neural['prediction_label']}` ({neural['confidence']:.2%})",
            f"- KG: `{kg['prediction_label']}` ({kg['confidence']:.2%})",
            f"- Fusion gate: **{gate:.1%} neural / {1.0 - gate:.1%} KG**",
            f"- Checkpoint epoch: `{report.get('checkpoint_epoch')}`",
            f"- Checkpoint validation accuracy: `{float(validation.get('accuracy', 0.0)):.2%}`",
        ]
    )


def build_demo_args(
    video_path: str,
    checkpoint_path: str,
    device: str,
    top_k: int,
) -> argparse.Namespace:
    if not video_path:
        raise gr.Error("Please upload a video before running the demo.")

    video = Path(video_path).resolve()
    checkpoint = resolve_user_path(checkpoint_path)
    if not video.exists():
        raise gr.Error(f"Input video does not exist: {video}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = FINAL_ROOT / "outputs" / "demo_ui_v4" / f"{video.stem}_{timestamp}"
    return argparse.Namespace(
        input_video=video,
        input_keypoints=None,
        input_cache=None,
        checkpoint=checkpoint,
        template_dir=DEFAULT_TEMPLATE_DIR,
        output_dir=output_dir,
        device=device.strip() or "cuda:0",
        top_k=int(top_k),
        disable_symbolic_match=False,
        min_score=0.3,
        window_size=5,
        motion_threshold=0.08,
        max_gap=2,
        min_event_confidence=0.3,
        composite_gap=4,
        repo_root=WORKSPACE_ROOT,
        extract_script=WORKSPACE_ROOT / "run_rtmw_splits_dual.py",
        mmpose_root=WORKSPACE_ROOT / "mmpose",
        pose_weights=(
            DEFAULT_POSE_WEIGHTS
        ),
    )


def label_name(label_names: dict, label_id: int) -> str:
    return str(label_names.get(label_id, label_names.get(str(label_id), label_id)))


def branch_prediction(logits, label_names: dict, top_k: int) -> dict:
    probabilities = logits.softmax(dim=-1)[0].detach().cpu()
    values, indices = probabilities.topk(min(top_k, probabilities.numel()))
    rows = [
        {
            "label_id": int(index),
            "label_name": label_name(label_names, int(index)),
            "probability": float(value),
        }
        for value, index in zip(values, indices)
    ]
    return {
        "prediction_id": rows[0]["label_id"],
        "prediction_label": rows[0]["label_name"],
        "confidence": rows[0]["probability"],
        "top_k": rows,
    }


def load_ui_model(checkpoint_path: Path, template_dir: Path, device) -> tuple[torch.nn.Module, dict, dict, str]:
    checkpoint = load_checkpoint_compat(checkpoint_path)
    saved = checkpoint.get("args", {})
    version = checkpoint.get("model_version")
    state = checkpoint.get("model", checkpoint)
    num_classes = int(saved.get("num_classes", 27))
    num_frames = int(saved.get("num_frames", 64))
    d_model = int(saved.get("d_model", 128))
    layers = int(saved.get("layers", 3))
    heads = int(saved.get("heads", 4))
    dropout = float(saved.get("dropout", 0.1))

    if version == "v4_neural_logits_only":
        model = NeuralBaselineAdapter(
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
        model_kind = "scenario_1"
    elif version == "v4_structured_requirement_dual_cross_attention":
        model = DualCrossAttentionV4NeuroSymbolicModel(
            template_dir=template_dir,
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=d_model,
            layers=layers,
            heads=heads,
            graph_layers=int(saved.get("graph_layers", 2)),
            decoder_layers=int(saved.get("decoder_layers", 1)),
            instance_tokens=int(saved.get("instance_tokens", 16)),
            dropout=dropout,
        )
        output_name = str(saved.get("output_dir") or checkpoint_path.parent)
        model_kind = "scenario_2" if "kg_only" in output_name.lower() else "scenario_3"
    elif "backbone" in saved:
        model = NeuralBaselineAdapter(
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
        model_kind = "scenario_1"
    else:
        model = NeuralKeypointTemplateCrossAttention(
            num_classes=num_classes,
            d_model=d_model,
            num_layers=layers,
            num_heads=heads,
            template_tokens=int(saved.get("template_tokens", 8)),
            dropout=dropout,
            max_frames=num_frames,
        )
        model_kind = "scenario_2"

    current_state = model.state_dict()
    compatible_state = {}
    mismatched = []
    for key, value in state.items():
        if key not in current_state:
            continue
        if current_state[key].shape == value.shape:
            compatible_state[key] = value
            continue
        if key.endswith("concept_emb.weight") and value.ndim == 2:
            merged = current_state[key].clone()
            rows = min(merged.shape[0], value.shape[0])
            cols = min(merged.shape[1], value.shape[1])
            merged[:rows, :cols] = value[:rows, :cols]
            compatible_state[key] = merged
            continue
        mismatched.append((key, tuple(value.shape), tuple(current_state[key].shape)))
    if mismatched:
        details = ", ".join(f"{key}: {src}->{dst}" for key, src, dst in mismatched[:5])
        raise RuntimeError(f"Checkpoint has incompatible tensors: {details}")
    model.load_state_dict(compatible_state, strict=False)
    return model.to(device).eval(), saved, checkpoint, model_kind


def empty_evidence(model_kind: str) -> dict:
    branch_labels = {
        "scenario_1": {
            "neural": "Neural-only model",
            "kg": "N/A",
            "fusion": "Mode",
            "fusion_value": "Direct keypoint classification",
        },
        "scenario_2": {
            "neural": "Neural branch",
            "kg": "V4 KG branch",
            "fusion": "Mode",
            "fusion_value": "KG-only prediction",
            "neural_value": "N/A",
        },
        "scenario_3": {
            "neural": "Neural branch",
            "kg": "Learned KG branch",
            "fusion": "Fusion ratio",
            "fusion_value": None,
        },
    }[model_kind]
    return {
        "model_kind": model_kind,
        "branch_labels": branch_labels,
        "final_class_neural_gate": 1.0 if model_kind == "scenario_1" else 0.0,
        "mean_neural_gate": 1.0 if model_kind == "scenario_1" else 0.0,
        "neural_temperature": 1.0,
        "kg_temperature": 1.0,
        "requirements": [],
        "attention": {"important_frames": [], "important_graph_tokens": []},
    }


@torch.no_grad()
def infer_ui_model(
    model: torch.nn.Module,
    saved_args: dict,
    checkpoint: dict,
    model_kind: str,
    normalized: dict,
    instance_graph: dict,
    events: dict,
    device,
    top_k: int,
) -> tuple[dict, dict]:
    label_names = checkpoint.get("label_names", {})
    keypoints = keypoints_to_tensor(normalized, int(saved_args.get("num_frames", 64))).unsqueeze(0).to(device)
    instance_graph_json = [json.dumps(instance_graph, ensure_ascii=False)]

    if model_kind == "scenario_1":
        outputs = model(keypoints)
        logits = outputs["logits"]
        predictions = {
            "final": branch_prediction(logits, label_names, top_k),
            "neural": branch_prediction(logits, label_names, top_k),
            "kg": branch_prediction(logits, label_names, top_k),
        }
        return predictions, empty_evidence(model_kind)

    if model_kind == "scenario_2":
        outputs = model(keypoints, instance_graph_json, return_attention=True)
        if isinstance(outputs, dict):
            logits = outputs["kg_logits"]
            neural_logits = outputs.get("neural_logits", logits)
        elif isinstance(outputs, tuple):
            logits = outputs[0]
            neural_logits = logits
        else:
            logits = outputs
            neural_logits = logits
        predictions = {
            "final": branch_prediction(logits, label_names, top_k),
            "neural": {
                "prediction_id": -1,
                "prediction_label": "N/A",
                "confidence": 0.0,
                "top_k": [],
            },
            "kg": branch_prediction(logits, label_names, top_k),
        }
        return predictions, empty_evidence(model_kind)

    outputs = model(keypoints, instance_graph_json, return_attention=True)
    predictions = {
        "final": branch_prediction(outputs["logits"], label_names, top_k),
        "neural": branch_prediction(outputs["neural_logits"], label_names, top_k),
        "kg": branch_prediction(outputs["kg_logits"], label_names, top_k),
    }
    final_id = predictions["final"]["prediction_id"]
    evidence = empty_evidence(model_kind)
    evidence.update(
        {
            "final_class_neural_gate": float(outputs["gate"][0, final_id]),
            "mean_neural_gate": float(outputs["gate"][0].mean()),
            "neural_temperature": float(outputs["neural_temperature"].detach().cpu())
            if "neural_temperature" in outputs
            else 1.0,
            "kg_temperature": float(outputs["kg_temperature"].detach().cpu())
            if "kg_temperature" in outputs
            else 1.0,
            "requirements": requirement_evidence(model, outputs, final_id),
        }
    )
    return predictions, evidence


def prepare_ui_demo(
    video_path: str,
    checkpoint_path: str,
    device: str,
    top_k: int,
    progress=gr.Progress(),
):
    args = build_demo_args(video_path, checkpoint_path, device, top_k)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        selected_device = select_device(args.device)
        args.device = str(selected_device)

        progress(0.03, desc="Extracting RTMW keypoints")
        normalized, source_path = load_input(args)
        progress(0.72, desc="Building predicates and instance graph")
        predicates = extract_predicates(
            normalized,
            min_score=args.min_score,
            window_size=args.window_size,
            motion_threshold=args.motion_threshold,
        )
        events = build_events(
            predicates,
            max_gap=args.max_gap,
            min_confidence=args.min_event_confidence,
            composite_gap=args.composite_gap,
        )
        graph = build_instance_graph(events)

        write_json(args.output_dir / "normalized_keypoints.json", normalized)
        write_json(args.output_dir / "predicates.json", predicates)
        write_json(args.output_dir / "events.json", events)
        write_json(args.output_dir / "instance_graph.json", graph)
        write_json(
            args.output_dir / "preparation.json",
            {
                "input_video": str(args.input_video),
                "keypoint_source": str(source_path),
                "device": args.device,
            },
        )

        progress(0.82, desc="Rendering keypoint video")
        overlay_path = render_keypoint_video(
            args.input_video,
            normalized,
            args.output_dir / "keypoint_overlay.mp4",
            confidence_threshold=0.3,
        )
        progress(1.0, desc="Keypoint video ready; starting prediction")
    except Exception as error:
        raise gr.Error(f"Keypoint preparation failed: {error}") from error

    return (
        str(overlay_path),
        build_predicate_chips_html(predicates),
        build_instance_graph_plot(graph),
        build_timeline_plot(events),
        predicate_rows(predicates),
        event_rows(events),
        graph_rows(graph),
        str(args.output_dir),
        gr.update(interactive=True, value="Run Model Prediction"),
    )


def preview_original_video(video_path: str):
    if not video_path:
        raise gr.Error("Please upload a video before running the demo.")
    return (
        str(Path(video_path).resolve()),
        None,
        gr.update(interactive=False, value="Preparing keypoints..."),
    )


def run_ui_prediction(
    output_dir_path: str,
    checkpoint_path: str,
    device: str,
    top_k: int,
    progress=gr.Progress(),
):
    if not output_dir_path:
        raise gr.Error("Keypoint video has not been prepared.")

    output_dir = Path(output_dir_path).resolve()
    checkpoint = resolve_user_path(checkpoint_path)
    if not checkpoint.exists():
        raise gr.Error(f"Checkpoint does not exist: {checkpoint}")
    template_dir = template_dir_for_checkpoint(checkpoint)

    try:
        progress(0.05, desc="Loading checkpoint")
        selected_device = select_device(device.strip() or "cuda:0")
        normalized = read_json(output_dir / "normalized_keypoints.json")
        graph = read_json(output_dir / "instance_graph.json")
        events = read_json(output_dir / "events.json")
        preparation = read_json(output_dir / "preparation.json")
        model, saved_args, checkpoint_data, model_kind = load_ui_model(
            checkpoint,
            template_dir,
            selected_device,
        )

        progress(0.42, desc="Running model prediction")
        predictions, evidence = infer_ui_model(
            model,
            saved_args,
            checkpoint_data,
            model_kind,
            normalized,
            graph,
            events,
            selected_device,
            int(top_k),
        )
        progress(0.76, desc="Matching symbolic templates")
        symbolic = match_instance_to_templates(
            graph,
            load_template_graphs(template_dir),
            top_k=int(top_k),
        )

        write_json(output_dir / "prediction.json", predictions)
        write_json(output_dir / "model_evidence.json", evidence)
        write_json(output_dir / "symbolic_match.json", symbolic)
        report = {
            "input": preparation.get("keypoint_source"),
            "input_video": preparation.get("input_video"),
            "checkpoint": str(checkpoint),
            "template_dir": str(template_dir),
            "checkpoint_epoch": checkpoint_data.get("epoch"),
            "model_version": checkpoint_data.get("model_version") or model_kind,
            "device": str(selected_device),
            "predictions": predictions,
            "artifacts": {
                "normalized_keypoints": "normalized_keypoints.json",
                "predicates": "predicates.json",
                "events": "events.json",
                "instance_graph": "instance_graph.json",
                "prediction": "prediction.json",
                "model_evidence": "model_evidence.json",
                "symbolic_match": "symbolic_match.json",
                "summary": "summary.md",
            },
        }
        write_json(output_dir / "demo_report.json", report)
        (output_dir / "summary.md").write_text(
            render_markdown(predictions, evidence, symbolic),
            encoding="utf-8",
        )
        progress(1.0, desc="Prediction complete")
    except Exception as error:
        raise gr.Error(f"Prediction failed: {error}") from error

    return (
        build_prediction_summary_html(predictions, evidence, report),
        build_requirements_cards_html(evidence, events),
        build_fusion_gate_plot(evidence),
        build_topk_probability_plot(predictions),
        build_attention_plot(evidence),
        decision_explanation_html(predictions, evidence, symbolic, events),
        status_markdown(predictions, evidence, report),
        prediction_rows(predictions),
        final_top_k_rows(predictions),
        requirement_rows(evidence),
        attention_rows(evidence),
        symbolic,
        str(output_dir),
        gr.update(interactive=True, value="Run Model Prediction"),
    )


CSS = """
/* Light palette with explicit contrast for Gradio components. */
:root {
  --ui-page: #f3f6fb;
  --ui-surface: #ffffff;
  --ui-surface-soft: #f8fafc;
  --ui-surface-muted: #eef2f7;
  --ui-text: #111827;
  --ui-text-muted: #475569;
  --ui-border: #cbd5e1;
  --ui-border-soft: #e2e8f0;
  --ui-primary: #1d4ed8;
  --ui-primary-hover: #1e40af;
  --ui-accent: #0891b2;
  --ui-success: #047857;
  --ui-success-soft: #ecfdf5;
  --ui-warning: #b45309;
  --ui-warning-soft: #fffbeb;
  --ui-danger: #b91c1c;
  --ui-danger-soft: #fef2f2;
}
#root, .gradio-container {
  background: var(--ui-page) !important;
  color: var(--ui-text) !important;
}
.gradio-container {
  max-width: 1280px !important;
  width: min(1280px, calc(100% - 32px)) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-inline: 0 !important;
}
body > gradio-app,
gradio-app {
  display: block;
  width: 100%;
}
.gradio-container,
.gradio-container p,
.gradio-container span,
.gradio-container label {
  color: var(--ui-text);
}
.hero {text-align: center; margin-bottom: 12px; color: var(--ui-text);}
.hero h1 {font-size: 30px; margin-bottom: 4px;}
.hero p {color: var(--ui-text-muted) !important;}
.result-card {
  border: 1px solid var(--ui-border-soft);
  border-radius: 14px;
  padding: 14px;
  background: var(--ui-surface);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}
#pipeline-status-card {display: none; margin: 8px 0 12px;}
.pipeline-status {
  --status-color: var(--ui-text-muted);
  --status-soft: var(--ui-surface-muted);
  display: flex;
  align-items: center;
  gap: 14px;
  min-height: 54px;
  padding: 10px 14px;
  border: 1px solid color-mix(in srgb, var(--status-color) 36%, white);
  border-left: 5px solid var(--status-color);
  border-radius: 14px;
  background: linear-gradient(135deg, var(--ui-surface) 0%, var(--status-soft) 100%);
  box-shadow: 0 8px 26px rgba(15, 23, 42, 0.07);
  transition: border-color 240ms ease, background 240ms ease, transform 240ms ease;
}
.pipeline-status.state-ready {--status-color: var(--ui-success); --status-soft: var(--ui-success-soft);}
.pipeline-status.state-extracting {--status-color: var(--ui-accent); --status-soft: #ecfeff;}
.pipeline-status.state-complete {--status-color: var(--ui-primary); --status-soft: #eff6ff;}
.pipeline-status.state-error {--status-color: var(--ui-danger); --status-soft: var(--ui-danger-soft);}
.status-icon {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  flex: 0 0 38px;
  border-radius: 11px;
  color: #ffffff !important;
  background: var(--status-color);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.07em;
  box-shadow: 0 8px 18px color-mix(in srgb, var(--status-color) 28%, transparent);
}
.status-copy {min-width: 0; flex: 1;}
.status-title {color: var(--ui-text); font-size: 17px; font-weight: 750;}
.status-detail {margin-top: 3px; color: var(--ui-text-muted); font-size: 13px;}
.status-progress {display: none;}
#processing-overlay {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: none;
  align-items: center;
  justify-content: center;
  background: rgba(243, 246, 251, 0.78);
  backdrop-filter: blur(3px);
}
.processing-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 230px;
  padding: 24px 30px;
  border: 1px solid var(--ui-border-soft);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.98);
  box-shadow: 0 20px 50px rgba(15, 23, 42, 0.14);
}
.processing-spinner {
  width: 46px;
  height: 46px;
  border: 5px solid #dbeafe;
  border-top-color: var(--ui-primary);
  border-right-color: var(--ui-accent);
  border-radius: 50%;
  animation: processing-spin 0.85s linear infinite;
}
.processing-title {
  margin-top: 14px;
  color: var(--ui-text);
  font-size: 15px;
  font-weight: 750;
}
.processing-detail {
  margin-top: 3px;
  color: var(--ui-text-muted);
  font-size: 12px;
}
@keyframes processing-spin {to {transform: rotate(360deg);}}
#run-prediction-button button {
  min-height: 42px;
  border: 0;
  color: #ffffff !important;
  background: linear-gradient(135deg, var(--ui-primary), var(--ui-accent));
  box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
  transition: transform 180ms ease, box-shadow 180ms ease, opacity 180ms ease;
}
#run-prediction-button button span,
#run-prediction-button button p {
  color: #ffffff !important;
}
#run-prediction-button button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 11px 22px rgba(37, 99, 235, 0.28);
}
#run-prediction-button button:disabled {
  cursor: not-allowed;
  color: #475569 !important;
  background: #cbd5e1 !important;
  box-shadow: none;
  opacity: 1;
}
#run-prediction-button button:disabled span,
#run-prediction-button button:disabled p {
  color: #475569 !important;
}
#video-preview-row {visibility: hidden;}
#prediction-panel {display: none;}
#preparation-details {display: none;}
#prediction-details {display: none;}
#artifact-output {display: none;}
#advanced-settings {
  margin-top: 4px;
  border: 1px solid var(--ui-border-soft);
  border-radius: 12px;
  background: var(--ui-surface);
}
#advanced-settings > button,
#advanced-settings summary {
  color: var(--ui-text) !important;
  background: var(--ui-surface) !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: var(--ui-text) !important;
  background: var(--ui-surface) !important;
  border-color: var(--ui-border) !important;
  caret-color: var(--ui-primary);
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: #64748b !important;
  opacity: 1;
}
.gradio-container input[type="range"] {
  accent-color: var(--ui-primary);
}
.gradio-container [data-testid="file"],
.gradio-container .file-preview,
.gradio-container .upload-container {
  color: var(--ui-text) !important;
  background: var(--ui-surface) !important;
  border-color: var(--ui-border) !important;
}
.gradio-container [data-testid="file"] button,
.gradio-container .upload-container button {
  color: var(--ui-primary) !important;
  background: #eff6ff !important;
  border-color: #bfdbfe !important;
}
.gradio-container [data-testid="file"] button span,
.gradio-container .upload-container button span {
  color: var(--ui-primary) !important;
}
.gradio-container table {
  color: var(--ui-text) !important;
  background: var(--ui-surface) !important;
}
.gradio-container thead,
.gradio-container th {
  color: #1e293b !important;
  background: var(--ui-surface-muted) !important;
  border-color: var(--ui-border) !important;
}
.gradio-container td {
  color: var(--ui-text) !important;
  background: var(--ui-surface) !important;
  border-color: var(--ui-border-soft) !important;
}
.gradio-container [role="tablist"] {
  background: var(--ui-surface-muted) !important;
  border: 1px solid var(--ui-border-soft);
  border-radius: 10px;
  padding: 4px;
}
.gradio-container [role="tab"] {
  color: var(--ui-text-muted) !important;
  background: transparent !important;
}
.gradio-container [role="tab"][aria-selected="true"] {
  color: var(--ui-primary) !important;
  background: var(--ui-surface) !important;
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.10);
}
.gradio-container .block,
.gradio-container .form {
  color: var(--ui-text);
}
.gradio-container code {
  color: #1e3a8a !important;
  background: #eff6ff !important;
}
.timeline-card,
.decision-flow {
  color: var(--ui-text);
  background: var(--ui-surface);
  border: 1px solid var(--ui-border-soft);
  border-radius: 14px;
  padding: 16px;
}
.timeline-heading {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}
.timeline-heading > div:first-child {
  display: flex;
  flex-direction: column;
}
.timeline-heading span,
.timeline-frames {
  color: var(--ui-text-muted) !important;
  font-size: 12px;
}
.timeline-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.timeline-legend span {
  padding: 3px 8px;
  border-radius: 999px;
  color: #ffffff !important;
  font-size: 11px;
}
.legend-motion {background: #7c3aed;}
.legend-position {background: #0369a1;}
.legend-shape {background: #047857;}
.legend-coordination {background: #b45309;}
.timeline-axis {
  display: flex;
  justify-content: space-between;
  margin: 0 44px 5px 230px;
  color: var(--ui-text-muted);
  font-size: 11px;
}
.timeline-row {
  display: grid;
  grid-template-columns: 220px minmax(180px, 1fr) 42px;
  align-items: center;
  gap: 10px;
  min-height: 30px;
  border-top: 1px solid #f1f5f9;
}
.timeline-label {
  min-width: 0;
  display: flex;
  flex-direction: column;
}
.timeline-label strong {
  color: var(--ui-text);
  font-size: 12px;
}
.timeline-label span {
  overflow: hidden;
  color: var(--ui-text-muted) !important;
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.timeline-track {
  position: relative;
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--ui-surface-muted);
}
.timeline-bar {
  position: absolute;
  top: 0;
  bottom: 0;
  min-width: 3px;
  border-radius: 999px;
}
.timeline-motion {background: #7c3aed;}
.timeline-position {background: #0369a1;}
.timeline-shape {background: #047857;}
.timeline-finger {background: #be185d;}
.timeline-coordination {background: #b45309;}
.timeline-sweep {background: #0f766e;}
.timeline-angle {background: #475569;}
.timeline-empty {
  padding: 18px;
  color: var(--ui-text-muted);
  text-align: center;
}
.decision-flow {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.decision-step {
  display: flex;
  gap: 12px;
  padding: 14px;
  border: 1px solid var(--ui-border-soft);
  border-left: 5px solid;
  border-radius: 12px;
  background: var(--ui-surface-soft);
}
.decision-step p {
  margin: 3px 0 7px;
  color: var(--ui-text-muted) !important;
}
.decision-step b {color: var(--ui-text);}
.step-number {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  border-radius: 50%;
  color: #ffffff !important;
  font-weight: 800;
}
.neural-step {border-left-color: #7c3aed;}
.neural-step .step-number {background: #7c3aed;}
.kg-step {border-left-color: #0369a1;}
.kg-step .step-number {background: #0369a1;}
.symbolic-step {border-left-color: #b45309;}
.symbolic-step .step-number {background: #b45309;}
.final-step {border-left-color: var(--ui-success);}
.final-step .step-number {background: var(--ui-success);}
.decision-arrow {
  color: var(--ui-text-muted);
  text-align: center;
  font-size: 18px;
}
.requirement-timeline {
  margin: 8px 0 0;
  padding-left: 18px;
}
.requirement-timeline li {
  margin: 3px 0;
  color: var(--ui-text);
}
.requirement-timeline li span {
  margin-left: 6px;
  color: var(--ui-text-muted) !important;
  font-size: 12px;
}
@media (max-width: 780px) {
  .timeline-row {grid-template-columns: 130px minmax(100px, 1fr) 38px;}
  .timeline-axis {margin-left: 140px;}
  .timeline-heading {flex-direction: column;}
}
.dashboard-card,
.video-card,
#technical-details {
  border: 1px solid var(--ui-border-soft);
  border-radius: 16px;
  background: var(--ui-surface);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}
.upload-card,
.control-card {padding: 14px;}
.section-heading {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 28px 0 12px;
}
.section-heading > span {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  border-radius: 10px;
  color: #ffffff !important;
  background: var(--ui-primary);
  font-size: 12px;
  font-weight: 800;
}
.section-heading h2 {
  margin: 0;
  color: var(--ui-text);
  font-size: 20px;
}
.section-heading p {
  margin: 2px 0 0;
  color: var(--ui-text-muted) !important;
  font-size: 13px;
}
.pipeline-overview {
  display: flex;
  align-items: stretch;
  justify-content: center;
  gap: 6px;
  padding: 18px;
  overflow-x: auto;
}
.pipeline-node {
  min-width: 105px;
  padding: 12px 10px;
  border: 1px solid var(--ui-border-soft);
  border-radius: 12px;
  background: var(--ui-surface-soft);
  text-align: center;
}
.pipeline-node span {
  display: block;
  color: var(--ui-text-muted) !important;
  font-size: 10px;
  font-weight: 800;
}
.pipeline-node strong {
  display: block;
  margin-top: 4px;
  color: var(--ui-text);
  font-size: 12px;
}
.pipeline-complete {
  border-color: #86efac;
  background: #f0fdf4;
}
.pipeline-complete span,
.pipeline-complete strong {color: #166534 !important;}
.pipeline-current {
  border-color: #93c5fd;
  background: #eff6ff;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10);
}
.pipeline-current span,
.pipeline-current strong {color: #1d4ed8 !important;}
.pipeline-arrow {
  display: grid;
  place-items: center;
  color: #94a3b8 !important;
  font-size: 18px;
}
.video-card {position: relative; padding: 10px;}
.video-card video {
  width: 100%;
  aspect-ratio: 16 / 9;
  object-fit: contain;
  border-radius: 12px;
  background: #0f172a;
}
.keypoint-processing-overlay {
  position: absolute;
  inset: 10px;
  z-index: 5;
  display: none;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.82);
}
.keypoint-processing-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: #ffffff;
  text-align: center;
}
.keypoint-processing-box .processing-spinner {
  width: 38px;
  height: 38px;
  border-width: 4px;
}
.keypoint-processing-title {
  color: #ffffff !important;
  font-size: 14px;
  font-weight: 750;
}
.keypoint-processing-detail {
  max-width: 220px;
  color: #cbd5e1 !important;
  font-size: 12px;
}
.prediction-dashboard {
  display: grid;
  grid-template-columns: minmax(280px, 1.25fr) minmax(360px, 2fr);
  gap: 16px;
  padding: 16px;
}
.final-prediction-card {
  padding: 24px;
  border-radius: 16px;
  color: #ffffff;
  background: linear-gradient(135deg, #1d4ed8, #0891b2);
  box-shadow: 0 14px 32px rgba(29, 78, 216, 0.22);
}
.final-prediction-card * {color: #ffffff !important;}
.final-prediction-card .eyebrow {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.final-prediction-card h2 {
  margin: 12px 0 4px;
  font-size: clamp(32px, 5vw, 54px);
  line-height: 1;
  text-transform: uppercase;
}
.confidence-line {
  display: flex;
  align-items: baseline;
  gap: 8px;
}
.confidence-line strong {font-size: 25px;}
.confidence-line span {font-size: 12px;}
.confidence-track {
  height: 8px;
  margin-top: 16px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.25);
}
.confidence-track div {
  height: 100%;
  border-radius: inherit;
  background: #ffffff;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(150px, 1fr));
  gap: 10px;
}
.metric-card {
  padding: 14px;
  border: 1px solid var(--ui-border-soft);
  border-left: 4px solid #94a3b8;
  border-radius: 12px;
  background: var(--ui-surface-soft);
}
.metric-card span {
  display: block;
  color: var(--ui-text-muted) !important;
  font-size: 11px;
  font-weight: 700;
}
.metric-card strong {
  display: block;
  margin-top: 6px;
  color: var(--ui-text);
  font-size: 20px;
}
.metric-neural {border-left-color: #7c3aed;}
.metric-kg {border-left-color: #0369a1;}
.metric-fusion {border-left-color: #b45309;}
.metric-success {border-left-color: #047857;}
.chip-grid,
.requirements-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 10px;
  padding: 14px;
}
.predicate-chip {
  display: grid;
  grid-template-columns: 28px 1fr auto;
  align-items: center;
  gap: 9px;
  padding: 11px;
  border: 1px solid var(--ui-border-soft);
  border-radius: 11px;
  background: var(--ui-surface-soft);
}
.chip-check {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  border-radius: 8px;
  color: #ffffff !important;
  background: var(--ui-success);
  font-weight: 800;
}
.predicate-chip strong,
.predicate-chip span {display: block;}
.predicate-chip strong {font-size: 12px;}
.predicate-chip div > span {
  color: var(--ui-text-muted) !important;
  font-size: 11px;
}
.predicate-chip small {color: var(--ui-text-muted);}
.requirement-card {
  padding: 13px;
  border: 1px solid var(--ui-border-soft);
  border-left: 4px solid;
  border-radius: 12px;
  background: var(--ui-surface-soft);
}
.requirement-pass {border-left-color: var(--ui-success);}
.requirement-miss {border-left-color: var(--ui-danger);}
.requirement-head {
  display: flex;
  justify-content: space-between;
  margin-bottom: 7px;
}
.requirement-head span {
  color: var(--ui-success) !important;
  font-size: 11px;
  font-weight: 800;
}
.requirement-miss .requirement-head span {color: var(--ui-danger) !important;}
.requirement-head small {color: var(--ui-text-muted);}
.requirement-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 9px;
}
.requirement-meta span {
  padding: 3px 7px;
  border-radius: 999px;
  color: var(--ui-text-muted) !important;
  background: var(--ui-surface-muted);
  font-size: 10px;
}
.empty-card {
  padding: 24px;
  color: var(--ui-text-muted);
  text-align: center;
}
#technical-details {margin-top: 24px; overflow: hidden;}
@media (max-width: 900px) {
  .prediction-dashboard {grid-template-columns: 1fr;}
  .pipeline-overview {justify-content: flex-start;}
}
"""

PREPARE_DEMO_JS = """
(video) => {
  const previewRow = document.querySelector("#video-preview-row");
  const predictionPanel = document.querySelector("#prediction-panel");
  const preparationDetails = document.querySelector("#preparation-details");
  const predictionDetails = document.querySelector("#prediction-details");
  const artifactOutput = document.querySelector("#artifact-output");
  const original = document.querySelector("#original-preview video");
  const keypoint = document.querySelector("#keypoint-preview video");
  const runButton = document.querySelector("#run-prediction-button button");
  const overlay = document.querySelector("#processing-overlay");
  const keypointCard = document.querySelector("#keypoint-preview")?.closest(".video-card, .gradio-column, .gr-column");
  let keypointOverlay = document.querySelector("#keypoint-card-processing");
  if (!keypointOverlay && keypointCard) {
    keypointCard.style.position = "relative";
    keypointOverlay = document.createElement("div");
    keypointOverlay.id = "keypoint-card-processing";
    keypointOverlay.className = "keypoint-processing-overlay";
    keypointOverlay.innerHTML =
      '<div class="keypoint-processing-box">' +
      '<div class="processing-spinner"></div>' +
      '<div class="keypoint-processing-title">Extracting keypoints</div>' +
      '<div class="keypoint-processing-detail">RTMW is processing this video</div>' +
      '</div>';
    keypointCard.appendChild(keypointOverlay);
  }
  const pickVideoUrl = (value) => {
    if (!value) return "";
    if (typeof value === "string") return value;
    if (value.url) return value.url;
    if (value.path) return value.path;
    if (value.name) return value.name;
    if (Array.isArray(value) && value.length) return pickVideoUrl(value[0]);
    return "";
  };
  const uploadedUrl = pickVideoUrl(video);
  if (previewRow) previewRow.style.visibility = "visible";
  if (predictionPanel) predictionPanel.style.display = "none";
  if (preparationDetails) preparationDetails.style.display = "none";
  if (predictionDetails) predictionDetails.style.display = "none";
  if (artifactOutput) artifactOutput.style.display = "none";
  const pipelineNodes = [...document.querySelectorAll(".pipeline-node")];
  const poseIndex = pipelineNodes.findIndex((node) => node.dataset.stage === "pose");
  pipelineNodes.forEach((node, index) => {
    node.classList.remove("pipeline-complete", "pipeline-current", "pipeline-pending");
    node.classList.add(index < poseIndex ? "pipeline-complete" : index === poseIndex ? "pipeline-current" : "pipeline-pending");
  });
  if (overlay) overlay.style.display = "none";
  if (keypointOverlay) keypointOverlay.style.display = "flex";
  if (runButton) {
    runButton.disabled = true;
    runButton.textContent = "Preparing keypoints...";
  }
  if (original) {
    original.pause();
    original.muted = true;
    original.loop = true;
    if (uploadedUrl) {
      original.src = uploadedUrl;
      original.load();
    }
    original.currentTime = 0;
    original.play().catch(() => {});
  }
  if (keypoint) {
    keypoint.pause();
    keypoint.removeAttribute("src");
    keypoint.load();
  }
  return [video];
}
"""

START_PREDICTION_JS = """
(outputDir, checkpoint, device, topK) => {
  const predictionPanel = document.querySelector("#prediction-panel");
  const predictionDetails = document.querySelector("#prediction-details");
  const artifactOutput = document.querySelector("#artifact-output");
  const runButton = document.querySelector("#run-prediction-button button");
  const statusCard = document.querySelector("#pipeline-status-card");
  const overlay = document.querySelector("#processing-overlay");
  if (predictionPanel) predictionPanel.style.display = "none";
  if (predictionDetails) predictionDetails.style.display = "none";
  if (artifactOutput) artifactOutput.style.display = "none";
  if (statusCard) statusCard.style.display = "none";
  const pipelineNodes = [...document.querySelectorAll(".pipeline-node")];
  const attentionIndex = pipelineNodes.findIndex((node) => node.dataset.stage === "attention");
  pipelineNodes.forEach((node, index) => {
    node.classList.remove("pipeline-complete", "pipeline-current", "pipeline-pending");
    node.classList.add(index < attentionIndex ? "pipeline-complete" : index === attentionIndex ? "pipeline-current" : "pipeline-pending");
  });
  if (overlay) {
    overlay.style.display = "flex";
    overlay.querySelector(".processing-title").textContent = "Running model prediction";
    overlay.querySelector(".processing-detail").textContent =
      "Neural and knowledge branches are being fused";
  }
  if (runButton) {
    runButton.disabled = true;
    runButton.textContent = "Predicting...";
  }
  return [outputDir, checkpoint, device, topK];
}
"""

SYNC_VIDEO_JS = """
() => {
  const revealAndSync = (attempt = 0) => {
    const original = document.querySelector("#original-preview video");
    const keypoint = document.querySelector("#keypoint-preview video");
    const previewRow = document.querySelector("#video-preview-row");
    const preparationDetails = document.querySelector("#preparation-details");
    const statusCard = document.querySelector("#pipeline-status-card");
    const overlay = document.querySelector("#processing-overlay");
    const keypointOverlay = document.querySelector("#keypoint-card-processing");
    if (!original || !keypoint) {
      if (attempt < 40) {
        setTimeout(() => revealAndSync(attempt + 1), 150);
      }
      return;
    }

    original.loop = true;
    keypoint.loop = true;
    original.muted = true;
    keypoint.muted = true;
    original.currentTime = 0;
    keypoint.currentTime = 0;

    let syncing = false;
    const align = (source, target) => {
      if (syncing || !Number.isFinite(source.currentTime)) return;
      if (Math.abs(source.currentTime - target.currentTime) > 0.08) {
        syncing = true;
        target.currentTime = source.currentTime;
        syncing = false;
      }
    };

    original.ontimeupdate = () => align(original, keypoint);
    keypoint.ontimeupdate = () => align(keypoint, original);
    original.onplay = () => {
      if (keypoint.paused) keypoint.play().catch(() => {});
    };
    original.onpause = () => {
      if (!keypoint.paused) keypoint.pause();
    };
    keypoint.onplay = () => {
      if (original.paused) original.play().catch(() => {});
    };
    keypoint.onpause = () => {
      if (!original.paused) original.pause();
    };

    const startTogether = () => {
      if (original.readyState < 2 || keypoint.readyState < 2) return false;
      original.currentTime = 0;
      keypoint.currentTime = 0;
      original.pause();
      keypoint.pause();
      if (previewRow) previewRow.style.visibility = "visible";
      if (preparationDetails) preparationDetails.style.display = "block";
      if (statusCard) statusCard.style.display = "block";
      if (overlay) overlay.style.display = "none";
      if (keypointOverlay) keypointOverlay.style.display = "none";
      requestAnimationFrame(() => {
        original.currentTime = 0;
        keypoint.currentTime = 0;
        Promise.allSettled([original.play(), keypoint.play()]);
      });
      return true;
    };

    if (!startTogether()) {
      const retryStart = () => startTogether();
      original.addEventListener("canplay", retryStart, {once: true});
      keypoint.addEventListener("canplay", retryStart, {once: true});
    }
  };
  revealAndSync();
  return [];
}
"""

SHOW_PREDICTION_JS = """
() => {
  const predictionPanel = document.querySelector("#prediction-panel");
  const predictionDetails = document.querySelector("#prediction-details");
  const artifactOutput = document.querySelector("#artifact-output");
  const statusCard = document.querySelector("#pipeline-status-card");
  const overlay = document.querySelector("#processing-overlay");
  if (predictionPanel) predictionPanel.style.display = "flex";
  if (predictionDetails) predictionDetails.style.display = "block";
  if (artifactOutput) artifactOutput.style.display = "block";
  if (statusCard) statusCard.style.display = "block";
  if (overlay) overlay.style.display = "none";
  return [];
}
"""

SHOW_PREPARATION_ERROR_JS = """
() => {
  let status = document.querySelector("#pipeline-status-card .pipeline-status");
  const statusCard = document.querySelector("#pipeline-status-card");
  const runButton = document.querySelector("#run-prediction-button button");
  const overlay = document.querySelector("#processing-overlay");
  const keypointOverlay = document.querySelector("#keypoint-card-processing");
  if (overlay) overlay.style.display = "none";
  if (keypointOverlay) keypointOverlay.style.display = "none";
  if (statusCard) statusCard.style.display = "block";
  if (!status && statusCard) {
    statusCard.innerHTML =
      '<div class="pipeline-status state-error"><div class="status-icon">ERROR</div>' +
      '<div class="status-copy"><div class="status-title"></div>' +
      '<div class="status-detail"></div><div class="status-progress"><span></span></div></div></div>';
    status = statusCard.querySelector(".pipeline-status");
  }
  if (status) {
    status.className = "pipeline-status state-error";
    status.querySelector(".status-icon").textContent = "ERROR";
    status.querySelector(".status-title").textContent = "Keypoint extraction failed";
    status.querySelector(".status-detail").textContent =
      "Check the RTMW error notification and logs, then upload the video again.";
  }
  if (runButton) {
    runButton.disabled = true;
    runButton.textContent = "Keypoints not ready";
  }
  return [];
}
"""

SHOW_PREDICTION_ERROR_JS = """
() => {
  let status = document.querySelector("#pipeline-status-card .pipeline-status");
  const statusCard = document.querySelector("#pipeline-status-card");
  const runButton = document.querySelector("#run-prediction-button button");
  const overlay = document.querySelector("#processing-overlay");
  const keypointOverlay = document.querySelector("#keypoint-card-processing");
  if (overlay) overlay.style.display = "none";
  if (keypointOverlay) keypointOverlay.style.display = "none";
  if (statusCard) statusCard.style.display = "block";
  if (!status && statusCard) {
    statusCard.innerHTML =
      '<div class="pipeline-status state-error"><div class="status-icon">ERROR</div>' +
      '<div class="status-copy"><div class="status-title"></div>' +
      '<div class="status-detail"></div><div class="status-progress"><span></span></div></div></div>';
    status = statusCard.querySelector(".pipeline-status");
  }
  if (status) {
    status.className = "pipeline-status state-error";
    status.querySelector(".status-icon").textContent = "ERROR";
    status.querySelector(".status-title").textContent = "Model prediction failed";
    status.querySelector(".status-detail").textContent =
      "The keypoint preview is still available. Check the checkpoint error and retry prediction.";
  }
  if (runButton) {
    runButton.disabled = false;
    runButton.textContent = "Retry Prediction";
  }
  return [];
}
"""


def build_app_legacy() -> gr.Blocks:
    with gr.Blocks(title="Demo", css=CSS) as app:
        gr.HTML(
            """
            <div id="processing-overlay">
              <div class="processing-box">
                <div class="processing-spinner"></div>
                <div class="processing-title">Processing</div>
                <div class="processing-detail">Please wait</div>
              </div>
            </div>
            """
        )
        gr.Markdown(
            """
            <div class="hero">
              <h1>Demo Gesture Recognition</h1>
            </div>
            """
        )

        input_video = gr.File(
            label="Upload Video",
            file_types=["video"],
            type="filepath",
        )

        pipeline_status = gr.HTML("", elem_id="pipeline-status-card")

        with gr.Row(elem_id="video-preview-row"):
            original_video = gr.Video(
                label="Original Video",
                interactive=False,
                elem_id="original-preview",
                autoplay=False,
            )
            keypoint_video = gr.Video(
                label="Keypoint Video Preview",
                interactive=False,
                elem_id="keypoint-preview",
                autoplay=False,
            )

        with gr.Accordion(
            "Advanced settings",
            open=False,
            elem_id="advanced-settings",
        ):
            with gr.Row():
                checkpoint = gr.Dropdown(
                    choices=checkpoint_choices(),
                    label="Checkpoint",
                    value=display_path(DEFAULT_CHECKPOINT),
                    scale=3,
                )
                device = gr.Textbox(label="Device", value="cuda:0", scale=1)
                top_k = gr.Slider(
                    label="Top-K",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    scale=1,
                )

        with gr.Row():
            run_button = gr.Button(
                "Run Model Prediction",
                variant="primary",
                scale=1,
                interactive=False,
                elem_id="run-prediction-button",
            )

        with gr.Row(elem_id="prediction-panel"):
            with gr.Column(scale=2, elem_classes=["result-card"]):
                summary = gr.Markdown("Upload a video, then run prediction.")
            with gr.Column(scale=1):
                prediction_table = gr.Dataframe(
                    headers=["Branch", "Prediction", "Confidence"],
                    datatype=["str", "str", "number"],
                    label="Branch Predictions",
                    interactive=False,
                )
            with gr.Column(scale=1):
                top_k_table = gr.Dataframe(
                    headers=["Rank", "Class", "Probability"],
                    datatype=["number", "str", "number"],
                    label="Final Top-K",
                    interactive=False,
                )

        with gr.Tabs(elem_id="preparation-details"):
            with gr.Tab("Timeline"):
                timeline_view = gr.HTML()
            with gr.Tab("Predicates"):
                predicates_table = gr.Dataframe(
                    headers=["Subject", "Predicate", "Start", "End", "Confidence"],
                    datatype=["str", "str", "number", "number", "number"],
                    interactive=False,
                )
            with gr.Tab("Events"):
                events_table = gr.Dataframe(
                    headers=[
                        "Event",
                        "Subject",
                        "Start",
                        "End",
                        "Duration",
                        "Concepts",
                        "Confidence",
                    ],
                    datatype=["str", "str", "number", "number", "number", "str", "number"],
                    interactive=False,
                )
            with gr.Tab("Instance Graph"):
                graph_table = gr.Dataframe(
                    headers=["Source", "Relation", "Target"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                )

        with gr.Tabs(elem_id="prediction-details"):
            with gr.Tab("Decision Explanation"):
                decision_explanation = gr.HTML()
            with gr.Tab("Requirements"):
                requirements_table = gr.Dataframe(
                    headers=[
                        "Status",
                        "Level",
                        "Structured Fact",
                        "Weight",
                        "Model Presence",
                        "Observed",
                    ],
                    datatype=["str", "str", "str", "number", "number", "bool"],
                    interactive=False,
                )
            with gr.Tab("Attention"):
                attention_table = gr.Dataframe(
                    headers=["Type", "Index", "Attention"],
                    datatype=["str", "number", "number"],
                    interactive=False,
                )
            with gr.Tab("Symbolic Match"):
                symbolic_json = gr.JSON(label="Direct Template Matching")

        output_path = gr.Textbox(
            label="Saved Artifact Directory",
            interactive=False,
            elem_id="artifact-output",
        )
        prepared_output_dir = gr.State("")

        prepare_event = input_video.change(
            fn=prepare_ui_demo,
            inputs=[input_video, checkpoint, device, top_k],
            outputs=[
                original_video,
                keypoint_video,
                pipeline_status,
                timeline_view,
                predicates_table,
                events_table,
                graph_table,
                prepared_output_dir,
                run_button,
            ],
            js=PREPARE_DEMO_JS,
        )
        prepare_event.success(
            fn=None,
            inputs=None,
            outputs=None,
            js=SYNC_VIDEO_JS,
        )
        prepare_event.failure(
            fn=None,
            inputs=None,
            outputs=None,
            js=SHOW_PREPARATION_ERROR_JS,
        )
        prediction_event = run_button.click(
            fn=run_ui_prediction,
            inputs=[prepared_output_dir, checkpoint, device, top_k],
            outputs=[
                pipeline_status,
                decision_explanation,
                summary,
                prediction_table,
                top_k_table,
                requirements_table,
                attention_table,
                symbolic_json,
                output_path,
                run_button,
            ],
            js=START_PREDICTION_JS,
        )
        prediction_event.success(
            fn=None,
            inputs=None,
            outputs=None,
            js=SHOW_PREDICTION_JS,
        )
        prediction_event.failure(
            fn=None,
            inputs=None,
            outputs=None,
            js=SHOW_PREDICTION_ERROR_JS,
        )
    return app


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Demo", css=CSS) as app:
        gr.HTML(
            """
            <div id="processing-overlay">
              <div class="processing-box">
                <div class="processing-spinner"></div>
                <div class="processing-title">Processing</div>
                <div class="processing-detail">Please wait</div>
              </div>
            </div>
            <div class="hero">
              <h1>Gesture Recognition</h1>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, elem_classes=["dashboard-card", "upload-card"]):
                input_video = gr.File(
                    label="Upload Gesture Video",
                    file_types=["video"],
                    type="filepath",
                )
            with gr.Column(scale=2, elem_classes=["dashboard-card", "control-card"]):
                run_button = gr.Button(
                    "Run Model Prediction",
                    variant="primary",
                    interactive=False,
                    elem_id="run-prediction-button",
                )
                with gr.Accordion("Advanced settings", open=False, elem_id="advanced-settings"):
                    checkpoint = gr.Dropdown(
                        choices=checkpoint_choices(),
                        label="Checkpoint",
                        value=display_path(DEFAULT_CHECKPOINT),
                    )
                    with gr.Row():
                        device = gr.Textbox(label="Device", value="cuda:0")
                        top_k = gr.Slider(
                            label="Top-K",
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                        )

        gr.HTML('<div class="section-heading"><span>01</span><div><h2>Pose Estimation</h2><p>Original video preview and Skeleton visualization</p></div></div>')
        with gr.Row(elem_id="video-preview-row", equal_height=True):
            with gr.Column(elem_classes=["video-card"]):
                original_video = gr.Video(
                    label="Original Video",
                    interactive=False,
                    elem_id="original-preview",
                    autoplay=False,
                )
            with gr.Column(elem_classes=["video-card"]):
                keypoint_video = gr.Video(
                    label="Skeleton Visualization",
                    interactive=False,
                    elem_id="keypoint-preview",
                    autoplay=False,
                )

        with gr.Column(elem_id="prediction-panel"):
            gr.HTML('<div class="section-heading"><span>02</span><div><h2>Prediction</h2><p>Final classification and branch-level confidence</p></div></div>')
            prediction_summary = gr.HTML(elem_classes=["dashboard-card"])

        with gr.Column(visible=False):
            predicate_chips = gr.HTML()
            instance_graph_plot = gr.Plot(show_label=False)
            timeline_plot = gr.Plot(show_label=False)
            decision_explanation = gr.HTML()
            requirements_cards = gr.HTML()
            fusion_gate_plot = gr.Plot(show_label=False)
            top_k_plot = gr.Plot(show_label=False)
            attention_plot = gr.Plot(show_label=False)

        with gr.Accordion("Technical details", open=False, elem_id="technical-details"):
            with gr.Tabs():
                with gr.Tab("Branch Predictions"):
                    summary = gr.Markdown()
                    prediction_table = gr.Dataframe(
                        headers=["Branch", "Prediction", "Confidence"],
                        datatype=["str", "str", "number"],
                        interactive=False,
                    )
                    top_k_table = gr.Dataframe(
                        headers=["Rank", "Class", "Probability"],
                        datatype=["number", "str", "number"],
                        interactive=False,
                    )
                with gr.Tab("Raw Predicates"):
                    predicates_table = gr.Dataframe(
                        headers=["Subject", "Predicate", "Start", "End", "Confidence"],
                        datatype=["str", "str", "number", "number", "number"],
                        interactive=False,
                    )
                with gr.Tab("Raw Events"):
                    events_table = gr.Dataframe(
                        headers=["Event", "Subject", "Start", "End", "Duration", "Concepts", "Confidence"],
                        datatype=["str", "str", "number", "number", "number", "str", "number"],
                        interactive=False,
                    )
                with gr.Tab("Raw Graph"):
                    graph_table = gr.Dataframe(
                        headers=["Source", "Relation", "Target"],
                        datatype=["str", "str", "str"],
                        interactive=False,
                    )

            output_path = gr.Textbox(
                label="Saved Artifact Directory",
                interactive=False,
                elem_id="artifact-output",
            )
            symbolic_json = gr.JSON(visible=False)
            requirements_table = gr.Dataframe(visible=False)
            attention_table = gr.Dataframe(visible=False)

        prepared_output_dir = gr.State("")
        preview_event = input_video.change(
            fn=preview_original_video,
            inputs=[input_video],
            outputs=[
                original_video,
                keypoint_video,
                run_button,
            ],
            js=PREPARE_DEMO_JS,
        )
        prepare_event = preview_event.then(
            fn=prepare_ui_demo,
            inputs=[input_video, checkpoint, device, top_k],
            outputs=[
                keypoint_video,
                predicate_chips,
                instance_graph_plot,
                timeline_plot,
                predicates_table,
                events_table,
                graph_table,
                prepared_output_dir,
                run_button,
            ],
        )
        prepare_event.success(fn=None, inputs=None, outputs=None, js=SYNC_VIDEO_JS)
        prepare_event.failure(fn=None, inputs=None, outputs=None, js=SHOW_PREPARATION_ERROR_JS)

        prediction_event = run_button.click(
            fn=run_ui_prediction,
            inputs=[prepared_output_dir, checkpoint, device, top_k],
            outputs=[
                prediction_summary,
                requirements_cards,
                fusion_gate_plot,
                top_k_plot,
                attention_plot,
                decision_explanation,
                summary,
                prediction_table,
                top_k_table,
                requirements_table,
                attention_table,
                symbolic_json,
                output_path,
                run_button,
            ],
            js=START_PREDICTION_JS,
        )
        prediction_event.success(fn=None, inputs=None, outputs=None, js=SHOW_PREDICTION_JS)
        prediction_event.failure(fn=None, inputs=None, outputs=None, js=SHOW_PREDICTION_ERROR_JS)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the demo UI.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def start_rtmw_warmup(device: str = "cuda:0") -> threading.Thread:
    def warmup() -> None:
        try:
            selected_device = str(select_device(device))
            initialize_rtmw(
                WORKSPACE_ROOT / "run_rtmw_splits_dual.py",
                WORKSPACE_ROOT / "mmpose",
                DEFAULT_POSE_WEIGHTS,
                selected_device,
            )
        except Exception as error:
            # The UI remains usable and will retry initialization on first upload.
            print(f"RTMW background preload failed: {error}")

    thread = threading.Thread(
        target=warmup,
        name="rtmw-background-preload",
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    cli_args = parse_args()
    start_rtmw_warmup()
    build_app().launch(
        server_name=cli_args.server_name,
        server_port=cli_args.server_port,
        share=cli_args.share,
        allowed_paths=[
            str(FINAL_ROOT / "outputs"),
            str(WORKSPACE_ROOT),
        ],
    )
