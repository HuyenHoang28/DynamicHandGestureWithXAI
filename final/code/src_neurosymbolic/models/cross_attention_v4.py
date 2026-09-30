import json
from pathlib import Path

import torch

from src_neurosymbolic.kg.graph_matching import (
    CONCEPT_RELATIONS,
    instance_facts,
    node_map,
    node_name,
)
from .cross_attention_v2 import GraphEncoderV2
from .cross_attention_v3 import DualCrossAttentionNeuroSymbolicModel


class StructuredRequirementGraphEncoder(GraphEncoderV2):
    """V3 graph encoder with structured requirement targets.

    V3 supervised requirement presence with concept-name lookup only. V4 keeps
    the same token encoder but targets requirements with (subject, relation,
    concept) facts, so left/right hand requirements no longer collapse into one
    generic concept hit.
    """

    def __init__(
        self,
        template_dir: str | Path,
        d_model: int,
        heads: int,
        layers: int = 2,
        dropout: float = 0.1,
        directed_temporal_edges: bool = False,
    ) -> None:
        super().__init__(template_dir, d_model, heads, layers, dropout, directed_temporal_edges)
        self.template_requirement_facts = self._load_template_requirement_facts(
            Path(template_dir)
        )

    @staticmethod
    def _event_subjects(graph: dict) -> dict[str, str]:
        nodes = node_map(graph)
        subjects = {}
        for node in graph.get("nodes", []):
            if node.get("type") == "Event":
                props = node.get("properties", {})
                if props.get("subject"):
                    subjects[node["id"]] = props["subject"]
        for edge in graph.get("edges", []):
            if edge.get("relation") != "INVOLVES":
                continue
            source = edge.get("source")
            target = edge.get("target")
            if source in subjects:
                subjects[source] = node_name(nodes, target)
        return subjects

    def _load_template_requirement_facts(
        self,
        template_dir: Path,
    ) -> list[list[tuple[str, str, str] | None]]:
        graphs = []
        for path in sorted(template_dir.glob("*.json")):
            if path.name == "index.json":
                continue
            graph = json.loads(path.read_text(encoding="utf-8"))
            if graph.get("graph_type") == "template_graph":
                graphs.append(graph)
        graphs.sort(key=lambda item: int(item.get("label_id", 0)))
        max_nodes = int(self.template_padding_mask.shape[1])
        all_facts = []
        for graph in graphs:
            nodes = graph.get("nodes", [])
            node_index = {node["id"]: idx for idx, node in enumerate(nodes)}
            node_lookup = node_map(graph)
            subjects = self._event_subjects(graph)
            facts: list[tuple[str, str, str] | None] = [None] * len(nodes)
            for edge in graph.get("edges", []):
                relation = edge.get("relation")
                props = edge.get("properties", {})
                if relation not in CONCEPT_RELATIONS or not props.get("requirement_level"):
                    continue
                target = node_index.get(edge.get("target"))
                if target is None:
                    continue
                source = edge.get("source")
                subject = subjects.get(source, "any")
                concept = node_name(node_lookup, edge.get("target"))
                facts[target] = (subject, relation, concept)
            all_facts.append(facts + [None] * (max_nodes - len(facts)))
        return all_facts

    def encode_instances(self, json_strings: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        graphs = [json.loads(value) for value in json_strings]
        encoded = super().encode_instances(graphs, device)
        encoded["evidence_facts"] = [instance_facts(graph)[0] for graph in graphs]
        return encoded

    def requirement_targets(
        self,
        evidence_facts: list[set[tuple[str, str, str]]],
        device: torch.device,
    ) -> torch.Tensor:
        values = []
        for facts in evidence_facts:
            wildcard_facts = {
                (relation, concept)
                for _, relation, concept in facts
            }
            values.append(
                [
                    [
                        self._fact_target(requirement_fact, facts, wildcard_facts)
                        for requirement_fact in class_facts
                    ]
                    for class_facts in self.template_requirement_facts
                ]
            )
        return torch.tensor(values, dtype=torch.float32, device=device)

    @staticmethod
    def _fact_target(
        requirement_fact: tuple[str, str, str] | None,
        evidence_facts: set[tuple[str, str, str]],
        wildcard_facts: set[tuple[str, str]],
    ) -> float:
        if requirement_fact is None:
            return 0.0
        subject, relation, concept = requirement_fact
        if subject == "any":
            return float((relation, concept) in wildcard_facts)
        return float(requirement_fact in evidence_facts)


class DualCrossAttentionV4NeuroSymbolicModel(DualCrossAttentionNeuroSymbolicModel):
    def __init__(
        self,
        template_dir: str | Path = "data/template_graphs",
        num_classes: int = 27,
        num_frames: int = 64,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        graph_layers: int = 2,
        decoder_layers: int = 1,
        instance_tokens: int = 16,
        dropout: float = 0.15,
        directed_temporal_edges: bool = False,
    ) -> None:
        super().__init__(
            template_dir=template_dir,
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=d_model,
            layers=layers,
            heads=heads,
            graph_layers=graph_layers,
            decoder_layers=decoder_layers,
            instance_tokens=instance_tokens,
            dropout=dropout,
            directed_temporal_edges=directed_temporal_edges,
        )
        self.graph_encoder = StructuredRequirementGraphEncoder(
            template_dir=template_dir,
            d_model=d_model,
            heads=heads,
            layers=graph_layers,
            dropout=dropout,
            directed_temporal_edges=directed_temporal_edges,
        )

    def forward(
        self,
        keypoints: torch.Tensor,
        instance_graph_json: list[str],
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        neural = self.neural(keypoints)
        batch, frames, _ = neural["tokens"].shape
        neural_memory = self.neural_projector(neural["tokens"])
        neural_mask = torch.zeros((batch, frames), dtype=torch.bool, device=keypoints.device)

        instance = self.graph_encoder.encode_instances(instance_graph_json, keypoints.device)
        graph_tokens, graph_mask = self.instance_compressor(instance["tokens"], instance["mask"])
        graph_memory = self.graph_projector(graph_tokens)

        templates = self.graph_encoder.encode_templates()
        classes, template_nodes, dim = templates["tokens"].shape
        template_tokens = self.template_projector(templates["tokens"])
        query = template_tokens.unsqueeze(0).expand(
            batch,
            classes,
            template_nodes,
            dim,
        ).reshape(batch * classes, template_nodes, dim)
        query_mask = templates["mask"].unsqueeze(0).expand(
            batch,
            classes,
            template_nodes,
        ).reshape(batch * classes, template_nodes)
        expanded_neural = self._expand_memory(neural_memory, classes)
        expanded_graph = self._expand_memory(graph_memory, classes)
        expanded_neural_mask = self._expand_mask(neural_mask, classes)
        expanded_graph_mask = self._expand_mask(graph_mask, classes)

        token_gate = None
        neural_attention = None
        graph_attention = None
        for decoder_layer in self.decoder:
            query, token_gate, neural_attention, graph_attention = decoder_layer(
                query,
                expanded_neural,
                expanded_graph,
                query_mask,
                expanded_neural_mask,
                expanded_graph_mask,
                return_attention,
            )
        matched_tokens = query.reshape(batch, classes, template_nodes, dim)
        class_mask = templates["mask"].unsqueeze(0).expand(batch, -1, -1)
        requirements = templates["requirements"].unsqueeze(0).expand(batch, -1, -1)
        requirement_weights = templates["requirement_weights"].unsqueeze(0).expand(batch, -1, -1)
        scored = self.scorer(
            matched_tokens,
            class_mask,
            requirements,
            requirement_weights,
        )
        kg_logits = scored["logits"]
        valid = (~class_mask).unsqueeze(-1)
        kg_features = (matched_tokens * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        final_logits, final_gate, calibration = self.final_fusion(
            neural["logits"],
            kg_logits,
            neural["pooled"],
            kg_features,
        )
        requirement_targets = self.graph_encoder.requirement_targets(
            instance["evidence_facts"],
            keypoints.device,
        )

        if token_gate is not None:
            token_gate = token_gate.reshape(batch, classes, template_nodes, dim)
        if neural_attention is not None:
            neural_attention = neural_attention.reshape(
                batch,
                classes,
                neural_attention.size(1),
                template_nodes,
                frames,
            )
        if graph_attention is not None:
            graph_attention = graph_attention.reshape(
                batch,
                classes,
                graph_attention.size(1),
                template_nodes,
                self.instance_tokens,
            )
        return {
            "logits": final_logits,
            "neural_logits": neural["logits"],
            "kg_logits": kg_logits,
            "gate": final_gate,
            "token_gate": token_gate,
            "presence": scored["presence"],
            "requirement_targets": requirement_targets,
            "requirement_ids": requirements,
            "requirement_weights": requirement_weights,
            "neural_attention": neural_attention,
            "graph_attention": graph_attention,
            "neural_temperature": calibration["neural_temperature"],
            "kg_temperature": calibration["kg_temperature"],
            "neural_confidence": calibration["neural_confidence"],
            "kg_confidence": calibration["kg_confidence"],
        }
