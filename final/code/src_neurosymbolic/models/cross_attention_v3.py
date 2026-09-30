import math
from pathlib import Path

import torch
from torch import nn

from .cross_attention_v2 import (
    GraphEncoderV2,
    NeuralBaselineAdapter,
    RequirementScorer,
)


class ModalityProjector(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(tokens + self.dropout(self.projection(tokens)))


class LearnedInstanceCompressor(nn.Module):
    def __init__(self, d_model: int, heads: int, output_tokens: int, dropout: float) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(output_tokens, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokens.size(0)
        memory = tokens
        safe_mask = padding_mask.clone()
        empty = safe_mask.all(dim=1)
        if empty.any():
            memory = memory.clone()
            memory[empty, 0] = 0.0
            safe_mask[empty, 0] = False
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.attention(
            queries,
            memory,
            memory,
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        output = self.norm1(queries + self.dropout(attended))
        output = self.norm2(output + self.dropout(self.ffn(output)))
        output_mask = torch.zeros(
            (batch, output.size(1)),
            dtype=torch.bool,
            device=tokens.device,
        )
        if empty.any():
            output = output.masked_fill(empty[:, None, None], 0.0)
            output_mask[empty] = True
            output_mask[empty, 0] = False
        return output, output_mask


class DualCrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.neural_attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.graph_attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.token_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        neural_memory: torch.Tensor,
        graph_memory: torch.Tensor,
        query_mask: torch.Tensor,
        neural_mask: torch.Tensor,
        graph_mask: torch.Tensor,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        self_output, _ = self.self_attention(
            query,
            query,
            query,
            key_padding_mask=query_mask,
            need_weights=False,
        )
        query = self.self_norm(query + self.dropout(self_output))
        neural_output, neural_attention = self.neural_attention(
            query,
            neural_memory,
            neural_memory,
            key_padding_mask=neural_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        graph_output, graph_attention = self.graph_attention(
            query,
            graph_memory,
            graph_memory,
            key_padding_mask=graph_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        gate_input = torch.cat(
            [
                query,
                neural_output,
                graph_output,
                torch.abs(neural_output - graph_output),
            ],
            dim=-1,
        )
        modality_gate = torch.sigmoid(self.token_gate(gate_input))
        fused = modality_gate * neural_output + (1.0 - modality_gate) * graph_output
        query = self.fusion_norm(query + self.dropout(fused))
        query = self.ffn_norm(query + self.dropout(self.ffn(query)))
        query = query.masked_fill(query_mask.unsqueeze(-1), 0.0)
        return query, modality_gate, neural_attention, graph_attention


class ConfidenceLogitFusion(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.neural_temperature_raw = nn.Parameter(torch.tensor(0.5413249))
        self.kg_temperature_raw = nn.Parameter(torch.tensor(0.5413249))
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + 4, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(0.8 / 0.2))
        self.num_classes = num_classes

    @staticmethod
    def _temperature(raw: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(raw) + 1e-4

    def forward(
        self,
        neural_logits: torch.Tensor,
        kg_logits: torch.Tensor,
        neural_feature: torch.Tensor,
        kg_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        neural_temperature = self._temperature(self.neural_temperature_raw)
        kg_temperature = self._temperature(self.kg_temperature_raw)
        neural_scaled = neural_logits / neural_temperature
        kg_scaled = kg_logits / kg_temperature
        neural_probability = torch.softmax(neural_scaled, dim=-1)
        kg_probability = torch.softmax(kg_scaled, dim=-1)
        neural_confidence = neural_probability.max(dim=-1, keepdim=True).values
        kg_confidence = kg_probability.max(dim=-1, keepdim=True).values
        neural_entropy = -(neural_probability * neural_probability.clamp_min(1e-8).log()).sum(
            dim=-1,
            keepdim=True,
        )
        kg_entropy = -(kg_probability * kg_probability.clamp_min(1e-8).log()).sum(
            dim=-1,
            keepdim=True,
        )
        batch, classes = neural_logits.shape
        neural_expanded = neural_feature.unsqueeze(1).expand(batch, classes, -1)
        confidence_features = torch.cat(
            [
                neural_confidence,
                kg_confidence,
                neural_entropy,
                kg_entropy,
            ],
            dim=-1,
        ).unsqueeze(1).expand(batch, classes, -1)
        gate_input = torch.cat([neural_expanded, kg_features, confidence_features], dim=-1)
        neural_gate = torch.sigmoid(self.gate(gate_input).squeeze(-1))
        final_logits = neural_gate * neural_scaled + (1.0 - neural_gate) * kg_scaled
        return final_logits, neural_gate, {
            "neural_temperature": neural_temperature,
            "kg_temperature": kg_temperature,
            "neural_confidence": neural_confidence.squeeze(-1),
            "kg_confidence": kg_confidence.squeeze(-1),
        }


class DualCrossAttentionNeuroSymbolicModel(nn.Module):
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
        super().__init__()
        self.num_classes = num_classes
        self.instance_tokens = instance_tokens
        self.neural = NeuralBaselineAdapter(
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
        self.graph_encoder = GraphEncoderV2(
            template_dir=template_dir,
            d_model=d_model,
            heads=heads,
            layers=graph_layers,
            dropout=dropout,
            directed_temporal_edges=directed_temporal_edges,
        )
        self.neural_projector = ModalityProjector(d_model, dropout)
        self.graph_projector = ModalityProjector(d_model, dropout)
        self.template_projector = ModalityProjector(d_model, dropout)
        self.instance_compressor = LearnedInstanceCompressor(
            d_model,
            heads,
            instance_tokens,
            dropout,
        )
        self.decoder = nn.ModuleList(
            [DualCrossAttentionLayer(d_model, heads, dropout) for _ in range(decoder_layers)]
        )
        self.scorer = RequirementScorer(d_model, dropout)
        self.final_fusion = ConfidenceLogitFusion(d_model, num_classes, dropout)

    def load_neural_checkpoint(self, path: str | Path) -> None:
        self.neural.load_baseline_checkpoint(path)

    def set_neural_trainable(self, trainable: bool) -> None:
        for parameter in self.neural.parameters():
            parameter.requires_grad = trainable

    @staticmethod
    def _expand_memory(memory: torch.Tensor, classes: int) -> torch.Tensor:
        batch, length, dim = memory.shape
        return memory.unsqueeze(1).expand(batch, classes, length, dim).reshape(
            batch * classes,
            length,
            dim,
        )

    @staticmethod
    def _expand_mask(mask: torch.Tensor, classes: int) -> torch.Tensor:
        batch, length = mask.shape
        return mask.unsqueeze(1).expand(batch, classes, length).reshape(
            batch * classes,
            length,
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
            instance["evidence_names"],
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
