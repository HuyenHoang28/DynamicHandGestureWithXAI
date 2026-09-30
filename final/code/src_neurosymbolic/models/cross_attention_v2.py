import json
import math
import os
import pathlib
from pathlib import Path

import torch
from torch import nn


NODE_TYPES = [
    "Video", "Gesture", "Phase", "Event", "BodyPart", "Motion", "Position",
    "JointAngle", "HandShape", "FingerMotion", "Coordination", "Sweep", "Evidence",
]
RELATION_TYPES = [
    "HAS_PHASE", "HAS_EVENT", "INVOLVES", "HAS_MOTION", "HAS_POSITION",
    "HAS_HAND_SHAPE", "HAS_FINGER_MOTION", "HAS_COORDINATION", "HAS_SWEEP",
    "HAS_JOINT_ANGLE", "BEFORE", "AFTER", "OVERLAPS", "DURING", "REPEATS",
    "HAS_EVIDENCE",
]
REQUIREMENT_LEVELS = {
    "structural": 0,
    "required": 1,
    "preferred": 2,
    "optional": 3,
    "negative": 4,
    "padding": 5,
}


def load_checkpoint_compat(path: str | Path) -> dict:
    # Older checkpoints contain pathlib.WindowsPath objects in saved CLI args.
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath
    return torch.load(path, map_location="cpu", weights_only=False)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NeuralBaselineAdapter(nn.Module):
    """Exact transformer baseline layout, with tokens exposed for fusion."""

    def __init__(
        self,
        num_classes: int = 27,
        num_keypoints: int = 52,
        input_dims: int = 3,
        num_frames: int = 64,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(num_keypoints * input_dims, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Keep names identical to neural_only_experiments.models.NeuralOnlyClassifier.
        self.encoder = nn.Module()
        self.encoder.pos = PositionalEncoding(d_model, num_frames)
        self.encoder.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.encoder.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, keypoints: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, frames, keypoints_count, dims = keypoints.shape
        x = keypoints.reshape(batch, frames, keypoints_count * dims)
        x = self.input_proj(x)
        tokens = self.encoder.norm(self.encoder.encoder(self.encoder.pos(x)))
        pooled = tokens.mean(dim=1)
        return {"tokens": tokens, "pooled": pooled, "logits": self.classifier(pooled)}

    def load_baseline_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_checkpoint_compat(checkpoint_path)
        state = checkpoint.get("model", checkpoint)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Neural checkpoint is not architecture-compatible. Missing={missing}, unexpected={unexpected}"
            )


class StrictGraphLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        relation_bias: torch.Tensor,
        edge_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, _ = x.shape
        neg_inf = torch.finfo(x.dtype).min
        mask = torch.full(
            (batch, self.heads, nodes, nodes),
            neg_inf,
            dtype=x.dtype,
            device=x.device,
        )
        mask = torch.where(edge_mask[:, None], relation_bias, mask)
        diagonal = torch.arange(nodes, device=x.device)
        mask[:, :, diagonal, diagonal] = 0.0
        mask = mask.masked_fill(padding_mask[:, None, None, :], neg_inf)
        mask = mask.masked_fill(padding_mask[:, None, :, None], neg_inf)
        mask[:, :, diagonal, diagonal] = 0.0
        mask = mask.reshape(batch * self.heads, nodes, nodes)

        attended, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        attended = torch.nan_to_num(attended)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class GraphEncoderV2(nn.Module):
    def __init__(
        self,
        template_dir: str | Path,
        d_model: int,
        heads: int,
        layers: int = 2,
        dropout: float = 0.1,
        directed_temporal_edges: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.directed_temporal_edges = directed_temporal_edges
        self.node_type_to_id = {name: idx for idx, name in enumerate(NODE_TYPES)}
        self.relation_to_id = {name: idx for idx, name in enumerate(RELATION_TYPES)}
        self.concept_to_id = {"<pad>": 0, "<unk>": 1}
        graph_data = self._load_templates(Path(template_dir))

        split = d_model // 2
        self.node_type_emb = nn.Embedding(len(NODE_TYPES) + 1, split, padding_idx=len(NODE_TYPES))
        self.concept_emb = nn.Embedding(len(self.concept_to_id), d_model - split, padding_idx=0)
        self.ordinal_emb = nn.Embedding(32, d_model, padding_idx=0)
        self.negative_emb = nn.Embedding(2, d_model)
        self.requirement_emb = nn.Embedding(len(REQUIREMENT_LEVELS), d_model)
        self.relation_emb = nn.Embedding(len(RELATION_TYPES) + 1, heads)
        self.layers = nn.ModuleList(
            [StrictGraphLayer(d_model, heads, dropout) for _ in range(layers)]
        )

        for name, value in graph_data.items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(name, value, persistent=False)
        self.template_concept_names = graph_data["template_concept_names"]

    def _concept_id(self, name: str) -> int:
        if name not in self.concept_to_id:
            self.concept_to_id[name] = len(self.concept_to_id)
        return self.concept_to_id[name]

    @staticmethod
    def _node_name(node: dict) -> str:
        props = node.get("properties", {})
        return str(props.get("name") or props.get("label_name") or node.get("id", "<unk>"))

    def _reverse_edge(self, relation_name: str | None) -> bool:
        if not self.directed_temporal_edges:
            return True
        return relation_name not in {"BEFORE", "AFTER"}

    def _load_templates(self, template_dir: Path) -> dict:
        graphs = []
        for path in sorted(template_dir.glob("*.json")):
            if path.name == "index.json":
                continue
            graph = json.loads(path.read_text(encoding="utf-8"))
            if graph.get("graph_type") == "template_graph":
                graphs.append(graph)
        if not graphs:
            raise FileNotFoundError(f"No template graphs found in {template_dir}")
        graphs.sort(key=lambda item: int(item.get("label_id", 0)))
        max_nodes = max(len(graph.get("nodes", [])) for graph in graphs)

        rows = {
            "template_node_types": [],
            "template_concepts": [],
            "template_ordinals": [],
            "template_negatives": [],
            "template_requirements": [],
            "template_requirement_weights": [],
            "template_adj": [],
            "template_edge_weights": [],
            "template_padding_mask": [],
            "template_label_ids": [],
            "template_concept_names": [],
        }
        for graph in graphs:
            nodes = graph.get("nodes", [])
            node_index = {node["id"]: idx for idx, node in enumerate(nodes)}
            requirements = [REQUIREMENT_LEVELS["structural"]] * len(nodes)
            requirement_weights = [0.0] * len(nodes)
            for edge in graph.get("edges", []):
                target = node_index.get(edge.get("target"))
                if target is None:
                    continue
                props = edge.get("properties", {})
                level = str(props.get("requirement_level", "")).lower()
                if level in REQUIREMENT_LEVELS:
                    requirements[target] = REQUIREMENT_LEVELS[level]
                    requirement_weights[target] = abs(float(props.get("weight", 1.0)))

            node_types = []
            concepts = []
            ordinals = []
            negatives = []
            names = []
            for node in nodes:
                props = node.get("properties", {})
                name = self._node_name(node)
                names.append(name)
                node_types.append(self.node_type_to_id.get(node.get("type"), len(NODE_TYPES)))
                concepts.append(self._concept_id(name))
                ordinals.append(min(max(int(props.get("ordinal", 0)), 0), 31))
                negatives.append(int(bool(props.get("negative", False))))

            pad = max_nodes - len(nodes)
            rows["template_node_types"].append(node_types + [len(NODE_TYPES)] * pad)
            rows["template_concepts"].append(concepts + [0] * pad)
            rows["template_ordinals"].append(ordinals + [0] * pad)
            rows["template_negatives"].append(negatives + [0] * pad)
            rows["template_requirements"].append(
                requirements + [REQUIREMENT_LEVELS["padding"]] * pad
            )
            rows["template_requirement_weights"].append(requirement_weights + [0.0] * pad)
            rows["template_padding_mask"].append([False] * len(nodes) + [True] * pad)
            rows["template_label_ids"].append(int(graph.get("label_id", 0)))
            rows["template_concept_names"].append(names + ["<pad>"] * pad)

            adj = torch.full(
                (max_nodes, max_nodes),
                len(RELATION_TYPES),
                dtype=torch.long,
            )
            weights = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)
            for edge in graph.get("edges", []):
                source = node_index.get(edge.get("source"))
                target = node_index.get(edge.get("target"))
                if source is None or target is None:
                    continue
                relation_name = edge.get("relation")
                relation = self.relation_to_id.get(relation_name, len(RELATION_TYPES))
                weight = float(edge.get("properties", {}).get("weight", 1.0))
                adj[source, target] = relation
                weights[source, target] = weight
                if self._reverse_edge(relation_name):
                    adj[target, source] = relation
                    weights[target, source] = weight
            rows["template_adj"].append(adj)
            rows["template_edge_weights"].append(weights)

        tensor_keys = {
            "template_node_types": torch.long,
            "template_concepts": torch.long,
            "template_ordinals": torch.long,
            "template_negatives": torch.long,
            "template_requirements": torch.long,
            "template_requirement_weights": torch.float32,
            "template_padding_mask": torch.bool,
            "template_label_ids": torch.long,
        }
        for key, dtype in tensor_keys.items():
            rows[key] = torch.tensor(rows[key], dtype=dtype)
        rows["template_adj"] = torch.stack(rows["template_adj"])
        rows["template_edge_weights"] = torch.stack(rows["template_edge_weights"])
        return rows

    def _encode(
        self,
        node_types: torch.Tensor,
        concepts: torch.Tensor,
        ordinals: torch.Tensor,
        negatives: torch.Tensor,
        requirements: torch.Tensor,
        adj: torch.Tensor,
        edge_weights: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([self.node_type_emb(node_types), self.concept_emb(concepts)], dim=-1)
        x = (
            x
            + self.ordinal_emb(ordinals)
            + self.negative_emb(negatives)
            + self.requirement_emb(requirements)
        )
        relation_bias = self.relation_emb(adj).permute(0, 3, 1, 2)
        relation_bias = relation_bias * edge_weights[:, None]
        edge_mask = edge_weights.ne(0)
        for layer in self.layers:
            x = layer(x, relation_bias, edge_mask, padding_mask)
        return x

    def encode_templates(self) -> dict[str, torch.Tensor]:
        tokens = self._encode(
            self.template_node_types,
            self.template_concepts,
            self.template_ordinals,
            self.template_negatives,
            self.template_requirements,
            self.template_adj,
            self.template_edge_weights,
            self.template_padding_mask,
        )
        return {
            "tokens": tokens,
            "mask": self.template_padding_mask,
            "requirements": self.template_requirements,
            "requirement_weights": self.template_requirement_weights,
            "label_ids": self.template_label_ids,
        }

    def encode_instances(
        self,
        json_strings: list[str] | list[dict],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        graphs = [
            json.loads(value) if isinstance(value, str) else value
            for value in json_strings
        ]
        max_nodes = max(1, max(len(graph.get("nodes", [])) for graph in graphs))
        rows = {key: [] for key in ("node_types", "concepts", "ordinals", "negatives", "mask")}
        all_adj = []
        all_weights = []
        evidence_names = []
        for graph in graphs:
            nodes = graph.get("nodes", [])
            index = {node["id"]: idx for idx, node in enumerate(nodes)}
            names = {self._node_name(node) for node in nodes}
            evidence_names.append(names)
            node_types = []
            concepts = []
            ordinals = []
            negatives = []
            for node in nodes:
                props = node.get("properties", {})
                name = self._node_name(node)
                node_types.append(self.node_type_to_id.get(node.get("type"), len(NODE_TYPES)))
                concepts.append(self.concept_to_id.get(name, self.concept_to_id["<unk>"]))
                ordinals.append(min(max(int(props.get("ordinal", 0)), 0), 31))
                negatives.append(int(bool(props.get("negative", False))))
            pad = max_nodes - len(nodes)
            rows["node_types"].append(node_types + [len(NODE_TYPES)] * pad)
            rows["concepts"].append(concepts + [0] * pad)
            rows["ordinals"].append(ordinals + [0] * pad)
            rows["negatives"].append(negatives + [0] * pad)
            rows["mask"].append([False] * len(nodes) + [True] * pad)

            adj = torch.full((max_nodes, max_nodes), len(RELATION_TYPES), dtype=torch.long)
            weights = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)
            for edge in graph.get("edges", []):
                source = index.get(edge.get("source"))
                target = index.get(edge.get("target"))
                if source is None or target is None:
                    continue
                relation_name = edge.get("relation")
                relation = self.relation_to_id.get(relation_name, len(RELATION_TYPES))
                weight = float(edge.get("properties", {}).get("weight", 1.0))
                adj[source, target] = relation
                weights[source, target] = weight
                if self._reverse_edge(relation_name):
                    adj[target, source] = relation
                    weights[target, source] = weight
            all_adj.append(adj)
            all_weights.append(weights)

        node_types = torch.tensor(rows["node_types"], dtype=torch.long, device=device)
        concepts = torch.tensor(rows["concepts"], dtype=torch.long, device=device)
        ordinals = torch.tensor(rows["ordinals"], dtype=torch.long, device=device)
        negatives = torch.tensor(rows["negatives"], dtype=torch.long, device=device)
        padding_mask = torch.tensor(rows["mask"], dtype=torch.bool, device=device)
        requirements = torch.full_like(node_types, REQUIREMENT_LEVELS["structural"])
        tokens = self._encode(
            node_types,
            concepts,
            ordinals,
            negatives,
            requirements,
            torch.stack(all_adj).to(device),
            torch.stack(all_weights).to(device),
            padding_mask,
        )
        return {"tokens": tokens, "mask": padding_mask, "evidence_names": evidence_names}

    def requirement_targets(self, evidence_names: list[set[str]], device: torch.device) -> torch.Tensor:
        values = []
        for names in evidence_names:
            values.append(
                [
                    [float(name in names) for name in class_names]
                    for class_names in self.template_concept_names
                ]
            )
        return torch.tensor(values, dtype=torch.float32, device=device)


def compress_instance_tokens(
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    output_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Order-preserving chunk pooling prevents large graphs dominating memory."""
    batch, _, dim = tokens.shape
    output = tokens.new_zeros((batch, output_tokens, dim))
    output_mask = torch.ones((batch, output_tokens), dtype=torch.bool, device=tokens.device)
    for batch_idx in range(batch):
        valid = tokens[batch_idx, ~padding_mask[batch_idx]]
        if valid.numel() == 0:
            continue
        chunks = min(output_tokens, valid.size(0))
        for chunk_idx, chunk in enumerate(torch.tensor_split(valid, chunks, dim=0)):
            output[batch_idx, chunk_idx] = chunk.mean(dim=0)
            output_mask[batch_idx, chunk_idx] = False
    return output, output_mask


class TemplateQueryDecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
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
        memory: torch.Tensor,
        query_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, _ = self.self_attn(
            query,
            query,
            query,
            key_padding_mask=query_mask,
            need_weights=False,
        )
        query = self.norm1(query + self.dropout(attended))
        attended, weights = self.cross_attn(
            query,
            memory,
            memory,
            key_padding_mask=memory_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        query = self.norm2(query + self.dropout(attended))
        query = self.norm3(query + self.dropout(self.ffn(query)))
        query = query.masked_fill(query_mask.unsqueeze(-1), 0.0)
        return query, weights


class RequirementScorer(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.presence_head = nn.Linear(d_model, 1)
        self.match_head = nn.Linear(d_model, 1)
        self.learned_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.positive_scale = nn.Parameter(torch.tensor(1.0))
        self.negative_scale = nn.Parameter(torch.tensor(1.0))
        self.required_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _weighted_mean(
        values: torch.Tensor,
        selector: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        selected_weights = weights * selector.float()
        return (values * selected_weights).sum(dim=-1) / selected_weights.sum(dim=-1).clamp(min=1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        requirements: torch.Tensor,
        requirement_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        presence = torch.sigmoid(self.presence_head(tokens).squeeze(-1))
        match = torch.tanh(self.match_head(tokens).squeeze(-1))
        valid = ~mask
        weights = requirement_weights * valid.float()
        positive = self._weighted_mean(
            presence + match,
            (requirements == REQUIREMENT_LEVELS["required"])
            | (requirements == REQUIREMENT_LEVELS["preferred"])
            | (requirements == REQUIREMENT_LEVELS["optional"]),
            weights,
        )
        negative = self._weighted_mean(
            presence,
            requirements == REQUIREMENT_LEVELS["negative"],
            weights,
        )
        required_penalty = self._weighted_mean(
            1.0 - presence,
            requirements == REQUIREMENT_LEVELS["required"],
            weights,
        )
        pooled = (tokens * valid.unsqueeze(-1)).sum(dim=2) / valid.sum(dim=2, keepdim=True).clamp(min=1)
        learned = self.learned_score(pooled).squeeze(-1)
        logits = (
            learned
            + self.positive_scale * positive
            - self.negative_scale * negative
            - self.required_scale * required_penalty
        )
        return {
            "logits": logits,
            "presence": presence,
            "positive": positive,
            "negative": negative,
            "required_penalty": required_penalty,
        }


class TemplateQueryNeuroSymbolicModel(nn.Module):
    def __init__(
        self,
        template_dir: str | Path = "data/template_graphs",
        num_classes: int = 27,
        num_frames: int = 64,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        graph_layers: int = 2,
        decoder_layers: int = 2,
        instance_tokens: int = 16,
        dropout: float = 0.1,
        fusion_mode: str = "residual",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"kg_only", "residual", "convex", "concat"}:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        self.num_classes = num_classes
        self.instance_tokens = instance_tokens
        self.fusion_mode = fusion_mode
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
        )
        self.modality_emb = nn.Embedding(2, d_model)
        self.decoder = nn.ModuleList(
            [TemplateQueryDecoderLayer(d_model, heads, dropout) for _ in range(decoder_layers)]
        )
        self.scorer = RequirementScorer(d_model, dropout)
        self.gate_head = nn.Linear(d_model, num_classes)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, math.log(0.05 / 0.95))
        self.concat_head = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes, num_classes),
        )

    def load_neural_checkpoint(self, path: str | Path) -> None:
        self.neural.load_baseline_checkpoint(path)

    def set_neural_trainable(self, trainable: bool) -> None:
        for parameter in self.neural.parameters():
            parameter.requires_grad = trainable

    def forward(
        self,
        keypoints: torch.Tensor,
        instance_graph_json: list[str],
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        neural = self.neural(keypoints)
        batch, frames, _ = neural["tokens"].shape
        keypoint_memory = neural["tokens"] + self.modality_emb.weight[0]
        keypoint_mask = torch.zeros((batch, frames), dtype=torch.bool, device=keypoints.device)

        instance = self.graph_encoder.encode_instances(instance_graph_json, keypoints.device)
        instance_tokens, instance_mask = compress_instance_tokens(
            instance["tokens"],
            instance["mask"],
            self.instance_tokens,
        )
        instance_tokens = instance_tokens + self.modality_emb.weight[1]
        memory = torch.cat([keypoint_memory, instance_tokens], dim=1)
        memory_mask = torch.cat([keypoint_mask, instance_mask], dim=1)

        templates = self.graph_encoder.encode_templates()
        classes, template_nodes, dim = templates["tokens"].shape
        query = (
            templates["tokens"]
            .unsqueeze(0)
            .expand(batch, classes, template_nodes, dim)
            .reshape(batch * classes, template_nodes, dim)
        )
        query_mask = (
            templates["mask"]
            .unsqueeze(0)
            .expand(batch, classes, template_nodes)
            .reshape(batch * classes, template_nodes)
        )
        expanded_memory = (
            memory.unsqueeze(1)
            .expand(batch, classes, memory.size(1), dim)
            .reshape(batch * classes, memory.size(1), dim)
        )
        expanded_memory_mask = (
            memory_mask.unsqueeze(1)
            .expand(batch, classes, memory.size(1))
            .reshape(batch * classes, memory.size(1))
        )
        attention = None
        for layer in self.decoder:
            query, attention = layer(
                query,
                expanded_memory,
                query_mask,
                expanded_memory_mask,
                return_attention,
            )
        query = query.reshape(batch, classes, template_nodes, dim)
        class_mask = templates["mask"].unsqueeze(0).expand(batch, -1, -1)
        requirements = templates["requirements"].unsqueeze(0).expand(batch, -1, -1)
        requirement_weights = templates["requirement_weights"].unsqueeze(0).expand(batch, -1, -1)
        scored = self.scorer(query, class_mask, requirements, requirement_weights)
        kg_logits = scored["logits"]
        gate = torch.sigmoid(self.gate_head(neural["pooled"]))

        if self.fusion_mode == "kg_only":
            logits = kg_logits
        elif self.fusion_mode == "residual":
            logits = neural["logits"] + gate * (kg_logits - neural["logits"].detach())
        elif self.fusion_mode == "convex":
            logits = (1.0 - gate) * neural["logits"] + gate * kg_logits
        else:
            logits = self.concat_head(torch.cat([neural["logits"], kg_logits], dim=-1))

        requirement_targets = self.graph_encoder.requirement_targets(
            instance["evidence_names"],
            keypoints.device,
        )
        if attention is not None:
            attention = attention.reshape(
                batch,
                classes,
                attention.size(1),
                template_nodes,
                memory.size(1),
            )
        return {
            "logits": logits,
            "neural_logits": neural["logits"],
            "kg_logits": kg_logits,
            "gate": gate,
            "presence": scored["presence"],
            "requirement_targets": requirement_targets,
            "requirement_ids": requirements,
            "requirement_weights": requirement_weights,
            "attention": attention,
        }
