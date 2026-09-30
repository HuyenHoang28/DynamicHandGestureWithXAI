import json
from pathlib import Path
import torch
import torch.nn as nn

NODE_TYPES = [
    "Video", "Gesture", "Phase", "Event", "BodyPart", "Motion", "Position", 
    "JointAngle", "HandShape", "FingerMotion", "Coordination", "Sweep", "Evidence"
]

RELATION_TYPES = [
    "HAS_PHASE", "HAS_EVENT", "INVOLVES", "HAS_MOTION", "HAS_POSITION",
    "HAS_HAND_SHAPE", "HAS_FINGER_MOTION", "HAS_COORDINATION", "HAS_SWEEP",
    "HAS_JOINT_ANGLE", "BEFORE", "AFTER", "OVERLAPS", "DURING", "REPEATS", "HAS_EVIDENCE"
]

class RelationalGraphTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor):
        attn_out, _ = self.self_attn(
            x, x, x, 
            attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TemplateGraphEncoder(nn.Module):
    def __init__(
        self, 
        d_model: int = 128, 
        num_heads: int = 4, 
        num_layers: int = 2, 
        dropout: float = 0.1, 
        template_dir: str = "data/template_graphs"
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.node_type_to_id = {nt: i for i, nt in enumerate(NODE_TYPES)}
        self.relation_type_to_id = {rt: i for i, rt in enumerate(RELATION_TYPES)}
        self.concept_to_id = {"<pad>": 0}
        
        self.graphs = self._load_and_build_graphs(template_dir)
        
        # Node Embeddings
        self.node_type_emb = nn.Embedding(len(NODE_TYPES) + 1, d_model // 2, padding_idx=len(NODE_TYPES))
        self.concept_emb = nn.Embedding(len(self.concept_to_id), d_model - (d_model // 2), padding_idx=0)
        
        # Temporal Ordinal Embedding (for Event sequence order)
        self.ordinal_emb = nn.Embedding(20, d_model, padding_idx=0) # Assume max 20 events
        # Negative flag embedding
        self.negative_emb = nn.Embedding(2, d_model)
        
        # Edge/Relation Embedding (added as bias to attention logits)
        self.relation_emb = nn.Embedding(len(RELATION_TYPES) + 1, num_heads, padding_idx=len(RELATION_TYPES))
        
        # Relational Graph Transformer Layers
        self.layers = nn.ModuleList([
            RelationalGraphTransformerLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.register_buffer("padded_node_types", self.graphs["node_types"], persistent=False)
        self.register_buffer("padded_concepts", self.graphs["concepts"], persistent=False)
        self.register_buffer("padded_ordinals", self.graphs["ordinals"], persistent=False)
        self.register_buffer("padded_negatives", self.graphs["negatives"], persistent=False)
        self.register_buffer("padded_adj", self.graphs["adj"], persistent=False)
        self.register_buffer("padded_weights", self.graphs["weights"], persistent=False)
        self.register_buffer("padding_mask", self.graphs["padding_mask"], persistent=False)
        self.register_buffer("label_ids", self.graphs["label_ids"], persistent=False)
        
    def _get_concept_id(self, name: str) -> int:
        if name not in self.concept_to_id:
            self.concept_to_id[name] = len(self.concept_to_id)
        return self.concept_to_id[name]

    def _load_and_build_graphs(self, template_dir: str) -> dict:
        template_path = Path(template_dir)
        graphs = []
        for path in sorted(template_path.glob("*.json")):
            if path.name == "index.json":
                continue
            with open(path, "r", encoding="utf-8") as f:
                graph = json.load(f)
            if graph.get("graph_type") == "template_graph":
                graphs.append(graph)
                
        if not graphs:
            raise FileNotFoundError(f"No template graphs found in {template_dir}. Please ensure JSON templates exist.")
        
        # Sort by label_id to ensure order matches logits
        graphs.sort(key=lambda x: x.get("label_id", 0))
        
        all_node_types = []
        all_concepts = []
        all_ordinals = []
        all_negatives = []
        all_adj = []
        all_weights = []
        label_ids = []
        
        max_nodes = max(len(g["nodes"]) for g in graphs)
        
        for g in graphs:
            label_ids.append(g["label_id"])
            node_id_to_idx = {}
            nt_list = []
            c_list = []
            ord_list = []
            neg_list = []
            
            for i, node in enumerate(g["nodes"]):
                node_id_to_idx[node["id"]] = i
                nt_id = self.node_type_to_id.get(node.get("type"), len(NODE_TYPES))
                nt_list.append(nt_id)
                props = node.get("properties", {})
                name = props.get("name") or props.get("label_name") or node["id"]
                c_list.append(self._get_concept_id(str(name)))
                
                # Extract ordinal and negative flag (mainly for Event nodes)
                ordinal = props.get("ordinal", 0)
                # Cap ordinal at 19 to fit embedding size 20
                ordinal = min(int(ordinal), 19)
                ord_list.append(ordinal)
                
                negative = 1 if props.get("negative", False) else 0
                neg_list.append(negative)
                
            pad_len = max_nodes - len(nt_list)
            nt_list.extend([len(NODE_TYPES)] * pad_len)
            c_list.extend([0] * pad_len)
            ord_list.extend([0] * pad_len)
            neg_list.extend([0] * pad_len)
            
            # Adjacency matrix for relations (max_nodes, max_nodes)
            adj = torch.full((max_nodes, max_nodes), len(RELATION_TYPES), dtype=torch.long)
            # Weights matrix (max_nodes, max_nodes)
            weights = torch.zeros((max_nodes, max_nodes), dtype=torch.float)
            
            for edge in g.get("edges", []):
                u = node_id_to_idx.get(edge["source"])
                v = node_id_to_idx.get(edge["target"])
                if u is not None and v is not None:
                    rel_id = self.relation_type_to_id.get(edge.get("relation"), len(RELATION_TYPES))
                    adj[u, v] = rel_id
                    # Default weight is 1.0 for standard edges
                    edge_weight = edge.get("properties", {}).get("weight", 1.0)
                    weights[u, v] = float(edge_weight)
                    
            all_node_types.append(nt_list)
            all_concepts.append(c_list)
            all_ordinals.append(ord_list)
            all_negatives.append(neg_list)
            all_adj.append(adj)
            all_weights.append(weights)
            
        padding_mask = [[False] * len(g["nodes"]) + [True] * (max_nodes - len(g["nodes"])) for g in graphs]
            
        return {
            "node_types": torch.tensor(all_node_types, dtype=torch.long),
            "concepts": torch.tensor(all_concepts, dtype=torch.long),
            "ordinals": torch.tensor(all_ordinals, dtype=torch.long),
            "negatives": torch.tensor(all_negatives, dtype=torch.long),
            "adj": torch.stack(all_adj),
            "weights": torch.stack(all_weights),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
            "label_ids": torch.tensor(label_ids, dtype=torch.long)
        }

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        nt_emb = self.node_type_emb(self.padded_node_types)
        c_emb = self.concept_emb(self.padded_concepts)
        ord_emb = self.ordinal_emb(self.padded_ordinals)
        neg_emb = self.negative_emb(self.padded_negatives)
        
        # Combine all node features
        x = torch.cat([nt_emb, c_emb], dim=-1) # [batch, max_nodes, d_model]
        x = x + ord_emb + neg_emb
        
        batch_size, max_nodes, _ = x.shape
        
        # relation biases: [batch, max_nodes, max_nodes, num_heads]
        rel_biases = self.relation_emb(self.padded_adj)
        # Scale biases by edge weights (broadcasting across num_heads)
        rel_biases = rel_biases * self.padded_weights.unsqueeze(-1)
        
        # Permute to [batch, num_heads, max_nodes, max_nodes]
        rel_biases = rel_biases.permute(0, 3, 1, 2)
        # Reshape to [batch * num_heads, max_nodes, max_nodes]
        rel_biases = rel_biases.reshape(batch_size * self.num_heads, max_nodes, max_nodes)

        
        for layer in self.layers:
            x = layer(x, attn_mask=rel_biases, key_padding_mask=self.padding_mask)
            
        return x, self.padding_mask


class InstanceGraphEncoder(nn.Module):
    def __init__(
        self, 
        template_encoder: TemplateGraphEncoder,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.template_encoder = template_encoder
        self.d_model = template_encoder.d_model
        self.num_heads = template_encoder.num_heads
        
        # Instance Graph Transformer Layers
        self.layers = nn.ModuleList([
            RelationalGraphTransformerLayer(self.d_model, self.num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(
        self, 
        node_types: torch.Tensor, 
        concepts: torch.Tensor, 
        ordinals: torch.Tensor, 
        negatives: torch.Tensor, 
        adj: torch.Tensor, 
        weights: torch.Tensor, 
        padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of dynamic instance graphs.
        All inputs are batched tensors.
        """
        # Share embeddings with TemplateGraphEncoder
        nt_emb = self.template_encoder.node_type_emb(node_types)
        c_emb = self.template_encoder.concept_emb(concepts)
        ord_emb = self.template_encoder.ordinal_emb(ordinals)
        neg_emb = self.template_encoder.negative_emb(negatives)
        
        x = torch.cat([nt_emb, c_emb], dim=-1) # [batch, max_nodes, d_model]
        x = x + ord_emb + neg_emb
        
        batch_size, max_nodes, _ = x.shape
        
        # relation biases: [batch, max_nodes, max_nodes, num_heads]
        rel_biases = self.template_encoder.relation_emb(adj)
        rel_biases = rel_biases * weights.unsqueeze(-1)
        
        # Permute to [batch, num_heads, max_nodes, max_nodes]
        rel_biases = rel_biases.permute(0, 3, 1, 2)
        # Reshape to [batch * num_heads, max_nodes, max_nodes]
        rel_biases = rel_biases.reshape(batch_size * self.num_heads, max_nodes, max_nodes)
        
        for layer in self.layers:
            x = layer(x, attn_mask=rel_biases, key_padding_mask=padding_mask)
            
        return x, padding_mask

    def parse_and_encode(self, json_strings: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        import json
        graphs = [json.loads(s) for s in json_strings]
        
        max_nodes = max(len(g.get("nodes", [])) for g in graphs)
        if max_nodes == 0:
            max_nodes = 1 # Avoid empty tensor issues
            
        all_node_types = []
        all_concepts = []
        all_ordinals = []
        all_negatives = []
        all_adj = []
        all_weights = []
        
        for g in graphs:
            node_id_to_idx = {}
            nt_list = []
            c_list = []
            ord_list = []
            neg_list = []
            
            nodes = g.get("nodes", [])
            for i, node in enumerate(nodes):
                node_id_to_idx[node["id"]] = i
                nt_id = self.template_encoder.node_type_to_id.get(node.get("type"), len(NODE_TYPES))
                nt_list.append(nt_id)
                props = node.get("properties", {})
                name = props.get("name") or props.get("label_name") or node["id"]
                
                # Use .get(..., 0) instead of _get_concept_id to prevent adding new concepts 
                # that exceed the pre-initialized Embedding matrix size
                c_id = self.template_encoder.concept_to_id.get(str(name), 0)
                c_list.append(c_id)
                
                ordinal = min(int(props.get("ordinal", 0)), 19)
                ord_list.append(ordinal)
                neg_list.append(1 if props.get("negative", False) else 0)
                
            pad_len = max_nodes - len(nt_list)
            nt_list.extend([len(NODE_TYPES)] * pad_len)
            c_list.extend([0] * pad_len)
            ord_list.extend([0] * pad_len)
            neg_list.extend([0] * pad_len)
            
            adj = torch.full((max_nodes, max_nodes), len(RELATION_TYPES), dtype=torch.long)
            weights = torch.zeros((max_nodes, max_nodes), dtype=torch.float)
            
            for edge in g.get("edges", []):
                u = node_id_to_idx.get(edge["source"])
                v = node_id_to_idx.get(edge["target"])
                if u is not None and v is not None:
                    rel_id = self.template_encoder.relation_type_to_id.get(edge.get("relation"), len(RELATION_TYPES))
                    adj[u, v] = rel_id
                    edge_weight = edge.get("properties", {}).get("weight", 1.0)
                    weights[u, v] = float(edge_weight)
                    
            all_node_types.append(nt_list)
            all_concepts.append(c_list)
            all_ordinals.append(ord_list)
            all_negatives.append(neg_list)
            all_adj.append(adj)
            all_weights.append(weights)
            
        padding_mask = [[False] * len(g.get("nodes", [])) + [True] * (max_nodes - len(g.get("nodes", []))) for g in graphs]
        
        node_types = torch.tensor(all_node_types, dtype=torch.long, device=device)
        concepts = torch.tensor(all_concepts, dtype=torch.long, device=device)
        ordinals = torch.tensor(all_ordinals, dtype=torch.long, device=device)
        negatives = torch.tensor(all_negatives, dtype=torch.long, device=device)
        adj = torch.stack(all_adj).to(device)
        weights = torch.stack(all_weights).to(device)
        pad_mask = torch.tensor(padding_mask, dtype=torch.bool, device=device)
        
        return self(node_types, concepts, ordinals, negatives, adj, weights, pad_mask)

