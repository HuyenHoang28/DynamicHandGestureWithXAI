import math

import torch
from torch import nn

from .template_graph_encoder import TemplateGraphEncoder, InstanceGraphEncoder


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class KeypointTemporalEncoder(nn.Module):
    def __init__(
        self,
        num_keypoints: int = 52,
        input_dims: int = 3,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(num_keypoints * input_dims, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = SinusoidalPositionEncoding(d_model, max_frames)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        batch, frames, keypoints_count, dims = keypoints.shape
        x = keypoints.reshape(batch, frames, keypoints_count * dims)
        x = self.input_proj(x)
        x = self.pos(x)
        return self.norm(self.encoder(x))


class KGCrossAttentionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 27,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.video_norm = nn.LayerNorm(d_model)
        self.template_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, 
        video_tokens: torch.Tensor, 
        template_tokens: torch.Tensor, 
        template_mask: torch.Tensor, 
        video_mask: torch.Tensor,
        return_attention: bool = False
    ):
        batch, video_len, dim = video_tokens.shape
        cls, max_nodes, _ = template_tokens.shape

        video = self.video_norm(video_tokens)
        templates = self.template_norm(template_tokens)

        q = video.unsqueeze(1).expand(batch, cls, video_len, dim).reshape(batch * cls, video_len, dim)
        kv = templates.unsqueeze(0).expand(batch, cls, -1, -1).reshape(batch * cls, max_nodes, dim)
        
        # template_mask is [cls, max_nodes], True where padded
        key_padding_mask = template_mask.unsqueeze(0).expand(batch, cls, -1).reshape(batch * cls, max_nodes)

        attended, attn_weights = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=return_attention, average_attn_weights=False)
        
        # Masked mean pool for video tokens
        # video_mask: [batch, video_len] -> [batch * cls, video_len, 1]
        mask_expanded = video_mask.unsqueeze(1).expand(batch, cls, video_len).reshape(batch * cls, video_len, 1)
        
        q_masked = q.masked_fill(mask_expanded, 0.0)
        attended_masked = attended.masked_fill(mask_expanded, 0.0)
        
        valid_lens = (~mask_expanded).sum(dim=1).clamp(min=1) # [batch * cls, 1]
        
        video_pool = q_masked.sum(dim=1) / valid_lens
        attended_pool = attended_masked.sum(dim=1) / valid_lens
        
        logits = self.score_head(torch.cat([video_pool, attended_pool], dim=-1)).reshape(batch, cls)

        if not return_attention:
            return logits
        return logits, attn_weights.reshape(batch, cls, attn_weights.size(1), video_len, max_nodes)


class NeuralKeypointTemplateCrossAttention(nn.Module):
    def __init__(
        self,
        num_classes: int = 27,
        num_keypoints: int = 52,
        input_dims: int = 3,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        template_tokens: int = 8,
        dropout: float = 0.1,
        max_frames: int = 256,
    ) -> None:
        super().__init__()
        self.keypoint_encoder = KeypointTemporalEncoder(
            num_keypoints=num_keypoints,
            input_dims=input_dims,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_frames=max_frames,
        )
        self.template_encoder = TemplateGraphEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout,
        )
        self.instance_encoder = InstanceGraphEncoder(
            template_encoder=self.template_encoder,
            num_layers=2,
            dropout=dropout,
        )
        self.classifier = KGCrossAttentionClassifier(
            num_classes=num_classes,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, keypoints: torch.Tensor, instance_graph_json: list[str] | None = None, return_attention: bool = False):
        video_tokens = self.keypoint_encoder(keypoints)
        batch, frames, _ = video_tokens.shape
        
        # Keypoints are always valid in this implementation, so mask is all False
        video_mask = torch.zeros((batch, frames), dtype=torch.bool, device=keypoints.device)
        
        if instance_graph_json is not None:
            instance_tokens, instance_mask = self.instance_encoder.parse_and_encode(instance_graph_json, keypoints.device)
            video_tokens = torch.cat([video_tokens, instance_tokens], dim=1)
            video_mask = torch.cat([video_mask, instance_mask], dim=1)
            
        template_tokens, template_mask = self.template_encoder()
        return self.classifier(video_tokens, template_tokens, template_mask, video_mask, return_attention=return_attention)
