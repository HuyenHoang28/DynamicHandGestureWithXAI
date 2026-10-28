import math

import torch
from torch import nn

from hand_geometry import HAND_GEOMETRY_DIMS, hand_geometry_inputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional capacity {self.pe.size(1)}"
            )
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, layers: int, heads: int, dropout: float, num_frames: int) -> None:
        super().__init__()
        self.pos = PositionalEncoding(d_model, num_frames)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.norm(
            self.encoder(self.pos(x), src_key_padding_mask=padding_mask)
        )


class TCNEncoder(nn.Module):
    def __init__(self, d_model: int, layers: int, dropout: float, num_frames: int) -> None:
        super().__init__()
        self.pos = PositionalEncoding(d_model, num_frames)
        blocks = []
        for idx in range(layers):
            dilation = 2 ** idx
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(d_model, d_model, kernel_size=1),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(x).transpose(1, 2)
        for block in self.blocks:
            x = x + block(x)
        return self.norm(x.transpose(1, 2))


class BiGRUEncoder(nn.Module):
    def __init__(self, d_model: int, layers: int, dropout: float, num_frames: int) -> None:
        super().__init__()
        self.pos = PositionalEncoding(d_model, num_frames)
        hidden = max(1, d_model // 2)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        out_dim = hidden * 2
        self.proj = nn.Linear(out_dim, d_model) if out_dim != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru(self.pos(x))
        return self.norm(self.proj(x))


class MixerBlock(nn.Module):
    def __init__(self, num_frames: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mixer = nn.Sequential(
            nn.Linear(num_frames, num_frames * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_frames * 2, num_frames),
            nn.Dropout(dropout),
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mixer(x)


class MLPMixerEncoder(nn.Module):
    def __init__(self, d_model: int, layers: int, dropout: float, num_frames: int) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.pos = PositionalEncoding(d_model, num_frames)
        self.blocks = nn.ModuleList([MixerBlock(num_frames, d_model, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.num_frames:
            raise ValueError(f"MLP-Mixer expects {self.num_frames} frames, got {x.size(1)}")
        x = self.pos(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class NeuralOnlyClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "transformer",
        num_classes: int = 27,
        num_keypoints: int = 52,
        input_dims: int = 3,
        num_frames: int = 64,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        use_motion_features: bool = True,
        pooling: str = "attention",
        hand_features: bool = False,
        aux_heads: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_keypoints = num_keypoints
        self.input_dims = input_dims
        self.use_motion_features = use_motion_features
        self.pooling = pooling
        # Optional scale-invariant finger/pinch geometry (see hand_geometry.py); off keeps old checkpoints loadable.
        self.hand_features = hand_features
        if hand_features and num_keypoints != 52:
            raise ValueError("hand_features needs the 52-keypoint layout")
        feature_dims = input_dims + (2 if use_motion_features else 0)
        extra_dims = HAND_GEOMETRY_DIMS * 2 if hand_features else 0
        self.input_proj = nn.Sequential(
            nn.Linear(num_keypoints * feature_dims + extra_dims, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if backbone == "transformer":
            self.encoder = TransformerEncoder(d_model, layers, heads, dropout, num_frames)
        elif backbone == "tcn":
            self.encoder = TCNEncoder(d_model, layers, dropout, num_frames)
        elif backbone == "bigru":
            self.encoder = BiGRUEncoder(d_model, layers, dropout, num_frames)
        elif backbone == "mlp_mixer":
            self.encoder = MLPMixerEncoder(d_model, layers, dropout, num_frames)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        if pooling == "attention":
            self.pool_score = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
            )
        elif pooling not in {"mean", "masked_mean"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        # Optional training-only heads on the pooled vector: click count (1 vs 2) and finger count (1 vs 2).
        self.aux_heads = aux_heads
        if aux_heads:
            self.count_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
            self.finger_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def forward(
        self,
        keypoints: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        batch, frames, keypoints_count, dims = keypoints.shape
        if keypoints_count != self.num_keypoints or dims != self.input_dims:
            raise ValueError(
                f"Expected [B,T,{self.num_keypoints},{self.input_dims}], got {tuple(keypoints.shape)}"
            )
        if padding_mask is not None:
            if padding_mask.shape != (batch, frames) or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be bool with shape [B,T]")
            if padding_mask.device != keypoints.device:
                raise ValueError("padding_mask and keypoints must be on the same device")
            if padding_mask.all(dim=1).any():
                raise ValueError("Every sequence must contain at least one real frame")
            frame_valid = ~padding_mask
            if self.backbone != "transformer":
                raise ValueError(
                    "Explicit variable-length padding masks are currently supported only by transformer"
                )
        else:
            frame_valid = keypoints[..., 2].amax(dim=2) > 0 if dims >= 3 else torch.ones(
                (batch, frames), dtype=torch.bool, device=keypoints.device
            )
        features = keypoints
        if self.use_motion_features:
            if velocity is None:
                velocity = torch.zeros_like(keypoints[..., :2])
                velocity[:, 1:] = keypoints[:, 1:, :, :2] - keypoints[:, :-1, :, :2]
                if dims >= 3:
                    point_valid = keypoints[..., 2] > 0
                    pair_valid = point_valid[:, 1:] & point_valid[:, :-1]
                    velocity[:, 1:] *= pair_valid.unsqueeze(-1)
            else:
                if velocity.shape != (batch, frames, keypoints_count, 2):
                    raise ValueError("velocity must have shape [B,T,K,2]")
                if velocity.device != keypoints.device or velocity.dtype != keypoints.dtype:
                    raise ValueError("velocity must match keypoints device and dtype")
            if padding_mask is not None:
                velocity = velocity.masked_fill(padding_mask[:, :, None, None], 0.0)
            features = torch.cat([keypoints, velocity], dim=-1)
        elif velocity is not None:
            raise ValueError("velocity was provided while motion features are disabled")
        x = features.reshape(batch, frames, -1)
        if self.hand_features:
            x = torch.cat([x, hand_geometry_inputs(keypoints)], dim=-1)
        x = self.input_proj(x)
        if self.backbone == "transformer":
            tokens = self.encoder(x, padding_mask=padding_mask)
        else:
            tokens = self.encoder(x)
        if self.pooling == "mean":
            # Exact legacy behavior for old checkpoints and controlled ablations.
            if padding_mask is None:
                pooled = tokens.mean(dim=1)
            else:
                weights = frame_valid.to(tokens.dtype)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        elif self.pooling == "masked_mean":
            weights = frame_valid.to(tokens.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        else:
            scores = self.pool_score(tokens).squeeze(-1)
            scores = scores.masked_fill(~frame_valid, torch.finfo(scores.dtype).min)
            all_invalid = ~frame_valid.any(dim=1)
            scores = torch.where(all_invalid.unsqueeze(1), torch.zeros_like(scores), scores)
            weights = scores.softmax(dim=1)
            pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        if return_aux:
            if not self.aux_heads:
                raise ValueError("return_aux requires a model built with aux_heads=True")
            return logits, {"count": self.count_head(pooled), "fingers": self.finger_head(pooled)}
        return logits
