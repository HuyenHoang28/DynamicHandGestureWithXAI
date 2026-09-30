import math

import torch
from torch import nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(self.pos(x)))


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
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Sequential(
            nn.Linear(num_keypoints * input_dims, d_model),
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
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        batch, frames, keypoints_count, dims = keypoints.shape
        x = keypoints.reshape(batch, frames, keypoints_count * dims)
        x = self.input_proj(x)
        tokens = self.encoder(x)
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)
