from __future__ import annotations

import torch
from torch import nn


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, embed_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RGBDTemporalStateNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        frame_embed_dim: int = 256,
        temporal_hidden_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.frame_encoder = FrameEncoder(in_channels, frame_embed_dim, dropout)
        self.temporal = nn.GRU(
            input_size=frame_embed_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(temporal_hidden_dim),
            nn.Linear(temporal_hidden_dim, temporal_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        emb = self.frame_encoder(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        out, _ = self.temporal(emb)
        return self.head(out[:, -1])


def build_model(cfg: dict) -> RGBDTemporalStateNet:
    return RGBDTemporalStateNet(
        in_channels=int(cfg.get("in_channels", 4)),
        frame_embed_dim=int(cfg.get("frame_embed_dim", 256)),
        temporal_hidden_dim=int(cfg.get("temporal_hidden_dim", 256)),
        dropout=float(cfg.get("dropout", 0.15)),
    )
