from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y[..., : x.size(-1)] + x


class TCNAutoencoder(nn.Module):
    """Dilated temporal convolutional autoencoder."""

    def __init__(self, input_size: int, channels: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, channels, 1)
        dilations = (1, 2, 4, 8)
        self.encoder = nn.Sequential(
            *(TemporalBlock(channels, d, dropout) for d in dilations)
        )
        self.decoder = nn.Sequential(
            *(TemporalBlock(channels, d, dropout) for d in reversed(dilations))
        )
        self.output_projection = nn.Conv1d(channels, input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.input_projection(y)
        y = self.encoder(y)
        y = self.decoder(y)
        return self.output_projection(y).transpose(1, 2)
