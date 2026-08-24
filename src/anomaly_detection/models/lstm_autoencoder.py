from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder for normal telemetry."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        latent_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        decoder_seed = self.from_latent(latent)
        decoder_input = x
        h0 = decoder_seed.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        decoded, _ = self.decoder(decoder_input, (h0, c0))
        return self.output(decoded)
