from __future__ import annotations

import torch


def reconstruction_error(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return per-window reconstruction error."""
    error = (x - x_hat).pow(2).mean(dim=-1)
    error = error.mean(dim=-1)
    if reduction == "mean":
        return error
    if reduction == "none":
        return error
    raise ValueError("reduction must be 'mean' or 'none'")
