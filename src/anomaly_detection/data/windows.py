from __future__ import annotations

import numpy as np


def make_windows(
    values: np.ndarray,
    sequence_length: int,
    stride: int = 1,
) -> np.ndarray:
    """Create [N, T, F] overlapping windows from [T, F] telemetry."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must have shape [time, features]")
    if sequence_length <= 0 or stride <= 0:
        raise ValueError("sequence_length and stride must be positive")
    if len(values) < sequence_length:
        return np.empty((0, sequence_length, values.shape[1]), dtype=np.float32)

    starts = range(0, len(values) - sequence_length + 1, stride)
    return np.stack([values[i : i + sequence_length] for i in starts]).astype(np.float32)
