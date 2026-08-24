from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TelemetryWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray) -> None:
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 3:
            raise ValueError("windows must have shape [N, T, F]")
        self.windows = torch.from_numpy(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]
