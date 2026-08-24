from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class Standardizer:
    scaler: StandardScaler

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        scaler = StandardScaler()
        scaler.fit(values)
        return cls(scaler)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.transform(values).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(values)
