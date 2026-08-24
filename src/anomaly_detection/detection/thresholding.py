from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Threshold:
    value: float
    method: str


def percentile_threshold(scores: np.ndarray, percentile: float = 99.5) -> Threshold:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("cannot calibrate a threshold from empty scores")
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    return Threshold(float(np.percentile(scores, percentile)), f"p{percentile:g}")


def robust_threshold(scores: np.ndarray, z: float = 3.5) -> Threshold:
    scores = np.asarray(scores, dtype=np.float64)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    # 1.4826 converts MAD to a robust standard-deviation estimate.
    scale = max(1.4826 * mad, np.finfo(float).eps)
    return Threshold(float(median + z * scale), f"median+{z:g}*MAD")
