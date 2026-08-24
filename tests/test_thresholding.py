import numpy as np

from anomaly_detection.detection.thresholding import percentile_threshold, robust_threshold


def test_percentile_threshold() -> None:
    result = percentile_threshold(np.arange(100, dtype=float), 99.0)
    assert result.value == 98.01


def test_robust_threshold_is_above_median() -> None:
    result = robust_threshold(np.array([1, 1, 1, 1, 2, 2, 2, 10], dtype=float))
    assert result.value >= 2.0
