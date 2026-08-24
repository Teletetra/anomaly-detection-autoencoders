import numpy as np

from anomaly_detection.data.windows import make_windows


def test_make_windows_shape() -> None:
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    windows = make_windows(values, sequence_length=4, stride=2)
    assert windows.shape == (4, 4, 3)
    np.testing.assert_array_equal(windows[0], values[:4])
    np.testing.assert_array_equal(windows[1], values[2:6])
