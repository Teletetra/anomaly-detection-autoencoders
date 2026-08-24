import torch

from anomaly_detection.detection.scoring import reconstruction_error


def test_reconstruction_error_is_zero_for_identical_sequences() -> None:
    x = torch.randn(3, 10, 4)
    score = reconstruction_error(x, x)
    assert torch.allclose(score, torch.zeros(3))


def test_reconstruction_error_is_positive() -> None:
    x = torch.zeros(2, 5, 3)
    x_hat = torch.ones(2, 5, 3)
    score = reconstruction_error(x, x_hat)
    assert torch.all(score > 0)
