import torch

from anomaly_detection.models import LSTMAutoencoder, TCNAutoencoder


def test_lstm_autoencoder_shape() -> None:
    model = LSTMAutoencoder(input_size=5, hidden_size=16, latent_size=8)
    x = torch.randn(4, 12, 5)
    assert model(x).shape == x.shape


def test_tcn_autoencoder_shape() -> None:
    model = TCNAutoencoder(input_size=5, channels=16)
    x = torch.randn(4, 12, 5)
    assert model(x).shape == x.shape
