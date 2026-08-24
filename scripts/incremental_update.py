from __future__ import annotations

import argparse

import numpy as np
import yaml
from torch.utils.data import DataLoader

from anomaly_detection.cli import build_model
from anomaly_detection.data.datasets import TelemetryWindowDataset
from anomaly_detection.training.incremental import incremental_update



def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally adapt a trained model on normal data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help=".npy windows containing verified normal data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    windows = np.load(args.data).astype(np.float32)
    dataset = TelemetryWindowDataset(windows)
    loader = DataLoader(dataset, batch_size=int(cfg["data"]["batch_size"]), shuffle=True)
    model = build_model(cfg, input_size=windows.shape[-1])

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    incremental_update(
        model,
        loader,
        args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
