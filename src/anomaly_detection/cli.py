from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.datasets import TelemetryWindowDataset
from .models import LSTMAutoencoder, TCNAutoencoder
from .training.trainer import AutoencoderTrainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, input_size: int):
    m = cfg["model"]
    if m["type"].lower() == "lstm":
        return LSTMAutoencoder(
            input_size=input_size,
            hidden_size=m.get("hidden_size", 128),
            latent_size=m.get("latent_size", 64),
            num_layers=m.get("num_layers", 2),
            dropout=m.get("dropout", 0.1),
        )
    if m["type"].lower() == "tcn":
        return TCNAutoencoder(
            input_size=input_size,
            channels=m.get("tcn_channels", 128),
            dropout=m.get("dropout", 0.1),
        )
    raise ValueError(f"unsupported model type: {m['type']}")


def train(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed_everything(int(cfg.get("seed", 42)))

    train_data = np.load(cfg["paths"]["train"])
    validation_data = np.load(cfg["paths"]["validation"])
    dataset_train = TelemetryWindowDataset(train_data)
    dataset_val = TelemetryWindowDataset(validation_data)
    batch_size = int(cfg["data"]["batch_size"])
    workers = int(cfg["data"].get("num_workers", 0))
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = build_model(cfg, input_size=train_data.shape[-1])
    trainer = AutoencoderTrainer(
        model,
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        gradient_clip_norm=float(cfg["training"].get("gradient_clip_norm", 1.0)),
    )
    trainer.fit(
        train_loader,
        val_loader,
        epochs=int(cfg["training"]["epochs"]),
        patience=int(cfg["training"]["early_stopping_patience"]),
        checkpoint_path=cfg["paths"]["checkpoint"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an autoencoder anomaly detector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    if args.command == "train":
        Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)
        train(args.config)


if __name__ == "__main__":
    main()
