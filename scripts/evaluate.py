from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from anomaly_detection.cli import build_model
from anomaly_detection.detection.scoring import reconstruction_error
from anomaly_detection.detection.thresholding import percentile_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Score telemetry windows with a trained model")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--data", default="data/processed/test.npy")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    windows = torch.from_numpy(np.load(args.data).astype(np.float32))
    model = build_model(cfg, input_size=windows.shape[-1])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        scores = reconstruction_error(windows, model(windows)).numpy()

    threshold = percentile_threshold(scores, float(cfg["threshold"]["percentile"]))
    anomalies = scores > threshold.value
    print(f"windows={len(scores):,}")
    print(f"threshold={threshold.value:.6f} ({threshold.method})")
    print(f"anomalous_windows={int(anomalies.sum()):,}")
    print(f"anomaly_rate={float(anomalies.mean()):.4%}")


if __name__ == "__main__":
    main()
