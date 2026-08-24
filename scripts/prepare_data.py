from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from anomaly_detection.data.preprocessing import Standardizer
from anomaly_detection.data.windows import make_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multivariate telemetry CSV")
    parser.add_argument("--input", required=True, help="CSV containing timestamp-free feature columns")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(args.input)
    values = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if values.shape[1] == 0:
        raise ValueError("input CSV contains no numeric telemetry features")

    n = len(values)
    val_ratio = float(cfg["data"]["validation_ratio"])
    test_ratio = float(cfg["data"]["test_ratio"])
    test_start = int(n * (1.0 - test_ratio))
    val_start = int(n * (1.0 - test_ratio - val_ratio))
    train, validation, test = values[:val_start], values[val_start:test_start], values[test_start:]

    scaler = Standardizer.fit(train)
    train = scaler.transform(train)
    validation = scaler.transform(validation)
    test = scaler.transform(test)

    length = int(cfg["data"]["sequence_length"])
    stride = int(cfg["data"]["stride"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train.npy", make_windows(train, length, stride))
    np.save(out / "validation.npy", make_windows(validation, length, stride))
    np.save(out / "test.npy", make_windows(test, length, stride))
    print(f"prepared {len(values):,} rows with {values.shape[1]} features")


if __name__ == "__main__":
    main()
