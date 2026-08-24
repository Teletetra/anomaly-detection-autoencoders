# Anomaly Detection with Autoencoders

Production-oriented unsupervised anomaly detection for multivariate industrial time-series telemetry using sequence autoencoders and reconstruction-error scoring.

The project is designed as a portfolio-grade implementation of an industrial anomaly-detection workflow: ingest telemetry, build normal-behavior windows, train an LSTM/TCN autoencoder, calibrate anomaly thresholds, evaluate detections, and support incremental learning from newly observed normal data.

## Why this project

Industrial systems generate high-dimensional telemetry where failures are rare, labels are incomplete, and the definition of normal behavior changes over time. This project treats anomaly detection as a reconstruction problem:

`normal sequence -> encoder -> latent representation -> decoder -> reconstructed sequence`

The reconstruction error becomes the anomaly score. Windows whose error exceeds a statistically calibrated threshold are flagged for investigation.

## Key capabilities

- Unsupervised learning from predominantly normal telemetry
- Multivariate sliding-window sequence construction
- LSTM Autoencoder baseline
- TCN Autoencoder architecture for fast temporal modeling
- Reconstruction-error anomaly scoring
- Threshold calibration from validation distributions
- Point-wise and event-level anomaly evaluation
- Precision, recall, F1, ROC-AUC and PR-AUC reporting when labels are available
- Incremental learning on newly verified normal samples
- Reproducible training configuration
- Checkpointing and experiment artifacts
- CLI-first training and inference workflow
- PyTorch implementation with modular datasets, models and trainers

## Architecture

```text
                 +----------------------+
Telemetry ------>| Preprocessing        |
                 | scaling / cleaning   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Sliding Window Builder|
                 | [T x Features]       |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Sequence Autoencoder  |
                 |                      |
                 | Encoder -> Latent    |
                 | Decoder -> X_hat      |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Reconstruction Error  |
                 | MAE / MSE per window  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Threshold Calibration|
                 | percentile / robust   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Anomaly Decision      |
                 | score > threshold      |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Events + Metrics       |
                 | alerts / reports       |
                 +------------------------+
```

## Training pipeline

```text
Raw telemetry
    |
    +--> schema validation
    |
    +--> missing-value handling
    |
    +--> train-only normalization fit
    |
    +--> time-ordered split
    |
    +--> sliding windows
    |
    +--> train on normal windows
    |
    +--> validation reconstruction scores
    |
    +--> threshold calibration
    |
    +--> test inference
    |
    +--> event aggregation
    |
    +--> evaluation + artifacts
```

The pipeline avoids fitting preprocessing statistics on the test set and keeps the temporal order intact where the dataset requires it.

## Models

### LSTM Autoencoder

The default baseline uses stacked LSTM layers to encode a temporal window into a compact latent vector and reconstruct the original sequence.

```text
X[T, F]
  |
  v
LSTM Encoder
  |
  v
Latent h
  |
  v
LSTM Decoder
  |
  v
X_hat[T, F]
```

### TCN Autoencoder

A dilated Temporal Convolutional Network provides an alternative to recurrent modeling. Dilations expand temporal receptive fields without requiring sequential recurrence, making the model attractive for higher-throughput inference.

The repository is structured so both models use the same dataset, trainer, scorer and evaluation interfaces.

## Detection strategy

For a window `X` and reconstruction `X_hat`, the anomaly score is:

`score(X) = mean((X - X_hat)^2)`

The decision rule is:

`anomaly = score(X) > threshold`

Thresholds are calibrated on validation-period scores rather than selected from the test set. Supported strategies are designed to include a high percentile baseline and robust statistics, with room for domain-specific calibration.

## Incremental learning

Industrial telemetry evolves. A deployment should not require training from zero whenever verified normal behavior changes.

The incremental workflow is designed to:

1. Load a previously trained checkpoint.
2. Collect a bounded batch of newly verified normal windows.
3. Continue optimization for a small number of epochs.
4. Refit normalization state only through an explicit versioned data update.
5. Recalibrate the anomaly threshold on a fresh validation slice.
6. Save a new versioned checkpoint and metadata.

This makes model adaptation explicit, reproducible and reversible rather than silently updating production state.

## Project structure

```text
.
├── configs/
│   ├── default.yaml
│   └── experiments/
├── data/
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   └── processed/.gitkeep
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   └── anomaly_detection/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── preprocessing.py
│       │   ├── datasets.py
│       │   └── windows.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── lstm_autoencoder.py
│       │   └── tcn_autoencoder.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── losses.py
│       │   ├── trainer.py
│       │   └── incremental.py
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── scoring.py
│       │   └── thresholding.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py
│       └── cli.py
├── tests/
│   ├── test_windows.py
│   ├── test_models.py
│   ├── test_scoring.py
│   └── test_thresholding.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── incremental_update.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── pyproject.toml
├── Makefile
└── README.md
```

## Dataset

The primary target is a multivariate telemetry benchmark such as NASA SMAP/MSL. The data loader is intentionally generic so the same pipeline can consume another industrial dataset such as SWaT after adapting the schema mapping.

Dataset files are not committed to this repository. Place downloaded data under `data/raw/` and run the preparation command.

## Quick start

### 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### 2. Prepare data

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

### 3. Train the LSTM model

```bash
python scripts/train.py --config configs/default.yaml --model lstm
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/best.pt \
  --config configs/default.yaml
```

### 5. Incrementally update from verified normal data

```bash
python scripts/incremental_update.py \
  --checkpoint artifacts/checkpoints/best.pt \
  --data data/processed/new_normal.parquet \
  --config configs/default.yaml
```

## Example configuration

```yaml
data:
  sequence_length: 100
  stride: 10
  validation_ratio: 0.15
  test_ratio: 0.15
  normalization: standard

model:
  type: lstm
  input_size: 55
  hidden_size: 128
  latent_size: 64
  num_layers: 2
  dropout: 0.1

training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.00001
  early_stopping_patience: 8

threshold:
  method: percentile
  percentile: 99.5
```

## Engineering decisions

### Why train primarily on normal data?

The autoencoder is expected to reconstruct normal operating states well. Unusual behavior should create higher reconstruction error. This keeps the training objective aligned with settings where anomaly labels are sparse or unreliable.

### Why keep labels out of training?

Labels, when available, are reserved for evaluation and threshold analysis. This preserves the unsupervised nature of the detector while still allowing rigorous benchmarking.

### Why event-level evaluation?

A single real incident can create many consecutive anomalous points. Counting every point equally can exaggerate performance. The evaluation layer therefore supports aggregation into anomaly events and reporting event-level precision/recall.

## Reproducibility

Experiments should record:

- Git commit SHA
- Dataset/version identifier
- Configuration file
- Random seed
- Preprocessing parameters
- Model architecture
- Training history
- Best checkpoint
- Threshold method and value
- Evaluation metrics

Artifacts should be written to `artifacts/` and excluded from Git unless a small example artifact is deliberately added.

## Planned production extensions

- Convolutional Variational Autoencoder (CVAE) baseline
- Multi-resolution temporal feature extraction
- Online drift monitoring
- Reservoir / replay buffer for incremental learning
- Model registry and checkpoint promotion
- FastAPI inference service
- Prometheus metrics
- Docker image and deployment manifest
- Streaming ingestion through Kafka
- Alert deduplication and event correlation

## Status

The repository is being built as a complete research-to-production pipeline rather than a single notebook experiment.

## License

MIT License. See `LICENSE` for details.
