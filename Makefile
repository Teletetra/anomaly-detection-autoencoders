.PHONY: install lint test train prepare clean

install:
	python -m pip install -e '.[dev]'

lint:
	ruff check src tests scripts

test:
	pytest -q

prepare:
	python scripts/prepare_data.py --input data/raw/telemetry.csv --config configs/default.yaml

train:
	python scripts/train.py train --config configs/default.yaml

clean:
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
