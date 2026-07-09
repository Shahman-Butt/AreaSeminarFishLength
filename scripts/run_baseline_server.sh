#!/usr/bin/env bash
set -euo pipefail

python scripts/build_autofish_index.py \
  --raw-dir data/raw/autofish \
  --out data/processed/index.csv \
  --splits-out data/processed/splits.json \
  --exclusions-out data/processed/exclusions.json

python scripts/make_crops.py \
  --raw-dir data/raw/autofish \
  --index data/processed/index.csv \
  --out-dir data/processed/crops

python scripts/check_processed.py \
  --index data/processed/index.csv \
  --crops-dir data/processed/crops

python -m src.autofish_vfm.train_baseline \
  --config configs/baseline_official.json \
  --index data/processed/index.csv \
  --crops-dir data/processed/crops \
  --out-dir results/baseline_mobilenetv2

python -m src.autofish_vfm.evaluate \
  --checkpoint results/baseline_mobilenetv2/best.pt \
  --config configs/baseline_official.json \
  --index data/processed/index.csv \
  --crops-dir data/processed/crops \
  --out results/baseline_mobilenetv2/test_metrics.json
