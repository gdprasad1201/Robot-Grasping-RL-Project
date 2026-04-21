#!/usr/bin/env bash
set -euo pipefail

echo "Starting training..."
python3 DexMobile/train.py

echo "Plotting learning curves..."
python3 DexMobile/plot_learning_curve.py

echo "Running evaluation and scenario generation..."
python3 DexMobile/evaluate.py

echo "Pipeline complete."