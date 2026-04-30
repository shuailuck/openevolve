#!/bin/bash
# NPS Symbolic Regression with OpenEvolve
# Run from examples/nps_apr28/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Prepare data (only needed once)
if [ ! -f "data/X_train.npy" ]; then
    echo "Preparing data..."
    python prepare_stage1.py
fi

# Step 2: Run OpenEvolve
echo "Starting Stage 1: Symbolic Regression..."
python ../../openevolve-run.py stage1_initial_program.py stage1_evaluator.py \
    --config stage1_config.yaml \
    --iterations 200
