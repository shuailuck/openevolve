#!/bin/bash
# Stage 2: Feature Engineering Evolution for NPS
# Run from examples/nps_apr28/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Verify Stage 1 data exists
if [ ! -f "data/X_train_all.npy" ]; then
    echo "Stage 1 data not found. Running prepare_stage1.py..."
    python prepare_stage1.py
fi

# Step 2: Extract Stage 1 expression
python prepare_stage2.py

# Step 2: Run OpenEvolve
echo "Starting Stage 2: Feature Engineering Evolution..."
python ../../openevolve-run.py stage2_initial_program.py stage2_evaluator.py \
    --config stage2_config.yaml \
    --iterations 200
