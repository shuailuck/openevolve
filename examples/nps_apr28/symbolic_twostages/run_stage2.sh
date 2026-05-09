#!/bin/bash
# Run Stage 2 OpenEvolve feature engineering.
# Must run after Stage 1 completes AND prepare_stage2.py has been executed.
#
# Usage: bash run_stage2.sh [num_iterations]

set -e

ITERATIONS=${1:-200}
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENVOLVE_RUN="${BASE_DIR}/../../../openevolve-run.py"

echo "============================================================"
echo "Stage 2 Runner"
echo "Iterations: ${ITERATIONS}"
echo "============================================================"

# Verify prerequisites
if [ ! -f "${BASE_DIR}/stage2_initial_program.py" ]; then
    echo "ERROR: stage2_initial_program.py not found."
    echo "Run 'python prepare_stage2.py' first."
    exit 1
fi

# Run OpenEvolve Stage 2
python "${OPENVOLVE_RUN}" \
    "${BASE_DIR}/stage2_initial_program.py" \
    "${BASE_DIR}/stage2_evaluator.py" \
    --config "${BASE_DIR}/stage2_config.yaml" \
    --iterations "${ITERATIONS}"

echo ""
echo "============================================================"
echo "Stage 2 complete. Best program in: ${BASE_DIR}/openevolve_output/best/"
echo "============================================================"
