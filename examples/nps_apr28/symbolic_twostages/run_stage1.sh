#!/bin/bash
# Batch runner for Stage 1 symbolic regression across all top groups.
# Usage: bash run_stage1.sh [num_iterations] [start_group] [end_group]
#
# Example:
#   bash run_stage1.sh 100        # run groups 0..9 with 100 iterations each
#   bash run_stage1.sh 100 0 4    # run groups 0..4 with 100 iterations each

set -e

ITERATIONS=${1:-100}
START_GROUP=${2:-0}
END_GROUP=${3:-9}

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENVOLVE_RUN="${BASE_DIR}/../../../openevolve-run.py"

echo "============================================================"
echo "Stage 1 Batch Runner"
echo "Iterations per group: ${ITERATIONS}"
echo "Group range: ${START_GROUP} .. ${END_GROUP}"
echo "============================================================"

mkdir -p "${BASE_DIR}/stage1_results"

for GROUP_ID in $(seq "${START_GROUP}" "${END_GROUP}"); do
    GROUP_DIR="${BASE_DIR}/data/group_${GROUP_ID}"
    if [ ! -d "${GROUP_DIR}" ]; then
        echo "Skipping group_${GROUP_ID}: data directory not found (run prepare_stage1.py first)"
        continue
    fi

    echo ""
    echo "========================================"
    echo "Running Stage 1 for Group ${GROUP_ID}"
    echo "========================================"

    export GROUP_ID

    # Run OpenEvolve for this group
    python "${OPENVOLVE_RUN}" \
        "${BASE_DIR}/stage1_initial_program.py" \
        "${BASE_DIR}/stage1_evaluator.py" \
        --config "${BASE_DIR}/stage1_config.yaml" \
        --iterations "${ITERATIONS}"

    # Archive best program
    BEST_SRC="${BASE_DIR}/openevolve_output/best/best_program.py"
    BEST_DST="${BASE_DIR}/stage1_results/group_${GROUP_ID}_best.py"
    if [ -f "${BEST_SRC}" ]; then
        cp "${BEST_SRC}" "${BEST_DST}"
        echo "[OK] Saved best program -> ${BEST_DST}"
    else
        echo "[WARN] No best program found for group ${GROUP_ID}"
    fi

    # Archive checkpoint (optional)
    CKPT_SRC="${BASE_DIR}/openevolve_output"
    CKPT_DST="${BASE_DIR}/stage1_results/group_${GROUP_ID}_output"
    if [ -d "${CKPT_SRC}" ]; then
        rm -rf "${CKPT_DST}"
        mv "${CKPT_SRC}" "${CKPT_DST}"
        echo "[OK] Moved output -> ${CKPT_DST}"
    fi
done

echo ""
echo "============================================================"
echo "Stage 1 complete. Results in: ${BASE_DIR}/stage1_results/"
echo "Next step: python prepare_stage2.py"
echo "============================================================"
