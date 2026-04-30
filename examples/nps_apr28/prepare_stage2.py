"""
Prepare Stage 2 (Feature Engineering Evolution).
- Verifies Stage 1 engineered data exists (X_train_all.npy etc.)
- Extracts Stage 1 best symbolic expression as prior knowledge

Run after prepare_stage1.py and (optionally) after Stage 1 evolution completes.
"""
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STAGE1_BEST_PATH = os.path.join(BASE_DIR, "openevolve_output", "best", "best_program.py")
STAGE1_INITIAL_PATH = os.path.join(BASE_DIR, "stage1_initial_program.py")


def extract_evolve_block(filepath):
    """Extract code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(
        r"# EVOLVE-BLOCK-START\s*\n(.*?)# EVOLVE-BLOCK-END",
        content,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return None


def main():
    # Verify Stage 1 data exists
    required = ["X_train_all.npy", "X_val_all.npy", "y_train.npy", "y_val.npy", "feature_names_all.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"Missing Stage 1 data: {missing}")
        print("Run `python prepare_stage1.py` first.")
        return

    import numpy as np
    X = np.load(os.path.join(DATA_DIR, "X_train_all.npy"))
    print(f"Stage 1 engineered data: X_train_all {X.shape}")

    # Extract Stage 1 best expression
    print("\n--- Stage 1 Prior Knowledge ---")
    if os.path.exists(STAGE1_BEST_PATH):
        expr = extract_evolve_block(STAGE1_BEST_PATH)
        source = STAGE1_BEST_PATH
    else:
        print(f"Stage 1 best not found at {STAGE1_BEST_PATH}")
        print(f"Falling back to initial program: {STAGE1_INITIAL_PATH}")
        expr = extract_evolve_block(STAGE1_INITIAL_PATH)
        source = STAGE1_INITIAL_PATH

    if expr:
        print(f"Source: {source}")
        print(f"Extracted expression:\n{expr}")
        with open(os.path.join(DATA_DIR, "stage1_expression.txt"), "w", encoding="utf-8") as f:
            f.write(expr)
        print(f"\nSaved to data/stage1_expression.txt")
    else:
        print("WARNING: Could not extract EVOLVE-BLOCK from Stage 1 program")


if __name__ == "__main__":
    main()
