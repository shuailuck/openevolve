"""
Prepare Stage 2 (Feature Engineering Evolution).
- Verifies Stage 1 engineered data exists
- Extracts Stage 1 best symbolic expression and translates x[:,i] to feature names
- Generates stage2_initial_program.py and stage2_config.yaml
"""
import os
import re
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STAGE1_BEST_PATH = os.path.join(BASE_DIR, "openevolve_output", "best", "best_program.py")
STAGE1_INITIAL_PATH = os.path.join(BASE_DIR, "stage1_initial_program.py")


def extract_evolve_block(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(
        r"# EVOLVE-BLOCK-START\s*\n(.*?)# EVOLVE-BLOCK-END",
        content, re.DOTALL,
    )
    return match.group(1).strip() if match else None


def expression_to_formula(expr, feature_names):
    """Convert Stage 1 func(x, params) into a readable math formula."""

    def _feat(m):
        i = int(m.group(1))
        return feature_names[i] if i < len(feature_names) else m.group(0)

    # Case 1: loop-based linear model
    loop_m = re.search(r"for\s+\w+\s+in\s+range\(min\((\d+)", expr)
    bias_m = re.search(r"params\[(\d+)\]", expr)
    if loop_m and bias_m:
        n_terms = int(loop_m.group(1))
        bias_idx = int(bias_m.group(1))
        lines = [f"logit = b + w0 * {feature_names[0]}"]
        for i in range(1, min(n_terms, len(feature_names))):
            lines.append(f"     + w{i} * {feature_names[i]}")
        return "\n".join(lines)

    # Case 2: evolved expression — strip boilerplate, replace indices
    body = expr
    body = re.sub(r'def\s+func\s*\([^)]*\)\s*:', '', body)
    body = re.sub(r'"""[\s\S]*?"""', '', body)
    body = re.sub(r"#[^\n]*", "", body)
    body = re.sub(r"return\s+\w+\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"x\[:,\s*(\d+)\]", _feat, body)
    body = re.sub(r"params\[(\d+)\]", r"w\1", body)
    body = body.replace("np.", "")
    lines = [l.strip() for l in body.strip().splitlines() if l.strip()]
    return "\n".join(lines)


def generate_config(stage1_formula, feature_names_s1, n_all):
    formula_indented = "\n".join(
        f"      {line}" for line in stage1_formula.split("\n")
    )

    content = f"""max_iterations: 200
checkpoint_interval: 10
log_level: "INFO"
target_score: "combined_score"

llm:
  primary_model: "gpt-4o"
  primary_model_weight: 0.8
  secondary_model: "o3"
  secondary_model_weight: 0.2
  api_base: "https://api.openai.com/v1"
  temperature: 0.7
  max_tokens: 8192
  timeout: 120

prompt:
  system_message: |
    You are an expert in feature engineering for binary classification.

    Your task is to evolve a Python function `feature_engineer` that selects and constructs
    features from ~{n_all} pre-engineered telecom customer features for XGBoost to predict NPS dissatisfaction.

    Function signature:
    ```python
    def feature_engineer(X_train, y_train, X_val, y_val, feature_names):
        # X_train: np.ndarray (1600, ~{n_all}) - engineered features
        # y_train: np.ndarray (1600,) - training labels (0=dissatisfied, 1=satisfied)
        # X_val: np.ndarray (400, ~{n_all}) - engineered features
        # y_val: np.ndarray (400,) - DO NOT USE for feature computation
        # feature_names: list of ~{n_all} feature name strings
        # Returns: X_train_new, y_train, X_val_new, y_val
    ```

    CRITICAL RULES:
    - NEVER use y_val for feature computation (label leakage)
    - y_train CAN be used for target encoding, but val features must use train statistics
    - Output sample counts must match input (1600 train, 400 val)
    - X_train_new and X_val_new must have the same number of columns
    - Use `idx = {{name: i for i, name in enumerate(feature_names)}}` for column lookup
    - Use `col(X, name)` pattern: `X[:, idx[name]] if name in idx else np.zeros(X.shape[0])`

    DATA STRUCTURE (already engineered from raw time-series):
    The ~{n_all} features are derived from ~401 base telecom features across 6 months:
    - `feature` : current month (T) value
    - `feature__trend5` : T minus T-5 months, positive = increasing
    - `feature__trend1` : T minus T-1 month, positive = recent increase
    - `feature__vol` : std across 6 months, higher = more volatile

    PRIOR KNOWLEDGE FROM STAGE 1 (Symbolic Regression):
    Stage 1 evolved a symbolic formula predicting NPS dissatisfaction:
      sigmoid(logit) = P(satisfied), where w0..wN and b are optimized weights.
{formula_indented}

    Use this formula as guidance: the features and combinations it uses are strong
    predictive signals. Select them and explore related features from the full {n_all} pool.

    FEATURE ENGINEERING STRATEGIES:
    - Feature selection: pick features from the {n_all} pool by name using idx dict
    - Interactions: multiply related features (e.g., col(X,"arpu") * col(X,"churn_risk"))
    - Ratios: divide related features (with safe denominator: denom + 1e-8)
    - Nonlinear transforms: np.tanh, np.log1p(np.abs(...)), np.sign, x**2
    - Piecewise: np.where(condition, val1, val2), np.maximum, np.clip
    - Target encoding: mean(y_train) per bin (train stats only, apply to val)
    - Explore beyond top-20: there are {n_all} features, many untapped combinations
    - Group features by domain: complaint features, usage features, payment features

    SCORING: combined_score = XGBoost validation AUC for class 0 (dissatisfied)
    Higher is better, max 1.0. Goal: push above 0.70.
  num_top_programs: 4
  use_template_stochasticity: true

database:
  population_size: 80
  archive_size: 30
  num_islands: 4
  elite_selection_ratio: 0.3
  exploitation_ratio: 0.6

evaluator:
  timeout: 180
  cascade_evaluation: false
  parallel_evaluations: 4
  use_llm_feedback: false

diff_based_evolution: true
allow_full_rewrites: false
"""
    path = os.path.join(BASE_DIR, "stage2_config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Generated: stage2_config.yaml")


def generate_initial_program(n_all):
    content = f'''"""
Stage 2: Feature Engineering for NPS Binary Classification
Evolves feature_engineer() over ~{n_all} pre-engineered features for XGBoost.

Input: X_train (~{n_all} cols), y_train, X_val (~{n_all} cols), y_val, feature_names
Output: X_train_new, y_train, X_val_new, y_val

IMPORTANT: y_val must NOT be used for feature computation.
"""
import numpy as np


# EVOLVE-BLOCK-START

def feature_engineer(X_train, y_train, X_val, y_val, feature_names):
    idx = {{name: i for i, name in enumerate(feature_names)}}

    def col(X, name):
        return X[:, idx[name]] if name in idx else np.zeros(X.shape[0])

    def build(X):
        feats = []
        # Select features from the ~{n_all} pool by name
        # Construct interaction / ratio / nonlinear features
        # Use Stage 1 prior knowledge to guide selection
        return np.column_stack(feats) if feats else X[:, :10]

    X_train_new = build(X_train)
    X_val_new = build(X_val)
    return X_train_new, y_train, X_val_new, y_val

# EVOLVE-BLOCK-END


def run_search():
    return feature_engineer
'''
    path = os.path.join(BASE_DIR, "stage2_initial_program.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Generated: stage2_initial_program.py")


def main():
    required = [
        "X_train_all.npy", "X_val_all.npy", "y_train.npy", "y_val.npy",
        "feature_names_all.json", "feature_names.json",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"Missing Stage 1 data: {missing}")
        print("Run `python prepare_stage1.py` first.")
        return

    with open(os.path.join(DATA_DIR, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names_s1 = json.load(f)
    with open(os.path.join(DATA_DIR, "feature_names_all.json"), "r", encoding="utf-8") as f:
        feature_names_all = json.load(f)

    n_all = len(feature_names_all)
    print(f"Stage 1 features: {len(feature_names_s1)}, All features: {n_all}")

    # Extract Stage 1 best expression
    print("\n--- Stage 1 Expression ---")
    if os.path.exists(STAGE1_BEST_PATH):
        expr = extract_evolve_block(STAGE1_BEST_PATH)
        source = STAGE1_BEST_PATH
    else:
        print(f"Stage 1 best not found, falling back to: {STAGE1_INITIAL_PATH}")
        expr = extract_evolve_block(STAGE1_INITIAL_PATH)
        source = STAGE1_INITIAL_PATH

    if not expr:
        print("WARNING: Could not extract EVOLVE-BLOCK from Stage 1 program")
        return

    print(f"Source: {source}")

    formula = expression_to_formula(expr, feature_names_s1)
    print(f"\nFormula:\n{formula}")

    with open(os.path.join(DATA_DIR, "stage1_expression.txt"), "w", encoding="utf-8") as f:
        f.write(f"# Source: {source}\n\n{formula}\n")
    print(f"\nSaved to data/stage1_expression.txt")

    print(f"\nGenerating Stage 2 files:")
    generate_config(formula, feature_names_s1, n_all)
    generate_initial_program(n_all)


if __name__ == "__main__":
    main()
