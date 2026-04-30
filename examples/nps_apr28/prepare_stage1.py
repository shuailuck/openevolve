"""
Prepare NPS data for symbolic regression.
Engineers time-aware features and selects top 20 via mutual information.
Run once before starting OpenEvolve.
"""
import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_FEATURES_TO_SELECT = 20
NUM_PARAMS = 15


def identify_base_features(columns):
    """Identify base feature names (current month T columns without _T_N_M suffix)."""
    base_features = []
    for c in columns:
        if c == "target":
            continue
        if not re.search(r"_T_[1-5]_M$", c):
            base_features.append(c)
    return base_features


def engineer_features(df, base_features):
    """Engineer time-aware features from raw data."""
    engineered = {}
    engineered_names = []

    for feat in base_features:
        # Current value (T)
        engineered[feat] = df[feat].values
        engineered_names.append(feat)

        t5_col = f"{feat}_T_5_M"
        t1_col = f"{feat}_T_1_M"

        if t5_col in df.columns:
            # 5-month trend: T minus T-5 (positive = increasing)
            name_t5 = f"{feat}__trend5"
            engineered[name_t5] = df[feat].values - df[t5_col].values
            engineered_names.append(name_t5)

            # 1-month trend: T minus T-1
            if t1_col in df.columns:
                name_t1 = f"{feat}__trend1"
                engineered[name_t1] = df[feat].values - df[t1_col].values
                engineered_names.append(name_t1)

            # Volatility: std across available time periods
            time_cols = [f"{feat}_T_{i}_M" for i in range(5, 0, -1)] + [feat]
            existing = [c for c in time_cols if c in df.columns]
            if len(existing) >= 3:
                name_vol = f"{feat}__vol"
                engineered[name_vol] = df[existing].std(axis=1).values
                engineered_names.append(name_vol)

    return engineered, engineered_names


def describe_feature(name):
    """Derive a short description from the feature name pattern."""
    if name.endswith("__trend5"):
        base = name[:-8]
        return f"5-month trend of {base}"
    elif name.endswith("__trend1"):
        base = name[:-8]
        return f"1-month trend of {base}"
    elif name.endswith("__vol"):
        base = name[:-5]
        return f"volatility of {base}"
    else:
        return f"{name}, current month"


def generate_initial_program(top_features):
    """Generate stage1_initial_program.py with dynamic feature mapping."""
    n = len(top_features)
    mapping_lines = []
    for i, feat in enumerate(top_features):
        desc = describe_feature(feat)
        mapping_lines.append(f"  x[:, {i:2d}]: {feat}  ({desc})")
    mapping_str = "\n".join(mapping_lines)

    content = f'''"""
NPS Binary Classification - Symbolic Regression
Predict telecom customer satisfaction (NPS promoter vs detractor).
func(x, params) returns raw logit scores.
sigmoid(func(x, params)) = P(satisfied=1).

Feature mapping (columns of x):
{mapping_str}

Notes:
  - Features with __trend5 suffix: current_value - value_5_months_ago (positive = increasing)
  - Features with __trend1 suffix: current_value - value_1_month_ago (positive = increasing)
  - Features with __vol suffix: std deviation across 6 monthly observations (higher = more volatile)
  - All continuous features are approximately standardized
  - params are optimized externally via L-BFGS-B on binary cross-entropy
"""
import numpy as np

NUM_PARAMS = {NUM_PARAMS}

# EVOLVE-BLOCK-START

def func(x, params):
    """
    Compute raw logit scores for NPS satisfaction prediction.

    Args:
        x: np.ndarray, shape (n_samples, {n}) - selected features
        params: np.ndarray, shape ({NUM_PARAMS},) - optimizable parameters

    Returns:
        np.ndarray, shape (n_samples,) - raw logit scores
    """
    logit = params[{NUM_PARAMS - 1}]  # bias
    for i in range(min({NUM_PARAMS - 1}, x.shape[1])):
        logit = logit + params[i] * x[:, i]
    return logit

# EVOLVE-BLOCK-END


def run_search():
    return func
'''
    path = os.path.join(BASE_DIR, "stage1_initial_program.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Generated: stage1_initial_program.py")


def generate_config(top_features, mi_series):
    """Generate stage1_config.yaml with dynamic feature mapping in system_message."""
    n = len(top_features)
    mapping_lines = []
    for i, feat in enumerate(top_features):
        desc = describe_feature(feat)
        mapping_lines.append(f"      x[:, {i:2d}]: {feat}  ({desc})")
    mapping_str = "\n".join(mapping_lines)

    content = f'''max_iterations: 200
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
    You are an expert in symbolic regression for binary classification.

    Your task is to evolve a Python function `func(x, params)` that predicts
    telecom customer satisfaction (NPS: Net Promoter Score).

    The function signature is:
    ```python
    def func(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    ```

    - `x` is a 2D NumPy array of shape `(n_samples, {n})` containing selected features
    - `params` is a 1D NumPy array of up to {NUM_PARAMS} optimizable parameters
    - The function returns raw logit scores of shape `(n_samples,)`
    - sigmoid(func(x, params)) = P(satisfied = 1)
    - Parameters are optimized externally via L-BFGS-B on binary cross-entropy

    Feature mapping (columns of x):
{mapping_str}

    Feature types:
      - __trend5 suffix: value_now - value_5_months_ago (positive = increasing over 5 months)
      - __trend1 suffix: value_now - value_1_month_ago (positive = increasing recently)
      - __vol suffix: std deviation across 6 monthly observations (higher = more volatile)
      - No suffix: current month value
      - All continuous features are approximately standardized (mean~0, std~1)

    Scoring: combined_score = AUC on validation set (higher is better, max 1.0)
    Parameters are optimized on training data only. Evaluation is on held-out validation data.
    A random baseline achieves ~0.5. The initial linear model achieves ~0.55-0.60.
    Your goal is to find an interpretable formula with validation AUC > 0.68.

    Strategies to try:
    - Feature interactions (e.g., x[:,2] * x[:,13] for arpu * churn_risk)
    - Nonlinear transforms (np.tanh, np.log1p, np.abs, np.sign, np.sqrt(np.abs(...)))
    - Piecewise functions (np.where, np.maximum, np.minimum)
    - Polynomial terms (x**2, x**3)
    - Combine trend and volatility features for richer temporal signals
    - Keep the expression concise and interpretable
    - Avoid overfitting: simpler expressions generalize better to validation
    - Ensure numerical stability (no division by zero, no log of negative numbers)
  num_top_programs: 4
  use_template_stochasticity: true

database:
  population_size: 80
  archive_size: 30
  num_islands: 4
  elite_selection_ratio: 0.3
  exploitation_ratio: 0.6

evaluator:
  timeout: 120
  cascade_evaluation: false
  parallel_evaluations: 4
  use_llm_feedback: false

diff_based_evolution: true
allow_full_rewrites: false
'''
    path = os.path.join(BASE_DIR, "stage1_config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Generated: stage1_config.yaml")


def main():
    print("Loading training data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv.gz"))
    y_train = df_train["target"].values

    print("Loading validation data...")
    df_val = pd.read_csv(os.path.join(DATA_DIR, "val.csv.gz"))
    y_val = df_val["target"].values

    # Identify base features
    base_features = identify_base_features(df_train.columns)
    print(f"Base features: {len(base_features)}")

    # Engineer features from training data
    print("Engineering features...")
    eng_train, feat_names = engineer_features(df_train, base_features)
    print(f"Engineered features: {len(feat_names)}")

    # Build feature matrix
    X_all_train = np.column_stack([eng_train[f] for f in feat_names])
    X_all_train = np.nan_to_num(X_all_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature selection via mutual information
    print("Computing mutual information (this may take a minute)...")
    mi_scores = mutual_info_classif(
        X_all_train, y_train, random_state=42, n_neighbors=5, discrete_features=False
    )

    mi_series = pd.Series(mi_scores, index=feat_names).sort_values(ascending=False)
    top_features = mi_series.head(NUM_FEATURES_TO_SELECT).index.tolist()

    # Select columns
    top_indices = [feat_names.index(f) for f in top_features]
    X_train = X_all_train[:, top_indices]

    # Apply same feature selection to validation data
    eng_val, _ = engineer_features(df_val, base_features)
    X_all_val = np.column_stack([eng_val[f] for f in feat_names])
    X_all_val = np.nan_to_num(X_all_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = X_all_val[:, top_indices]

    # Save full engineered features (for Stage 2)
    np.save(os.path.join(DATA_DIR, "X_train_all.npy"), X_all_train.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "X_val_all.npy"), X_all_val.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val.astype(np.float32))

    with open(os.path.join(DATA_DIR, "feature_names_all.json"), "w", encoding="utf-8") as f:
        json.dump(feat_names, f, ensure_ascii=False)

    # Save top-20 selected features (for Stage 1)
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val.astype(np.float32))

    with open(os.path.join(DATA_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=2, ensure_ascii=False)

    # Print feature mapping
    print(f"\n{'='*60}")
    print(f"Selected {NUM_FEATURES_TO_SELECT} features:")
    print(f"{'='*60}")
    for i, feat in enumerate(top_features):
        print(f"  x[:, {i:2d}]: {feat:<50s} MI={mi_series[feat]:.4f}")
    print(f"{'='*60}")
    print(f"\nSaved to {DATA_DIR}:")
    print(f"  X_train_all.npy  shape={X_all_train.shape}  (all engineered, for Stage 2)")
    print(f"  X_val_all.npy    shape={X_all_val.shape}")
    print(f"  X_train.npy      shape={X_train.shape}  (top {NUM_FEATURES_TO_SELECT}, for Stage 1)")
    print(f"  X_val.npy        shape={X_val.shape}")
    print(f"  y_train.npy      shape={y_train.shape}")
    print(f"  y_val.npy        shape={y_val.shape}")
    print(f"  feature_names_all.json  ({len(feat_names)} names)")
    print(f"  feature_names.json      ({len(top_features)} names)")

    # Generate Stage 1 program and config with dynamic feature mapping
    print(f"\nGenerating Stage 1 files:")
    generate_initial_program(top_features)
    generate_config(top_features, mi_series)


if __name__ == "__main__":
    main()
