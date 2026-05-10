"""
Prepare residual data for symbolic regression.

This script:
1. Trains an XGBoost baseline (matching xgb_model.py exactly)
2. Computes residuals = y_true - predicted_probability
3. Selects top 40 features by XGBoost importance
4. Saves train/val residual datasets for OpenEvolve symbolic regression
5. Saves baseline predictions for evaluator correction

Run once before starting OpenEvolve.
"""
import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TARGET_COL = "target"
TRAIN_PATH = os.path.join(DATA_DIR, "train_derived.csv.gz")
VAL_PATH = os.path.join(DATA_DIR, "val_derived.csv.gz")
TOP_K = 40


def load_data(path, target_col):
    df = pd.read_csv(path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def main():
    print("Loading data...")
    X_train, y_train = load_data(TRAIN_PATH, TARGET_COL)
    X_val, y_val = load_data(VAL_PATH, TARGET_COL)

    # Compute scale_pos_weight
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    spw = neg_count / max(pos_count, 1)
    if spw < 1:
        spw = 1.0
    print(f"Train Stats -> Positive: {pos_count}, Negative: {neg_count}")
    print(f"Scale Pos Weight: {spw:.2f}")

    # Train XGBoost baseline (exactly matching xgb_model.py)
    print("\nTraining XGBoost baseline...")
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=spw,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # Predict probabilities for positive class
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]

    baseline_auc = roc_auc_score(y_val, val_prob)
    print(f"\nBaseline XGBoost Val AUC: {baseline_auc:.4f}")

    # Compute residuals
    train_residual = y_train.values.astype(float) - train_prob
    val_residual = y_val.values.astype(float) - val_prob

    print(f"Train residuals -> mean={train_residual.mean():.4f}, std={train_residual.std():.4f}")
    print(f"Val residuals   -> mean={val_residual.mean():.4f}, std={val_residual.std():.4f}")

    # Select top-K features by importance
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(TOP_K).index.tolist()

    print(f"\nTop {TOP_K} features selected by XGBoost importance:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"  {i}. {feat} ({importances[feat]:.4f})")
    print(f"  ... ({TOP_K - 10} more)")

    # Save selected features
    features_path = os.path.join(DATA_DIR, "selected_features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=2)
    print(f"\nSaved selected features to {features_path}")

    # Save residual datasets (only top features + residual)
    train_residuals = X_train[top_features].copy()
    train_residuals["residual"] = train_residual
    val_residuals = X_val[top_features].copy()
    val_residuals["residual"] = val_residual

    train_out = os.path.join(DATA_DIR, "train_residuals.csv.gz")
    val_out = os.path.join(DATA_DIR, "val_residuals.csv.gz")
    train_residuals.to_csv(train_out, index=False, compression="gzip")
    val_residuals.to_csv(val_out, index=False, compression="gzip")
    print(f"Saved train_residuals.csv.gz ({train_residuals.shape})")
    print(f"Saved val_residuals.csv.gz ({val_residuals.shape})")

    # Save baseline predictions and labels for evaluator
    baseline_path = os.path.join(DATA_DIR, "xgb_baseline.npz")
    np.savez(
        baseline_path,
        train_prob=train_prob,
        val_prob=val_prob,
        train_y=y_train.values,
        val_y=y_val.values,
    )
    print(f"Saved baseline predictions to {baseline_path}")

    print("\n" + "=" * 50)
    print("Preparation complete. You can now run OpenEvolve:")
    print("  python openevolve-run.py \\")
    print("    examples/nps_apr28/symbolic_residual/initial_program.py \\")
    print("    examples/nps_apr28/symbolic_residual/evaluator.py \\")
    print("    --config examples/nps_apr28/symbolic_residual/config.yaml \\")
    print("    --iterations 200")
    print("=" * 50)


if __name__ == "__main__":
    main()
