"""
Prepare Stage 1 data for multi-group symbolic regression.

Steps:
1. Load train_derived.csv.gz / val_derived.csv.gz
2. Train an XGBoost baseline to compute per-feature importances
3. Group features by base name (each base has up to 4 variants:
   base, base__trend5, base__trend1, base__vol)
4. Select top-K groups by summed importance
5. Save group metadata as JSON (no per-group data files — evaluators read
   the full CSV and extract group columns on the fly)

Run once before starting Stage 1.
"""
import os
import json
import pandas as pd
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data lives in the parent directory (examples/nps_apr28/data)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
TARGET_COL = "target"
TRAIN_PATH = os.path.join(DATA_DIR, "train_derived.csv.gz")
VAL_PATH = os.path.join(DATA_DIR, "val_derived.csv.gz")
TOP_K = 10  # Number of top groups to evolve


def group_features_by_base(feature_cols):
    """Group features by their base name."""
    groups = {}
    for col in feature_cols:
        if col.endswith("__trend5"):
            base = col[:-8]
        elif col.endswith("__trend1"):
            base = col[:-8]
        elif col.endswith("__vol"):
            base = col[:-5]
        else:
            base = col

        if base not in groups:
            groups[base] = []
        groups[base].append(col)
    return groups


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # Consistent with xgb_model.py: flip labels so original 0 -> 1 (positive)
    train_df[TARGET_COL] = train_df[TARGET_COL].map({0: 1, 1: 0})
    val_df[TARGET_COL] = val_df[TARGET_COL].map({0: 1, 1: 0})

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    print(f"Total features: {len(feature_cols)}")

    # Group features
    groups = group_features_by_base(feature_cols)
    print(f"Total groups: {len(groups)}")

    # Train XGBoost baseline for feature importances
    print("\nTraining XGBoost baseline for importance ranking...")
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]

    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    spw = neg_count / max(pos_count, 1)
    if spw < 1:
        spw = 1.0

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

    importances = pd.Series(model.feature_importances_, index=feature_cols)

    # Compute group importance as sum of member importances
    group_importance = {}
    for base, cols in groups.items():
        group_importance[base] = importances[cols].sum()

    # Select top-K groups
    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
    top_groups = sorted_groups[:TOP_K]

    print(f"\nTop {TOP_K} groups selected:")
    for rank, (base, imp) in enumerate(top_groups, 1):
        cols = groups[base]
        print(f"  {rank}. {base}: importance={imp:.4f}, cols={cols}")

    # Save metadata only (no per-group data files — evaluators read full CSV on demand)
    print("\nSaving group metadata...")
    metadata = {"num_groups": TOP_K, "groups": []}
    for group_id, (base, imp) in enumerate(top_groups):
        group_cols = groups[base]
        metadata["groups"].append({
            "id": group_id,
            "base": base,
            "cols": group_cols,
            "importance": float(imp),
        })

    with open(os.path.join(DATA_DIR, "stage1_groups.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Stage 1 data preparation complete.")
    print(f"  Top groups: {TOP_K}")
    print(f"  Full data: {TRAIN_PATH}, {VAL_PATH}")
    print(f"  Metadata: {os.path.join(DATA_DIR, 'stage1_groups.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
