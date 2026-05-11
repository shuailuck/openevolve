"""
Prepare data for Stage 1 symbolic regression with feature grouping.

Trains XGBoost to get feature importances, aggregates to base-feature level,
groups top base features, and saves group metadata to stage1_groups.json.

All groups share the same initial_program.py and evaluator.py — the group
is selected at runtime via the STAGE1_GROUP_ID environment variable.
"""
import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
TARGET_COL = "target"

NUM_GROUPS = 2
BASES_PER_GROUP = 4  # ~16 derived features per group (4 bases * 4 derived = 16)


def load_data():
    train_path = os.path.join(DATA_DIR, "train_derived.csv.gz")
    val_path = os.path.join(DATA_DIR, "val_derived.csv.gz")
    return pd.read_csv(train_path), pd.read_csv(val_path)


def train_xgboost(train_df, val_df):
    """Train a lightweight XGBoost to get feature importances."""
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / max(pos, 1)
    if spw < 1:
        spw = 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=spw,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, feature_cols


def aggregate_base_importance(model, feature_cols):
    """Aggregate feature importance to base-feature level.

    Each base feature has 4 derived columns: {base}, {base}__trend5,
    {base}__trend1, {base}__vol.  We sum their importances.
    """
    importances = dict(zip(feature_cols, model.feature_importances_))

    base_importance = {}
    for col in feature_cols:
        base = col.rsplit("__", 1)[0] if "__" in col else col

        if base not in base_importance:
            base_importance[base] = {"importance": 0.0, "cols": []}
        base_importance[base]["importance"] += importances[col]
        base_importance[base]["cols"].append(col)

    return base_importance


def create_groups(base_importance):
    """Group top base features consecutively by importance."""
    sorted_bases = sorted(
        base_importance.items(), key=lambda x: x[1]["importance"], reverse=True
    )

    num_bases = NUM_GROUPS * BASES_PER_GROUP
    top_bases = sorted_bases[:num_bases]

    groups = []
    for g in range(NUM_GROUPS):
        start = g * BASES_PER_GROUP
        end = start + BASES_PER_GROUP
        group_bases = top_bases[start:end]

        group_cols = []
        group_importance = 0.0
        for base_name, info in group_bases:
            group_cols.extend(sorted(info["cols"]))
            group_importance += info["importance"]

        groups.append({
            "id": g,
            "bases": [b[0] for b in group_bases],
            "cols": group_cols,
            "importance": float(group_importance),
        })

    return groups


def main():
    print("=" * 60)
    print("Stage 1 Preparation: Feature Grouping for Symbolic Regression")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    train_df, val_df = load_data()
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    print(f"  Train: {train_df.shape}, Val: {val_df.shape}")
    print(f"  Features: {len(feature_cols)} derived columns")

    print("\n[2/4] Training XGBoost for feature importance...")
    model, feature_cols = train_xgboost(train_df, val_df)

    print("\n[3/4] Aggregating importance by base feature + creating groups...")
    base_imp = aggregate_base_importance(model, feature_cols)
    print(f"  Unique base features: {len(base_imp)}")

    sorted_bases = sorted(
        base_imp.items(), key=lambda x: x[1]["importance"], reverse=True
    )
    print("  Top 15 base features by importance:")
    for i, (name, info) in enumerate(sorted_bases[:15], 1):
        print(f"    {i:2d}. {name:<45s} ({info['importance']:.5f})")

    groups = create_groups(base_imp)
    print(f"\n  Created {len(groups)} groups ({BASES_PER_GROUP} bases each):")
    for g in groups:
        print(
            f"    Group {g['id']}: {len(g['cols'])} cols, "
            f"importance={g['importance']:.4f}"
        )

    results_dir = os.path.join(HERE, "stage1_results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n[4/4] Saving group metadata...")
    metadata_path = os.path.join(results_dir, "stage1_groups.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"num_groups": len(groups), "groups": groups}, f, indent=2)
    print(f"  Saved to {metadata_path}")

    print("\n" + "=" * 60)
    print("Preparation complete!")
    print("Next: python run_stage1.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
