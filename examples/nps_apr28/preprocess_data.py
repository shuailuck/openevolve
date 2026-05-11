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

    engineered['target'] = df['target'].values

    for feat in base_features:
        # Current value (T)
        engineered[feat] = df[feat].values

        t5_col = f"{feat}_T_5_M"
        t1_col = f"{feat}_T_1_M"

        if t5_col in df.columns:
            # 5-month trend: T minus T-5 (positive = increasing)
            name_t5 = f"{feat}__trend5"
            engineered[name_t5] = df[feat].values - df[t5_col].values

            # 1-month trend: T minus T-1
            if t1_col in df.columns:
                name_t1 = f"{feat}__trend1"
                engineered[name_t1] = df[feat].values - df[t1_col].values

            # Volatility: std across available time periods
            time_cols = [f"{feat}_T_{i}_M" for i in range(5, 0, -1)] + [feat]
            existing = [c for c in time_cols if c in df.columns]
            if len(existing) >= 3:
                name_vol = f"{feat}__vol"
                engineered[name_vol] = df[existing].std(axis=1).values

    eng_df = pd.DataFrame(engineered)         
    eng_df = eng_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return eng_df


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


def main():
    print("Loading training data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv.gz"))

    print("Loading validation data...")
    df_val = pd.read_csv(os.path.join(DATA_DIR, "val.csv.gz"))

    # Identify base features
    base_features = identify_base_features(df_train.columns)
    print(f"Base features: {len(base_features)}")

    # Engineer features from training data
    print("Engineering features...")
    eng_train = engineer_features(df_train, base_features)
    # Apply same feature selection to validation data
    eng_val = engineer_features(df_val, base_features)

    eng_train.to_csv(os.path.join(DATA_DIR, "train_derived.csv.gz"), index=False, compression='gzip')
    eng_val.to_csv(os.path.join(DATA_DIR, "val_derived.csv.gz"), index=False, compression='gzip')

    # Print feature mapping
    print(f"\n{'='*60}")
    print(f"\nSaved to {DATA_DIR}:")
    print(f"  train_derived.csv.gz      shape={eng_train.shape}")
    print(f"  val_derived.csv.gz       shape={eng_val.shape}")

if __name__ == "__main__":
    main()