import os
import json
import importlib.util
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_curve, auc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data lives in the parent directory (examples/nps_apr28/data)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
STAGE1_RESULTS_DIR = os.path.join(BASE_DIR, "stage1_results")

# ---------------------------------------------------------------------------
# Load Stage 1 discovered formulas
# ---------------------------------------------------------------------------
with open(os.path.join(DATA_DIR, "stage1_groups.json"), "r", encoding="utf-8") as f:
    _GROUPS = json.load(f)["groups"]

_STAGE1_MODULES = {}
_STAGE1_PARAMS = {}

for _g in _GROUPS:
    _gid = _g["id"]
    _mod_path = os.path.join(BASE_DIR, "stage1_results", f"group_{_gid}_best.py")
    _param_path = os.path.join(BASE_DIR, "stage1_results", f"group_{_gid}_params.npy")
    if os.path.exists(_mod_path) and os.path.exists(_param_path):
        _spec = importlib.util.spec_from_file_location(f"group_{_gid}", _mod_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _STAGE1_MODULES[_gid] = _mod
        _STAGE1_PARAMS[_gid] = np.load(_param_path)


def _apply_stage1_formula(df, group_info):
    """Apply a single Stage 1 symbolic formula to a DataFrame."""
    gid = group_info["id"]
    mod = _STAGE1_MODULES.get(gid)
    params = _STAGE1_PARAMS.get(gid)
    if mod is None or params is None:
        return None
    cols = group_info["cols"]
    x = df[cols].values.astype(np.float32)
    return mod.func(x, params)


# ---------------------------------------------------------------------------
# Stage 2: Feature Engineering (to be evolved by OpenEvolve)
# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START
def engineer_features(train_df, val_df):
    """
    Construct high-order features guided by Stage 1 symbolic expressions.

    Stage 1 discovered powerful symbolic formulas for top feature groups.
    Use them as engineered features, then add your own interactions,
    nonlinear transforms, and aggregations.

    Args:
        train_df: Training DataFrame with all original features + target.
        val_df: Validation DataFrame with the same columns.

    Returns:
        (train_df, val_df) with additional engineered feature columns.
        Do NOT drop the target column.
    """
    results = []
    for df in [train_df, val_df]:
        new_feats = {}

        # --- Prior knowledge: Stage 1 symbolic formulas ---
        # These formulas were discovered by OpenEvolve in Stage 1.
        # They capture nonlinear patterns within each feature group.
        # It is strongly recommended to KEEP them as baseline engineered features.
        # Group 0 (arpu)
        _v = _apply_stage1_formula(df, _GROUPS[0])
        if _v is not None:
            new_feats['eng_s1_arpu'] = _v
        # Group 1 (tariff_complaint_count)
        _v = _apply_stage1_formula(df, _GROUPS[1])
        if _v is not None:
            new_feats['eng_s1_tariff_complaint_count'] = _v
        # Group 2 (churn_risk)
        _v = _apply_stage1_formula(df, _GROUPS[2])
        if _v is not None:
            new_feats['eng_s1_churn_risk'] = _v
        # Group 3 (network_quality_complaint_count)
        _v = _apply_stage1_formula(df, _GROUPS[3])
        if _v is not None:
            new_feats['eng_s1_network_quality_complaint_count'] = _v
        # Group 4 (black_user_flag)
        _v = _apply_stage1_formula(df, _GROUPS[4])
        if _v is not None:
            new_feats['eng_s1_black_user_flag'] = _v
        # Group 5 (acct_balance)
        _v = _apply_stage1_formula(df, _GROUPS[5])
        if _v is not None:
            new_feats['eng_s1_acct_balance'] = _v
        # Group 6 (staff_flag)
        _v = _apply_stage1_formula(df, _GROUPS[6])
        if _v is not None:
            new_feats['eng_s1_staff_flag'] = _v
        # Group 7 (family_circle_number_count)
        _v = _apply_stage1_formula(df, _GROUPS[7])
        if _v is not None:
            new_feats['eng_s1_family_circle_number_count'] = _v
        # Group 8 (flag_32)
        _v = _apply_stage1_formula(df, _GROUPS[8])
        if _v is not None:
            new_feats['eng_s1_flag_32'] = _v
        # Group 9 (mnp_port_out_risk)
        _v = _apply_stage1_formula(df, _GROUPS[9])
        if _v is not None:
            new_feats['eng_s1_mnp_port_out_risk'] = _v
        # --- Your evolved features go here ---
        # Ideas: pairwise products, ratios, log-transforms, conditionals,
        #        cross-group interactions, row-wise stats, etc.
        # Example (replace with evolved code):
        # new_feats["eng_arpu_x_tenure"] = df["arpu"].values * df["tenure"].values

        if new_feats:
            df = pd.concat([df, pd.DataFrame(new_feats, index=df.index)], axis=1)
        results.append(df)
    return results[0], results[1]
# EVOLVE-BLOCK-END


# ---------------------------------------------------------------------------
# Fixed training pipeline (not evolved)
# ---------------------------------------------------------------------------
def run(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train_derived.csv.gz"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_derived.csv.gz"))

    # Label flip (original 0 -> 1 positive)
    train_df["target"] = train_df["target"].map({0: 1, 1: 0})
    val_df["target"] = val_df["target"].map({0: 1, 1: 0})

    orig_cols = set(train_df.columns)
    train_df, val_df = engineer_features(train_df, val_df)

    new_cols = [c for c in train_df.columns if c not in orig_cols]
    if new_cols:
        for df in [train_df, val_df]:
            df[new_cols] = df[new_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]

    # Scale pos weight
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    fpr, tpr, _ = roc_curve(y_val, y_prob, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    f1_macro = float(f1_score(y_val, y_pred, average="macro"))
    num_eng = len(new_cols)

    return {"roc_auc": roc_auc, "f1_macro": f1_macro, "num_eng_features": num_eng}


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    results = run(data_dir)
    print(results)
