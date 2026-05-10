"""
Stage 2 Evaluator: Feature Engineering guided by Stage 1 priors.

Evaluates the `make_features` function by:
1. Generating engineered features from train/val data
2. Concatenating with original features
3. Training XGBoost on the combined feature set
4. Measuring validation AUC

The combined_score is the validation AUC (primary) with penalties
for overfitting and excessive feature count.
"""
import importlib.util
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from openevolve.evaluation_result import EvaluationResult

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data")
TARGET_COL = "target"

# Max number of engineered features to keep evaluation fast
MAX_ENGINEERED_FEATURES = 200


def _compute_metrics(program_path):
    """Load program, generate features, train XGBoost, evaluate AUC."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    if not hasattr(program, "make_features"):
        return {"combined_score": 0.0, "error": "Missing make_features function"}

    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_derived.csv.gz"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_derived.csv.gz"))

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train_orig = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_val_orig = val_df[feature_cols]
    y_val = val_df[TARGET_COL].values

    # Generate engineered features
    try:
        train_eng = program.make_features(train_df[feature_cols])
        val_eng = program.make_features(val_df[feature_cols])
    except Exception as e:
        return {"combined_score": 0.0, "error": f"make_features crashed: {e}"}

    if not isinstance(train_eng, pd.DataFrame):
        return {
            "combined_score": 0.0,
            "error": f"make_features must return DataFrame, got {type(train_eng)}",
        }

    if len(train_eng) != len(train_df):
        return {
            "combined_score": 0.0,
            "error": f"Row count mismatch: {len(train_eng)} vs {len(train_df)}",
        }

    # Limit engineered features
    num_eng = train_eng.shape[1]
    if num_eng > MAX_ENGINEERED_FEATURES:
        train_eng = train_eng.iloc[:, :MAX_ENGINEERED_FEATURES]
        val_eng = val_eng.iloc[:, :MAX_ENGINEERED_FEATURES]
        num_eng = MAX_ENGINEERED_FEATURES

    # Ensure index alignment
    train_eng = train_eng.reset_index(drop=True)
    val_eng = val_eng.reset_index(drop=True)
    X_train_orig = X_train_orig.reset_index(drop=True)
    X_val_orig = X_val_orig.reset_index(drop=True)

    # Combine original + engineered features
    X_train = pd.concat([X_train_orig, train_eng], axis=1)
    X_val = pd.concat([X_val_orig, val_eng], axis=1)

    # Handle non-numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])

    # Train XGBoost
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
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Predict and evaluate
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_prob)
    val_auc = roc_auc_score(y_val, val_prob)

    # Feature importance of engineered vs original
    total_orig = len(feature_cols)
    importance_eng = sum(
        model.feature_importances_[total_orig:]
    )
    importance_total = sum(model.feature_importances_)
    eng_contribution = (
        importance_eng / importance_total if importance_total > 0 else 0
    )

    return {
        "train_auc": float(train_auc),
        "val_auc": float(val_auc),
        "overfit": max(0.0, train_auc - val_auc),
        "num_engineered": num_eng,
        "eng_feature_importance_share": float(eng_contribution),
    }


def evaluate(program_path):
    """Full evaluation of feature engineering program."""
    try:
        m = _compute_metrics(program_path)

        if "error" in m:
            return EvaluationResult(metrics=m)

        # Combined score: val_auc is primary, penalize overfit and reward
        # engineered features that actually contribute
        auc_score = max(0.0, m["val_auc"])
        overfit_penalty = min(0.3, m["overfit"]) * 0.5
        # Bonus for engineered features being useful
        contribution_bonus = min(0.05, m["eng_feature_importance_share"]) * 0.5

        combined_score = auc_score - overfit_penalty + contribution_bonus
        m["combined_score"] = max(0.0, min(1.0, combined_score))
        return EvaluationResult(metrics=m)

    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)}
        )


def evaluate_stage1(program_path):
    """Stage 1: quick smoke test."""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "make_features"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "Missing make_features"}
            )

        # Test with small sample
        sample_df = pd.read_csv(
            os.path.join(DATA_DIR, "train_derived.csv.gz"), nrows=100
        )
        feature_cols = [c for c in sample_df.columns if c != TARGET_COL]
        result = program.make_features(sample_df[feature_cols])

        if not isinstance(result, pd.DataFrame):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "Must return DataFrame"}
            )
        if len(result) != 100:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "Row count mismatch"}
            )

        return EvaluationResult(metrics={"combined_score": 0.2})

    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)}
        )


def evaluate_stage2(program_path):
    """Stage 2: full evaluation with XGBoost training."""
    return evaluate(program_path)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    program_path = os.path.join(current_dir, "initial_program.py")
    result = evaluate(program_path)
    print(result.to_dict())
