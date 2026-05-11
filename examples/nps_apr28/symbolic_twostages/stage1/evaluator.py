"""
Stage 1 shared evaluator — classification symbolic regression with BFGS.

Reads STAGE1_GROUP_ID from environment to select the correct feature columns.
All groups share this single evaluator file.
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from openevolve.evaluation_result import EvaluationResult

DATA_DIR = os.environ["STAGE1_DATA_DIR"]
GROUPS_PATH = os.environ["STAGE1_GROUPS_PATH"]
TARGET_COL = "target"

# --- Load group config at import time ---
_group_id = int(os.environ.get("STAGE1_GROUP_ID", 0))
with open(GROUPS_PATH, "r") as f:
    _all = json.load(f)["groups"]
_group = next(g for g in _all if g["id"] == _group_id)

GROUP_ID = _group["id"]
GROUP_COLS = _group["cols"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _binary_cross_entropy(y_true, prob):
    eps = 1e-8
    prob = np.clip(prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))


def _compute_metrics(program_path):
    """Load program, run BFGS optimization, compute AUC."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_derived.csv.gz"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_derived.csv.gz"))

    X_train = train_df[GROUP_COLS]
    y_train = train_df[TARGET_COL].values
    X_val = val_df[GROUP_COLS]
    y_val = val_df[TARGET_COL].values

    max_nparams = getattr(program, "MAX_NPARAMS", 20)

    def loss_func(p):
        try:
            logits = program.predict_nps(X_train, p)
            prob = _sigmoid(logits)
            return _binary_cross_entropy(y_train, prob)
        except Exception:
            return 1e10

    res = minimize(loss_func, np.zeros(max_nparams), method="BFGS", tol=1e-3)
    best_params = res.x

    # Train metrics
    train_logits = program.predict_nps(X_train, best_params)
    train_prob = _sigmoid(train_logits)
    train_loss = _binary_cross_entropy(y_train, train_prob)
    train_auc = roc_auc_score(y_train, train_prob)

    # Val metrics
    val_logits = program.predict_nps(X_val, best_params)
    val_prob = _sigmoid(val_logits)
    val_loss = _binary_cross_entropy(y_val, val_prob)
    val_auc = roc_auc_score(y_val, val_prob)

    return {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "train_auc": float(train_auc),
        "val_auc": float(val_auc),
        "overfit": max(0.0, train_auc - val_auc),
    }


def evaluate(program_path):
    """Full evaluation with BFGS optimization."""
    try:
        m = _compute_metrics(program_path)
        auc_score = max(0.0, min(1.0, m["val_auc"]))
        penalty = min(0.5, m["overfit"])
        m["combined_score"] = max(0.0, auc_score - 0.3 * penalty)
        return EvaluationResult(metrics=m)
    except Exception as e:
        return EvaluationResult(metrics={"combined_score": 0.0, "error": str(e)})


def evaluate_stage1(program_path):
    """Stage 1 smoke test — check code runs on 100 rows without error."""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "predict_nps"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "Missing predict_nps"}
            )

        sample_df = pd.read_csv(
            os.path.join(DATA_DIR, "train_derived.csv.gz"), nrows=100
        )
        X_sample = sample_df[GROUP_COLS]

        max_nparams = getattr(program, "MAX_NPARAMS", 20)
        logits = program.predict_nps(X_sample, np.zeros(max_nparams))

        if not isinstance(logits, np.ndarray) or logits.shape[0] != 100:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "Invalid output shape"}
            )
        if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "NaN/Inf in output"}
            )

        return EvaluationResult(metrics={"combined_score": 0.2})
    except Exception as e:
        return EvaluationResult(metrics={"combined_score": 0.0, "error": str(e)})


def evaluate_stage2(program_path):
    """Stage 2 full evaluation."""
    return evaluate(program_path)


if __name__ == "__main__":
    result = evaluate(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_program.py")
    )
    print(result.to_dict())
