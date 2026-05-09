"""
Stage 1 Evaluator: single-group symbolic regression.

Uses BFGS to optimize free parameters, then reports:
- R^2 on train / val
- AUC on train / val (sigmoid applied to logits)
- overfit penalty

The group is selected via the GROUP_ID environment variable.
Data is read from the full CSV files on demand — no per-group .npy files.
"""
import os
import json
import importlib.util
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from openevolve.evaluation_result import EvaluationResult

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUP_ID = os.environ.get("GROUP_ID", "0")
# Data lives in the parent directory (examples/nps_apr28/data)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

# Load group metadata once
with open(os.path.join(DATA_DIR, "stage1_groups.json"), "r", encoding="utf-8") as f:
    _GROUPS_META = json.load(f)["groups"]


def _sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _load_group_data(group_id):
    """Load full CSVs and extract columns for the specified group."""
    group_info = _GROUPS_META[group_id]
    cols = group_info["cols"]

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_derived.csv.gz"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_derived.csv.gz"))

    # Label flip consistent with xgb_model.py
    train_df["target"] = train_df["target"].map({0: 1, 1: 0})
    val_df["target"] = val_df["target"].map({0: 1, 1: 0})

    X_train = train_df[cols].values.astype(np.float32)
    X_val = val_df[cols].values.astype(np.float32)
    y_train = train_df["target"].values.astype(np.int32)
    y_val = val_df["target"].values.astype(np.int32)
    return X_train, X_val, y_train, y_val


def _compute_metrics(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    X_train, X_val, y_train, y_val = _load_group_data(int(GROUP_ID))
    max_nparams = getattr(program, "MAX_NPARAMS", 20)

    # --- BFGS parameter optimization ---
    def loss_func(p):
        try:
            pred = program.func(X_train, p)
            if not isinstance(pred, np.ndarray) or pred.shape != y_train.shape:
                return 1e10
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e10
            return np.mean(np.abs(pred - y_train))
        except Exception:
            return 1e10

    res = minimize(loss_func, np.zeros(max_nparams), method="BFGS", tol=1e-3)
    best_params = res.x

    # --- Predictions ---
    train_pred = program.func(X_train, best_params)
    val_pred = program.func(X_val, best_params)

    train_prob = _sigmoid(train_pred)
    val_prob = _sigmoid(val_pred)

    r2_train = _r2(y_train, train_pred)
    r2_val = _r2(y_val, val_pred)
    auc_train = roc_auc_score(y_train, train_prob)
    auc_val = roc_auc_score(y_val, val_prob)

    overfit = max(0.0, r2_train - r2_val)

    return {
        "r2_train": float(r2_train),
        "r2_val": float(r2_val),
        "auc_train": float(auc_train),
        "auc_val": float(auc_val),
        "overfit": float(overfit),
        "bfgs_success": 1.0 if res.success else 0.0,
    }, best_params


def evaluate(program_path):
    try:
        m, best_params = _compute_metrics(program_path)

        r2_score = max(0.0, min(1.0, m["r2_val"]))
        auc_score = max(0.0, min(1.0, m["auc_val"]))
        penalty = min(1.0, m["overfit"])

        combined_score = 0.4 * r2_score + 0.4 * auc_score - 0.2 * penalty
        m["combined_score"] = max(0.0, combined_score)

        summary = (
            f"Val R²={m['r2_val']:.4f}, Val AUC={m['auc_val']:.4f}, "
            f"Overfit={m['overfit']:.4f}"
        )
        return EvaluationResult(metrics=m, artifacts={"summary": summary})
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)},
            artifacts={"error": str(e)},
        )


def evaluate_stage1(program_path):
    """Quick smoke test on 100 samples — no BFGS."""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "func"):
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": "Missing func"},
            )

        X_train, _, _, _ = _load_group_data(int(GROUP_ID))
        sample = X_train[:100]
        params = np.zeros(getattr(program, "MAX_NPARAMS", 20))
        pred = program.func(sample, params)

        if not isinstance(pred, np.ndarray) or pred.shape != (100,):
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": f"Invalid output shape {getattr(pred, 'shape', 'N/A')}"},
            )
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": "NaN/Inf in output"},
            )

        return EvaluationResult(
            metrics={"combined_score": 0.3},
            artifacts={"status": "Stage 1 passed"},
        )
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": str(e)},
        )


def evaluate_stage2(program_path):
    return evaluate(program_path)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    program_path = os.path.join(current_dir, "stage1_initial_program.py")
    result = evaluate(program_path)
    print(result.to_dict())
