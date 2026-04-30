"""
Stage 2 Evaluator: Feature Engineering → XGBoost → Validation AUC
"""
import os
import sys
import json
import importlib.util
import traceback
import numpy as np
from sklearn.metrics import roc_curve, auc as sk_auc
import concurrent.futures

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "verbosity": 0,
}


def run_with_timeout(fn, args=(), timeout_seconds=30):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        return future.result(timeout=timeout_seconds)


def evaluate(program_path):
    metrics = {"combined_score": 0.0, "auc_val": 0.0, "num_features": 0.0}

    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train_all.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        X_val = np.load(os.path.join(DATA_DIR, "X_val_all.npy"))
        y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
        with open(os.path.join(DATA_DIR, "feature_names_all.json"), "r", encoding="utf-8") as f:
            column_names = json.load(f)
    except Exception as e:
        print(f"[evaluator] Failed to load data: {e}")
        return metrics

    # Load program
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
    except Exception as e:
        print(f"[evaluator] Failed to load program: {e}")
        return metrics

    if not hasattr(program, "run_search") or not callable(program.run_search):
        print("[evaluator] Program missing callable run_search()")
        return metrics

    feature_engineer = program.run_search()

    # Run feature engineering with timeout
    try:
        result = run_with_timeout(
            feature_engineer,
            args=(X_train, y_train, X_val, y_val, column_names),
            timeout_seconds=60,
        )
    except concurrent.futures.TimeoutError:
        print("[evaluator] feature_engineer timed out")
        return metrics
    except Exception as e:
        print(f"[evaluator] feature_engineer failed: {e}")
        traceback.print_exc()
        return metrics

    if not isinstance(result, tuple) or len(result) != 4:
        print(f"[evaluator] Expected 4-tuple return, got {type(result)}")
        return metrics

    X_train_new, y_train_out, X_val_new, y_val_out = result

    # Validate shapes
    if X_train_new.shape[0] != X_train.shape[0]:
        print(f"[evaluator] X_train sample count changed: {X_train.shape[0]} -> {X_train_new.shape[0]}")
        return metrics
    if X_val_new.shape[0] != X_val.shape[0]:
        print(f"[evaluator] X_val sample count changed: {X_val.shape[0]} -> {X_val_new.shape[0]}")
        return metrics
    if X_train_new.shape[1] == 0:
        print("[evaluator] No features produced")
        return metrics
    if X_train_new.shape[1] != X_val_new.shape[1]:
        print(f"[evaluator] Feature count mismatch: train={X_train_new.shape[1]}, val={X_val_new.shape[1]}")
        return metrics

    # Clean data
    X_train_new = np.nan_to_num(X_train_new, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_val_new = np.nan_to_num(X_val_new, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Train XGBoost
    try:
        from xgboost import XGBClassifier

        pos_count = np.sum(y_train_out == 1)
        neg_count = np.sum(y_train_out == 0)
        spw = float(pos_count) / max(float(neg_count), 1.0)

        model = XGBClassifier(**XGBOOST_PARAMS, scale_pos_weight=spw)
        model.fit(X_train_new, y_train_out, eval_set=[(X_val_new, y_val_out)], verbose=False)
        val_probs = model.predict_proba(X_val_new)[:, 0]
        fpr, tpr, _ = roc_curve(y_val_out, val_probs, pos_label=0)
        auc_val = float(sk_auc(fpr, tpr))

        metrics = {
            "combined_score": float(auc_val),
            "auc_val": float(auc_val),
            "num_features": float(X_train_new.shape[1]),
        }
    except Exception as e:
        print(f"[evaluator] XGBoost training/eval failed: {e}")
        traceback.print_exc()

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        prog = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_initial_program.py")
    else:
        prog = sys.argv[1]

    print(f"Evaluating: {prog}")
    result = evaluate(prog)
    print("\nResults:")
    for k, v in result.items():
        if k == "num_features":
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.4f}")
