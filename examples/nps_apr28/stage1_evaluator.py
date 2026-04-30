"""
Evaluator for NPS symbolic regression.
Optimizes params via L-BFGS-B on binary cross-entropy, scores by AUC-ROC.
"""
import os
import sys
import importlib.util
import traceback
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import concurrent.futures

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NUM_FEATURES = 20
NUM_PARAMS = 15
NUM_RESTARTS = 10


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(params, func, X, y):
    try:
        logits = func(X, params)
        if not isinstance(logits, np.ndarray) or logits.shape != y.shape:
            return 1e9
        probs = sigmoid(logits)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        if np.isnan(loss) or np.isinf(loss):
            return 1e9
        return loss
    except Exception:
        return 1e9


def run_with_timeout(fn, args=(), timeout_seconds=5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        return future.result(timeout=timeout_seconds)


def evaluate(program_path):
    metrics = {
        "combined_score": 0.0,
        "auc_train": 0.0,
        "auc_val": 0.0,
        "logloss_train": 1.0,
        "logloss_val": 1.0,
    }

    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
        y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
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

    func = program.run_search()
    if not callable(func):
        print("[evaluator] run_search() did not return a callable")
        return metrics

    # Test with dummy data
    try:
        dummy_out = run_with_timeout(
            func, args=(np.random.randn(5, NUM_FEATURES), np.random.randn(NUM_PARAMS)), timeout_seconds=5
        )
        if not isinstance(dummy_out, np.ndarray) or dummy_out.shape != (5,):
            print(f"[evaluator] func returned wrong shape: {getattr(dummy_out, 'shape', type(dummy_out))}")
            return metrics
    except concurrent.futures.TimeoutError:
        print("[evaluator] func timed out on dummy test")
        return metrics
    except Exception as e:
        print(f"[evaluator] func failed on dummy test: {e}")
        return metrics

    # Optimize params with multiple restarts
    best_params = None
    best_loss = float("inf")

    for _ in range(NUM_RESTARTS):
        init_params = np.random.randn(NUM_PARAMS) * 0.1
        try:
            result = minimize(
                binary_cross_entropy,
                init_params,
                args=(func, X_train, y_train),
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-8},
            )
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        print("[evaluator] All optimization restarts failed")
        return metrics

    # Compute final metrics
    try:
        train_probs = sigmoid(func(X_train, best_params))
        val_probs = sigmoid(func(X_val, best_params))

        train_probs = np.clip(train_probs, 1e-15, 1 - 1e-15)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)

        auc_train = roc_auc_score(y_train, train_probs)
        auc_val = roc_auc_score(y_val, val_probs)

        logloss_val = -np.mean(
            y_val * np.log(val_probs) + (1 - y_val) * np.log(1 - val_probs)
        )

        metrics = {
            "combined_score": float(auc_val),
            "auc_train": float(auc_train),
            "auc_val": float(auc_val),
            "logloss_val": float(logloss_val),
        }
    except Exception as e:
        print(f"[evaluator] Error computing metrics: {e}")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        prog = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1_initial_program.py")
    else:
        prog = sys.argv[1]

    print(f"Evaluating: {prog}")
    result = evaluate(prog)
    print("\nResults:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
