import importlib.util
import os
import traceback

import numpy as np
import pandas as pd
from openevolve.evaluation_result import EvaluationResult

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def evaluate(program_path):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "run"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "Missing run function"},
                artifacts={"error_type": "MissingFunction"},
            )

        results = program.run(DATA_DIR)

        roc_auc = results["roc_auc"]
        f1_macro = results["f1_macro"]
        num_eng = results.get("num_eng_features", 0)

        combined_score = 0.6 * roc_auc + 0.4 * f1_macro

        return EvaluationResult(
            metrics={
                "roc_auc": roc_auc,
                "f1_macro": f1_macro,
                "num_eng_features": float(num_eng),
                "combined_score": combined_score,
            },
            artifacts={
                "summary": f"ROC AUC={roc_auc:.4f}, F1 macro={f1_macro:.4f}, eng features={num_eng}"
            },
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)},
            artifacts={
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )


def evaluate_stage1(program_path):
    """Quick validation: load module and run engineer_features on a small sample."""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "engineer_features") or not hasattr(program, "run"):
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": "Missing required functions"},
            )

        train_sample = pd.read_csv(
            os.path.join(DATA_DIR, "train_data_500f.csv.gz"), nrows=100
        )
        val_sample = pd.read_csv(
            os.path.join(DATA_DIR, "val_data_500f.csv.gz"), nrows=50
        )

        train_out, val_out = program.engineer_features(
            train_sample.copy(), val_sample.copy()
        )

        numeric = train_out.select_dtypes(include=[np.number])
        has_nan = numeric.isnull().any().any()
        has_inf = np.isinf(numeric.values).any()

        if has_nan or has_inf:
            return EvaluationResult(
                metrics={"combined_score": 0.3},
                artifacts={"warning": "Features contain NaN/inf (will be cleaned in run)"},
            )

        return EvaluationResult(
            metrics={"combined_score": 0.5},
            artifacts={"status": "Stage 1 passed"},
        )
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        traceback.print_exc()
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": str(e), "traceback": traceback.format_exc()},
        )


def evaluate_stage2(program_path):
    """Full training and evaluation."""
    return evaluate(program_path)
