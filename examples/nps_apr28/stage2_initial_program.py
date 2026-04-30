"""
Stage 2: Feature Engineering for NPS Binary Classification
Evolves feature_engineer() over ~1604 pre-engineered features for XGBoost.

Input: X_train (~1604 cols), y_train, X_val (~1604 cols), y_val, feature_names
Output: X_train_new, y_train, X_val_new, y_val

IMPORTANT: y_val must NOT be used for feature computation.
"""
import numpy as np


# EVOLVE-BLOCK-START

def feature_engineer(X_train, y_train, X_val, y_val, feature_names):
    idx = {name: i for i, name in enumerate(feature_names)}

    def col(X, name):
        return X[:, idx[name]] if name in idx else np.zeros(X.shape[0])

    def build(X):
        feats = []
        # Select features from the ~1604 pool by name
        # Construct interaction / ratio / nonlinear features
        # Use Stage 1 prior knowledge to guide selection
        return np.column_stack(feats) if feats else X[:, :10]

    X_train_new = build(X_train)
    X_val_new = build(X_val)
    return X_train_new, y_train, X_val_new, y_val

# EVOLVE-BLOCK-END


def run_search():
    return feature_engineer
