"""
Stage 2: Feature Engineering for NPS Binary Classification
Evolves a feature engineering function based on Stage 1's ~1604 engineered features.

Stage 1 engineered 1604 features from raw data:
  - 401 current-month (T) values
  - ~401 five-month trends (__trend5 = T - T_5_M)
  - ~401 one-month trends (__trend1 = T - T_1_M)
  - ~401 volatilities (__vol = std across 6 months)

Stage 1 symbolic regression found the top-20 most predictive features and
discovered key patterns. Stage 2 uses that prior to select and construct
features from the full 1604 pool for XGBoost modeling.

Input: X_train (1600, ~1604), y_train (1600,), X_val (400, ~1604), y_val (400,)
       feature_names: list of ~1604 feature name strings
Output: X_train_new, y_train, X_val_new, y_val (same sample counts)

IMPORTANT: y_val must NOT be used for feature computation (label leakage).
           y_train CAN be used for target-based features, but val must use train statistics.
"""
import numpy as np


# Stage 1 top-20 feature names (for reference in evolution)
STAGE1_TOP20 = [
    "data_usage_high_frequency_low_saturation_high_saturation__trend1",
    "long_term_data_silence__vol",
    "arpu",
    "network_quality_complaint_count__trend5",
    "is_changed_from_non_delegated_to_delegated_payment",
    "flag_act_user__trend5",
    "cross_network_outgoing_call_ratio",
    "arpu__trend5",
    "mnp_port_out_risk__trend5",
    "card2_dou_hb",
    "is_sub_card__trend5",
    "primary_tariff_discount_rate__trend5",
    "family_cnt__vol",
    "churn_risk",
    "last_month_bill__trend1",
    "zztd_cnt",
    "flag_64__trend1",
    "credit_class_name",
    "card2_call_num_hb__trend1",
    "suspected_illegal_outbound_call_reach__vol",
]


# EVOLVE-BLOCK-START

def feature_engineer(X_train, y_train, X_val, y_val, feature_names):
    """
    Select and construct features from ~1604 engineered features for XGBoost.

    Args:
        X_train: np.ndarray (n_train, ~1604)
        y_train: np.ndarray (n_train,) - can use for target encoding (train only)
        X_val: np.ndarray (n_val, ~1604)
        y_val: np.ndarray (n_val,) - DO NOT use for feature computation
        feature_names: list of ~1604 feature name strings

    Returns:
        X_train_new, y_train, X_val_new, y_val
    """
    idx = {name: i for i, name in enumerate(feature_names)}

    def col(X, name):
        return X[:, idx[name]] if name in idx else np.zeros(X.shape[0])

    def build(X):
        feats = []

        # Stage 1 top-20 features as baseline
        for name in STAGE1_TOP20:
            feats.append(col(X, name))

        # Key interactions discovered by Stage 1 prior
        feats.append(col(X, "arpu") * col(X, "churn_risk"))
        feats.append(col(X, "arpu__trend5") * col(X, "churn_risk"))
        feats.append(col(X, "network_quality_complaint_count__trend5") * col(X, "mnp_port_out_risk__trend5"))
        feats.append(col(X, "cross_network_outgoing_call_ratio") * col(X, "churn_risk"))

        return np.column_stack(feats)

    X_train_new = build(X_train)
    X_val_new = build(X_val)

    return X_train_new, y_train, X_val_new, y_val

# EVOLVE-BLOCK-END


def run_search():
    return feature_engineer
