import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_curve, auc


# EVOLVE-BLOCK-START
def engineer_features(train_df, val_df):
    """
    Construct high-order features via symbolic regression.

    Args:
        train_df: Training DataFrame with feat_0..feat_499 and target columns.
        val_df: Validation DataFrame with the same columns.

    Returns:
        (train_df, val_df) with additional engineered feature columns.
        You may drop original feat_* columns if they are redundant or noisy.
        Do NOT drop the target column.
        Use pd.concat to add new columns to avoid DataFrame fragmentation.
    """
    results = []
    for df in [train_df, val_df]:
        new_feats = pd.DataFrame(
            {
                "eng_0": df["feat_0"].values * df["feat_1"].values,
                "eng_1": df["feat_2"].values / (df["feat_3"].values + 1e-8),
                "eng_2": np.sqrt(np.abs(df["feat_4"].values)),
            },
            index=df.index,
        )
        results.append(pd.concat([df, new_feats], axis=1))
    return results[0], results[1]
# EVOLVE-BLOCK-END


def focal_loss(y_true, y_pred, alpha=0.5, gamma=2):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    gradient = -alpha * (1 - p_t) ** gamma * (y_true - y_pred)
    hessian = alpha * (1 - p_t) ** gamma * y_pred * (1 - y_pred)
    return gradient, hessian


def run(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train_data_500f.csv.gz"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_data_500f.csv.gz"))

    train_df, val_df = engineer_features(train_df, val_df)

    for df in [train_df, val_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = neg_count / max(pos_count, 1)

    model = XGBClassifier(
        objective=focal_loss,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        seed=42,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.6,
        colsample_bytree=0.6,
        n_estimators=300,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 0]

    fpr, tpr, _ = roc_curve(y_val, y_prob, pos_label=0)
    roc_auc = float(auc(fpr, tpr))
    f1_macro = float(f1_score(y_val, y_pred, average="macro"))
    num_eng = len([c for c in feature_cols if not c.startswith("feat_")])

    return {"roc_auc": roc_auc, "f1_macro": f1_macro, "num_eng_features": num_eng}


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    results = run(data_dir)
    print(json.dumps(results, indent=2))
