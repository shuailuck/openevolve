import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_curve, auc, classification_report, confusion_matrix
from utils import calc_recall_precision_auc_and_save, save_precision_recall_curve


def xgb_model(X_train, y_train, X_val, y_val):
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = pos_count / max(neg_count, 1)

    model = XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        scale_pos_weight=scale_pos_weight, seed=42,
        max_depth=5, learning_rate=0.01, subsample=0.6,
        colsample_bytree=0.6, n_estimators=500,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 0]

    # fpr, tpr, _ = roc_curve(y_val, y_prob, pos_label=0)
    # roc_auc = float(auc(fpr, tpr))
    # f1_macro = float(f1_score(y_val, y_pred, average="macro"))
    # print(f'roc_auc: {roc_auc}, f1_macro: {f1_macro}')
    return y_pred, y_prob

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved')
    os.makedirs(SAVE_PATH, exist_ok=True)
    X_train = np.load(os.path.join(DATA_DIR, "X_train_all.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val_all.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_pred, y_prob = xgb_model(X_train, y_train, X_val, y_val)
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    precision, recall, results = calc_recall_precision_auc_and_save(SAVE_PATH, y_prob, y_val)
    save_precision_recall_curve(SAVE_PATH, precision, recall, results)

