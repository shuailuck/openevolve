import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV

from xgb_util import focal_loss
from train_util import save_precision_recall_curve

warnings.filterwarnings('ignore')


def load_data(config):
    train_data_path = config['DATA']['train_data_path']
    val_data_path = config['DATA']['val_data_path']
    label_column = config['DATA']['label_column'].strip()

    print("加载训练数据...")
    train_df = pd.read_csv(train_data_path)
    print(f"训练集大小: {train_df.shape}")

    print("加载验证数据...")
    val_df = pd.read_csv(val_data_path)
    print(f"验证集大小: {val_df.shape}")

    feat_cols = [c for c in train_df.columns if c.startswith('feat_')]
    print(f"特征数量: {len(feat_cols)}")

    X_train = train_df[feat_cols]
    y_train = train_df[label_column]
    X_val = val_df[feat_cols]
    y_val = val_df[label_column]

    print(f"训练集标签分布:\n{y_train.value_counts()}")
    print(f"验证集标签分布:\n{y_val.value_counts()}")

    return X_train, y_train, X_val, y_val


def train_model(config, X_train, y_train, X_val, y_val):
    hyper_tuning = config['TRAIN'].getboolean('hyperparameter_tuning', fallback=False)
    model_save_path = config['TRAIN']['model_save_path']
    top_n_features = config['TRAIN'].getint('top_n_features', fallback=10)
    tree_method = config['TRAIN'].get('tree_method', fallback=None)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    model, model_train_time, y_prob, y_test = train_and_eval_model(
        X_train, y_train, X_val, y_val, hyper_tuning, tree_method
    )

    precision, recall, results = calc_recall_precision_auc_and_save(model_save_path, y_prob, y_test)

    save_precision_recall_curve(model_save_path, precision, recall, results)

    save_model_and_feature(X_train, model, model_save_path)

    show_important_feature(X_train, model, model_train_time, top_n_features)


def train_and_eval_model(X_train, y_train, X_test, y_test, hyper_tuning, tree_method):
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = neg_count / max(pos_count, 1) if pos_count > 0 else 1
    print(f"Training data shape: {X_train.shape}")
    print(f"Positive samples: {pos_count}, Negative samples: {neg_count}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    model_train_time = time.time()

    params = {
        'objective': focal_loss,
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'seed': 42,
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'n_estimators': 300
    }
    if tree_method is not None:
        params["tree_method"] = tree_method
        if 'gpu' in tree_method:
            params["device"] = 'cuda'
            params["predictor"] = 'gpu_predictor'

    model = select_xgboost_model(X_train, hyper_tuning, params, y_train)

    print("\nTraining model")
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
    except Exception as e:
        print(f"Warning: {e}. Using simple training.")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 0]
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return model, model_train_time, y_prob, y_test


def calc_recall_precision_auc_and_save(model_save_path, y_prob, y_test):
    print("Precision at Different Recall Levels: ")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=0)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC (target=0): {roc_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob, pos_label=0)

    target_recalls = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    results = []
    for target_recall in target_recalls:
        idx = np.argmin(np.abs(recall - target_recall))
        if recall[idx] < target_recall and idx < len(recall) - 1:
            idx += 1
        prec_at_rec = precision[idx]
        thresh = thresholds[idx] if idx < len(thresholds) else 0.0
        results.append((target_recall, prec_at_rec, thresh))
        print(f"Recall: {target_recall:4.2f} | Precision: {prec_at_rec:.4f} | Threshold: {thresh:.4f}")

    recall_precision_path = os.path.splitext(model_save_path)[0] + "_recall_precision.csv"
    results_df = pd.DataFrame(results, columns=['Recall', 'Precision', 'Threshold'])
    results_df.to_csv(recall_precision_path, index=False)
    print(f"Recall-Precision data saved to {recall_precision_path}")
    return precision, recall, results


def save_model_and_feature(X, model, model_save_path):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    feature_names_path = os.path.splitext(model_save_path)[0] + "_features.txt"
    with open(feature_names_path, 'w') as f:
        f.write("\n".join(X.columns))
    print(f"Feature names saved to {feature_names_path}")


def show_important_feature(X, model, model_train_time, top_n_features):
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    print(f"Top {top_n_features} features")
    for i in range(min(top_n_features, len(X.columns))):
        print(f"{i + 1}. {X.columns[sorted_idx[i]]}: {feature_importances[sorted_idx[i]]:.4f}")
    print(f"model train time is :{time.time() - model_train_time}")


def select_xgboost_model(X_train, hyper_tuning, params, y_train):
    if hyper_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200]
        }
        model = xgb.XGBClassifier(**params)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        params.update(grid_search.best_params_)
        model = grid_search.best_estimator_
    else:
        model = xgb.XGBClassifier(**params)
    return model
