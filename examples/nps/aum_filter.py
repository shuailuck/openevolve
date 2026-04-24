import numpy as np
import xgboost as xgb
from xgb_util import focal_loss


def filter_noisy_samples_aum(X_train, y_train, X_val, y_val,
                              n_estimators=300, aum_percentile=5):
    print("=" * 50)
    print("AUM 标签噪声识别")
    print("=" * 50)

    model = _train_aum_model(X_train, y_train, n_estimators)

    print("计算训练集 AUM...")
    train_aum = _compute_aum(model, X_train, y_train, n_estimators)
    print("计算验证集 AUM...")
    val_aum = _compute_aum(model, X_val, y_val, n_estimators)

    _print_aum_stats("训练集", train_aum, y_train)
    _print_aum_stats("验证集", val_aum, y_val)

    train_threshold = np.percentile(train_aum, aum_percentile)
    val_threshold = np.percentile(val_aum, aum_percentile)

    train_mask = train_aum >= train_threshold
    val_mask = val_aum >= val_threshold

    train_removed = (~train_mask).sum()
    val_removed = (~val_mask).sum()
    print(f"\n剔除阈值 (percentile={aum_percentile}%):")
    print(f"  训练集: AUM < {train_threshold:.4f}, 剔除 {train_removed} 个样本 "
          f"(label=0: {((~train_mask) & (y_train == 0)).sum()}, "
          f"label=1: {((~train_mask) & (y_train == 1)).sum()})")
    print(f"  验证集: AUM < {val_threshold:.4f}, 剔除 {val_removed} 个样本 "
          f"(label=0: {((~val_mask) & (y_val == 0)).sum()}, "
          f"label=1: {((~val_mask) & (y_val == 1)).sum()})")

    X_train_clean = X_train[train_mask].reset_index(drop=True)
    y_train_clean = y_train[train_mask].reset_index(drop=True)
    X_val_clean = X_val[val_mask].reset_index(drop=True)
    y_val_clean = y_val[val_mask].reset_index(drop=True)

    print(f"\n清洗后训练集: {len(y_train_clean)} 样本 (label=0: {(y_train_clean==0).sum()}, label=1: {(y_train_clean==1).sum()})")
    print(f"清洗后验证集: {len(y_val_clean)} 样本 (label=0: {(y_val_clean==0).sum()}, label=1: {(y_val_clean==1).sum()})")
    print("=" * 50)

    return X_train_clean, y_train_clean, X_val_clean, y_val_clean


def _train_aum_model(X_train, y_train, n_estimators):
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = neg_count / max(pos_count, 1) if pos_count > 0 else 1

    print(f"训练 AUM 模型 (n_estimators={n_estimators})...")
    model = xgb.XGBClassifier(
        objective=focal_loss,
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        seed=42,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.6,
        colsample_bytree=0.6,
        n_estimators=n_estimators
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def _compute_aum(model, X, y, n_estimators):
    n_samples = len(y)
    margins_sum = np.zeros(n_samples)
    y_arr = y.values.astype(int)
    idx = np.arange(n_samples)

    for i in range(1, n_estimators + 1):
        proba = model.predict_proba(X, iteration_range=(0, i))
        p_correct = proba[idx, y_arr]
        p_other = proba[idx, 1 - y_arr]
        margins_sum += (p_correct - p_other)

    return margins_sum / n_estimators


def _print_aum_stats(name, aum_values, y):
    print(f"\n{name} AUM 统计:")
    print(f"  整体: mean={aum_values.mean():.4f}, std={aum_values.std():.4f}, "
          f"min={aum_values.min():.4f}, max={aum_values.max():.4f}")
    for label in sorted(y.unique()):
        mask = y == label
        vals = aum_values[mask]
        print(f"  label={label}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, AUM<0 数量={int((vals < 0).sum())}")
