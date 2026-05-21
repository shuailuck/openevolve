import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    roc_curve,
    auc
)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved')



def load_data(path, target_col):
    # pandas 会自动处理 .gz 压缩
    df = pd.read_csv(path)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

# ---------------------------------------------------------
# 2. 建模逻辑
# ---------------------------------------------------------
def run_modeling():
    # A. 加载数据 (直接使用 DataFrame)
    print("Loading data...")
    X_train, y_train = load_data(TRAIN_PATH, TARGET_COL)
    X_val, y_val = load_data(VAL_PATH, TARGET_COL)
    
    # B. 计算权重
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    spw = neg_count / max(pos_count, 1)
    if spw < 1:
        spw = 1.0
    
    print(f"Train Stats -> Positive: {pos_count}, Negative: {neg_count}")
    print(f"Scale Pos Weight: {spw:.2f}")

    # C. 定义模型
    # 直接支持 DataFrame，且会自动识别 X_train 的列名
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=spw,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=42,
    )

    # D. 训练
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    print(f"\nTraining took: {time.time() - start_time:.2f}s")

    # -----------------------------------------------------
    # 3. 评估分析
    # -----------------------------------------------------
    # 预测概率和标签
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    print("\n" + "="*40)
    print("CLASSIFICATION REPORT (Target = Original 0)")
    print(classification_report(y_val, y_pred))
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_val, y_pred))
    print("="*40)

    # AUC计算
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    # PR 曲线计算
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    
    # 寻找特定 Recall 水平下的 Precision
    target_recalls = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    pr_results = []
    print("\nRecall-Precision Analysis:")
    for tr in target_recalls:
        # 找到第一个满足 recall >= tr 的索引
        idx = np.where(recall >= tr)[0][-1]
        p_at_r = precision[idx]
        t_at_r = thresholds[idx] if idx < len(thresholds) else 1.0
        pr_results.append([tr, p_at_r, t_at_r])
        print(f"Recall: {tr:4.2f} | Precision: {p_at_r:.4f} | Threshold: {t_at_r:.4f}")

    # 保存结果
    pd.DataFrame(pr_results, columns=['Recall', 'Precision', 'Threshold']).to_csv(
        os.path.join(SAVE_DIR, "precision_recall_curve.csv"), index=False
    )

    # 特征重要性分析 (得益于 DataFrame，这里直接带列名)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(100)

    # 提取特征名列表
    top_100_feature_names = top_features.index.tolist()

    # 推荐存为文本文件，简单直观
    with open(os.path.join(SAVE_DIR, "top_100_features.txt"), "w", encoding="utf-8") as f:
        for feat in top_100_feature_names:
            f.write(feat + "\n")

    print(f"成功保存 {len(top_100_feature_names)} 个特征名到 top_100_features.txt")

    # 可视化 PR 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', label=f'PR Curve (AUC = {auc(recall, precision):.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(SAVE_DIR, "pr_curve.png"))
    print(f"\nResults saved to {SAVE_DIR}")

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    TARGET_COL = "target"  
    TRAIN_PATH = os.path.join(DATA_DIR, "train_derived.csv.gz")
    VAL_PATH = os.path.join(DATA_DIR, "val_derived.csv.gz")
    os.makedirs(SAVE_DIR, exist_ok=True)
    run_modeling()