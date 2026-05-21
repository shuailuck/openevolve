import os
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from xgb_model import load_data
from interpret import show, preserve


SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved')

def run_modeling():
    # 1. 读取保存的特征名
    with open(os.path.join(SAVE_DIR, "top_100_features.txt"), "r", encoding="utf-8") as f:
        selected_features = [line.strip() for line in f.readlines()]
    # A. 加载数据 (直接使用 DataFrame)
    print("Loading data...")
    X_train, y_train = load_data(TRAIN_PATH, TARGET_COL)
    X_val, y_val = load_data(VAL_PATH, TARGET_COL)

    # 2. 初始化 EBM (可以指定交互项的数量)
    ebm = ExplainableBoostingClassifier(
        n_jobs=-1, 
        interactions=10, # 即使只有100个特征，限制交互项也能让运行飞快
        max_bins=128
    ) 

    X_train = X_train[selected_features]
    X_val = X_val[selected_features]

    # 3. 训练模型
    ebm.fit(X_train, y_train)

    # 4. 评估性能
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, ebm.predict_proba(X_val)[:, 1])
    print(f"EBM AUC: {auc:.4f}")
    ebm_global = ebm.explain_global()
    preserve(ebm_global, file_name=os.path.join(SAVE_DIR, "EBM_NPS_Analysis_Report.html"))

    print("完整的交互式报告已保存至当前目录下的: EBM_NPS_Analysis_Report.html")
    # show(ebm_global)
    # # print(ebm_global)
    # # 重要：在脚本末尾加上这句，防止脚本运行完直接关闭服务器导致网页打不开
    # print("按 Enter 键退出...")
    # input()



if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    TARGET_COL = "target"  
    TRAIN_PATH = os.path.join(DATA_DIR, "train_derived.csv.gz")
    VAL_PATH = os.path.join(DATA_DIR, "val_derived.csv.gz")
    os.makedirs(SAVE_DIR, exist_ok=True)
    run_modeling()