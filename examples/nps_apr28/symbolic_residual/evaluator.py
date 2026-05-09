import importlib.util
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from openevolve.evaluation_result import EvaluationResult

# 获取数据路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _compute_metrics(program_path):
    """加载程序并执行 BFGS 优化与指标计算"""
    # 动态加载进化中的代码
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # 加载数据集
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_residuals.csv.gz"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_residuals.csv.gz"))
    
    feature_cols = [c for c in train_df.columns if c != "residual"]
    X_train, y_train = train_df[feature_cols], train_df["residual"].values
    X_val, y_val = val_df[feature_cols], val_df["residual"].values

    # --- 核心改进：BFGS 参数优化 ---
    def loss_func(p):
        # 让 AI 的公式跑在当前的参数 p 下
        try:
            pred = program.predict_residual(X_train, p)
            return np.mean(np.abs(pred - y_train))
        except:
            return 1e10 # 容错：公式写错时返回极大的 loss

    # 从全 0 开始优化
    res = minimize(loss_func, np.zeros(program.MAX_NPARAMS), method='BFGS', tol=1e-3)
    best_params = res.x

    # --- 性能计算 ---
    train_pred = program.predict_residual(X_train, best_params)
    val_pred = program.predict_residual(X_val, best_params)

    def get_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_train = get_r2(y_train, train_pred)
    r2_val = get_r2(y_val, val_pred)

    # 载入基准线计算 AUC 提升
    baseline = np.load(os.path.join(DATA_DIR, "xgb_baseline.npz"))
    val_prob, val_y = baseline["val_prob"], baseline["val_y"]
    
    # 修正预测概率并计算提升
    corrected_prob = np.clip(val_prob + val_pred, 0.001, 0.999)
    baseline_auc = roc_auc_score(val_y, val_prob)
    corrected_auc = roc_auc_score(val_y, corrected_prob)
    delta_auc = corrected_auc - baseline_auc

    return {
        "r2_train": r2_train,
        "r2_val": r2_val,
        "delta_auc": delta_auc,
        "overfit": max(0.0, r2_train - r2_val),
        "baseline_auc": baseline_auc,
        "corrected_auc": corrected_auc
    }

def evaluate(program_path):
    """计算最终综合分数（得分越高越好）"""
    try:
        m = _compute_metrics(program_path)
        
        # 归一化得分 (0.0 到 1.0 之间)
        r2_score = max(0.0, min(1.0, m["r2_val"]))
        auc_score = min(1.0, max(0.0, m["delta_auc"]) * 50.0) # 0.02提升即满分
        penalty = min(1.0, m["overfit"])

        # 综合评分逻辑：权重分布为 40% 准确度, 40% 业务提升, -20% 过拟合惩罚
        combined_score = (0.4 * r2_score) + (0.4 * auc_score) - (0.2 * penalty)
        
        m["combined_score"] = max(0.0, combined_score)
        return EvaluationResult(metrics=m)

    except Exception as e:
        return EvaluationResult(metrics={"combined_score": 0.0, "error": str(e)})


def evaluate_stage2(program_path):
    """
    第二阶段评估：全量评估 + BFGS 优化。
    """
    return evaluate(program_path)


def evaluate_stage1(program_path):
    """
    第一阶段评估：快速冒烟测试。
    只检查代码是否能正常运行 100 行数据，不运行 BFGS。
    """
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # 检查结构要求
        if not hasattr(program, "predict_residual"):
            return EvaluationResult(metrics={"combined_score": 0.0, "error": "Missing predict_residual"})

        # 加载极少量数据做快速验证
        sample_df = pd.read_csv(os.path.join(DATA_DIR, "train_residuals.csv.gz"), nrows=100)
        feature_cols = [c for c in sample_df.columns if c != "residual"]
        X_sample = sample_df[feature_cols]

        # 模拟全 0 参数运行，检查是否报错
        params = np.zeros(getattr(program, "MAX_NPARAMS", 20))
        pred = program.predict_residual(X_sample, params)

        # 检查输出格式
        if not isinstance(pred, np.ndarray) or pred.shape[0] != 100:
            return EvaluationResult(metrics={"combined_score": 0.0, "error": "Invalid output shape/type"})
        
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            return EvaluationResult(metrics={"combined_score": 0.0, "error": "NaN/Inf in output"})

        # Stage 1 通过，给一个基础分引导进入 Stage 2
        return EvaluationResult(metrics={"combined_score": 0.2})

    except Exception as e:
        return EvaluationResult(metrics={"combined_score": 0.0, "error": str(e)})
    
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    program_path = os.path.join(current_dir, "initial_program.py")
    result = evaluate(program_path)
    print(result.to_dict())