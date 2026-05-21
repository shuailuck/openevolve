"""
Symbolic feature construction via gplearn for NPS prediction.

Groups 2406 features into 401 base-field groups (6 monthly values each),
ranks groups by XGBoost importance, then applies gplearn.SymbolicTransformer
to the top groups to generate higher-order features.

Two grouping strategies are supported:
  1. within-field (default): each group = 6 monthly values of one base field
  2. cross-field (--pair-top N): pair up top-N bases → 12-feature groups

Overfitting is controlled by:
  - Splitting training data: 60% for GP evolution, 40% for feature validation
  - Only keeping features with individual AUC > 0.51 on the holdout
  - Filtering constant and redundant (corr > 0.99) features

Usage:
    python run.py --top-groups 50 --n-components 5 --generations 15
    python run.py --top-groups 30 --pair-top 15  # cross-field pairs
"""

import argparse
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from gplearn.functions import make_function
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
SAVE_DIR = HERE / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature grouping
# ---------------------------------------------------------------------------

def build_feature_groups(df):
    """Group columns by base field.

    Each group has 6 monthly values sorted chronologically:
    {base}_T_5_M, ..., {base}_T_1_M, {base}
    """
    cols = [c for c in df.columns if c != TARGET_COL]
    time_pattern = re.compile(r"^(.+)_T_\d+_M$")

    base_to_cols = {}
    for c in cols:
        m = time_pattern.match(c)
        base = m.group(1) if m else c
        base_to_cols.setdefault(base, []).append(c)

    month_pattern = re.compile(r"_T_(\d+)_M$")

    groups = []
    for base, group_cols in base_to_cols.items():
        def sort_key(c):
            if c == base:
                return (1,)
            m = month_pattern.search(c)
            if m:
                return (0, -int(m.group(1)))
            return (0, 0)

        group_cols.sort(key=sort_key)
        groups.append({"name": base, "cols": group_cols})

    return groups


def build_pair_groups(ranked_bases, groups, top_n=20):
    """Create groups from pairs of top base features (12 columns each).

    Cross-field pairs allow GP to discover interactions between different
    business metrics that XGBoost might miss.
    """
    pair_groups = []
    for i in range(min(top_n, len(ranked_bases))):
        for j in range(i + 1, min(top_n, len(ranked_bases))):
            base_i, base_j = ranked_bases[i][0], ranked_bases[j][0]
            g_i = next(g for g in groups if g["name"] == base_i)
            g_j = next(g for g in groups if g["name"] == base_j)
            pair_groups.append({
                "name": f"pair_{base_i[:30]}__{base_j[:30]}",
                "cols": g_i["cols"] + g_j["cols"],
            })
    return pair_groups


# ---------------------------------------------------------------------------
# Group ranking
# ---------------------------------------------------------------------------

def rank_groups(X_train, y_train, X_val, y_val, feature_cols, groups):
    """Quick XGBoost -> aggregate importance to base-field level -> rank."""
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / max(pos, 1)
    if spw < 1:
        spw = 1.0

    model = XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        scale_pos_weight=spw, n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.6, colsample_bytree=0.6,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    col_imp = dict(zip(feature_cols, model.feature_importances_))
    group_importances = {}
    for g in groups:
        total = sum(col_imp.get(c, 0.0) for c in g["cols"])
        group_importances[g["name"]] = total

    ranked = sorted(group_importances.items(), key=lambda x: x[1], reverse=True)
    return ranked, model


# ---------------------------------------------------------------------------
# XGBoost helpers
# ---------------------------------------------------------------------------

def make_xgb(n_estimators, max_depth, spw):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        scale_pos_weight=spw, n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.01, subsample=0.6, colsample_bytree=0.6,
        random_state=RANDOM_STATE, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# GP feature generation
# ---------------------------------------------------------------------------

def _group_ts_delay(x):
    try:
        if x.ndim == 2:
            res = np.roll(x, 1, axis=1)
            res[:, 0] = np.mean(x, axis=1)
            return res
        elif x.ndim == 1:
            res = np.roll(x, 1)
            res[0] = x[0]
            return res
    except Exception:
        pass
    return x


def _group_ts_delta(x):
    try:
        if x.ndim == 2:
            return x - np.roll(x, 1, axis=1)
        elif x.ndim == 1:
            return x - np.roll(x, 1)
    except Exception:
        pass
    return x

group_ts_delay = make_function(function=_group_ts_delay, name='ts_delay', arity=1)
group_ts_delta = make_function(function=_group_ts_delta, name='ts_delta', arity=1)

FUNCTION_SET = ["add", "sub", "mul", "div", "log", "abs", "neg", group_ts_delay, group_ts_delta]


def extract_formula(transformer):
    """Best program string from a fitted SymbolicTransformer."""
    if hasattr(transformer, "_best_programs") and transformer._best_programs:
        return str(transformer._best_programs[0])
    return "<no program>"


def _individual_auc_safe(y_true, scores):
    """AUC in either direction (max(auc, 1-auc)), robust to edge cases."""
    try:
        auc_val = roc_auc_score(y_true, scores)
        return max(auc_val, 1.0 - auc_val)
    except ValueError:
        return 0.5


def run_gp_for_groups(
    groups_to_process, df_train_A, y_train_A, df_train_B, y_train_B,
    df_val, n_components, generations, population_size,
):
    from gplearn.genetic import SymbolicTransformer

    all_train_new = []
    all_val_new = []
    col_names = []
    formulas = []
    success_count = 0

    for idx, g in enumerate(groups_to_process):
        cols = g["cols"]
        X_A = df_train_A[cols].values.astype(np.float64)
        X_B = df_train_B[cols].values.astype(np.float64)
        X_V = df_val[cols].values.astype(np.float64)
        n_features_in = X_A.shape[1]

        t0 = time.time()
        try:
            hof = min(100, population_size // 2)
            st = SymbolicTransformer(
                population_size=population_size,
                generations=generations,
                hall_of_fame=hof,
                tournament_size=min(20, max(2, population_size // 5)),
                function_set=FUNCTION_SET,
                metric="spearman",
                parsimony_coefficient=0.005,
                n_components=n_components,
                random_state=RANDOM_STATE,
                n_jobs=1,       
                verbose=0,
            )
            st.fit(X_A, y_train_A)

            X_B_t = st.transform(X_B)
            keep_mask = np.zeros(X_B_t.shape[1], dtype=bool)
            
            for j in range(X_B_t.shape[1]):
                col = X_B_t[:, j]
                if np.std(col) > 1e-8:
                    keep_mask[j] = _individual_auc_safe(y_train_B, col) > 0.51

            n_kept = keep_mask.sum()
            if n_kept == 0:
                elapsed = time.time() - t0
                print(f"  [{idx+1:3d}/{len(groups_to_process)}] {g['name']:<45s} SKIP ({elapsed:.1f}s)  全部未通过Holdout验证")
                continue

            X_A_t = st.transform(X_A)
            X_V_t = st.transform(X_V)
            X_tr_all = np.vstack([X_A_t, X_B_t]) 
            
            final_keep = np.copy(keep_mask)
            X_orig_all = np.vstack([X_A, X_B])  
            
            for j in range(X_tr_all.shape[1]):
                if not final_keep[j]:
                    continue
                if np.std(X_tr_all[:, j]) <= 1e-8:
                    final_keep[j] = False
                    continue
                for k in range(n_features_in):
                    if np.std(X_orig_all[:, k]) < 1e-8:
                        continue
                    if np.abs(np.corrcoef(X_tr_all[:, j], X_orig_all[:, k])[0, 1]) > 0.99:
                        final_keep[j] = False
                        break

            if final_keep.sum() == 0:
                print(f"  [{idx+1:3d}/{len(groups_to_process)}] {g['name']:<45s} SKIP ({time.time()-t0:.1f}s)  后置过滤剔除了所有高冗余特征")
                continue

            X_tr_filtered = X_tr_all[:, final_keep]
            X_val_filtered = X_V_t[:, final_keep]

            all_train_new.append(X_tr_filtered)
            all_val_new.append(X_val_filtered)

            survived_indices = np.where(final_keep)[0]
            for rank_id, comp_idx in enumerate(survived_indices):
                program = st._best_programs[comp_idx]
                formula_str = str(program) if program is not None else "<error>"
                
                for idx_col in reversed(range(len(cols))):
                    formula_str = formula_str.replace(f"X{idx_col}", f"[{cols[idx_col]}]")
                
                col_n = f"gp_{g['name'][:40]}_c{comp_idx}"
                col_names.append(col_n)
                formulas.append((col_n, formula_str))

            success_count += 1
            elapsed = time.time() - t0
            print(f"  [{idx+1:3d}/{len(groups_to_process)}] {g['name']:<45s} OK ({elapsed:.1f}s) 精选出 {X_tr_filtered.shape[1]} 个高阶时序特征")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{idx+1:3d}/{len(groups_to_process)}] {g['name']:<45s} FAIL ({elapsed:.1f}s): {e}")

    return all_train_new, all_val_new, col_names, formulas, success_count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="gplearn symbolic feature construction")
    p.add_argument("--top-groups", type=int, default=50,
                   help="Within-field groups: top N base features (default: 50)")
    p.add_argument("--pair-top", type=int, default=0,
                   help="Cross-field pairs: pair top N bases (0=disabled)")
    p.add_argument("--n-components", type=int, default=5,
                   help="GP candidate features per group (default: 5)")
    p.add_argument("--generations", type=int, default=15,
                   help="GP generations per group (default: 15)")
    p.add_argument("--population-size", type=int, default=500,
                   help="GP population size (default: 500)")
    p.add_argument("--xgb-estimators", type=int, default=500,
                   help="Final XGBoost n_estimators (default: 500)")
    p.add_argument("--xgb-depth", type=int, default=5,
                   help="Final XGBoost max_depth (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()

    train_path = DATA_DIR / "train.csv.gz"
    val_path = DATA_DIR / "val.csv.gz"

    # ---- 1. Load data -------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    y_train_full = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    print(f"  Train: {train_df.shape}, Val: {val_df.shape}")

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    # ---- 2. Train/validation split for GP -----------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Splitting training data for GP")
    print("=" * 60)

    rng = np.random.RandomState(RANDOM_STATE)
    n_train = len(y_train_full)
    perm = rng.permutation(n_train)
    split_a = int(n_train * 0.6)
    idx_A = perm[:split_a]
    idx_B = perm[split_a:]

    df_train_A = train_df.iloc[idx_A].reset_index(drop=True)
    df_train_B = train_df.iloc[idx_B].reset_index(drop=True)
    y_train_A = y_train_full[idx_A]
    y_train_B = y_train_full[idx_B]

    X_train_A = df_train_A[feature_cols].values.astype(np.float64)
    X_train_B = df_train_B[feature_cols].values.astype(np.float64)
    X_val = val_df[feature_cols].values.astype(np.float64)

    # Full training data (aligned order for final XGBoost)
    X_train_full = np.vstack([X_train_A, X_train_B])
    y_train_aligned = np.concatenate([y_train_A, y_train_B])

    print(f"  train_A (GP evolution):     {len(idx_A)} samples")
    print(f"  train_B (feature selection): {len(idx_B)} samples")
    print(f"  val (final evaluation):      {len(y_val)} samples")

    # ---- 3. Build groups + rank ---------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Feature groups & ranking")
    print("=" * 60)
    groups = build_feature_groups(train_df)
    total_in_groups = sum(len(g["cols"]) for g in groups)
    print(f"  Base-field groups: {len(groups)}, columns: {total_in_groups}")
    assert total_in_groups == 2406

    ranked, rank_model = rank_groups(X_train_A, y_train_A, X_train_B, y_train_B,
                                     feature_cols, groups)
    print(f"  Top 15 bases by importance:")
    for i, (name, imp) in enumerate(ranked[:15], 1):
        print(f"    {i:2d}. {name:<50s} {imp:.5f}")

    # ---- 4. Baselines -------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Computing baselines")
    print("=" * 60)

    y_prob_quick = rank_model.predict_proba(X_val)[:, 1]
    quick_auc = roc_auc_score(y_val, y_prob_quick)
    print(f"  Quick baseline (200 trees, depth=4): {quick_auc:.4f}")

    pos = (y_train_aligned == 1).sum()
    neg = (y_train_aligned == 0).sum()
    spw = neg / max(pos, 1)
    if spw < 1:
        spw = 1.0

    base_full = make_xgb(args.xgb_estimators, args.xgb_depth, spw)
    t0 = time.time()
    base_full.fit(X_train_full, y_train_aligned,
                  eval_set=[(X_val, y_val)], verbose=False)
    full_baseline_auc = roc_auc_score(y_val, base_full.predict_proba(X_val)[:, 1])
    print(f"  Full baseline ({args.xgb_estimators} trees, depth={args.xgb_depth}): "
          f"{full_baseline_auc:.4f}  ({time.time() - t0:.0f}s)")

    # ---- 5. GP: within-field groups -----------------------------------------
    print("\n" + "=" * 60)
    print(f"Step 5: GP on top {args.top_groups} within-field groups")
    print(f"  n_components={args.n_components}, generations={args.generations}, "
          f"pop_size={args.population_size}")
    print("=" * 60)

    top_names = [name for name, _ in ranked[: args.top_groups]]
    within_groups = [g for g in groups if g["name"] in top_names]

    all_tr, all_va, col_n, form, succ = run_gp_for_groups(
        within_groups, df_train_A, y_train_A, df_train_B, y_train_B,
        val_df, args.n_components, args.generations, args.population_size,
    )

    all_train_new = list(all_tr)
    all_val_new = list(all_va)
    all_col_names = list(col_n)
    all_formulas = list(form)

    # ---- 6. Optional: cross-field pair groups -------------------------------
    if args.pair_top > 0:
        print("\n" + "=" * 60)
        print(f"Step 6: GP on cross-field pairs (top {args.pair_top} bases)")
        print("=" * 60)

        pair_groups = build_pair_groups(ranked, groups, top_n=args.pair_top)
        max_pairs = min(len(pair_groups), 200)
        pair_groups = pair_groups[:max_pairs]
        print(f"  {len(pair_groups)} pair groups (12 features each)")

        pair_comp = max(1, args.n_components // 2)
        pair_gen = max(5, args.generations // 2)
        pair_pop = max(100, args.population_size // 3)

        all_tr_p, all_va_p, col_n_p, form_p, succ_p = run_gp_for_groups(
            pair_groups, df_train_A, y_train_A, df_train_B, y_train_B,
            val_df, pair_comp, pair_gen, pair_pop,
        )
        all_train_new.extend(all_tr_p)
        all_val_new.extend(all_va_p)
        all_col_names.extend(col_n_p)
        all_formulas.extend(form_p)
        print(f"  Cross-field pairs: {succ_p}/{len(pair_groups)} successful")

    # ---- 7. Combine features & Advanced Clipping (Fixed & Linked) -----------
    print("\n" + "=" * 60)
    print("Step 7: Augmented feature matrix & Class 1 Oriented Selection")
    print("=" * 60)

    total_gp = sum(a.shape[1] for a in all_train_new)
    if total_gp == 0:
        print("  ERROR: No GP features survived filtering.")
        print("  Try: --n-components 10 --top-groups 100")
        sys.exit(1)

    X_train_aug = np.hstack([X_train_full] + all_train_new)
    X_val_aug = np.hstack([X_val] + all_val_new)
    X_train_aug = np.nan_to_num(X_train_aug, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_aug = np.nan_to_num(X_val_aug, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  合并后初始总特征数: {X_train_aug.shape[1]} (原始: {X_train_full.shape[1]}, 衍生: {total_gp})")

    # 创建完整的合并特征名，用于裁剪阶段追溯
    merged_feature_names = list(feature_cols) + all_col_names

    # 1a. 剔除常数项与超低方差
    selector = VarianceThreshold(threshold=1e-5)
    X_train_s1 = selector.fit_transform(X_train_aug)
    X_val_s1 = selector.transform(X_val_aug)
    retained_indices = selector.get_support(indices=True)
    current_cols = [merged_feature_names[i] for i in retained_indices]
    print(f"  -> [阶段 1a] 移除常数/低方差特征后，剩余特征数: {len(current_cols)}")

    # 1b. 快速抽样消除近乎完全重合的特征 (相关系数 > 0.98)
    sample_size = min(10000, X_train_s1.shape[0])
    sample_idx = np.random.choice(X_train_s1.shape[0], sample_size, replace=False)
    corr_matrix = np.abs(np.corrcoef(X_train_s1[sample_idx, :], rowvar=False))

    upper_tri = np.triu(corr_matrix, k=1)
    to_drop_corr = [i for i in range(upper_tri.shape[1]) if any(upper_tri[:, i] > 0.98)]

    keep_indices_s1 = [i for i in range(X_train_s1.shape[1]) if i not in to_drop_corr]
    X_train_s2 = X_train_s1[:, keep_indices_s1]
    X_val_s2 = X_val_s1[:, keep_indices_s1]
    current_cols = [current_cols[i] for i in keep_indices_s1]
    print(f"  -> [阶段 1b] 移除高度共线性冗余特征后，剩余特征数: {len(current_cols)}")

    # 1c. 基于 LightGBM 的 Class 1 敏感特征重要性精选
    y_train_series = pd.Series(y_train_aligned)
    class_counts = y_train_series.value_counts()
    weight_for_1 = class_counts.max() / class_counts.min()
    sample_weights = np.where(y_train_aligned == 1, weight_for_1, 1.0)

    lgb_train = lgb.Dataset(X_train_s2, label=y_train_aligned, weight=sample_weights)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'seed': RANDOM_STATE,
        'verbose': -1
    }

    print("  正在通过多核 LightGBM 提取特征对 Class 1（不满意）的贡献度评估...")
    lgbm_model = lgb.train(lgb_params, lgb_train, num_boost_round=150)
    lgb_importances = lgbm_model.feature_importance(importance_type='gain')

    threshold_gain = np.mean(lgb_importances) * 0.05
    useful_indices = np.where(lgb_importances > threshold_gain)[0]

    # 兜底：保留至少 150 个特征，但不超过剩余总特征数
    min_keep = min(150, X_train_s2.shape[1])
    if len(useful_indices) < min_keep:
        useful_indices = np.argsort(lgb_importances)[::-1][:min_keep]

    # 【深度对齐】最终裁剪后的黄金特征阵列与列名映射
    X_train_final = X_train_s2[:, useful_indices]
    X_val_final = X_val_s2[:, useful_indices]
    final_feature_names = [current_cols[i] for i in useful_indices]

    print(f"  -> [阶段 1c] LightGBM 贡献度精选完成！")
    print(f"  最终注入最终 XGBoost 的黄金特征总数: {X_train_final.shape[1]}")

    # ---- 8. Final XGBoost (Updated to use final pruned matrices) ------------
    print("\n" + "=" * 60)
    print("Step 8: Final XGBoost on augmented & pruned features")
    print("=" * 60)

    final_model = make_xgb(args.xgb_estimators, args.xgb_depth, spw)
    t0 = time.time()
    # 核心级联更新：使用过滤剪裁后的黄金数据集进行最终拟合
    final_model.fit(X_train_final, y_train_aligned,
                    eval_set=[(X_val_final, y_val)], verbose=100)
    print(f"\n  Training: {time.time() - t0:.0f}s")

    # ---- 9. Evaluate (Updated matrices) -------------------------------------
    print("\n" + "=" * 60)
    print("Step 9: Evaluation")
    print("=" * 60)

    y_prob = final_model.predict_proba(X_val_final)[:, 1]
    y_pred = final_model.predict(X_val_final)
    final_auc = roc_auc_score(y_val, y_prob)

    print(f"\n  Quick baseline AUC:  {quick_auc:.4f}")
    print(f"  Full baseline AUC:   {full_baseline_auc:.4f}")
    print(f"  Final AUC (with GP):  {final_auc:.4f}")
    print(f"  Improvement:          {final_auc - full_baseline_auc:+.4f}")

    print("\n" + "-" * 40)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("-" * 40)

    # ---- 10. Feature importance analysis (Aligned with final features) ------
    print("\n" + "=" * 60)
    print("Step 10: Feature importance")
    print("=" * 60)

    importances = final_model.feature_importances_
    gp_mask = np.array([c.startswith("gp_") for c in final_feature_names])
    sorted_idx = np.argsort(importances)[::-1]
    
    # 适配保留特征可能小于 100 的场景
    top_k_count = min(100, len(final_feature_names))
    top_k_indices = sorted_idx[:top_k_count]

    gp_in_top20 = [idx for idx in top_k_indices[:20] if gp_mask[idx]]
    gp_in_top50 = [idx for idx in top_k_indices[:50] if gp_mask[idx]]
    gp_in_top100 = [idx for idx in top_k_indices if gp_mask[idx]]

    print(f"  GP features in top 20/50/100:  "
          f"{len(gp_in_top20)}/{len(gp_in_top50)}/{len(gp_in_top100)}")

    gp_indices = np.where(gp_mask)[0]
    if len(gp_indices) > 0:
        avg_rank = np.mean([np.where(sorted_idx == i)[0][0] + 1 for i in gp_indices])
        print(f"  Avg rank of GP features: {avg_rank:.1f} / {len(final_feature_names)}")

    if gp_in_top20:
        print("\n  GP features in top 20:")
        for idx in gp_in_top20:
            print(f"    {final_feature_names[idx]:<60s} importance={importances[idx]:.5f}")

    # ---- 11. Save outputs ---------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 11: Saving outputs")
    print("=" * 60)

    with open(SAVE_DIR / "top_100_features.txt", "w", encoding="utf-8") as f:
        for idx in top_k_indices:
            tag = "[GP]" if gp_mask[idx] else "[ORIG]"
            f.write(f"{final_feature_names[idx]}\t{importances[idx]:.6f}\t{tag}\n")

    with open(SAVE_DIR / "formulas.txt", "w", encoding="utf-8") as f:
        f.write("Best GP formulas per group:\n" + "=" * 60 + "\n\n")
        for name, formula in all_formulas:
            f.write(f"--- {name} ---\n{formula}\n\n")

    with open(SAVE_DIR / "summary.txt", "w") as f:
        f.write(f"Quick baseline AUC:  {quick_auc:.4f}\n")
        f.write(f"Full baseline AUC:   {full_baseline_auc:.4f}\n")
        f.write(f"Final AUC (with GP): {final_auc:.4f}\n")
        f.write(f"Improvement:         {final_auc - full_baseline_auc:+.4f}\n")
        f.write(f"\nConfig: top_groups={args.top_groups}, "
                f"pair_top={args.pair_top}, "
                f"n_components={args.n_components}, "
                f"generations={args.generations}, "
                f"pop_size={args.population_size}\n")
        f.write(f"GP successful groups: {len(all_formulas)}\n")
        f.write(f"GP features total (before lgb prune):    {total_gp}\n")
        f.write(f"Final features total (after lgb prune): {X_train_final.shape[1]}\n")
        f.write(f"GP in top 20/50/100:  "
                f"{len(gp_in_top20)}/{len(gp_in_top50)}/{len(gp_in_top100)}\n")

    print(f"   -> {SAVE_DIR / 'summary.txt'}")
    print(f"  -> {SAVE_DIR / 'top_100_features.txt'}")
    print(f"  -> {SAVE_DIR / 'formulas.txt'}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()