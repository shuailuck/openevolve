"""
Prepare Stage 2 (Feature Engineering Evolution).

Workflow:
1. Collect Stage 1 best symbolic expressions from all groups.
2. For each group, run BFGS to recover optimized parameters (reading full CSV on demand).
3. Translate x[:,i] to real feature names and build readable formulas.
4. Save stage1_expressions.json, stage1_formulas.txt, and per-group params as JSON.
5. Generate stage2_initial_program.py that embeds Stage 1 formulas as engineered features.

Run after Stage 1 completes (i.e., after run_stage1.sh finishes).
"""
import os
import re
import json
import importlib.util
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data lives in the parent directory (examples/nps_apr28/data)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
STAGE1_RESULTS_DIR = os.path.join(BASE_DIR, "stage1_results")
STAGE1_INITIAL_PATH = os.path.join(BASE_DIR, "stage1_initial_program.py")


def extract_evolve_block(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(
        r"# EVOLVE-BLOCK-START\s*\n(.*?)# EVOLVE-BLOCK-END",
        content,
        re.DOTALL,
    )
    return match.group(1).strip() if match else None


def expression_to_formula(expr, feature_names):
    """Convert Stage 1 func(x, params) into a single readable math formula."""

    def _feat(m):
        i = int(m.group(1))
        return feature_names[i] if i < len(feature_names) else m.group(0)

    # Case 1: loop-based linear model
    loop_m = re.search(r"for\s+\w+\s+in\s+range\(min\((\d+)", expr)
    bias_m = re.search(r"params\[(\d+)\]\s*\*?\s*[^[]*$", expr, re.MULTILINE)
    if loop_m and bias_m:
        n_terms = int(loop_m.group(1))
        terms = [
            f"w{i} * {feature_names[i]}"
            for i in range(min(n_terms, len(feature_names)))
        ]
        return "logit = b + " + " + ".join(terms)

    # Case 2: evolved expression — parse into one continuous formula
    body = expr
    body = re.sub(r"def\s+func\s*\([^)]*\)\s*:", "", body)
    body = re.sub(r'"""[\s\S]*?"""', "", body)
    body = re.sub(r"#[^\n]*", "", body)
    body = re.sub(r"return\s+\w+\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"x\[:,\s*(\d+)\]", _feat, body)
    body = re.sub(r"params\[(\d+)\]", r"w\1", body)
    body = body.replace("np.", "")

    lines = [l.strip() for l in body.strip().splitlines() if l.strip()]

    intermediates = {}
    initial = None
    terms = []

    for line in lines:
        m = re.match(r"logit\s*=\s*(.+)", line)
        if m:
            initial = m.group(1).strip()
            continue
        m = re.match(r"logit\s*\+=\s*(.+)", line)
        if m:
            terms.append(m.group(1).strip())
            continue
        m = re.match(r"(\w+)\s*=\s*(.+)", line)
        if m:
            intermediates[m.group(1).strip()] = m.group(2).strip()

    for var, val in intermediates.items():
        if initial:
            initial = initial.replace(var, f"({val})")
        terms = [t.replace(var, f"({val})") for t in terms]

    if initial and re.match(r"w\d+$", initial):
        initial = "b"

    all_terms = ([initial] if initial else []) + terms
    return "logit = " + " + ".join(all_terms)


def optimize_group_params(program_path, group_cols):
    """Run BFGS on a saved Stage 1 best program to recover optimal params.
    Reads full CSV and extracts group columns on the fly."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_derived.csv.gz"))
    train_df["target"] = train_df["target"].map({0: 1, 1: 0})
    X_train = train_df[group_cols].values.astype(np.float32)
    y_train = train_df["target"].values.astype(np.int32)
    max_nparams = getattr(program, "MAX_NPARAMS", 20)

    def loss_func(p):
        try:
            pred = program.func(X_train, p)
            if not isinstance(pred, np.ndarray) or pred.shape != y_train.shape:
                return 1e10
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e10
            return np.mean((pred - y_train) ** 2)
        except Exception:
            return 1e10

    res = minimize(loss_func, np.zeros(max_nparams), method="BFGS", tol=1e-3)
    return res.x


def generate_stage2_initial(expressions, output_path):
    """Generate stage2_initial_program.py with embedded Stage 1 formulas."""

    header = '''import os
import json
import importlib.util
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_curve, auc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data lives in the parent directory (examples/nps_apr28/data)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
STAGE1_RESULTS_DIR = os.path.join(BASE_DIR, "stage1_results")

# ---------------------------------------------------------------------------
# Load Stage 1 discovered formulas
# ---------------------------------------------------------------------------
with open(os.path.join(DATA_DIR, "stage1_groups.json"), "r", encoding="utf-8") as f:
    _GROUPS = json.load(f)["groups"]

_STAGE1_MODULES = {}
_STAGE1_PARAMS = {}

for _g in _GROUPS:
    _gid = _g["id"]
    _mod_path = os.path.join(BASE_DIR, "stage1_results", f"group_{_gid}_best.py")
    _param_path = os.path.join(BASE_DIR, "stage1_results", f"group_{_gid}_params.json")
    if os.path.exists(_mod_path) and os.path.exists(_param_path):
        _spec = importlib.util.spec_from_file_location(f"group_{_gid}", _mod_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _STAGE1_MODULES[_gid] = _mod
        with open(_param_path, "r", encoding="utf-8") as _pf:
            _STAGE1_PARAMS[_gid] = np.array(json.load(_pf)["params"], dtype=np.float32)


def _apply_stage1_formula(df, group_info):
    """Apply a single Stage 1 symbolic formula to a DataFrame."""
    gid = group_info["id"]
    mod = _STAGE1_MODULES.get(gid)
    params = _STAGE1_PARAMS.get(gid)
    if mod is None or params is None:
        return None
    cols = group_info["cols"]
    x = df[cols].values.astype(np.float32)
    return mod.func(x, params)


# ---------------------------------------------------------------------------
# Stage 2: Feature Engineering (to be evolved by OpenEvolve)
# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START
def engineer_features(train_df, val_df):
    """
    Construct high-order features guided by Stage 1 symbolic expressions.

    Stage 1 discovered powerful symbolic formulas for top feature groups.
    Use them as engineered features, then add your own interactions,
    nonlinear transforms, and aggregations.

    Args:
        train_df: Training DataFrame with all original features + target.
        val_df: Validation DataFrame with the same columns.

    Returns:
        (train_df, val_df) with additional engineered feature columns.
        Do NOT drop the target column.
    """
    results = []
    for df in [train_df, val_df]:
        new_feats = {}

        # --- Prior knowledge: Stage 1 symbolic formulas ---
        # These formulas were discovered by OpenEvolve in Stage 1.
        # They capture nonlinear patterns within each feature group.
        # It is strongly recommended to KEEP them as baseline engineered features.
'''

    # Embed each Stage 1 formula as an engineered feature
    body_lines = []
    for expr_info in expressions:
        base = expr_info["base_name"]
        gid = expr_info["group_id"]
        body_lines.append(f"        # Group {gid} ({base})")
        body_lines.append(f"        _v = _apply_stage1_formula(df, _GROUPS[{gid}])")
        body_lines.append(f"        if _v is not None:")
        body_lines.append(f"            new_feats['eng_s1_{base}'] = _v")

    footer = '''
        # --- Your evolved features go here ---
        # Ideas: pairwise products, ratios, log-transforms, conditionals,
        #        cross-group interactions, row-wise stats, etc.
        # Example (replace with evolved code):
        # new_feats["eng_arpu_x_tenure"] = df["arpu"].values * df["tenure"].values

        if new_feats:
            df = pd.concat([df, pd.DataFrame(new_feats, index=df.index)], axis=1)
        results.append(df)
    return results[0], results[1]
# EVOLVE-BLOCK-END


# ---------------------------------------------------------------------------
# Fixed training pipeline (not evolved)
# ---------------------------------------------------------------------------
def run(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train_derived.csv.gz"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_derived.csv.gz"))

    # Label flip (original 0 -> 1 positive)
    train_df["target"] = train_df["target"].map({0: 1, 1: 0})
    val_df["target"] = val_df["target"].map({0: 1, 1: 0})

    orig_cols = set(train_df.columns)
    train_df, val_df = engineer_features(train_df, val_df)

    new_cols = [c for c in train_df.columns if c not in orig_cols]
    if new_cols:
        for df in [train_df, val_df]:
            df[new_cols] = df[new_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]

    # Scale pos weight
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    spw = neg_count / max(pos_count, 1)
    if spw < 1:
        spw = 1.0

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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    fpr, tpr, _ = roc_curve(y_val, y_prob, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    f1_macro = float(f1_score(y_val, y_pred, average="macro"))
    num_eng = len(new_cols)

    return {"roc_auc": roc_auc, "f1_macro": f1_macro, "num_eng_features": num_eng}


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    results = run(data_dir)
    print(results)
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(body_lines))
        f.write(footer)

    print(f"Generated {output_path}")


def main():
    # Verify Stage 1 data exists
    metadata_path = os.path.join(DATA_DIR, "stage1_groups.json")
    if not os.path.exists(metadata_path):
        print(f"Missing {metadata_path}. Run prepare_stage1.py first.")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Ensure stage1_results directory exists
    os.makedirs(STAGE1_RESULTS_DIR, exist_ok=True)

    print(f"Collecting Stage 1 expressions for {metadata['num_groups']} groups...")

    expressions = []
    for group_info in metadata["groups"]:
        gid = group_info["id"]
        base_name = group_info["base"]
        feature_names = group_info["cols"]

        best_path = os.path.join(STAGE1_RESULTS_DIR, f"group_{gid}_best.py")
        fallback_path = STAGE1_INITIAL_PATH

        if os.path.exists(best_path):
            expr = extract_evolve_block(best_path)
            source = best_path
        else:
            print(f"  group_{gid} best not found, falling back to initial program")
            expr = extract_evolve_block(fallback_path)
            source = fallback_path
            # Copy fallback program so Stage 2 can import it
            shutil.copy(fallback_path, best_path)
            print(f"    Copied fallback -> {best_path}")

        if not expr:
            print(f"  WARNING: Could not extract EVOLVE-BLOCK for group {gid}")
            continue

        # Run BFGS to get optimized params for this program
        param_save_path = os.path.join(STAGE1_RESULTS_DIR, f"group_{gid}_params.json")
        print(f"  group_{gid}: optimizing params via BFGS...")
        try:
            best_params = optimize_group_params(best_path, group_info["cols"])
            with open(param_save_path, "w", encoding="utf-8") as f:
                json.dump({"params": best_params.tolist()}, f)
            print(f"    Saved optimized params -> {param_save_path}")
        except Exception as e:
            print(f"    BFGS failed: {e}, using zeros")
            with open(param_save_path, "w", encoding="utf-8") as f:
                json.dump({"params": [0.0] * 20}, f)

        formula = expression_to_formula(expr, feature_names)
        expressions.append(
            {
                "group_id": gid,
                "base_name": base_name,
                "feature_names": feature_names,
                "formula": formula,
                "source": source,
            }
        )
        print(f"  group_{gid} ({base_name}): extracted formula")

    # Save expressions JSON
    expr_path = os.path.join(DATA_DIR, "stage1_expressions.json")
    with open(expr_path, "w", encoding="utf-8") as f:
        json.dump(expressions, f, indent=2, ensure_ascii=False)
    print(f"\nSaved expressions -> {expr_path}")

    # Save human-readable formulas
    formulas_txt = os.path.join(DATA_DIR, "stage1_formulas.txt")
    with open(formulas_txt, "w", encoding="utf-8") as f:
        f.write("# Stage 1 Discovered Symbolic Formulas\n\n")
        for expr_info in expressions:
            f.write(f"## Group {expr_info['group_id']}: {expr_info['base_name']}\n")
            f.write(f"Features: {', '.join(expr_info['feature_names'])}\n")
            f.write(f"Formula:\n{expr_info['formula']}\n\n")
    print(f"Saved readable formulas -> {formulas_txt}")

    # Generate Stage 2 initial program
    stage2_initial_path = os.path.join(BASE_DIR, "stage2_initial_program.py")
    generate_stage2_initial(expressions, stage2_initial_path)

    print("\n" + "=" * 60)
    print("Stage 2 preparation complete.")
    print("Next step: run Stage 2 OpenEvolve")
    print("=" * 60)


if __name__ == "__main__":
    main()
