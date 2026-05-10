"""
Extract formula expressions from Stage 1 best programs.

Parses each group's best evolved program and extracts the mathematical
expression used for NPS prediction. These formulas serve as prior knowledge
for Stage 2 feature engineering.

Output: stage1_formulas.json — structured formula data for Stage 2 prompts.
"""
import ast
import importlib.util
import inspect
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "stage1_results")
GROUPS_PATH = os.path.join(RESULTS_DIR, "stage1_groups.json")
OUTPUT_PATH = os.path.join(HERE, "stage1_formulas.json")


def load_group_metadata():
    """Load group column definitions."""
    with open(GROUPS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_program(group_dir):
    """Find the best program from an OpenEvolve output directory.

    Looks for the highest-scoring checkpoint's best program.
    Falls back to any .py file in the checkpoints.
    """
    if not os.path.isdir(group_dir):
        return None, None

    # Look for checkpoints directory
    checkpoints_dir = os.path.join(group_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        return None, None

    # Find all checkpoint dirs and sort by number
    ckpt_dirs = []
    for name in os.listdir(checkpoints_dir):
        ckpt_path = os.path.join(checkpoints_dir, name)
        if os.path.isdir(ckpt_path) and name.startswith("checkpoint_"):
            try:
                num = int(name.split("_")[1])
                ckpt_dirs.append((num, ckpt_path))
            except (IndexError, ValueError):
                continue

    if not ckpt_dirs:
        return None, None

    # Latest checkpoint
    ckpt_dirs.sort(key=lambda x: x[0], reverse=True)
    _, latest_ckpt = ckpt_dirs[0]

    # Find best program in this checkpoint
    for fname in os.listdir(latest_ckpt):
        if fname.endswith(".py"):
            prog_path = os.path.join(latest_ckpt, fname)
            param_path = prog_path.replace(".py", ".npy")
            if os.path.exists(param_path):
                return prog_path, param_path
            # Try without .npy
            return prog_path, None

    return None, None


def extract_expression_from_source(source_code):
    """Extract the logits expression from predict_nps source code.

    Handles two styles:
    1. Evolved: `logits = (<big expression>)`
    2. Initial: `for i in range(...): logits = logits + params[i] * X[:, i]`

    Returns (expression_ast, index→col_name map).
    For X[:, i] style, index_map is built from col_names later.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None, {}

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "predict_nps":
            func_def = node
            break
    if func_def is None:
        return None, {}

    # Strategy 1: direct logits = <expression> (skip trivial initializers)
    for node in func_def.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "logits":
                    # Skip logits = np.zeros(...) — that's just initialization
                    if (isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Attribute)
                            and node.value.func.attr == "zeros"):
                        continue
                    return node.value, {}

    # Strategy 2: loop-body accumulation (logits = logits + ...) or AugAssign
    for node in func_def.body:
        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name) and node.target.id == "i":
                for inner in node.body:
                    # logits += <expr>
                    if isinstance(inner, ast.AugAssign) and isinstance(
                        inner.target, ast.Name
                    ) and inner.target.id == "logits":
                        return inner.value, {}
                    # logits = logits + <expr>  (extract the RHS expression)
                    if (isinstance(inner, ast.Assign)
                            and len(inner.targets) == 1
                            and isinstance(inner.targets[0], ast.Name)
                            and inner.targets[0].id == "logits"
                            and isinstance(inner.value, ast.BinOp)):
                        # Return the part after the + operator
                        binop = inner.value
                        if (isinstance(binop.left, ast.Name)
                                and binop.left.id == "logits"):
                            return binop.right, {}

    # Strategy 3: return value
    for node in func_def.body:
        if isinstance(node, ast.Return):
            return node.value, {}

    # Strategy 4: any non-trivial logits assignment (last resort)
    for node in func_def.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "logits":
                    return node.value, {}

    return None, {}


def _extract_index(node):
    """Extract integer value from an AST index node (Num or Constant)."""
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _format_ast(node, col_names, params):
    """Recursively format an AST expression to a readable formula string.

    - `X[:, i]` → col_names[i] (the actual column name)
    - `params[i]` → the optimized parameter value
    """
    if node is None:
        return "unknown"

    if isinstance(node, ast.BinOp):
        left = _format_ast(node.left, col_names, params)
        right = _format_ast(node.right, col_names, params)
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(
            type(node.op), "?"
        )
        return f"({left} {op} {right})"

    if isinstance(node, ast.UnaryOp):
        operand = _format_ast(node.operand, col_names, params)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        return f"(+{operand})"

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            func_name = (
                f"{node.func.value.id}.{node.func.attr}"
                if hasattr(node.func.value, "id")
                else node.func.attr
            )
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
        else:
            func_name = "?"
        args = [_format_ast(a, col_names, params) for a in node.args]
        return f"{func_name}({', '.join(args)})"

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, float):
            return f"{val:.4g}"
        return str(val)

    if isinstance(node, ast.Num):
        val = node.n
        if isinstance(val, float):
            return f"{val:.4g}"
        return str(val)

    if isinstance(node, ast.Subscript):
        obj = node.value.id if isinstance(node.value, ast.Name) else None

        # Handle params[i] → optimized value or params[idx]
        if obj == "params":
            idx = None
            if isinstance(node.slice, ast.Index):
                idx = _extract_index(node.slice.value)
            elif isinstance(node.slice, ast.Constant):
                idx = _extract_index(node.slice)
            elif isinstance(node.slice, ast.Name):
                idx = node.slice.id  # e.g., params[i] → "params[i]"
            if isinstance(idx, int) and params is not None and idx < len(params):
                return f"{params[idx]:.4g}"
            return f"params[{idx}]"

        # Handle X[:, i] → col_names[i]
        if obj == "X":
            idx = None
            if isinstance(node.slice, ast.Tuple):
                # Python 3.9+: X[:, i] → Tuple(Slice, Constant(i) or Name(i))
                elts = node.slice.elts
                if len(elts) == 2:
                    idx = _extract_index(elts[1])
                    if idx is None and isinstance(elts[1], ast.Name):
                        idx = elts[1].id  # loop variable
            elif isinstance(node.slice, ast.Index):
                idx = _extract_index(node.slice.value)
            if isinstance(idx, int) and idx < len(col_names):
                return col_names[idx]
            return f"X[:, {idx}]"

        return f"{obj}[?]"

    if isinstance(node, ast.Compare):
        left = _format_ast(node.left, col_names, params)
        parts = []
        for op, comp in zip(node.ops, node.comparators):
            op_str = {
                ast.Gt: ">", ast.Lt: "<", ast.GtE: ">=",
                ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!=",
            }.get(type(op), "?")
            parts.append(f"{op_str} {_format_ast(comp, col_names, params)}")
        return f"({left} {' and '.join(parts)})"

    if isinstance(node, ast.BoolOp):
        op = " & " if isinstance(node.op, ast.And) else " | "
        return op.join([_format_ast(v, col_names, params) for v in node.values])

    if isinstance(node, ast.IfExp):
        body = _format_ast(node.body, col_names, params)
        test = _format_ast(node.test, col_names, params)
        orelse = _format_ast(node.orelse, col_names, params)
        return f"({body} if {test} else {orelse})"

    return f"<{type(node).__name__}>"


def ast_to_formula_text(node, _feat_map, params, col_names):
    """Convert an AST expression node to a readable formula string."""
    return _format_ast(node, col_names, params if params is not None else [])


def extract_group_formulas(group_id, groups_metadata):
    """Extract formula from a group's best evolved program."""
    group_dir = os.path.join(RESULTS_DIR, f"group_{group_id}")
    prog_path, param_path = find_best_program(group_dir)

    if prog_path is None:
        return {"group_id": group_id, "status": "no_results", "formula_text": None}

    # Load params
    params = None
    if param_path:
        try:
            params = np.load(param_path)
        except Exception:
            params = None

    # Read source
    with open(prog_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Extract expression
    logits_expr, feat_map = extract_expression_from_source(source)

    # Get col names for this group
    group_meta = None
    for g in groups_metadata["groups"]:
        if g["id"] == group_id:
            group_meta = g
            break

    col_names = group_meta["cols"] if group_meta else []

    # Convert to readable formula
    formula_text = ast_to_formula_text(logits_expr, feat_map, params or [], col_names)

    return {
        "group_id": group_id,
        "status": "ok",
        "program_path": prog_path,
        "formula_text": formula_text,
        "num_params_used": len(params) if params is not None else 0,
        "bases": group_meta["bases"] if group_meta else [],
    }


def build_stage2_prior_knowledge(formulas):
    """Build the prior knowledge text block for Stage 2's system message.

    This text summarizes what Stage 1 discovered about feature relationships,
    guiding Stage 2's feature engineering.
    """
    ok_formulas = [f for f in formulas if f["status"] == "ok"]

    if not ok_formulas:
        return "# No Stage 1 priors available."

    lines = [
        "# Stage 1 Prior Knowledge: Discovered Feature Relationships",
        "",
        "The following formulas were discovered by symbolic regression in Stage 1.",
        "Each formula captures nonlinear interactions within a feature group that",
        "are predictive of NPS. Use these patterns to guide feature engineering:",
        "",
    ]

    for f in ok_formulas:
        bases_str = ", ".join(f["bases"][:8])
        lines.append(f"## Group {f['group_id']} (base features: {bases_str})")
        lines.append(f"```")
        lines.append(f"logits = {f['formula_text']}")
        lines.append(f"```")
        lines.append("")

    lines.append("## How to Use These Priors")
    lines.append("")
    lines.append(
        "These formulas reveal multiplicative interactions, ratios, and "
        "nonlinear transformations that XGBoost may have missed."
    )
    lines.append(
        "When creating new features, consider:"
    )
    lines.append(
        "1. The interaction terms (e.g., feat_a * feat_b) — these are "
        "candidate cross-features."
    )
    lines.append(
        "2. The conditional patterns (np.where) — these identify subgroups "
        "that need special treatment."
    )
    lines.append(
        "3. The nonlinear transforms (log, sqrt, exp) — these suggest "
        "appropriate transformations for specific features."
    )

    return "\n".join(lines)


def main():
    metadata = load_group_metadata()
    groups = metadata["groups"]
    print(f"Extracting formulas from {len(groups)} groups...")

    formulas = []
    for g in groups:
        gid = g["id"]
        print(f"  Group {gid}...", end=" ")
        formula = extract_group_formulas(gid, metadata)
        formulas.append(formula)
        print(formula["status"])

    # Save structured formulas
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(formulas, f, indent=2, ensure_ascii=False)
    print(f"\nSaved formulas to {OUTPUT_PATH}")

    # Build and print prior knowledge
    prior = build_stage2_prior_knowledge(formulas)
    prior_path = os.path.join(HERE, "stage2_priors.txt")
    with open(prior_path, "w", encoding="utf-8") as f:
        f.write(prior)
    print(f"Saved Stage 2 prior knowledge to {prior_path}")

    # Summary
    ok_count = sum(1 for f in formulas if f["status"] == "ok")
    print(f"\n{'=' * 60}")
    print(f"Extraction complete: {ok_count}/{len(formulas)} groups succeeded")
    for f in formulas:
        if f["status"] == "ok":
            print(f"\n  Group {f['group_id']} (bases: {f['bases']}):")
            print(f"  {f['formula_text'][:200]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
