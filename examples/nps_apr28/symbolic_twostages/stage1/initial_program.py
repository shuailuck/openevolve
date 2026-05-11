"""
Stage 1 symbolic regression — per-group initial program template.

Since each group has different feature columns, the actual initial_program.py
is generated at runtime by `write_initial_program()`. Callers:

  - run_stage1.py : generates temp file before each group's evolution
  - test_program.py : generates temp file for local testing

Usage:
  from initial_program import write_initial_program
  prog_path = write_initial_program(group, output_dir)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data")
GROUPS_PATH = os.path.join(HERE, "..", "stage1_results", "stage1_groups.json")


def make_initial_program_source(group):
    """Generate the source code for a group-specific initial_program.py."""
    cols = group["cols"]
    bases = group["bases"]
    gid = group["id"]
    num_features = len(cols)
    max_nparams = max(num_features, 20)

    params_block = ",\n    ".join(
        f"{c}: np.ndarray = None" for c in cols
    )
    terms_block = "\n".join(
        f"        + params[{i}] * {c}" for i, c in enumerate(cols)
    )

    return f'''"""
Stage 1 initial program — Group {gid}
Base features: {json.dumps(bases)}
Feature columns ({num_features}): {json.dumps(cols)}
"""
import numpy as np

MAX_NPARAMS = {max_nparams}

# EVOLVE-BLOCK-START
def predict_nps(
    params,
    *,
    {params_block},
):
    """
    Predict NPS logits via symbolic regression.

    Each feature above is passed as a keyword argument by the evaluator.
    Use the named parameters directly in the formula — no df access needed.

    Output: raw logits (1D ndarray).
    Evaluator applies sigmoid + binary cross-entropy + BFGS.
    NEVER hardcode constants — use params[i] for coefficients.
    """
    logits = (
{terms_block}
    )

    return logits
# EVOLVE-BLOCK-END
'''


def write_initial_program(group, output_dir):
    """Write a group-specific initial_program.py and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    source = make_initial_program_source(group)
    prog_path = os.path.join(output_dir, "initial_program.py")
    with open(prog_path, "w", encoding="utf-8") as f:
        f.write(source)
    return prog_path


# For backward compatibility (test_program.py creates its own temp file,
# but this module still needs to be importable with a predict_nps function).
# Load group 0 by default.
_group_id = int(os.environ.get("STAGE1_GROUP_ID", 0))
_groups_path = os.environ.get("STAGE1_GROUPS_PATH", GROUPS_PATH)
if os.path.exists(_groups_path):
    with open(_groups_path, "r") as f:
        _all = json.load(f)["groups"]
    _group = next(g for g in _all if g["id"] == _group_id)
    _src = make_initial_program_source(_group)
    # Extract and exec just the function from EVOLVE block
    _block = _src.split("# EVOLVE-BLOCK-START\n")[1].split("# EVOLVE-BLOCK-END\n")[0]
    exec(_block, globals())
