"""
Stage 1 symbolic regression — shared initial program for all groups.

Each group has its own feature subset. The group is selected at runtime via
the STAGE1_GROUP_ID environment variable. Set it before running OpenEvolve.

Module-level variables (visible to the LLM for context):
  GROUP_ID    — integer group id
  GROUP_COLS  — list of column names in this group's feature subset
  GROUP_BASES — list of base feature names that make up this group
  NUM_FEATURES — number of features in this group
  MAX_NPARAMS — max params for BFGS (== NUM_FEATURES, at least 20)
"""
import json
import os

import numpy as np

GROUPS_PATH = os.environ["STAGE1_GROUPS_PATH"]

_group_id = int(os.environ.get("STAGE1_GROUP_ID", 0))
with open(GROUPS_PATH, "r") as f:
    _all = json.load(f)["groups"]
_group = next(g for g in _all if g["id"] == _group_id)

GROUP_ID = _group["id"]
GROUP_COLS = _group["cols"]
GROUP_BASES = _group["bases"]
NUM_FEATURES = len(GROUP_COLS)
MAX_NPARAMS = max(NUM_FEATURES, 20)

# EVOLVE-BLOCK-START
def predict_nps(df, params):
    """
    Predict NPS logits via symbolic regression.

    See module-level GROUP_COLS for the feature names available in this group.
    Output: raw logits (1D ndarray, same length as df).
    The evaluator applies sigmoid + binary cross-entropy loss, then BFGS.
    NEVER hardcode constants — always use params[i] as tunable coefficients.
    """
    X = df.values.astype(np.float64)

    logits = np.zeros(len(df))
    for i in range(NUM_FEATURES):
        logits = logits + params[i] * X[:, i]

    return logits
# EVOLVE-BLOCK-END
