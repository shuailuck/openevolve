import numpy as np

# Maximum number of free parameters for BFGS optimization
MAX_NPARAMS = 20


# EVOLVE-BLOCK-START
def func(x, params):
    """
    Symbolic regression to predict the binary target from a small group of features.

    Args:
        x: numpy array of shape (n_samples, n_features), where n_features is small
           (typically 2-4 columns belonging to the same base feature group).
        params: 1D numpy array of free parameters. Use params[i] instead of
                hard-coded constants so BFGS can optimize them.

    Returns:
        1D numpy array of predictions (logits). The evaluator will apply sigmoid
        to obtain probabilities for AUC computation.

    Tips:
    - This group has very few features; exploit interactions and nonlinearities.
    - Use np.where for piecewise behavior, np.log1p / np.sqrt for transforms.
    - Always ensure numerical stability (clip before exp, epsilon before division).
    """
    n_features = x.shape[1]

    # Linear baseline (replace / extend this with evolved structure)
    logit = params[-1]  # bias term
    n_params = min(n_features, len(params) - 1)
    for i in range(n_params):
        logit = logit + params[i] * x[:, i]

    return logit
# EVOLVE-BLOCK-END
