import numpy as np


def focal_loss(y_true, y_pred, alpha=0.5, gamma=2):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    gradient = -alpha * (1 - p_t) ** gamma * (y_true - y_pred)
    hessian = alpha * (1 - p_t) ** gamma * y_pred * (1 - y_pred)
    return gradient, hessian
