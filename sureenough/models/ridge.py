import numpy as np
from sklearn.linear_model import Ridge


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    return Ridge(alpha=alpha).fit(x, y)
