import numpy as np

from .pvalues import hb_pvalue


def fixed_sequence(pvals: np.ndarray, delta: float) -> int:
    leading_rejections = int(np.cumprod(np.asarray(pvals) <= delta).sum())
    return max(leading_rejections - 1, 0)


def calibrate(losses: np.ndarray, image_ids: np.ndarray, alpha: float, delta: float) -> int:
    losses = np.asarray(losses, np.float64)
    n = len(losses)
    assert len(image_ids) == n == len(np.unique(image_ids))
    assert np.all((losses >= 0) & (losses <= 1))
    return fixed_sequence(hb_pvalue(losses.mean(0), n, alpha), delta)
