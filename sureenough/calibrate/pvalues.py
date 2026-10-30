import numpy as np
from scipy.special import xlogy
from scipy.stats import binom

FLOAT_NOISE = 1e-9


def hb_pvalue(risk_hat: float | np.ndarray, n: int, alpha: float) -> float | np.ndarray:
    r = np.asarray(risk_hat, np.float64)
    a = np.minimum(r, alpha)
    h1 = xlogy(a, a / alpha) + xlogy(1 - a, (1 - a) / (1 - alpha))
    hoeffding = np.exp(-n * h1)
    bentkus = np.e * binom.cdf(np.ceil(n * r - FLOAT_NOISE), n, alpha)
    p = np.minimum(np.minimum(hoeffding, bentkus), 1.0)
    return p if p.ndim else float(p)
