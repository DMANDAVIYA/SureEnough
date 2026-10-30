import math

import numpy as np
from scipy.stats import wilcoxon


def cvar(values: np.ndarray, fraction: float = 0.2) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[: max(1, math.ceil(fraction * len(ordered)))].mean())


def h5a(hedged: np.ndarray, top1: np.ndarray) -> dict:
    hedged, top1 = np.asarray(hedged, np.float64), np.asarray(top1, np.float64)
    diff = hedged - top1
    return {
        "n": int(len(diff)),
        "p": float(wilcoxon(diff, alternative="greater").pvalue),
        "median_diff": float(np.median(diff)),
        "cvar20_hedged": cvar(hedged),
        "cvar20_top1": cvar(top1),
    }
