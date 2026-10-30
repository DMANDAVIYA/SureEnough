import math

import numpy as np

from .pvalues import FLOAT_NOISE
from .sets import mass_above, policy_tables, top_m_mass


def fixed_k_tables(probs: np.ndarray, k: int) -> dict:
    tables = policy_tables(probs[:, :, k - 1 : k], K=1)
    return {"loss_count": tables["loss_count"], "set_size": tables["set_size"]}


def conformal_scores(probs: np.ndarray, K: int) -> dict:
    steps = probs[:, :, :K]
    images = np.arange(len(probs))
    return {"a_true": mass_above(steps)[images, :, :, images], "top3": top_m_mass(steps, 3)}


def conformal_quantile(calibration_scores: np.ndarray, alpha: float) -> np.ndarray:
    n_cal = len(calibration_scores)
    rank = math.ceil((n_cal + 1) * (1 - alpha) - FLOAT_NOISE)
    if rank > n_cal:
        return np.full(calibration_scores.shape[1:], np.inf)
    return np.sort(calibration_scores, 0)[rank - 1]


def first_hit_or_last(hit: np.ndarray) -> np.ndarray:
    return np.where(hit.any(-1), hit.argmax(-1), hit.shape[-1] - 1)[..., None]


def naive_policy(scores: dict, cal_idx: np.ndarray, eval_idx: np.ndarray, alpha: float, K: int) -> dict:
    a_true, top3 = scores["a_true"][:, :, :K], scores["top3"][:, :, :K]
    q = conformal_quantile(a_true[cal_idx], alpha)
    stop = first_hit_or_last(top3 > q)
    pop_loss = np.take_along_axis(a_true > q, stop, -1)[..., 0].mean(1)
    stop_t = (stop[..., 0] + 1).mean(1)
    return {"loss": pop_loss[eval_idx], "stop_t": stop_t[eval_idx], "pop_loss": pop_loss}


def bonferroni_policy(scores: dict, cal_idx: np.ndarray, eval_idx: np.ndarray, alpha: float, K: int) -> dict:
    return naive_policy(scores, cal_idx, eval_idx, alpha / K, K)


def uncalibrated_policy(probs: np.ndarray, K: int, thresh: float = 0.8) -> dict:
    steps = np.asarray(probs[:, :, :K], np.float64)
    stop = first_hit_or_last(steps.max(-1) >= thresh)
    guess = np.take_along_axis(steps.argmax(-1), stop, -1)[..., 0]
    truth = np.arange(len(probs))[:, None]
    return {"loss": (guess != truth).mean(1), "stop_t": (stop[..., 0] + 1).mean(1)}
