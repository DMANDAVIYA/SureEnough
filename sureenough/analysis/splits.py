from collections.abc import Callable

import numpy as np

from sureenough.calibrate.baselines import bonferroni_policy, naive_policy
from sureenough.calibrate.ltt import calibrate


def split(subject: int, s: int, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng([7, subject, s]).permutation(n)
    return np.sort(order[: n // 2]), np.sort(order[n // 2 :])


def ltt_records(tables: dict, cal: np.ndarray, ev: np.ndarray, streams: int, alpha: float, delta: float) -> dict:
    losses = tables["loss_count"] / streams
    lam_idx = np.array([calibrate(losses[c], c, alpha, delta) for c in cal])
    deployed = lam_idx[:, None]
    return {
        "lam_idx": lam_idx,
        "eval_risk": losses[ev, deployed].mean(1),
        "pop_risk": losses[:, lam_idx].mean(0),
    } | {name: table[ev, deployed].mean(1) for name, table in tables.items() if name != "loss_count"}


def conformal_records(policy: Callable, scores: dict, cal: np.ndarray, ev: np.ndarray, alpha: float, K: int) -> dict:
    results = [policy(scores, c, e, alpha, K) for c, e in zip(cal, ev)]
    fields = {"eval_risk": "loss", "pop_risk": "pop_loss", "stop_t": "stop_t"}
    return {name: np.array([result[field].mean() for result in results]) for name, field in fields.items()}


def run_splits(
    tables_K20: dict,
    fixed_tables: dict[int, dict],
    scores: dict,
    subject: int,
    n_splits: int = 1000,
    alpha: float = 0.2,
    delta: float = 0.1,
) -> dict:
    n, streams, K = scores["a_true"].shape
    cal, ev = map(np.array, zip(*(split(subject, s, n) for s in range(n_splits))))
    return {
        "sureenough": ltt_records(tables_K20, cal, ev, streams, alpha, delta),
        "fixed": {k: ltt_records(tables, cal, ev, streams, alpha, delta) for k, tables in fixed_tables.items()},
        "naive": conformal_records(naive_policy, scores, cal, ev, alpha, K),
        "bonferroni": conformal_records(bonferroni_policy, scores, cal, ev, alpha, K),
    }
