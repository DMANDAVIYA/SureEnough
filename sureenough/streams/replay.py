from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from sureenough.models.retrieval import unit_rows

STREAM_SEED = 20261030
N_STREAMS = 10
N_REPEATS = 80


def stream_orders(subject: int, image: int, n_streams: int = N_STREAMS, n_repeats: int = N_REPEATS) -> np.ndarray:
    return np.stack([
        np.random.default_rng([STREAM_SEED, subject, image, r]).permutation(n_repeats) for r in range(n_streams)
    ])


def _zscore(sims: np.ndarray) -> np.ndarray:
    return (sims - sims.mean(axis=-1, keepdims=True)) / sims.std(axis=-1, keepdims=True)


def zsoftmax(sims: np.ndarray, beta: float) -> np.ndarray:
    return softmax(beta * _zscore(sims), axis=-1)


def fit_beta(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    scored = [(_zscore(sims), true) for sims, true in pairs]

    def neg_log_lik(beta: float) -> float:
        logp = [log_softmax(beta * z, axis=-1)[np.arange(len(true)), true] for z, true in scored]
        return -float(np.mean(np.concatenate(logp)))

    return float(minimize_scalar(neg_log_lik, bounds=(0.1, 50.0), method="bounded").x)


def make_streams(
    embed: Callable[[np.ndarray], np.ndarray],
    eeg: np.ndarray,
    targets: np.ndarray,
    subject: int,
    beta: float,
    n_streams: int = N_STREAMS,
    n_steps: int = N_REPEATS,
) -> np.ndarray:
    n_img = len(eeg)
    gallery = unit_rows(targets)
    steps = np.arange(1, n_steps + 1, dtype=np.float64)[:, None, None]
    out = np.empty((n_img, n_streams, n_steps, n_img), dtype=np.float16)
    for i in range(n_img):
        orders = stream_orders(subject, i, n_streams, eeg.shape[1])[:, :n_steps]
        running = np.concatenate([np.cumsum(eeg[i, o], axis=0, dtype=np.float64) / steps for o in orders])
        sims = unit_rows(embed(running.astype(np.float32))) @ gallery.T
        out[i] = zsoftmax(sims, beta).reshape(n_streams, n_steps, n_img)
    return out
