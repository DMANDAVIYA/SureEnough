import numpy as np


def unit_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def cosine_sims(pred: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return unit_rows(pred) @ unit_rows(gallery).T


def cosine_top1(pred: np.ndarray, gallery: np.ndarray) -> float:
    hits = np.argmax(cosine_sims(pred, gallery), axis=1) == np.arange(len(pred))
    return float(hits.mean())
