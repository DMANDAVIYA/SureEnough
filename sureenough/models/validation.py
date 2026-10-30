from collections.abc import Callable

import numpy as np

from sureenough.models.retrieval import cosine_top1


def val_top1(
    predict: Callable[[np.ndarray], np.ndarray],
    eeg: np.ndarray,
    targets: np.ndarray,
    galleries: np.ndarray,
    k: int = 4,
) -> float:
    scores = []
    for gallery in galleries:
        x = eeg[gallery, :k].mean(axis=1).reshape(len(gallery), -1)
        scores.append(cosine_top1(predict(x), targets[gallery]))
    return float(np.mean(scores))
