import numpy as np

from sureenough.models.retrieval import unit_rows


def pe_guess(embedding: np.ndarray, text_features: np.ndarray, names: list[str]) -> str:
    sims = unit_rows(text_features) @ (embedding / np.linalg.norm(embedding))
    return names[int(np.argmax(sims))]
