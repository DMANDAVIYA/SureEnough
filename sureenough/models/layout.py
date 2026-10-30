import numpy as np


def validation_layout(concept_of_image: np.ndarray, val_concepts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    is_val = np.isin(concept_of_image, val_concepts)
    train_idx = np.flatnonzero(~is_val)
    per_concept = [np.flatnonzero(concept_of_image == c) for c in val_concepts]
    galleries = np.stack(per_concept, axis=1)
    return train_idx, galleries
