import numpy as np

from sureenough.models.layout import validation_layout


def _concept_of_image(n_concepts=30, per_concept=10):
    return np.repeat(np.arange(n_concepts), per_concept)


def test_layout_shapes_and_disjointness():
    concept_of_image = _concept_of_image()
    val_concepts = np.array([2, 5, 11, 29])
    train_idx, galleries = validation_layout(concept_of_image, val_concepts)
    assert galleries.shape == (10, 4)
    assert len(train_idx) == (30 - 4) * 10
    assert not np.isin(concept_of_image[train_idx], val_concepts).any()
    for gallery in galleries:
        assert np.array_equal(concept_of_image[gallery], val_concepts)
    assert len(np.unique(galleries)) == 40


def test_gallery_g_takes_the_gth_image_of_each_concept():
    concept_of_image = _concept_of_image()
    _, galleries = validation_layout(concept_of_image, np.array([3]))
    assert galleries[:, 0].tolist() == list(range(30, 40))
