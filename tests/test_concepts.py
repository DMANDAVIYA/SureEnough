import numpy as np
import pytest

from sureenough.paths import DERIVED, RAW

needs_data = pytest.mark.skipif(
    not (DERIVED / "image_meta.json").exists() or not (RAW / "osf-jum2f").exists(),
    reason="converted metadata or THINGS metadata not present",
)


@pytest.fixture(scope="module")
def concepts():
    from sureenough.data.concepts import load_concepts

    return load_concepts()


@needs_data
def test_counts(concepts):
    assert len(concepts.test_names) == 200
    assert len(concepts.test_things_ids) == 200
    assert len(set(concepts.train_concept_of_image.tolist())) == 1654
    assert len(concepts.train_concept_of_image) == 16540
    assert len(concepts.all_names) == 1854 and len(concepts.all_things_ids) == 1854


@needs_data
def test_every_test_concept_has_cc0_photo(concepts):
    assert all(t in concepts.cc0_member for t in concepts.test_things_ids)


@needs_data
def test_category_coverage_matches_spec(concepts):
    test = set(concepts.test_things_ids)
    categorised = {t for t in test if any(t in m for m in concepts.categories53.values())}
    assert len(categorised) == 195
    assert len(concepts.categories53["food"] & test) == 53
    assert len(concepts.categories53) == 53


@needs_data
def test_validation_split_is_deterministic_and_sized(concepts):
    from sureenough.data.concepts import load_concepts

    again = load_concepts()
    assert concepts.val_concepts.shape == (200,)
    assert np.array_equal(concepts.val_concepts, again.val_concepts)
    assert np.all(np.diff(concepts.val_concepts) > 0)
    assert concepts.val_concepts.min() >= 0 and concepts.val_concepts.max() < 1654
