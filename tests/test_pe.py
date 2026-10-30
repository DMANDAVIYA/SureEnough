import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def pe():
    from sureenough.features.pe import load_pe

    return load_pe()


def _images():
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)) for _ in range(2)]


def test_preprocess_is_224_with_half_normalisation(pe):
    _, preprocess = pe
    x = preprocess(Image.new("RGB", (300, 200), (255, 255, 255)))
    assert tuple(x.shape) == (3, 224, 224)
    assert np.allclose(x.numpy(), 1.0, atol=1e-5)


def test_image_feature_shapes_and_determinism(pe):
    from sureenough.features.pe import pe_image_features

    model, preprocess = pe
    inter, final = pe_image_features(model, preprocess, _images())
    assert inter.shape == (2, 12, 2, 768) and inter.dtype == np.float16
    assert final.shape == (2, 1024) and final.dtype == np.float16
    assert np.isfinite(inter).all() and np.isfinite(final).all()
    inter2, final2 = pe_image_features(model, preprocess, _images())
    assert np.array_equal(inter, inter2) and np.array_equal(final, final2)


def test_text_feature_shape(pe):
    from sureenough.features.pe import pe_text_features

    model, _ = pe
    t = pe_text_features(model, ["dog", "aircraft carrier"])
    assert t.shape == (2, 1024) and t.dtype == np.float16
