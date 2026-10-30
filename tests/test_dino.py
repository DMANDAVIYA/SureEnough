import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def dino():
    from sureenough.features.dino import load_dino

    return load_dino()


def test_dino_embeds_unit_rows_at_224(dino):
    from sureenough.features.dino import dino_embed

    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for h, w in ((300, 200), (64, 640))]
    emb = dino_embed(dino, images)
    assert emb.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_dino_identical_images_match(dino):
    from sureenough.features.dino import dino_embed

    img = Image.new("RGB", (128, 128), (40, 120, 200))
    emb = dino_embed(dino, [img, img.copy()])
    assert float(emb[0] @ emb[1]) > 0.9999
