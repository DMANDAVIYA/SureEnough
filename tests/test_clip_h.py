import numpy as np
import pytest

from sureenough.features.clip_h import ATM_DIR, load_clip_h, order_agreement
from sureenough.features.targets import FEATURES, load_targets


def test_order_agreement_survives_rotation_and_catches_permutation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 32))
    rotation, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    y = x @ rotation + 0.05 * rng.normal(size=x.shape)
    assert order_agreement(x, y) > 0.95
    assert order_agreement(x, y[rng.permutation(100)]) < 0.2


@pytest.mark.skipif(not (ATM_DIR / "ViT-H-14_features_test.pt").exists(), reason="ATM features not downloaded")
def test_clip_h_shapes_unit_rows_and_row_order():
    train, test = load_clip_h()
    assert train.shape == (16540, 1024) and test.shape == (200, 1024)
    assert np.allclose(np.linalg.norm(test, axis=1), 1, atol=1e-5)
    if (FEATURES / "pe_test_final.npy").exists():
        assert order_agreement(test, load_targets("final")[1]) >= 0.5


@pytest.mark.skipif(not (ATM_DIR / "ViT-H-14_features_test.pt").exists(), reason="ATM features not downloaded")
def test_load_targets_knows_clip_h():
    train, test = load_targets("clip_h")
    assert test.shape == (200, 1024)
