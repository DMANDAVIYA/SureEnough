import numpy as np
import pytest
import torch

from sureenough.models.encoder import (
    MLPEncoder,
    Standardizer,
    encode,
    sample_repeat_average,
    train_mlp,
)
from sureenough.models.retrieval import cosine_top1
from sureenough.features.targets import parse_target_name, target_names


def _toy(n_classes=200, per_class=8, in_dim=64, out_dim=32, seed=0):
    rng = np.random.default_rng(seed)
    protos_in = rng.normal(size=(n_classes, in_dim))
    protos_out = rng.normal(size=(n_classes, out_dim))
    labels = np.repeat(np.arange(n_classes), per_class)
    x = protos_in[labels][:, None, :] + 0.6 * rng.normal(size=(len(labels), 4, in_dim))
    return x.astype(np.float32), protos_out[labels].astype(np.float32), protos_in, protos_out


def test_train_mlp_learns_toy_prototypes():
    x, y, protos_in, protos_out = _toy()
    model = train_mlp(x, y, epochs=30, batch=256, lr=1e-3, seed=0, hidden=256)
    rng = np.random.default_rng(1)
    probe = (protos_in + 0.6 * rng.normal(size=protos_in.shape) / 2).astype(np.float32)
    assert cosine_top1(encode(model, probe), protos_out) > 0.9


def test_encode_returns_unit_rows():
    model = MLPEncoder(in_dim=10, out_dim=4, hidden=16)
    out = encode(model, np.random.default_rng(0).normal(size=(5, 10)).astype(np.float32))
    assert out.shape == (5, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_sample_repeat_average_uses_one_to_four_repeats():
    x = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    gen = torch.Generator().manual_seed(0)
    seen = set()
    for _ in range(50):
        avg, counts = sample_repeat_average(x, gen)
        seen.update(counts.tolist())
        assert avg.shape == (3, 2)
        assert torch.all((counts >= 1) & (counts <= 4))
    assert seen == {1, 2, 3, 4}


def test_sample_repeat_average_full_count_is_plain_mean():
    x = torch.randn(2, 4, 5)
    gen = torch.Generator().manual_seed(0)
    for _ in range(40):
        avg, counts = sample_repeat_average(x, gen)
        full = counts == 4
        if full.any():
            assert torch.allclose(avg[full], x[full].mean(dim=1), atol=1e-6)
            return
    pytest.fail("never sampled all four repeats")


def test_standardizer_zero_mean_unit_std_per_channel():
    rng = np.random.default_rng(0)
    train = (rng.normal(size=(50, 4, 3, 7)) * np.array([1.0, 5.0, 0.1])[None, None, :, None] + 3).astype(np.float32)
    st = Standardizer.fit(train)
    z = st.transform(train)
    assert np.allclose(z.mean(axis=(0, 1, 3)), 0, atol=1e-4)
    assert np.allclose(z.std(axis=(0, 1, 3)), 1, atol=1e-3)


def test_cosine_top1_identity_is_perfect():
    g = np.random.default_rng(0).normal(size=(20, 8))
    assert cosine_top1(g, g) == 1.0


def test_target_names_and_parsing():
    names = target_names()
    assert names[0] == "final" and len(names) == 13
    assert parse_target_name("final") is None
    assert parse_target_name("L7_mean") == (7, 1)
    assert parse_target_name("L11_cls") == (11, 0)
    with pytest.raises(ValueError):
        parse_target_name("L12_cls")
