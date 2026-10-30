import inspect

import numpy as np
import pytest

from sureenough.streams.replay import fit_beta, make_streams, stream_orders, zsoftmax


def test_stream_orders_are_fixed_permutations():
    a = stream_orders(subject=3, image=17)
    b = stream_orders(subject=3, image=17)
    assert a.shape == (10, 80)
    assert np.array_equal(a, b)
    assert all(np.array_equal(np.sort(row), np.arange(80)) for row in a)
    assert not np.array_equal(a, stream_orders(subject=3, image=18))
    assert not np.array_equal(a[0], a[1])


def test_zsoftmax_rows_sum_to_one_and_ignore_offsets():
    sims = np.random.default_rng(0).normal(size=(4, 200))
    p = zsoftmax(sims, beta=3.0)
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-5)
    assert np.allclose(p, zsoftmax(sims + 7.5, beta=3.0), atol=1e-6)


def test_fit_beta_recovers_generating_beta():
    rng = np.random.default_rng(0)
    sims = rng.normal(size=(4000, 50))
    p = zsoftmax(sims, beta=5.0)
    true = np.array([rng.choice(50, p=row) for row in p])
    assert abs(fit_beta([(sims, true)]) - 5.0) < 0.5


def test_make_streams_shape_and_running_average():
    rng = np.random.default_rng(0)
    eeg = rng.normal(size=(3, 80, 2, 5)).astype(np.float32)
    targets = rng.normal(size=(3, 10)).astype(np.float32)
    seen = []

    def embed(x):
        seen.append(x.copy())
        return np.tile(x.reshape(len(x), -1)[:, :1], (1, 10))

    probs = make_streams(embed, eeg, targets, subject=1, beta=2.0, n_streams=2, n_steps=4)
    assert probs.shape == (3, 2, 4, 3) and probs.dtype == np.float16
    order = stream_orders(subject=1, image=0, n_streams=2)[0]
    expected = eeg[0, order[:3]].mean(axis=0)
    assert np.allclose(seen[0][2], expected, atol=1e-5)


def test_split_takes_no_data_arguments():
    from sureenough.analysis.splits import split

    params = inspect.signature(split).parameters
    assert all(p.annotation in (int, "int") for p in params.values())


def test_load_frozen_detects_tampering(tmp_path):
    from sureenough.config import freeze, load_frozen

    cfg = tmp_path / "primary.json"
    freeze({"K": 20, "alpha": 0.2}, cfg)
    assert load_frozen(cfg)["K"] == 20
    cfg.write_text(cfg.read_text().replace("20", "21"))
    with pytest.raises(ValueError):
        load_frozen(cfg)
