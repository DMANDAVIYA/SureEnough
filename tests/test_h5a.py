import numpy as np

from sureenough.analysis.h5a import cvar, h5a


def test_cvar_is_mean_of_lowest_fraction():
    assert cvar(np.arange(10, dtype=float), 0.2) == 0.5
    assert cvar(np.array([3.0]), 0.2) == 3.0


def test_h5a_detects_consistent_hedging_gain():
    rng = np.random.default_rng(0)
    top1 = rng.uniform(0.2, 0.6, size=60)
    out = h5a(top1 + 0.05, top1)
    assert out["p"] < 0.01
    assert abs(out["median_diff"] - 0.05) < 1e-9
    assert out["cvar20_hedged"] > out["cvar20_top1"]
    assert out["n"] == 60


def test_h5a_null_when_no_difference():
    rng = np.random.default_rng(1)
    top1 = rng.uniform(0.2, 0.6, size=60)
    out = h5a(top1 + rng.normal(0, 0.01, size=60), top1)
    assert out["p"] > 0.01
