import numpy as np
import pytest
from scipy.special import xlogy
from scipy.stats import binom, binomtest

from sureenough.calibrate.ltt import calibrate, fixed_sequence
from sureenough.calibrate.pvalues import hb_pvalue


def hand_written_hb(loss_sum: int, n: int, alpha: float) -> float:
    r = min(loss_sum / n, alpha)
    h1 = xlogy(r, r / alpha) + xlogy(1 - r, (1 - r) / (1 - alpha))
    return min(np.exp(-n * h1), np.e * binom.cdf(loss_sum, n, alpha), 1.0)


def test_hb_zero_risk_is_hoeffding_term():
    assert hb_pvalue(0.0, 100, 0.2) == pytest.approx(0.8**100, rel=1e-9)


def test_hb_at_or_above_alpha_is_one():
    assert hb_pvalue(0.2, 100, 0.2) == 1.0
    assert hb_pvalue(0.9, 100, 0.2) == 1.0


def test_hb_keeps_scalar_and_array_shapes():
    assert isinstance(hb_pvalue(0.1, 100, 0.2), float)
    assert hb_pvalue(np.zeros((3, 4)), 100, 0.2).shape == (3, 4)


@pytest.mark.parametrize("n", [10, 100, 200])
def test_hb_monotone_in_risk(n):
    p = hb_pvalue(np.linspace(0, 1, 2001), n, 0.2)
    assert np.all(np.diff(p) >= 0)
    assert np.all((p > 0) & (p <= 1))


@pytest.mark.parametrize("n,alpha", [(100, 0.2), (100, 0.8), (37, 0.3)])
def test_hb_matches_hand_formula_despite_float_noise(n, alpha):
    loss_sums = np.arange(n + 1)
    expected = [hand_written_hb(s, n, alpha) for s in loss_sums]
    np.testing.assert_allclose(hb_pvalue(loss_sums / n, n, alpha), expected, rtol=1e-12)


def test_fixed_sequence():
    assert fixed_sequence(np.array([0.01, 0.02, 0.5, 0.01]), 0.1) == 1
    assert fixed_sequence(np.array([0.5, 0.01, 0.01]), 0.1) == 0
    assert fixed_sequence(np.array([0.01, 0.5, 0.01]), 0.1) == 0
    assert fixed_sequence(np.array([0.01, 0.01, 0.01]), 0.1) == 2
    assert fixed_sequence(np.array([0.01, 0.1, 0.2]), 0.1) == 1
    assert fixed_sequence(np.array([0.01, np.nan, 0.01]), 0.1) == 0


def test_calibrate_hand_cases():
    assert calibrate(np.zeros((4, 3)), np.arange(4), 0.2, 0.1) == 0
    assert calibrate(np.zeros((20, 3)), np.arange(20), 0.2, 0.1) == 2
    losses = np.zeros((50, 4))
    losses[:20, 2:] = 1.0
    assert calibrate(losses, np.arange(50) + 100, 0.2, 0.1) == 1


def test_calibrate_certifies_stream_averaged_losses_despite_float_noise():
    losses = np.zeros((100, 2))
    losses[:, 1] = 7 / 10
    assert hand_written_hb(70, 100, 0.8) < 0.04 < hand_written_hb(71, 100, 0.8)
    assert calibrate(losses, np.arange(100), 0.8, 0.04) == 1


def test_calibrate_rejects_duplicate_images():
    with pytest.raises(AssertionError):
        calibrate(np.zeros((4, 3)), np.array([0, 1, 1, 2]), 0.2, 0.1)


def test_calibrate_rejects_losses_outside_unit_interval():
    with pytest.raises(AssertionError):
        calibrate(np.full((4, 3), 1.5), np.arange(4), 0.2, 0.1)


def staircase_population(rng: np.random.Generator, images: int = 200, streams: int = 10) -> np.ndarray:
    hard = rng.random(images) < 0.18
    onset = np.where(hard, rng.integers(20, 240, images), rng.integers(255, 700, images))
    stream_onset = onset[:, None] + rng.integers(-10, 11, (images, streams))
    return (np.arange(500) >= stream_onset[..., None]).mean(1)


def test_ltt_is_valid_on_without_replacement_splits():
    alpha, delta, draws = 0.2, 0.1, 1000
    rng = np.random.default_rng(20261030)
    losses = staircase_population(rng)
    population_risk = losses.mean(0)
    assert population_risk[0] == 0 and np.all(np.diff(population_risk) >= 0)
    assert 150 <= np.argmax(population_risk > alpha) <= 350
    deployed = np.empty(draws, int)
    for b in range(draws):
        calibration_images = rng.choice(200, 100, replace=False)
        deployed[b] = calibrate(losses[calibration_images], calibration_images, alpha, delta)
    violations = int((population_risk[deployed] > alpha).sum())
    upper = binomtest(violations, draws).proportion_ci(0.95, method="exact").high
    assert deployed.min() > 0
    assert upper <= delta + 0.02
