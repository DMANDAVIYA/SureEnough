import numpy as np
import pytest

from sureenough.analysis.hypotheses import h1_test, h2a_check, h2b_test, holm, hull_repeats
from sureenough.analysis.splits import run_splits, split
from sureenough.calibrate.baselines import bonferroni_policy, conformal_scores, fixed_k_tables, naive_policy
from sureenough.calibrate.ltt import calibrate
from sureenough.calibrate.sets import policy_tables


def running_average_streams(rng: np.random.Generator, images: int, streams: int, steps: int) -> np.ndarray:
    evidence = rng.normal(size=(images, streams, steps, images)).cumsum(2) / np.arange(1, steps + 1)[:, None]
    evidence[np.arange(images), :, :, np.arange(images)] += 1.0
    weights = np.exp(4.0 * (evidence - evidence.max(-1, keepdims=True)))
    return (weights / weights.sum(-1, keepdims=True)).astype(np.float16)


def population_crossing_alpha(rng: np.random.Generator, images: int = 200, streams: int = 10) -> np.ndarray:
    onset = rng.integers(11, 500, images)[:, None] + rng.integers(-10, 11, (images, streams))
    return (np.arange(500) >= onset[..., None]).mean(1)


@pytest.fixture(scope="module")
def subject_tables():
    K, streams, images = 4, 3, 60
    probs = running_average_streams(np.random.default_rng(21), images, streams, K)
    fixed = {k: fixed_k_tables(probs, k) for k in (1, 2, 4)}
    return policy_tables(probs, K), fixed, conformal_scores(probs, K), streams, K, images


def test_split_halves_are_disjoint_and_deterministic():
    cal, ev = split(4, 17)
    assert len(cal) == len(ev) == 100
    np.testing.assert_array_equal(np.sort(np.concatenate([cal, ev])), np.arange(200))
    np.testing.assert_array_equal(split(4, 17)[0], cal)
    np.testing.assert_array_equal(split(4, 17)[1], ev)
    assert not np.array_equal(split(4, 18)[0], cal)
    assert not np.array_equal(split(5, 17)[0], cal)


def test_split_uses_the_preregistered_seed():
    order = np.random.default_rng([7, 4, 17]).permutation(200)
    cal, ev = split(4, 17)
    np.testing.assert_array_equal(cal, np.sort(order[:100]))
    np.testing.assert_array_equal(ev, np.sort(order[100:]))


def test_run_splits_records_every_policy_per_split(subject_tables):
    tables, fixed, scores, *_ = subject_tables
    result = run_splits(tables, fixed, scores, subject=3, n_splits=4)
    assert set(result) == {"sureenough", "fixed", "naive", "bonferroni"}
    assert set(result["sureenough"]) == {"lam_idx", "eval_risk", "pop_risk", "stop_t", "set_size"}
    assert set(result["fixed"]) == {1, 2, 4}
    assert all(set(record) == {"lam_idx", "eval_risk", "pop_risk", "set_size"} for record in result["fixed"].values())
    assert set(result["naive"]) == set(result["bonferroni"]) == {"eval_risk", "pop_risk", "stop_t"}
    for records in [result["sureenough"], result["naive"], *result["fixed"].values()]:
        assert all(values.shape == (4,) for values in records.values())
    assert result["sureenough"]["lam_idx"].min() > 0


def test_run_splits_calibrates_stream_averaged_losses_on_each_split(subject_tables):
    tables, fixed, scores, streams, _, images = subject_tables
    result = run_splits(tables, fixed, scores, subject=3, n_splits=4, alpha=0.3, delta=0.2)
    cal, ev = split(3, 2, images)
    for record, table in [(result["sureenough"], tables), (result["fixed"][2], fixed[2])]:
        losses = table["loss_count"] / streams
        lam = calibrate(losses[cal], cal, 0.3, 0.2)
        assert record["lam_idx"][2] == lam
        assert record["eval_risk"][2] == pytest.approx(losses[ev, lam].mean())
        assert record["pop_risk"][2] == pytest.approx(losses[:, lam].mean())
        assert record["set_size"][2] == pytest.approx(table["set_size"][ev, lam].mean())
    lam = result["sureenough"]["lam_idx"][2]
    assert result["sureenough"]["stop_t"][2] == pytest.approx(tables["stop_t"][ev, lam].mean())


def test_run_splits_scores_naive_and_bonferroni_on_each_split(subject_tables):
    tables, fixed, scores, _, K, images = subject_tables
    result = run_splits(tables, fixed, scores, subject=3, n_splits=4, alpha=0.3)
    cal, ev = split(3, 1, images)
    for name, policy in [("naive", naive_policy), ("bonferroni", bonferroni_policy)]:
        expected = policy(scores, cal, ev, 0.3, K)
        assert result[name]["eval_risk"][1] == pytest.approx(expected["loss"].mean())
        assert result[name]["pop_risk"][1] == pytest.approx(expected["pop_loss"].mean())
        assert result[name]["stop_t"][1] == pytest.approx(expected["stop_t"].mean())


def test_h2a_check_is_valid_on_a_population_crossing_alpha():
    losses = population_crossing_alpha(np.random.default_rng(31))
    population_risk = losses.mean(0)
    assert population_risk[0] == 0 and 50 <= np.argmax(population_risk > 0.2) <= 150
    result = h2a_check(losses, 0.2, 0.1, rng=np.random.default_rng(32))
    assert set(result) == {"violations", "cp_upper", "mean_risk", "conservativeness"}
    assert isinstance(result["violations"], int)
    assert result["cp_upper"] <= 0.12
    assert 0 < result["mean_risk"] < 0.2
    assert result["conservativeness"] == pytest.approx(0.2 - result["mean_risk"])


def test_h2a_check_hand_case():
    losses = np.zeros((200, 3))
    losses[:, 2] = 1.0
    result = h2a_check(losses, 0.2, 0.1, B=50, rng=np.random.default_rng(0))
    assert result["violations"] == 0
    assert result["mean_risk"] == 0.0
    assert result["conservativeness"] == pytest.approx(0.2)
    assert result["cp_upper"] == pytest.approx(1 - 0.025 ** (1 / 50))


def test_hull_repeats_interpolates_the_lower_convex_hull():
    points = [(10, 1), (5, 4), (2, 20)]
    assert hull_repeats(points, 5) == (4, False)
    assert hull_repeats(points, 7.5) == (2.5, False)
    assert hull_repeats(points, 1) == (20, True)
    assert hull_repeats(points, 12) == (1, True)
    assert hull_repeats([(10, 1), (5, 15), (2, 20)], 5) == (pytest.approx(12.875), False)
    assert hull_repeats([(4, 9), (4, 3), (8, 1)], 4) == (3, False)


def test_h1_test_is_one_sided_wilcoxon_against_seven_tenths():
    assert h1_test(np.linspace(0.3, 0.6, 10)) == pytest.approx(0.5**10)
    assert h1_test(np.linspace(0.8, 1.1, 10)) == pytest.approx(1.0)


def test_h2b_test_counts_subjects_above_alpha():
    assert h2b_test(np.full(10, 0.3), 0.2) == (10, 0.5**10)
    count, p = h2b_test(np.array([0.3] * 8 + [0.2, 0.1]), 0.2)
    assert count == 8 and p == pytest.approx(56 / 1024)


def test_holm_steps_down():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": True, "b": False, "c": False}
    assert holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert holm({"a": 0.2, "b": 0.001}, level=0.1) == {"a": False, "b": True}
