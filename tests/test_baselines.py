import numpy as np
import pytest

from sureenough.calibrate.baselines import (
    bonferroni_policy,
    conformal_quantile,
    conformal_scores,
    fixed_k_tables,
    naive_policy,
    uncalibrated_policy,
)
from sureenough.calibrate.sets import LAM_GRID, mass_above


def running_average_streams(rng: np.random.Generator, images: int, streams: int, steps: int) -> np.ndarray:
    repeats = np.arange(1, steps + 1)[:, None]
    evidence = rng.normal(size=(images, streams, steps, images)).cumsum(2) / repeats
    evidence[np.arange(images), :, :, np.arange(images)] += 1.0
    logits = 10.0 * evidence
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    return (weights / weights.sum(-1, keepdims=True)).astype(np.float16)


def true_item_ranked_last(images: int = 4, streams: int = 2, steps: int = 3) -> np.ndarray:
    probs = np.zeros((images, streams, steps, images), dtype=np.float16)
    for image in range(images):
        others = [j for j in range(images) if j != image]
        for item, value in zip(others + [image], [0.5, 0.375, 0.125, 0.0005]):
            probs[image, :, :, item] = value
    return probs


def test_fixed_k_tables_score_the_set_at_step_k():
    rng = np.random.default_rng(3)
    images, k = 5, 3
    probs = rng.dirichlet(np.ones(6), size=(images, 2, 4)).astype(np.float16)
    tables = fixed_k_tables(probs, k)
    above = mass_above(probs[:, :, k - 1])
    a_true = above[np.arange(images), :, np.arange(images)]
    assert set(tables) == {"loss_count", "set_size"}
    assert tables["loss_count"].dtype == np.uint8 and tables["loss_count"].shape == (images, 500)
    assert tables["set_size"].dtype == np.float32 and tables["set_size"].shape == (images, 500)
    np.testing.assert_array_equal(tables["loss_count"][:, 1:], (a_true[..., None] >= LAM_GRID[1:]).sum(1))
    expected_size = (above[..., None] < LAM_GRID[1:]).sum(-2).mean(1)
    np.testing.assert_allclose(tables["set_size"][:, 1:], expected_size, rtol=0, atol=1e-6)


def test_fixed_k_tables_lambda_one_is_full_gallery_with_no_loss():
    probs = true_item_ranked_last()
    assert np.all(mass_above(probs)[0, :, :, 0] >= 1.0)
    tables = fixed_k_tables(probs, 2)
    assert np.all(tables["loss_count"][:, 0] == 0)
    assert np.all(tables["set_size"][:, 0] == 4)


def test_conformal_scores_hand_case():
    probs = np.zeros((2, 1, 3, 4))
    probs[:, :, :2] = [0.5, 0.3, 0.15, 0.05]
    probs[:, :, 2] = [0.05, 0.15, 0.3, 0.5]
    scores = conformal_scores(probs, K=2)
    assert set(scores) == {"a_true", "top3"}
    np.testing.assert_allclose(scores["a_true"], [[[0.0, 0.0]], [[0.5, 0.5]]])
    np.testing.assert_allclose(scores["top3"], np.full((2, 1, 2), 0.95))


def test_conformal_quantile_rank_and_infinity():
    calibration = np.arange(9.0)[::-1, None] / 10
    np.testing.assert_array_equal(conformal_quantile(calibration, 0.7), [0.2])
    np.testing.assert_array_equal(conformal_quantile(calibration, 0.2), [0.7])
    np.testing.assert_array_equal(conformal_quantile(calibration, 0.05), [np.inf])


def test_naive_policy_hand_case():
    a_true = np.array([[0.1, 0.0], [0.3, 0.05], [0.5, 0.4], [0.2, 0.6]])[:, None, :]
    top3 = np.array([[0.9, 0.0], [0.2, 0.0], [0.35, 0.0], [0.31, 0.0]])[:, None, :]
    result = naive_policy({"a_true": a_true, "top3": top3}, np.array([0, 1, 2]), np.array([3, 1]), 0.5, K=2)
    np.testing.assert_array_equal(result["pop_loss"], [0, 0, 1, 0])
    np.testing.assert_array_equal(result["loss"], [0, 0])
    np.testing.assert_array_equal(result["stop_t"], [1, 2])


def test_naive_policy_at_a_forced_step_keeps_split_conformal_coverage():
    alpha, splits = 0.2, 500
    rng = np.random.default_rng(11)
    scores = conformal_scores(running_average_streams(rng, 200, 3, 4), K=4)
    never_small = {"a_true": scores["a_true"], "top3": np.full_like(scores["top3"], -np.inf)}
    for step in range(1, 5):
        coverage = np.empty(splits)
        for s in range(splits):
            order = rng.permutation(200)
            result = naive_policy(never_small, order[:100], order[100:], alpha, K=step)
            assert np.all(result["stop_t"] == step)
            coverage[s] = 1 - result["loss"].mean()
        assert coverage.mean() >= 1 - alpha - 0.02


def test_naive_policy_with_stopping_reports_its_miss_rate():
    alpha, K = 0.2, 8
    rng = np.random.default_rng(12)
    scores = conformal_scores(running_average_streams(rng, 200, 3, K), K=K)
    miss, stop = [], []
    for _ in range(50):
        order = rng.permutation(200)
        result = naive_policy(scores, order[:100], order[100:], alpha, K)
        assert result["loss"].shape == result["stop_t"].shape == (100,)
        assert result["pop_loss"].shape == (200,)
        np.testing.assert_array_equal(result["loss"], result["pop_loss"][order[100:]])
        assert np.all((result["stop_t"] >= 1) & (result["stop_t"] <= K))
        miss.append(result["pop_loss"].mean())
        stop.append(result["stop_t"].mean())
    assert np.mean(stop) < K - 1
    print(f"naive repeated look population miss rate {np.mean(miss):.3f} at alpha {alpha}")


def test_bonferroni_threshold_dominates_naive():
    K = 6
    rng = np.random.default_rng(13)
    scores = conformal_scores(running_average_streams(rng, 200, 3, K), K=K)
    cal, ev = np.arange(100), np.arange(100, 200)
    a_cal = scores["a_true"][cal]
    assert np.all(conformal_quantile(a_cal, 0.2 / K) >= conformal_quantile(a_cal, 0.2))
    bonferroni = bonferroni_policy(scores, cal, ev, 0.2, K)
    naive = naive_policy(scores, cal, ev, 0.2, K)
    assert np.all(bonferroni["stop_t"] >= naive["stop_t"])
    for key in ("loss", "stop_t", "pop_loss"):
        np.testing.assert_array_equal(bonferroni[key], naive_policy(scores, cal, ev, 0.2 / K, K)[key])


def test_uncalibrated_policy_hand_case():
    probs = np.zeros((3, 2, 3, 4), dtype=np.float16)
    probs[0, 0] = [0.9, 0.05, 0.05, 0.0]
    probs[0, 1] = [0.1, 0.2, 0.7, 0.0]
    probs[1, :, 0] = [0.2, 0.5, 0.2, 0.1]
    probs[1, :, 1:] = [0.05, 0.05, 0.05, 0.85]
    probs[2, :, :2] = [0.0, 0.45, 0.45, 0.1]
    probs[2, :, 2] = [0.0, 0.0, 0.95, 0.05]
    result = uncalibrated_policy(probs, K=2)
    np.testing.assert_array_equal(result["loss"], [0.5, 1.0, 1.0])
    np.testing.assert_array_equal(result["stop_t"], [1.5, 2.0, 2.0])


@pytest.mark.parametrize("stored, stops", [(0.8, False), (0.875, True)])
def test_uncalibrated_threshold_compares_the_stored_probability(stored, stops):
    probs = np.zeros((1, 1, 2, 4), dtype=np.float16)
    probs[0, 0, :, 0] = stored
    probs[0, 0, :, 1] = 1 - stored
    assert uncalibrated_policy(probs, K=2, thresh=0.8)["stop_t"][0] == (1 if stops else 2)
