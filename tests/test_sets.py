import numpy as np
import pytest

from sureenough.calibrate.sets import LAM_GRID, mass_above, policy_tables, rank_desc, top_m_mass


def _set(p, lam):
    return set(np.flatnonzero(mass_above(np.asarray(p)) < lam).tolist())


def test_lam_grid():
    assert LAM_GRID.shape == (500,)
    assert LAM_GRID[0] == 1.0 and LAM_GRID[-1] == pytest.approx(0.002)
    assert np.all(np.diff(LAM_GRID) < 0)
    np.testing.assert_array_equal(LAM_GRID, [1 - j / 500 for j in range(500)])


def test_hand_case():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    np.testing.assert_allclose(mass_above(p), [0, 0.5, 0.8, 0.95])
    assert _set(p, 0.5) == {0}
    assert _set(p, 0.6) == {0, 1}
    assert _set(p, 0.81) == {0, 1, 2}
    assert top_m_mass(p, 3) == pytest.approx(0.95)
    assert top_m_mass(p, 1) == pytest.approx(0.5)


def test_hand_case_permuted_and_batched():
    p = np.array([[0.05, 0.5, 0.15, 0.3], [0.5, 0.3, 0.15, 0.05]])
    np.testing.assert_array_equal(rank_desc(p), [[1, 3, 2, 0], [0, 1, 2, 3]])
    np.testing.assert_allclose(mass_above(p), [[0.95, 0, 0.8, 0.5], [0, 0.5, 0.8, 0.95]])
    np.testing.assert_allclose(top_m_mass(p, 3), [0.95, 0.95])


def test_ties_break_by_lower_index():
    np.testing.assert_array_equal(rank_desc(np.array([0.25] * 4)), [0, 1, 2, 3])
    np.testing.assert_allclose(mass_above(np.array([0.25] * 4)), [0, 0.25, 0.5, 0.75])
    q = np.array([0.1, 0.3, 0.3, 0.3], dtype=np.float16)
    np.testing.assert_array_equal(rank_desc(q), [1, 2, 3, 0])


def _lam1_probs(kind: str, n: int = 4, R: int = 3, T: int = 6) -> np.ndarray:
    G = n
    probs = np.zeros((n, R, T, G), dtype=np.float16)
    for i in range(n):
        others = [j for j in range(G) if j != i]
        if kind == "sum_below_1":
            row = {others[0]: 0.5, others[1]: 0.25, others[2]: 0.125, i: 0.12451171875}
        elif kind == "sum_above_1":
            row = {others[0]: 0.5, others[1]: 0.375, others[2]: 0.125, i: 0.0005}
        else:
            row = {others[0]: 1.0}
        for j, v in row.items():
            probs[i, :, :, j] = v
    return probs


@pytest.mark.parametrize("kind", ["sum_below_1", "sum_above_1", "one_hot_wrong"])
def test_lambda_one_is_full_gallery_never_stops(kind):
    probs = _lam1_probs(kind)
    n, R, T, G = probs.shape
    if kind == "sum_below_1":
        total = probs.astype(np.float64).sum(-1)
        assert np.all(total < 1) and np.allclose(total, 0.9995, atol=1e-4)
    K = 5
    tab = policy_tables(probs, K=K)
    assert np.all(tab["loss_count"][:, 0] == 0)
    assert np.all(tab["stop_t"][:, 0] == K)
    assert np.all(tab["set_size"][:, 0] == G)


def test_stopping_time():
    G, T, K = 6, 8, 8
    probs = np.zeros((1, 1, T, G))
    probs[0, 0, :4] = [0.25, 0.2, 0.15, 0.14, 0.13, 0.13]
    probs[0, 0, 4:] = [0.5, 0.3, 0.15, 0.03, 0.01, 0.01]
    tab = policy_tables(probs, K=K, lam=np.array([1.0, 0.999, 0.9]))
    np.testing.assert_array_equal(tab["stop_t"][0], [K, K, 5])
    np.testing.assert_array_equal(tab["set_size"][0], [G, G, 3])
    np.testing.assert_array_equal(tab["loss_count"][0], [0, 0, 0])
    tab = policy_tables(probs, K=4, lam=np.array([0.9]))
    assert tab["stop_t"][0, 0] == 4 and tab["set_size"][0, 0] == G


def test_output_types():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(9), size=(3, 2, 5)).astype(np.float16)
    tab = policy_tables(probs, K=5)
    assert tab["loss_count"].dtype == np.uint8 and tab["loss_count"].shape == (3, 500)
    assert tab["stop_t"].dtype == np.float32 and tab["stop_t"].shape == (3, 500)
    assert tab["set_size"].dtype == np.float32 and tab["set_size"].shape == (3, 500)


def test_monotone_in_lambda():
    rng = np.random.default_rng(0)
    n, R, T, G = 30, 4, 20, 30
    base = rng.dirichlet(np.full(G, 0.5), size=(n, R))
    boost = np.exp(0.25 * np.arange(1, T + 1))
    w = np.repeat(base[:, :, None, :], T, axis=2)
    w[np.arange(n), :, :, np.arange(n)] *= boost
    probs = (w / w.sum(-1, keepdims=True)).astype(np.float16)
    tab = policy_tables(probs, K=T)
    loss = tab["loss_count"].mean(0)
    stop = tab["stop_t"].mean(0)
    assert np.all(np.diff(loss) >= 0)
    assert np.all(np.diff(stop) <= 0)
    assert loss[0] == 0 and loss[-1] > 0 and stop[0] == T and stop[-1] < T


def _reference(probs, K, m, lam):
    n, R, T, G = probs.shape
    L = len(lam)
    loss = np.zeros((n, L), dtype=np.int64)
    stop = np.zeros((n, L))
    size = np.zeros((n, L))
    for i in range(n):
        for r in range(R):
            for l, lm in enumerate(lam):
                if lm >= 1.0:
                    stop[i, l] += K
                    size[i, l] += G
                    continue
                for t in range(K):
                    p = [float(x) for x in probs[i, r, t]]
                    order = sorted(range(G), key=lambda j: (-p[j], j))
                    a, s = [0.0] * G, 0.0
                    for j in order:
                        a[j] = s
                        s += p[j]
                    c = [j for j in range(G) if a[j] < lm]
                    if len(c) <= m or t == K - 1:
                        break
                loss[i, l] += a[i] >= lm
                stop[i, l] += t + 1
                size[i, l] += len(c)
    return loss, stop / R, size / R


@pytest.mark.parametrize("quantised", [False, True])
@pytest.mark.parametrize("K,m", [(4, 3), (6, 3), (6, 1)])
def test_matches_loop_reference(quantised, K, m):
    rng = np.random.default_rng([5, K, m, quantised])
    probs = rng.dirichlet(np.ones(7), size=(5, 2, 6))
    if quantised:
        probs = np.round(probs * 16) / 16
    probs = probs.astype(np.float16)
    tab = policy_tables(probs, K=K, m=m)
    loss, stop, size = _reference(probs, K, m, LAM_GRID)
    np.testing.assert_array_equal(tab["loss_count"], loss)
    np.testing.assert_allclose(tab["stop_t"], stop, rtol=0, atol=1e-6)
    np.testing.assert_allclose(tab["set_size"], size, rtol=0, atol=1e-6)
