import numpy as np

LAM_GRID = 1 - np.arange(500) / 500
LAM_GRID.flags.writeable = False


def rank_desc(p: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(p), axis=-1, kind="stable")


def _ranked(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = rank_desc(p)
    cs = np.cumsum(np.take_along_axis(np.asarray(p, np.float64), order, -1), -1)
    return order, cs, np.concatenate([np.zeros_like(cs[..., :1]), cs[..., :-1]], -1)


def mass_above(p: np.ndarray) -> np.ndarray:
    order, _, excl = _ranked(p)
    a = np.empty_like(excl)
    np.put_along_axis(a, order, excl, -1)
    return a


def top_m_mass(p: np.ndarray, m: int = 3) -> np.ndarray:
    return _ranked(p)[1][..., m - 1]


def policy_tables(probs: np.ndarray, K: int, m: int = 3, lam: np.ndarray = LAM_GRID) -> dict:
    n, R, T, G = probs.shape
    lam = np.asarray(lam, np.float64)
    assert T >= K >= 1 and 1 <= m < G and n <= G and R <= 255
    assert np.all((lam > 0) & (lam <= 1))
    full = lam >= 1.0
    rows = np.arange(R)[:, None]
    loss, stop, size = (np.empty((n, len(lam)), d) for d in (np.uint8, np.float32, np.float32))
    for i in range(n):
        order, cs, excl = _ranked(probs[i, :, :K])
        hit = cs[..., m - 1, None] >= lam
        t = np.where(hit.any(1), hit.argmax(1), K - 1)
        t[:, full] = K - 1
        a_true = np.take_along_axis(excl, np.argmax(order == i, -1)[..., None], -1)[..., 0]
        miss = (np.take_along_axis(a_true, t, 1) >= lam) & ~full
        n_in = np.where(full, G, (excl[rows, t] < lam[:, None]).sum(-1))
        loss[i], stop[i], size[i] = miss.sum(0), (t + 1).mean(0), n_in.mean(0)
    return {"loss_count": loss, "stop_t": stop, "set_size": size}
