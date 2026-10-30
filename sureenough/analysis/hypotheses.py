import numpy as np
from scipy.stats import binom, binomtest, wilcoxon

from sureenough.calibrate.ltt import calibrate


def h2a_check(losses_200: np.ndarray, alpha: float, delta: float, B: int = 1000, *, rng: np.random.Generator) -> dict:
    n = len(losses_200)
    population_risk = losses_200.mean(0)
    draw_ids = np.arange(n // 2)
    draws = rng.integers(0, n, (B, n // 2))
    deployed_risk = population_risk[[calibrate(losses_200[d], draw_ids, alpha, delta) for d in draws]]
    violations = int((deployed_risk > alpha).sum())
    mean_risk = float(deployed_risk.mean())
    return {
        "violations": violations,
        "cp_upper": float(binomtest(violations, B).proportion_ci(0.95, method="exact").high),
        "mean_risk": mean_risk,
        "conservativeness": alpha - mean_risk,
    }


def hull_repeats(points: list[tuple[float, float]], at_size: float) -> tuple[float, bool]:
    sizes, repeats = np.asarray(points, np.float64).T
    size = float(np.clip(at_size, sizes.min(), sizes.max()))
    left, right = np.nonzero((sizes[:, None] <= size) & (sizes[None, :] >= size))
    width = sizes[right] - sizes[left]
    share = np.divide(size - sizes[left], width, out=np.zeros_like(width), where=width > 0)
    mixtures = (1 - share) * repeats[left] + share * repeats[right]
    return float(mixtures.min()), size != at_size


def h1_test(ratios: np.ndarray) -> float:
    return float(wilcoxon(np.asarray(ratios) - 0.7, alternative="less").pvalue)


def h2b_test(naive_pop_risk: np.ndarray, alpha: float) -> tuple[int, float]:
    count = int((np.asarray(naive_pop_risk) > alpha).sum())
    return count, float(binom.sf(count - 1, len(naive_pop_risk), 0.5))


def holm(pvals: dict[str, float], level: float = 0.05) -> dict[str, bool]:
    ranked = sorted(pvals, key=pvals.get)
    within = [pvals[name] <= level / (len(ranked) - i) for i, name in enumerate(ranked)]
    return dict(zip(ranked, map(bool, np.logical_and.accumulate(within))))
