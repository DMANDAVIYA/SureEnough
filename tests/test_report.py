from types import SimpleNamespace

import numpy as np
import pytest

from sureenough.analysis.report import (
    combine,
    final_sets,
    hier_bootstrap_ci,
    outcome_mix,
    subject_summary,
)


def synthetic_probs(n=200, R=4, T=20, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, R, T, n))
    strength = rng.uniform(0.0, 0.6, size=(n, 1, 1))
    logits[np.arange(n), :, :, np.arange(n)] += strength * np.arange(1, T + 1)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)).astype(np.float16)


def concepts(n=200):
    ids = [f"c{i}" for i in range(n)]
    return SimpleNamespace(
        test_things_ids=ids,
        test_names=[f"thing{i}" for i in range(n)],
        categories53={"small": set(ids[:10]), "big": set(ids[:120])},
    )


@pytest.fixture(scope="module")
def probs():
    return synthetic_probs()


def test_final_sets_full_gallery_at_lambda_one(probs):
    sets = final_sets(probs[:3], lam=1.0, K=20, m=3)
    assert len(sets) == 3 and len(sets[0]) == probs.shape[1]
    assert all(len(s) == 200 for row in sets for s in row)


def test_final_sets_shrink_as_lambda_falls(probs):
    big = final_sets(probs[:5], lam=0.99, K=20, m=3)
    small = final_sets(probs[:5], lam=0.5, K=20, m=3)
    assert sum(map(len, sum(small, []))) <= sum(map(len, sum(big, [])))


def test_outcome_mix_shares_sum_to_one(probs):
    mix = outcome_mix(probs, np.arange(50), lam=0.9, K=20, m=3, concepts=concepts())
    assert set(mix) == {"single", "category", "list", "full"}
    assert abs(sum(mix.values()) - 1.0) < 1e-9


def test_subject_summary_keys_and_validity(probs):
    out = subject_summary(probs, subject=1, concepts=concepts(), alpha=0.2, delta=0.1, K=20, m=3, n_splits=40, B=40)
    s = out["summary"]
    for key in ("eval_risk", "pop_risk", "mean_stop", "mean_set_size", "naive_pop_risk", "bonferroni_pop_risk",
                "uncalibrated_risk", "fixed_points", "h1_ratio", "h1_clamped", "h2a", "outcome_mix", "pinned_lam_idx"):
        assert key in s
    assert s["h2a"]["cp_upper"] <= 1.0
    assert set(out["tables"]) >= {"K20", "K80", "fixed", "scores"} or set(out["tables"]) >= {"K20", "fixed", "scores"}


def test_combine_runs_confirmatory_family(probs):
    summaries = [subject_summary(probs, subject=s, concepts=concepts(), alpha=0.2, delta=0.1, K=20, m=3, n_splits=20, B=20)["summary"]
                 for s in (1, 2)]
    out = combine({1: summaries[0], 2: summaries[1]}, alpha=0.2)
    assert set(out["pvalues"]) == {"H1", "H2b"}
    assert set(out["holm"]) == {"H1", "H2b"}
    assert len(out["per_subject"]) == 2


def test_hier_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    values = [rng.normal(loc, 1, size=50) for loc in (0.0, 0.2, 0.4)]
    lo, hi = hier_bootstrap_ci(values, B=500, rng=np.random.default_rng(1))
    assert lo < 0.2 < hi
