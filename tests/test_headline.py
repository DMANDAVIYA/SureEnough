import time
from types import SimpleNamespace

import numpy as np

from sureenough.analysis.headline import headline, load_tables
from sureenough.analysis.report import subject_summary
from tests.test_report import concepts, synthetic_probs


def _save(tables, path):
    flat = {f"K20_{k}": v for k, v in tables["K20"].items()}
    flat |= {f"fixed{k}_{name}": v for k, t in tables["fixed"].items() for name, v in t.items()}
    flat |= {f"scores_{k}": v for k, v in tables["scores"].items()}
    np.savez_compressed(path, **flat)


def test_load_tables_round_trip(tmp_path):
    out = subject_summary(synthetic_probs(), 1, concepts(), 0.2, 0.1, 20, 3, n_splits=10, B=10)
    _save(out["tables"], tmp_path / "sub-01.npz")
    tables = load_tables(tmp_path / "sub-01.npz")
    assert set(tables) == {"K20", "fixed", "scores"}
    assert np.array_equal(tables["K20"]["loss_count"], out["tables"]["K20"]["loss_count"])
    assert sorted(tables["fixed"]) == [1, 2, 4, 8, 16, 20]


def test_headline_matches_full_report_and_is_fast(tmp_path):
    summaries = {}
    for s in (1, 2):
        out = subject_summary(synthetic_probs(seed=s), s, concepts(), 0.2, 0.1, 20, 3, n_splits=30, B=10)
        _save(out["tables"], tmp_path / f"sub-{s:02d}.npz")
        summaries[s] = out["summary"]
    cfg = SimpleNamespace(alpha=0.2, delta=0.1, n_splits=30)
    t0 = time.time()
    numbers = headline(tmp_path, [1, 2], cfg.alpha, cfg.delta, cfg.n_splits)
    assert time.time() - t0 < 60
    for s in (1, 2):
        row = numbers["per_subject"][s]
        assert row["pop_risk"] == summaries[s]["pop_risk"]
        assert row["naive_pop_risk"] == summaries[s]["naive_pop_risk"]
        assert row["mean_stop"] == summaries[s]["mean_stop"]
