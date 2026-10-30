import re
from pathlib import Path

import numpy as np

from sureenough.analysis.splits import run_splits

_FIXED = re.compile(r"fixed(\d+)_(\w+)")


def load_tables(path: Path) -> dict:
    tables = {"K20": {}, "fixed": {}, "scores": {}}
    with np.load(path) as npz:
        for key in npz.files:
            if fixed := _FIXED.fullmatch(key):
                tables["fixed"].setdefault(int(fixed[1]), {})[fixed[2]] = npz[key]
            elif key.startswith(("K20_", "scores_")):
                group, name = key.split("_", 1)
                tables[group][name] = npz[key]
    return tables


def headline(tables_dir: Path, subjects: list[int], alpha: float, delta: float, n_splits: int) -> dict:
    per_subject = {}
    for s in subjects:
        t = load_tables(Path(tables_dir) / f"sub-{s:02d}.npz")
        runs = run_splits(t["K20"], t["fixed"], t["scores"], s, n_splits, alpha, delta)
        ours = runs["sureenough"]
        per_subject[s] = {
            "pop_risk": float(ours["pop_risk"].mean()),
            "naive_pop_risk": float(runs["naive"]["pop_risk"].mean()),
            "mean_stop": float(ours["stop_t"].mean()),
            "mean_set_size": float(ours["set_size"].mean()),
        }
    means = {k: float(np.mean([row[k] for row in per_subject.values()])) for k in next(iter(per_subject.values()))}
    return {"per_subject": per_subject, "mean": means}
