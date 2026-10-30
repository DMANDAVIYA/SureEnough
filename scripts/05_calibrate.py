import argparse
import json

import numpy as np

from sureenough.analysis.figures import write_figures
from sureenough.analysis.hypotheses import holm
from sureenough.analysis.report import combine, subject_summary
from sureenough.config import load_frozen
from sureenough.data.concepts import load_concepts
from sureenough.paths import OUTPUTS, subject_ids

STREAMS = OUTPUTS / "streams"
TABLES = OUTPUTS / "tables"
RESULTS = OUTPUTS / "results.json"


def save_tables(subject: int, tables: dict) -> None:
    flat = {f"K20_{k}": v for k, v in tables["K20"].items()}
    flat |= {f"K80_{k}": v for k, v in tables.get("K80", {}).items()}
    flat |= {f"fixed{k}_{name}": v for k, t in tables["fixed"].items() for name, v in t.items()}
    flat |= {f"scores_{k}": v for k, v in tables["scores"].items()}
    np.savez_compressed(TABLES / f"sub-{subject:02d}.npz", **flat)


def run() -> None:
    cfg = load_frozen()
    concepts = load_concepts()
    TABLES.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for s in subject_ids():
        probs = np.load(STREAMS / f"sub-{s:02d}.npy")
        out = subject_summary(probs, s, concepts, cfg["alpha"], cfg["delta"], cfg["K"], cfg["m"],
                              n_splits=cfg["n_splits"], B=1000)
        save_tables(s, out["tables"])
        summaries[s] = out["summary"]
        print(f"sub-{s:02d}: risk {summaries[s]['pop_risk']:.3f} naive {summaries[s]['naive_pop_risk']:.3f} "
              f"stop {summaries[s]['mean_stop']:.1f} size {summaries[s]['mean_set_size']:.1f}", flush=True)
    results = combine(summaries, cfg["alpha"]) | {"config": cfg, "cp_bound": "two-sided 95% Clopper-Pearson upper bound"}
    RESULTS.write_text(json.dumps(results, indent=1))
    write_figures(results, cfg["alpha"], OUTPUTS / "figures")
    print(json.dumps({"pvalues": results["pvalues"], "holm": results["holm"]}, indent=1))


def add_h5a(path: str) -> None:
    results = json.loads(RESULTS.read_text())
    h5a = json.loads(open(path, encoding="utf-8").read())
    results["h5a"] = h5a
    results["pvalues"]["H5a"] = h5a["p"]
    results["holm"] = holm(results["pvalues"])
    RESULTS.write_text(json.dumps(results, indent=1))
    print(json.dumps({"pvalues": results["pvalues"], "holm": results["holm"]}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5a", metavar="JSON")
    args = ap.parse_args()
    if args.h5a:
        add_h5a(args.h5a)
    else:
        run()


if __name__ == "__main__":
    main()
