import json
import time

from sureenough.analysis.headline import headline
from sureenough.config import load_frozen
from sureenough.paths import OUTPUTS, subject_ids


def main() -> None:
    t0 = time.time()
    cfg = load_frozen()
    numbers = headline(OUTPUTS / "tables", subject_ids(), cfg["alpha"], cfg["delta"], cfg["n_splits"])
    stored = json.loads((OUTPUTS / "results.json").read_text())["per_subject"]
    for s, row in numbers["per_subject"].items():
        for key, value in row.items():
            assert abs(value - stored[str(s)][key]) < 1e-12, (s, key, value, stored[str(s)][key])
    print(json.dumps(numbers["mean"], indent=1))
    print(f"recomputed from cached tables and matched results.json in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
