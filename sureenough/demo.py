import json

import numpy as np

from sureenough.calibrate.sets import LAM_GRID
from sureenough.config import load_frozen
from sureenough.paths import OUTPUTS

N_DEMO_SUBJECTS = 2


def demo_subjects(n: int = N_DEMO_SUBJECTS) -> list[int]:
    target = load_frozen()["target"]
    gate = json.loads((OUTPUTS / f"gate1_{target}.json").read_text())
    scores = {int(s): v["0"] for s, v in gate["per_subject"].items()}
    return sorted(scores, key=scores.get, reverse=True)[:n]


def pinned_lambda(subject: int) -> float:
    results = json.loads((OUTPUTS / "results.json").read_text())
    return float(LAM_GRID[results["per_subject"][str(subject)]["pinned_lam_idx"]])


def load_streams(subject: int) -> np.ndarray:
    return np.load(OUTPUTS / "streams" / f"sub-{subject:02d}.npy")
