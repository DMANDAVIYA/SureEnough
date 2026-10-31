import json
from pathlib import Path

import numpy as np

from sureenough.analysis.splits import split
from sureenough.calibrate.baselines import conformal_scores
from sureenough.calibrate.ltt import calibrate
from sureenough.calibrate.sets import policy_tables
from sureenough.export.web_data import naive_eval_risk_u16, table_payload

OUT = Path(__file__).resolve().parents[1] / "src" / "lib" / "fixtures" / "ltt_fixture.json"
ALPHA, DELTA, K, M = 0.2, 0.1, 8, 3
IMAGES, STREAMS, SUBJECT, SPLITS = 100, 4, 1, 5
LAM = 1 - np.arange(100) / 100
BETA = 3.0


def synthetic_streams(rng: np.random.Generator) -> np.ndarray:
    strength = rng.gamma(2.0, 1.5, IMAGES)[:, None, None, None]
    truth = np.eye(IMAGES)[:, None, None, :]
    noise = rng.normal(size=(IMAGES, STREAMS, K, IMAGES)).cumsum(2) / np.arange(1, K + 1)[:, None]
    logits = BETA * (strength * truth + noise)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)).astype(np.float16)


def fixture_table(probs: np.ndarray) -> dict:
    tables = policy_tables(probs, K=K, m=M, lam=LAM)
    scores = conformal_scores(probs, K)
    return table_payload({
        "K20_loss_count": tables["loss_count"], "K20_stop_t": tables["stop_t"], "K20_set_size": tables["set_size"],
        "scores_a_true": scores["a_true"], "scores_top3": scores["top3"],
    })


def fixture_split(table: dict, s: int) -> dict:
    losses = np.asarray(table["loss_count"]) / STREAMS
    cal, ev = split(SUBJECT, s, IMAGES)
    lam_idx = calibrate(losses[cal], cal, ALPHA, DELTA)
    naive = naive_eval_risk_u16(table["naive"]["a_true_u16"], table["naive"]["top3_u16"], cal, ev, ALPHA, K)
    return {"cal": cal.tolist(), "eval": ev.tolist(), "lam_idx": lam_idx,
            "eval_risk": float(losses[ev, lam_idx].mean()), "naive_eval_risk": naive}


def main() -> None:
    table = fixture_table(synthetic_streams(np.random.default_rng(20261030)))
    splits = [fixture_split(table, s) for s in range(SPLITS)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"alpha": ALPHA, "delta": DELTA, "table": table, "splits": splits}, separators=(",", ":")))
    print(OUT, [(s["lam_idx"], round(s["eval_risk"], 3), round(s["naive_eval_risk"], 3)) for s in splits])


if __name__ == "__main__":
    main()
