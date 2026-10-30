import numpy as np

from sureenough.analysis.hypotheses import h1_test, h2a_check, h2b_test, holm, hull_repeats
from sureenough.analysis.splits import run_splits, split
from sureenough.calibrate.baselines import conformal_scores, fixed_k_tables, uncalibrated_policy
from sureenough.calibrate.ltt import calibrate
from sureenough.calibrate.sets import LAM_GRID, mass_above, policy_tables, rank_desc, top_m_mass
from sureenough.narrate.display import display_claim

FIXED_KS = (1, 2, 4, 8, 16, 20)
KINDS = ("single", "category", "list", "full")


def stop_step(stream: np.ndarray, lam: float, K: int, m: int) -> int:
    if lam >= 1.0:
        return K - 1
    hits = top_m_mass(stream[:K], m) >= lam
    return int(hits.argmax()) if hits.any() else K - 1


def set_at(step_probs: np.ndarray, lam: float) -> np.ndarray:
    order = rank_desc(step_probs)
    if lam >= 1.0:
        return order
    inside = mass_above(step_probs) < lam
    return order[inside[order]]


def final_sets(probs: np.ndarray, lam: float, K: int, m: int) -> list[list[np.ndarray]]:
    return [[set_at(stream[stop_step(stream, lam, K, m)], lam) for stream in image] for image in probs]


def outcome_mix(probs: np.ndarray, eval_idx: np.ndarray, lam: float, K: int, m: int, concepts) -> dict:
    kinds = [display_claim(members.tolist(), concepts)["kind"]
             for row in final_sets(probs[eval_idx], lam, K, m) for members in row]
    return {kind: kinds.count(kind) / len(kinds) for kind in KINDS}


def subject_tables(probs: np.ndarray, K: int, m: int) -> dict:
    tables = {
        "K20": policy_tables(probs, K=K, m=m),
        "fixed": {k: fixed_k_tables(probs, k) for k in FIXED_KS if k <= K},
        "scores": conformal_scores(probs, K),
    }
    if probs.shape[2] >= 80:
        tables["K80"] = policy_tables(probs, K=80, m=m)
    return tables


def subject_summary(probs, subject, concepts, alpha, delta, K, m, n_splits=1000, B=1000) -> dict:
    R = probs.shape[1]
    tables = subject_tables(probs, K, m)
    runs = run_splits(tables["K20"], tables["fixed"], tables["scores"], subject, n_splits, alpha, delta)
    ours, naive, bonf = runs["sureenough"], runs["naive"], runs["bonferroni"]
    losses = tables["K20"]["loss_count"] / R
    points = [(float(runs["fixed"][k]["set_size"].mean()), float(k)) for k in tables["fixed"]]
    mean_size, mean_stop = float(ours["set_size"].mean()), float(ours["stop_t"].mean())
    hull, clamped = hull_repeats(points, mean_size)
    cal, ev = split(subject, 0, len(probs))
    pinned = calibrate(losses[cal], cal, alpha, delta)
    uncalibrated = uncalibrated_policy(probs, K)
    summary = {
        "eval_risk": float(ours["eval_risk"].mean()),
        "pop_risk": float(ours["pop_risk"].mean()),
        "split_violation_rate": float((ours["pop_risk"] > alpha).mean()),
        "conservativeness": float(alpha - ours["pop_risk"].mean()),
        "mean_stop": mean_stop,
        "mean_set_size": mean_size,
        "naive_pop_risk": float(naive["pop_risk"].mean()),
        "naive_mean_stop": float(naive["stop_t"].mean()),
        "bonferroni_pop_risk": float(bonf["pop_risk"].mean()),
        "bonferroni_mean_stop": float(bonf["stop_t"].mean()),
        "uncalibrated_risk": float(uncalibrated["loss"].mean()),
        "uncalibrated_mean_stop": float(uncalibrated["stop_t"].mean()),
        "fixed_points": points,
        "fixed_eval_risk": {k: float(runs["fixed"][k]["eval_risk"].mean()) for k in tables["fixed"]},
        "h1_ratio": mean_stop / hull,
        "h1_clamped": clamped,
        "h2a": h2a_check(losses, alpha, delta, B, rng=np.random.default_rng([11, subject])),
        "outcome_mix": outcome_mix(probs, ev, float(LAM_GRID[pinned]), K, m, concepts),
        "pinned_lam_idx": int(pinned),
        "pinned_lam": float(LAM_GRID[pinned]),
        "per_image": {
            "stop": tables["K20"]["stop_t"][:, pinned].tolist(),
            "size": tables["K20"]["set_size"][:, pinned].tolist(),
            "loss": losses[:, pinned].tolist(),
        },
    }
    return {"summary": summary, "tables": tables}


def hier_bootstrap_ci(values: list[np.ndarray], B: int = 2000, rng=None, level: float = 0.95) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    groups = [np.asarray(v, np.float64) for v in values]
    stats = np.empty(B)
    for b in range(B):
        picked = rng.integers(0, len(groups), len(groups))
        stats[b] = np.mean([rng.choice(groups[g], len(groups[g])).mean() for g in picked])
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))


def combine(summaries: dict[int, dict], alpha: float) -> dict:
    subjects = sorted(summaries)
    ratios = np.array([summaries[s]["h1_ratio"] for s in subjects])
    count, p_h2b = h2b_test(np.array([summaries[s]["naive_pop_risk"] for s in subjects]), alpha)
    pvalues = {"H1": h1_test(ratios), "H2b": p_h2b}
    headline = {}
    for key in ("stop", "size", "loss"):
        values = [np.asarray(summaries[s]["per_image"][key]) for s in subjects]
        headline[key] = {"mean": float(np.mean([v.mean() for v in values])), "ci95": hier_bootstrap_ci(values)}
    return {
        "per_subject": {str(s): summaries[s] for s in subjects},
        "pvalues": pvalues,
        "h2b_count": count,
        "holm": holm(pvalues),
        "headline": headline,
    }
