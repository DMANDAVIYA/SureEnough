import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from sureenough.analysis.report import stop_step
from sureenough.calibrate.baselines import naive_policy
from sureenough.calibrate.sets import mass_above, rank_desc

FORBIDDEN_KEY_PARTS = ("embedding", "feature")
EEG_POINTS = 50


class BundleError(RuntimeError):
    pass


def u16(x) -> np.ndarray:
    return np.round(np.clip(np.asarray(x, np.float64), 0.0, 1.0) * 65535).astype(np.int64)


def table_payload(tables: dict) -> dict:
    return {
        "loss_count": tables["K20_loss_count"].astype(int).tolist(),
        "stop_x10": np.round(tables["K20_stop_t"].astype(np.float64) * 10).astype(int).tolist(),
        "size_x10": np.round(tables["K20_set_size"].astype(np.float64) * 10).astype(int).tolist(),
        "naive": {
            "a_true_u16": u16(tables["scores_a_true"]).tolist(),
            "top3_u16": u16(tables["scores_top3"]).tolist(),
        },
    }


def example_payload(stream, eeg, order, lam: float, K: int, m: int, channels: list[int], top: int = 30) -> dict:
    p = np.asarray(stream[:K], np.float64)
    ranks = rank_desc(p)[:, :top]
    sizes = [p.shape[1] if lam >= 1.0 else int((mass_above(step) < lam).sum()) for step in p]
    running = np.cumsum(eeg[order[:K]][:, channels].mean(axis=1), axis=0) / np.arange(1, K + 1)[:, None]
    trace = running.reshape(K, EEG_POINTS, -1).mean(axis=-1)
    scale = 127.0 / max(float(np.abs(trace).max()), 1e-12)
    return {
        "top": [[[int(j), int(v)] for j, v in zip(row, u16(p[t, row]))] for t, row in enumerate(ranks)],
        "set_size": sizes,
        "stopped_at": stop_step(stream, lam, K, m) + 1,
        "eeg": np.round(trace * scale).astype(int).tolist(),
    }


def naive_eval_risk_u16(a_true_u16, top3_u16, cal_idx, eval_idx, alpha: float, K: int) -> float:
    scores = {"a_true": np.asarray(a_true_u16) / 65535, "top3": np.asarray(top3_u16) / 65535}
    return float(naive_policy(scores, np.asarray(cal_idx), np.asarray(eval_idx), alpha, K)["loss"].mean())


def _keys(obj) -> Iterator[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _keys(value)
    elif isinstance(obj, list) and obj and isinstance(obj[0], (dict, list)):
        for value in obj:
            yield from _keys(value)


def check_bundle(out_dir: Path, forbidden_hashes: set[str], max_mb: float = 150) -> float:
    total = 0
    for path in sorted(Path(out_dir).rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        total += len(data)
        if hashlib.sha256(data).hexdigest() in forbidden_hashes:
            raise BundleError(f"{path} is byte-identical to a THINGS stimulus")
        if path.suffix == ".json":
            bad = sorted({k for k in _keys(json.loads(data)) if any(part in k.lower() for part in FORBIDDEN_KEY_PARTS)})
            if bad:
                raise BundleError(f"{path} contains forbidden keys {bad}")
    if total > max_mb * 1024 * 1024:
        raise BundleError(f"bundle is {total / 2**20:.1f} MB, over {max_mb} MB")
    return total / 2**20
