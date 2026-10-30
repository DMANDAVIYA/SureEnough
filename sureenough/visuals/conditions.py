import numpy as np

from sureenough.analysis.report import set_at, stop_step

CONDITIONS = ("hedged", "top1", "answer_key", "blind_guess")


def stopped_state(stream: np.ndarray, lam: float, K: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    step = np.asarray(stream[stop_step(stream, lam, K, m)], np.float64)
    return set_at(step, lam), step


def condition_members(probs: np.ndarray, image: int, lam: float, K: int, m: int) -> dict[str, tuple[list[int], list[float]]]:
    members, step = stopped_state(probs[image, 0], lam, K, m)
    blind, blind_step = stopped_state(probs[(image + 1) % len(probs), 0], lam, K, m)
    return {
        "hedged": (members.tolist(), step[members].tolist()),
        "top1": ([int(members[0])], [1.0]),
        "answer_key": ([image], [1.0]),
        "blind_guess": (blind.tolist(), blind_step[blind].tolist()),
    }
