import numpy as np

from sureenough.visuals.conditions import condition_members


def _probs():
    p = np.full((4, 2, 20, 4), 0.01)
    p[:, :, :, 0] = 0.97
    p[1, 0, :, :] = [0.4, 0.35, 0.2, 0.05]
    return (p / p.sum(-1, keepdims=True)).astype(np.float16)


def test_conditions_follow_the_stopped_set():
    out = condition_members(_probs(), image=1, lam=0.9, K=20, m=3)
    hedged, weights = out["hedged"]
    assert hedged == [0, 1, 2] and abs(sum(weights) - 0.95) < 1e-2
    assert out["top1"] == ([0], [1.0])
    assert out["answer_key"] == ([1], [1.0])


def test_blind_guess_uses_next_image_stream():
    out = condition_members(_probs(), image=3, lam=0.9, K=20, m=3)
    assert out["blind_guess"][0] == [0]
