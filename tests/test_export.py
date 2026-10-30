import json

import numpy as np
import pytest

from sureenough.export.web_data import (
    BundleError,
    check_bundle,
    example_payload,
    naive_eval_risk_u16,
    table_payload,
    u16,
)


def _probs(n=6, R=3, T=20, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(n), size=(n, R, T))
    return p.astype(np.float16)


def test_u16_quantises_unit_interval():
    assert u16(np.array([0.0, 0.5, 1.0])).tolist() == [0, 32768, 65535]


def test_table_payload_types_and_shapes():
    tables = {
        "K20_loss_count": np.array([[0, 3], [10, 1]], np.uint8),
        "K20_stop_t": np.array([[20.0, 3.26], [1.0, 7.04]], np.float32),
        "K20_set_size": np.array([[200.0, 2.5], [1.0, 3.0]], np.float32),
        "scores_a_true": np.full((2, 3, 20), 0.25),
        "scores_top3": np.full((2, 3, 20), 0.75),
    }
    out = table_payload(tables)
    assert out["loss_count"] == [[0, 3], [10, 1]]
    assert out["stop_x10"] == [[200, 33], [10, 70]]
    assert out["size_x10"] == [[2000, 25], [10, 30]]
    assert np.array(out["naive"]["a_true_u16"]).shape == (2, 3, 20)
    assert out["naive"]["top3_u16"][0][0][0] == 49151
    json.dumps(out)


def test_example_payload_steps_and_stop():
    probs = _probs()
    eeg = np.random.default_rng(1).normal(size=(80, 17, 100)).astype(np.float32)
    ex = example_payload(probs[0, 0], eeg, np.arange(80), lam=0.5, K=20, m=3, channels=[3, 4, 5], top=4)
    assert len(ex["top"]) == 20 and all(len(step) == 4 for step in ex["top"])
    first = ex["top"][0]
    assert [p for _, p in first] == sorted((p for _, p in first), reverse=True)
    assert 1 <= ex["stopped_at"] <= 20
    assert len(ex["set_size"]) == 20 and len(ex["eeg"]) == 20 and len(ex["eeg"][0]) == 50
    assert max(abs(v) for step in ex["eeg"] for v in step) <= 127
    json.dumps(ex)


def test_example_payload_lambda_one_never_stops_early():
    ex = example_payload(_probs()[0, 0], np.zeros((80, 17, 100), np.float32), np.arange(80), lam=1.0, K=20, m=3, channels=[0], top=3)
    assert ex["stopped_at"] == 20 and set(ex["set_size"]) == {6}


def test_naive_eval_risk_from_u16_matches_float_policy():
    rng = np.random.default_rng(2)
    a_true = rng.uniform(size=(20, 3, 20))
    top3 = rng.uniform(size=(20, 3, 20))
    from sureenough.calibrate.baselines import naive_policy

    a_q, t_q = u16(a_true), u16(top3)
    cal, ev = np.arange(10), np.arange(10, 20)
    expected = naive_policy({"a_true": a_q / 65535, "top3": t_q / 65535}, cal, ev, 0.2, 20)["loss"].mean()
    assert naive_eval_risk_u16(a_q, t_q, cal, ev, 0.2, 20) == pytest.approx(expected, abs=0)


def test_check_bundle_rejects_forbidden_content(tmp_path):
    (tmp_path / "ok.json").write_text(json.dumps({"loss_count": [1]}))
    check_bundle(tmp_path, forbidden_hashes=set(), max_mb=1)
    (tmp_path / "bad.json").write_text(json.dumps({"pe_embedding": [0.1]}))
    with pytest.raises(BundleError):
        check_bundle(tmp_path, forbidden_hashes=set(), max_mb=1)


def test_check_bundle_rejects_stimulus_bytes(tmp_path):
    import hashlib

    blob = b"stimulus-bytes"
    (tmp_path / "img.webp").write_bytes(blob)
    with pytest.raises(BundleError):
        check_bundle(tmp_path, forbidden_hashes={hashlib.sha256(blob).hexdigest()}, max_mb=1)
