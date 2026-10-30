import numpy as np

from sureenough.models.encoder import MLPEncoder, Standardizer, encode
from sureenough.models.ridge import fit_ridge
from sureenough.models.store import load_decoder, save_decoder
from sureenough.models.validation import val_top1


def _eeg_matching_targets(n=40, k_signal=1, seed=0):
    rng = np.random.default_rng(seed)
    eeg = rng.normal(size=(n, 4, 17, 100)).astype(np.float32)
    signal = eeg[:, :k_signal].mean(axis=1).reshape(n, -1)
    return eeg, signal


def test_val_top1_identity_predictor_is_perfect_on_signal_repeats():
    eeg, targets = _eeg_matching_targets(k_signal=1)
    galleries = np.arange(40).reshape(2, 20)
    assert val_top1(lambda x: x, eeg, targets, galleries, k=1) == 1.0


def test_val_top1_uses_only_first_k_repeats():
    eeg, _ = _eeg_matching_targets()
    targets = eeg[:, 3].reshape(len(eeg), -1)
    galleries = np.arange(40).reshape(2, 20)
    assert val_top1(lambda x: x, eeg, targets, galleries, k=1) < 0.5
    assert val_top1(lambda x: x, eeg, targets, galleries, k=4) == 1.0


def test_fit_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 20))
    w = rng.normal(size=(20, 8))
    model = fit_ridge(x, x @ w, alpha=1e-3)
    assert np.allclose(model.predict(x[:5]), x[:5] @ w, atol=1e-2)


def test_decoder_round_trip(tmp_path):
    net = MLPEncoder(in_dim=1700, out_dim=6, hidden=8)
    st = Standardizer(np.arange(17, dtype=np.float32), np.ones(17, dtype=np.float32))
    path = tmp_path / "sub-01.pt"
    save_decoder(path, net, st, target="final", seed=3)
    decoder = load_decoder(path)
    x = np.random.default_rng(0).normal(size=(3, 17, 100)).astype(np.float32)
    assert decoder.target == "final" and decoder.seed == 3
    assert np.allclose(decoder.embed(x), encode(net, st.transform(x)), atol=1e-6)
