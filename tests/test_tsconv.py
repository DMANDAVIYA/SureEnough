import numpy as np

from sureenough.models.encoder import Standardizer, encode, train_contrastive
from sureenough.models.retrieval import cosine_top1
from sureenough.models.store import load_decoder, save_decoder
from sureenough.models.tsconv import TSConvEncoder


def test_tsconv_outputs_unit_rows():
    net = TSConvEncoder(out_dim=8)
    out = encode(net, np.random.default_rng(0).normal(size=(5, 17, 100)).astype(np.float32))
    assert out.shape == (5, 8)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_tsconv_learns_toy_spatiotemporal_patterns():
    rng = np.random.default_rng(0)
    n_classes, per_class = 40, 12
    protos = rng.normal(size=(n_classes, 17, 100)).astype(np.float32)
    targets = rng.normal(size=(n_classes, 16)).astype(np.float32)
    labels = np.repeat(np.arange(n_classes), per_class)
    x = protos[labels][:, None] + 0.8 * rng.normal(size=(len(labels), 4, 17, 100)).astype(np.float32)
    net = train_contrastive(TSConvEncoder(out_dim=16), x.reshape(len(x), 4, -1), targets[labels], epochs=25, batch=128, seed=0)
    probe = protos + 0.4 * rng.normal(size=protos.shape).astype(np.float32)
    assert cosine_top1(encode(net, probe), targets) > 0.8


def test_tsconv_decoder_round_trip(tmp_path):
    net = TSConvEncoder(out_dim=6)
    st = Standardizer(np.zeros(17, np.float32), np.ones(17, np.float32))
    save_decoder(tmp_path / "d.pt", net, st, target="final", seed=1)
    x = np.random.default_rng(0).normal(size=(3, 17, 100)).astype(np.float32)
    assert np.allclose(load_decoder(tmp_path / "d.pt").embed(x), encode(net, x), atol=1e-6)
