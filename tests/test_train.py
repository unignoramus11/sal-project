import math

import numpy as np
import torch

from sal.probes import models as M
from sal.probes import tp_data as T
from sal.probes.train import TrainConfig, train_probe

LOCS = np.radians(np.array([[28.6, 77.2], [26.9, 75.8], [25.6, 85.1], [23.3, 77.4]]))


def synthetic(n=600, seed=0):
    """Location is signalled by an ordered pair of symbols (order matters, inventory does not)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    seqs = []
    for c in y:
        base = list(rng.integers(10, 20, rng.integers(6, 15)))
        a, b = (4 + c, 8 + c) if c < 2 else (8 + c - 2, 4 + c - 2)  # classes 2,3 reuse symbols reversed
        pos = rng.integers(0, len(base))
        base[pos:pos] = [a, b]
        seqs.append(np.array(base, dtype=np.int64))
    return seqs, LOCS[y]


def _split(n):
    idx = np.arange(n)
    return {"train": idx[: int(0.7 * n)], "val": idx[int(0.7 * n): int(0.85 * n)], "test": idx[int(0.85 * n):]}


def test_tp_gru_learns_and_is_deterministic():
    seqs, t = synthetic()
    sp = _split(len(seqs))
    cfg = TrainConfig(lr=3e-3, batch_size=32, min_epochs=3, max_epochs=30, patience=12, val_per_epoch=2, seed=1)
    r1 = train_probe(lambda: M.TPGRU(24, hidden=32, emb_dim=16), T.SequenceBatcher(seqs, t), t, sp, cfg, log=lambda s: None)
    r2 = train_probe(lambda: M.TPGRU(24, hidden=32, emb_dim=16), T.SequenceBatcher(seqs, t), t, sp, cfg, log=lambda s: None)
    assert r1["metrics"]["test"]["recall1"] > 80
    assert r1["metrics"]["test"] == r2["metrics"]["test"]


def test_shuffle_hurts_on_order_task():
    """Classes 0/2 and 1/3 share symbols in opposite order, so shuffling must cost accuracy."""
    seqs, t = synthetic()
    sp = _split(len(seqs))
    cfg = TrainConfig(lr=3e-3, batch_size=32, min_epochs=3, max_epochs=30, patience=12, val_per_epoch=2, seed=1)
    ordered = train_probe(lambda: M.TPGRU(24, hidden=32, emb_dim=16), T.SequenceBatcher(seqs, t), t, sp, cfg,
                          log=lambda s: None)
    shuffled = train_probe(lambda: M.TPGRU(24, hidden=32, emb_dim=16), T.SequenceBatcher(seqs, t, shuffle=True), t, sp,
                           cfg, log=lambda s: None)
    assert ordered["metrics"]["test"]["recall1"] > shuffled["metrics"]["test"]["recall1"] + 15


def test_min_epochs_respected():
    seqs, t = synthetic(200)
    sp = _split(len(seqs))
    cfg = TrainConfig(lr=1e-9, batch_size=32, min_epochs=4, max_epochs=20, patience=1, val_per_epoch=2)
    r = train_probe(lambda: M.TPGRU(24, hidden=8, emb_dim=8), T.SequenceBatcher(seqs, t), t, sp, cfg, log=lambda s: None)
    assert 4 <= r["epochs_run"] < 20


def test_bigram_bag_path():
    seqs, t = synthetic()
    sp = _split(len(seqs))
    x, info = T.bag_features(seqs, 24, sp["train"], bigrams=True, min_bigram=2)
    assert info["bigrams"] > 0 and math.isclose(x[0, :24].sum(), 1.0, rel_tol=1e-5)
    cfg = TrainConfig(lr=3e-3, batch_size=32, min_epochs=3, max_epochs=10, patience=8, val_per_epoch=2)
    r = train_probe(lambda: M.BagMLP(x.shape[1], hidden=64), T.DenseBatcher(x, t), t, sp, cfg, log=lambda s: None)
    assert r["metrics"]["test"]["recall1"] > 60   # bigrams encode local order, so the task is learnable


def test_char_vocab_matches_prism_order():
    v = T.build_vocab([list("kʰa"), list("ab")])
    assert list(v)[:4] == ["<pad>", "<unk>", "<s>", "</s>"]
    assert list(v)[4:] == sorted(set("kʰab"))
    assert T.encode(list("z"), v).tolist() == [T.UNK]


def test_vector_mlp_trains_in_loop():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    x = (np.eye(4)[y] + 0.1 * rng.normal(size=(400, 4))).astype(np.float32)
    t = LOCS[y]
    sp = _split(400)
    cfg = TrainConfig(lr=3e-3, batch_size=32, min_epochs=3, max_epochs=40, patience=15, val_per_epoch=2)
    r = train_probe(lambda: M.VectorMLP(4, hidden=16), T.DenseBatcher(x, t), t, sp, cfg, log=lambda s: None)
    assert r["metrics"]["test"]["recall1"] > 90
