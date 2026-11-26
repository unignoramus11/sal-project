"""A run interrupted after some epochs and resumed from its checkpoint ends exactly like an uninterrupted run."""
import numpy as np
import pytest
import torch

from sal.probes import models as M
from sal.probes import tp_data as T
from sal.probes.train import TrainConfig, train_probe


class Interrupt(Exception):
    pass


def _data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(0.35, 0.55, n), rng.uniform(1.25, 1.55, n)
    targets = np.stack([lat, lon], 1)
    seqs = [rng.integers(1, 20, rng.integers(3, 15)) for _ in range(n)]
    splits = {"train": np.arange(0, 160), "val": np.arange(160, 200), "test": np.arange(200, n)}
    return seqs, targets, splits


def _run(tmp_path, ckpt, stop_after=None, shuffle=True):
    seqs, targets, splits = _data()
    cfg = TrainConfig(lr=1e-2, batch_size=32, min_epochs=2, max_epochs=6, patience=50, val_per_epoch=2,
                      warmup_steps=5, seed=7, device="cpu")
    batcher = T.SequenceBatcher(seqs, targets, shuffle=shuffle, seed=7)
    seen = []

    def log(msg):
        seen.append(msg)
        if stop_after is not None and msg.startswith(f"epoch {stop_after}:"):
            raise Interrupt                       # like a kill: this epoch's checkpoint is never written
    return train_probe(lambda: M.TPGRU(20), batcher, targets, splits, cfg, log=log, checkpoint=ckpt), seen


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_matches_uninterrupted(tmp_path, shuffle):
    full, _ = _run(tmp_path, None, shuffle=shuffle)
    ck = tmp_path / "run.pt"
    with pytest.raises(Interrupt):
        _run(tmp_path, ck, stop_after=4, shuffle=shuffle)
    assert ck.exists()                            # holds epochs 1-3
    resumed, log = _run(tmp_path, ck, shuffle=shuffle)
    assert any(m.startswith("resumed from run.pt after epoch 3") for m in log)
    assert not ck.exists()                        # removed once the run ends
    assert resumed["history"] == full["history"]
    assert resumed["best_epoch"] == full["best_epoch"] and resumed["epochs_run"] == full["epochs_run"]
    for split in ("val", "test"):
        assert np.array_equal(resumed["predictions"][split][1], full["predictions"][split][1])


def test_checkpoint_with_other_config_is_refused(tmp_path):
    ck = tmp_path / "run.pt"
    with pytest.raises(Interrupt):
        _run(tmp_path, ck, stop_after=2)
    seqs, targets, splits = _data()
    cfg = TrainConfig(lr=5e-3, batch_size=32, min_epochs=2, max_epochs=6, patience=50, val_per_epoch=2,
                      warmup_steps=5, seed=7, device="cpu")
    with pytest.raises(ValueError, match="different configuration"):
        train_probe(lambda: M.TPGRU(20), T.SequenceBatcher(seqs, targets, seed=7), targets, splits, cfg,
                    log=lambda s: None, checkpoint=ck)


def test_no_checkpoint_by_default(tmp_path, monkeypatch):
    from sal.probes.train import checkpoint_for
    monkeypatch.delenv("SAL_CKPT_DIR", raising=False)
    assert checkpoint_for("x") is None
    monkeypatch.setenv("SAL_CKPT_DIR", str(tmp_path))
    assert checkpoint_for("x") == tmp_path / "x.pt"
    torch.manual_seed(0)


def test_batcher_generators_round_trip():
    from types import SimpleNamespace

    from sal.probes.train import _batcher_rng, _set_batcher_rng
    b = SimpleNamespace(rng=np.random.default_rng(3), gen=torch.Generator().manual_seed(3))
    st = _batcher_rng(b)
    a1, t1 = b.rng.random(5), torch.rand(5, generator=b.gen)
    _set_batcher_rng(b, st)
    assert np.array_equal(b.rng.random(5), a1) and torch.equal(torch.rand(5, generator=b.gen), t1)


def test_resume_ignores_extra_and_device(tmp_path):
    """RP jobs extract features into a per-job scratch folder, recorded in cfg.extra: a resubmitted job must resume."""
    seqs, targets, splits = _data()
    ck = tmp_path / "run.pt"

    def go(extra, stop_after=None):
        cfg = TrainConfig(lr=1e-2, batch_size=32, min_epochs=2, max_epochs=4, patience=50, val_per_epoch=2,
                          warmup_steps=5, seed=7, device="cpu", extra=extra)

        def log(msg):
            if stop_after is not None and msg.startswith(f"epoch {stop_after}:"):
                raise Interrupt
        return train_probe(lambda: M.TPGRU(20), T.SequenceBatcher(seqs, targets, seed=7), targets, splits, cfg,
                           log=log, checkpoint=ck)
    with pytest.raises(Interrupt):
        go({"features": "/scratch/me/sal_101/w2v2_rebuild"}, stop_after=2)
    r = go({"features": "/scratch/me/sal_202/w2v2_rebuild"})
    assert r["epochs_run"] == 4 and not ck.exists()


def test_checkpoint_loaded_on_cpu(tmp_path, monkeypatch):
    """A run on CUDA must resume: the checkpoint is loaded with map_location='cpu' (RNG states must stay on the CPU)."""
    import sal.probes.train as tr
    ck = tmp_path / "run.pt"
    with pytest.raises(Interrupt):
        _run(tmp_path, ck, stop_after=2)
    seen = {}
    real = torch.load

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(tr.torch, "load", spy)
    _run(tmp_path, ck)
    assert seen.get("map_location") == "cpu"
