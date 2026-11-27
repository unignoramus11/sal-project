import numpy as np
import torch

from sal.probes import tp_data as T
from sal.probes.models import TPGRU
from sal.probes.train import TrainConfig, train_probe


def test_post_eval_gets_the_best_model_and_its_scores_are_kept():
    g = np.random.default_rng(0)
    seqs = [g.integers(4, 12, g.integers(3, 15)) for _ in range(60)]
    targets = np.radians(np.c_[g.uniform(20, 30, 60), g.uniform(70, 85, 60)])
    splits = {"train": np.arange(40), "val": np.arange(40, 50), "test": np.arange(50, 60)}
    cfg = TrainConfig(min_epochs=1, max_epochs=2, val_per_epoch=1, batch_size=16)
    seen = {}

    def post_eval(model):
        seen["state"] = {k: v.clone() for k, v in model.state_dict().items()}
        return {"original": {"recall1": 1.0}}

    res = train_probe(lambda: TPGRU(12), T.SequenceBatcher(seqs, targets), targets, splits, cfg, log=lambda s: None,
                      post_eval=post_eval)
    assert res["order_ablation"] == {"original": {"recall1": 1.0}}
    assert all(torch.equal(seen["state"][k], res["model_state"][k]) for k in seen["state"])   # the restored best model
    res2 = train_probe(lambda: TPGRU(12), T.SequenceBatcher(seqs, targets), targets, splits, cfg, log=lambda s: None)
    assert "order_ablation" not in res2                                                        # off by default
