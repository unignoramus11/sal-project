import json

import numpy as np
import pandas as pd
import torch

from sal.probes.rp_data import FeatureStore, FrameBatcher


def make_store(tmp_path, widths=(4, 6), lengths=(5, 30, 12)):
    offs = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    total = int(sum(lengths))
    for l, w in enumerate(widths):
        arr = np.arange(total * w, dtype=np.float16).reshape(total, w) % 997 + l * 1000
        np.save(tmp_path / f"frames_L{l}.npy", arr.astype(np.float16))
    pd.DataFrame({"metadata_idx": [10, 11, 12], "n_frames": lengths, "offset": offs}).to_parquet(tmp_path / "index.parquet")
    (tmp_path / "meta.json").write_text(json.dumps({"layers": ["a", "b"], "frame_rate": 50}))
    return tmp_path


def test_get_and_padding(tmp_path):
    d = make_store(tmp_path)
    s = FeatureStore(d, [0, 1])
    x = s.get(11)
    assert x.shape == (2, 30, 6)
    assert (x[0, :, 4:] == 0).all()                       # narrower layer zero-padded
    full = np.load(d / "frames_L1.npy")
    assert (x[1] == full[5:35]).all()


def test_batch_shapes_and_last_layer(tmp_path):
    d = make_store(tmp_path)
    t = np.zeros((3, 2))
    b = FrameBatcher(FeatureStore(d, [-1]), np.array([10, 11, 12]), t)
    x, lengths, _ = b(np.array([0, 2]), train=False)
    assert x.shape == (2, 12, 6) and lengths.tolist() == [5, 12]
    assert (x[0, 5:] == 0).all()


def test_eval_chunk_shuffle_independent_of_batch(tmp_path):
    d = make_store(tmp_path)
    t = np.zeros((3, 2))
    b = FrameBatcher(FeatureStore(d, [0]), np.array([10, 11, 12]), t, chunk_shuffle_ms=100, seed=3)  # 5 frames
    alone, _, _ = b(np.array([1]), train=False)
    together, _, _ = b(np.array([2, 1, 0]), train=False)
    assert torch.equal(alone[0], together[1, :30])
    orig = torch.from_numpy(FeatureStore(d, [0]).get(11)[0].astype(np.float32))
    assert not torch.equal(alone[0], orig)                # actually shuffled
    assert torch.equal(alone[0].sum(0), orig.sum(0))      # same frames
