import numpy as np
import pandas as pd
import pytest

from sal.benchmark import RUN, Scorer, find_runs, pool_folds, speaker_macro
from sal.probes import geo


def xyz(lat, lon):
    import torch
    return geo.latlon_to_xyz(torch.tensor(np.radians([lat])), torch.tensor(np.radians([lon]))).numpy()[0]


LOCS = {1: (28.9, 76.6, "Haryana"), 2: (23.3, 77.4, "MadhyaPradesh"), 3: (23.4, 85.3, "Jharkhand")}


@pytest.fixture
def track():
    rows = []
    for i in range(30):
        pin = 1 + i % 3
        rows.append({"metadata_idx": i, "latitude": LOCS[pin][0], "longitude": LOCS[pin][1], "state": LOCS[pin][2],
                     "group": f"g{i // 2}", "fold": i % 5, "pin": pin})
    return pd.DataFrame(rows)


def write_run(root, name, idx, pred_xyz):
    d = root / name
    d.mkdir(parents=True)
    pd.DataFrame({"split": "test", "metadata_idx": idx, "x": pred_xyz[:, 0], "y": pred_xyz[:, 1],
                  "z": pred_xyz[:, 2]}).to_parquet(d / "predictions.parquet")


def test_run_names():
    m = RUN.match("tp_w2v2_rebuild_phone_bigram_shared_f3_bench_s44")
    assert (m["cfg"], m["track"], m["fold"], m["tag"], m["seed"]) == ("tp_w2v2_rebuild_phone_bigram", "shared", "3", "", "44")
    assert RUN.match("rp_zipa_rebuild_L-1_attn_all_f0_bench_smoke_s42")["tag"] == "_smoke"


def test_pool_folds_averages_seeds_and_rejects_overlap(tmp_path, track):
    for f in range(5):
        idx = track.loc[track["fold"] == f, "metadata_idx"].to_numpy()
        for s, scale in ((42, 1.0), (43, 3.0)):           # scale must not matter: predictions are normalised
            write_run(tmp_path, f"tp_x_bigram_all_f{f}_bench_s{s}", idx, scale * np.tile(xyz(28.9, 76.6), (len(idx), 1)))
    runs = find_runs(tmp_path)
    pooled = pool_folds(runs[("tp_x_bigram", "all")])
    assert len(pooled) == 30 and (pooled["seeds"] == 2).all()
    assert np.allclose(np.linalg.norm(pooled[["x", "y", "z"]].to_numpy(), axis=1), 1)
    write_run(tmp_path, "tp_x_bigram_all_f1_bench_s44", np.array([0]), xyz(28.9, 76.6)[None])   # utterance 0 is fold 0
    with pytest.raises(ValueError):
        pool_folds(find_runs(tmp_path)[("tp_x_bigram", "all")])


def test_speaker_macro_weights_speakers_equally():
    hit = np.array([100, 100, 100, 0])
    assert speaker_macro(hit, np.array(["a", "a", "a", "b"])) == 50


def test_scorer_perfect_and_majority(track):
    perfect = np.stack([xyz(*LOCS[p][:2]) for p in track["pin"]])
    majority = np.tile(xyz(*LOCS[1][:2]), (len(track), 1))
    sc = Scorer(track)
    sc.reference = {"majority": sc.hits(majority)}
    r = sc.score(perfect, n=200)
    assert r["recall1"] == 100 and r["state"] == 100 and r["km"] < 1
    assert r["recall1_vs_majority"] == pytest.approx(100 * 2 / 3)
    assert r["km_vs_majority"] > 0 and r["km_vs_majority_lo"] <= r["km_vs_majority"] <= r["km_vs_majority_hi"]
    m = sc.score(majority, n=200)
    assert m["recall1"] == pytest.approx(100 / 3) and m["recall1_vs_majority"] == 0


def test_scorer_requires_every_utterance(track):
    sc = Scorer(track)
    pred = pd.DataFrame({"metadata_idx": track["metadata_idx"][:-1], "x": 1.0, "y": 0.0, "z": 0.0})
    with pytest.raises(ValueError):
        sc.align(pred)
