import numpy as np

from sal.stats import cluster_bootstrap, paired_cluster_bootstrap


def test_cluster_ci_wider_than_naive_for_correlated_data():
    rng = np.random.default_rng(0)
    spk = np.repeat(np.arange(40), 25)                     # 40 speakers x 25 utterances
    v = rng.normal(size=40)[spk] + 0.1 * rng.normal(size=len(spk))   # strongly speaker-driven
    c = cluster_bootstrap(v, spk, n=500)
    u = cluster_bootstrap(v, np.arange(len(v)), n=500)      # each utterance its own cluster
    assert (c["hi"] - c["lo"]) > 3 * (u["hi"] - u["lo"])
    assert c["lo"] <= c["estimate"] <= c["hi"] and c["n_clusters"] == 40


def test_paired_difference_detects_real_gap_and_not_noise():
    rng = np.random.default_rng(1)
    spk = np.repeat(np.arange(60), 10)
    base = rng.random(len(spk))
    better = (base < 0.6).astype(float)
    worse = (base < 0.4).astype(float)
    r = paired_cluster_bootstrap(better, worse, spk, n=500)
    assert r["diff"] > 0.15 and r["lo"] > 0 and r["p"] < 0.01
    same = paired_cluster_bootstrap(better, better.copy(), spk, n=200)
    assert same["diff"] == 0 and same["p"] == 1.0
