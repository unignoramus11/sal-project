"""Speaker leakage and pseudo-speakers for the official Vaani-Hi from ECAPA embeddings, calibrated on the rebuild.

The official release has no speaker labels and its sources cannot be recovered (docs/official_data.md), so speaker
identity is estimated from ECAPA-TDNN embeddings (scripts/speaker_embed.py). Everything is calibrated on the
rebuilt Vaani-Hi, where speakers are known from the Vaani file names. Speakers never cross districts, so all
comparisons are within district.

  1. verification   cosine scores of same-speaker and different-speaker pairs within district (rebuild);
                    equal-error-rate threshold
  2. leakage        a test clip is called "speaker seen" if some train clip of its district scores above the
                    threshold; hit rate and false-alarm rate measured on the rebuild's two splits (true leakage
                    97.9% and 0%), then applied to the official split and corrected:
                    true = (observed - false alarm) / (hit - false alarm)
  3. pseudo-speakers  agglomerative clustering (average linkage, cosine distance) within district, threshold chosen
                    on the rebuild to maximise the adjusted Rand index against true speakers; applied to the
                    official release and used for a pincode-stratified speaker-disjoint split

Output: results/official/speaker_calibration.txt, speaker_leakage.csv, data/vaani_hi_official/pseudo_speakers.parquet
Run:    uv run python scripts/speaker_calibration.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from sal.paths import DATA, OUTPUTS, RESULTS

OUT = RESULTS / "official"
rng = np.random.default_rng(0)
lines = []


def load(feat, meta):
    x = np.load(OUTPUTS / "features" / feat / "emb.npy").astype(np.float32)
    idx = pd.read_parquet(OUTPUTS / "features" / feat / "index.parquet")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, idx.merge(meta, on="metadata_idx", how="left")


# ---- rebuild: embeddings + true speakers
rb_meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet")
xr, ir = load("ecapa_rebuild", rb_meta)

# 1. verification threshold (equal error rate) from within-district pairs
same, diff = [], []
for _, g in ir.groupby("district"):
    pos = g.index.to_numpy()
    spk = g["speaker"].to_numpy()
    a, b = rng.integers(0, len(pos), 20000), rng.integers(0, len(pos), 20000)
    keep = a != b
    s = np.einsum("ij,ij->i", xr[pos[a[keep]]], xr[pos[b[keep]]])
    eq = spk[a[keep]] == spk[b[keep]]
    diff.append(s[~eq])
    for _, gs in g.groupby("speaker"):          # all same-speaker pairs (sub-sampled for large speakers)
        p = gs.index.to_numpy()
        if len(p) > 1:
            k = rng.integers(0, len(p), (min(200, len(p) * 3), 2))
            k = k[k[:, 0] != k[:, 1]]
            same.append(np.einsum("ij,ij->i", xr[p[k[:, 0]]], xr[p[k[:, 1]]]))
same, diff = np.concatenate(same), np.concatenate(diff)
grid = np.linspace(-0.2, 1, 1201)
fnr = np.array([(same < t).mean() for t in grid])
fpr = np.array([(diff >= t).mean() for t in grid])
thr = grid[np.argmin(np.abs(fnr - fpr))]
lines.append(f"verification (rebuild, within district): {len(same)} same-speaker and {len(diff)} different-speaker "
             f"pairs; EER {100 * fnr[np.argmin(np.abs(fnr - fpr))]:.1f}% at cosine {thr:.3f}")


def seen(x, info, split_col, thr):
    """Per val/test clip: best cosine to a train clip of the same district, and whether it passes thr."""
    out = []
    for _, g in info.groupby("district"):
        tr = g.index[g[split_col] == "train"].to_numpy()
        te = g.index[g[split_col].isin(["val", "test"])].to_numpy()
        if len(tr) == 0 or len(te) == 0:
            continue
        best = (x[te] @ x[tr].T).max(1)
        out.append(pd.DataFrame({"row": te, "best": best, "seen": best >= thr}))
    return pd.concat(out).set_index("row")


# 2. leakage estimator on the rebuild (truth known)
cal = {}
for proto in ("prism", "speaker_split"):
    s = seen(xr, ir, proto, thr)
    train_pairs = set(zip(ir.loc[ir[proto] == "train", "district"], ir.loc[ir[proto] == "train", "speaker"], strict=True))
    sub = ir.loc[s.index]
    truth = pd.Series([(d, sp) in train_pairs for d, sp in zip(sub["district"], sub["speaker"], strict=True)], index=s.index)
    cal[proto] = {"observed": s["seen"].mean(), "true": truth.mean(),
                  "hit": s.loc[truth, "seen"].mean() if truth.any() else np.nan,
                  "false_alarm": s.loc[~truth, "seen"].mean() if (~truth).any() else np.nan}
hit, fa = cal["prism"]["hit"], cal["speaker_split"]["false_alarm"]
for k, v in cal.items():
    lines.append(f"rebuild {k}: true leakage {v['true']:.1%}, observed 'speaker seen' {v['observed']:.1%}")
lines.append(f"estimator: hit rate {hit:.1%} (rebuild random split), false-alarm rate {fa:.1%} (speaker-disjoint split)")

# apply to the official release
of_meta = pd.read_parquet(DATA / "vaani_hi_official" / "splits.parquet")
xo, io_ = load("ecapa_official", of_meta)
so = seen(xo, io_, "prism", thr)
obs = so["seen"].mean()
est = (obs - fa) / (hit - fa)
lines.append(f"official released split: observed 'speaker seen' {obs:.1%} of val/test clips -> estimated true speaker "
             f"leakage {100 * min(max(est, 0), 1):.1f}% (rebuild PRiSM-recipe split, measured: {cal['prism']['true']:.1%})")
pd.DataFrame([{"data": f"rebuild_{k}", **v} for k, v in cal.items()] +
             [{"data": "official_released", "observed": obs, "true": est, "hit": hit, "false_alarm": fa}]).to_csv(
    OUT / "speaker_leakage.csv", index=False)

# 3. pseudo-speakers: choose the clustering threshold on the rebuild
best = None
for dt in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
    aris = []
    for _, g in ir.groupby("district"):
        if len(g) < 3:
            continue
        lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                      distance_threshold=dt).fit_predict(xr[g.index])
        aris.append((adjusted_rand_score(g["speaker"], lab), len(g)))
    ari = np.average([a for a, _ in aris], weights=[n for _, n in aris])
    lines.append(f"pseudo-speakers, distance threshold {dt}: rebuild ARI {ari:.3f}")
    if best is None or ari > best[1]:
        best = (dt, ari)
lines.append(f"chosen threshold {best[0]} (ARI {best[1]:.3f})")
labels = pd.Series(index=io_.index, dtype=object)
for di, g in io_.groupby("district"):
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                  distance_threshold=best[0]).fit_predict(xo[g.index])
    labels.loc[g.index] = [f"{di}_{k}" for k in lab]
ps = io_[["metadata_idx"]].assign(pseudo_speaker=labels.values)
ps.to_parquet(DATA / "vaani_hi_official" / "pseudo_speakers.parquet", index=False)
per = ps.groupby("pseudo_speaker").size()
lines.append(f"official: {len(per)} pseudo-speakers, clips per pseudo-speaker median {per.median():.0f}, max {per.max()}")
(OUT / "speaker_calibration.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
