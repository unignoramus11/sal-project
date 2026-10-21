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


# 2. leakage estimator: a threshold on each clip's best score against the train clips of its district, set so that
# 5% of clips are called "seen" on the rebuild's speaker-disjoint split (no clip truly seen there); its hit rate is
# measured on the rebuild's random split (98% truly seen). The pairwise EER threshold is far too low for a maximum
# over hundreds of train clips.
best_prism = seen(xr, ir, "prism", 0)["best"]
best_spk = seen(xr, ir, "speaker_split", 0)["best"]
train_pairs = set(zip(ir.loc[ir["prism"] == "train", "district"], ir.loc[ir["prism"] == "train", "speaker"], strict=True))
sub = ir.loc[best_prism.index]
truth = np.array([(d, sp) in train_pairs for d, sp in zip(sub["district"], sub["speaker"], strict=True)])
t_seen = float(np.quantile(best_spk, 0.95))
fa = float((best_spk >= t_seen).mean())
hit = float((best_prism[truth] >= t_seen).mean())
lines.append(f"leakage estimator: best-score threshold {t_seen:.3f} (5% false alarms on the rebuild speaker-disjoint "
             f"split); hit rate {hit:.1%} on rebuild random-split clips whose speaker is in train (true leakage there "
             f"{truth.mean():.1%})")
of_meta = pd.read_parquet(DATA / "vaani_hi_official" / "splits.parquet")
xo, io_ = load("ecapa_official", of_meta)
best_off = seen(xo, io_, "prism", 0)["best"]
obs = float((best_off >= t_seen).mean())
est = (obs - fa) / (hit - fa)
obs_rb = float((best_prism >= t_seen).mean())
lines.append(f"rebuild random split: observed {obs_rb:.1%}, corrected {(obs_rb - fa) / (hit - fa):.1%} (true {truth.mean():.1%})")
lines.append(f"official released split: observed {obs:.1%} of val/test clips -> estimated speaker leakage "
             f"{100 * min(max(est, 0), 1):.1f}%")
pd.DataFrame([{"data": "rebuild_random", "observed": obs_rb, "estimated": (obs_rb - fa) / (hit - fa), "true": truth.mean()},
              {"data": "rebuild_speaker_disjoint", "observed": fa, "estimated": 0.0, "true": 0.0},
              {"data": "official_released", "observed": obs, "estimated": est, "true": np.nan}]).assign(
    threshold=t_seen, hit=hit, false_alarm=fa).to_csv(OUT / "speaker_leakage.csv", index=False)

# 3. pseudo-speakers: cluster within district; the threshold is chosen by what matters for a split built on the
# clusters: the true speaker leakage of a pincode-stratified "speaker-disjoint" split over pseudo-speakers, measured
# on the rebuild with its real speakers (merging speakers is harmless, splitting one speaker leaks)
from sal.data.splits import leakage, speaker_disjoint_splits  # noqa: E402

best = None
for dt in (0.5, 0.6, 0.7, 0.8, 0.9):
    lab = pd.Series(index=ir.index, dtype=object)
    for di, g in ir.groupby("district"):
        lab.loc[g.index] = [f"{di}_{k}" for k in AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=dt).fit_predict(xr[g.index])] \
            if len(g) > 1 else [f"{di}_0"]
    tmp = ir.assign(pseudo=lab.values)
    tmp["split_ps"] = speaker_disjoint_splits(tmp, speaker_col="pseudo", group_col="pincode", seed=42)
    lk = leakage(tmp, "split_ps", speaker_col="speaker")
    sizes = tmp["split_ps"].value_counts(normalize=True)
    ari = adjusted_rand_score(tmp["speaker"], tmp["pseudo"])
    lines.append(f"pseudo-speakers, distance threshold {dt}: rebuild ARI {ari:.3f}, clusters {tmp['pseudo'].nunique()}, "
                 f"true test leakage of the pseudo-speaker split {lk['test']:.1%}, test share {sizes.get('test', 0):.1%}")
    if best is None and lk["test"] <= 0.05 and sizes.get("test", 0) >= 0.10:
        best = (dt, lk["test"])
best = best or (0.9, np.nan)
lines.append(f"chosen threshold {best[0]} (rebuild true test leakage {best[1]:.1%})")
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
