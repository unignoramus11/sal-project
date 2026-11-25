"""Checks of the benchmark's negative result: is there location information that the headline scores could miss?

Our main claim is negative: once speakers and topic are controlled, probes do not beat the majority location. This
script asks whether that holds up, using only existing predictions (no new runs).

  transfer      Models trained on all pictures (the `all` track, three times the data of the `shared` track) scored
                only on test utterances about shared pictures, which carry no topic cue. Two references on the same
                utterances: the majority location of the model's own training fold, and a stronger one that knows
                the shared-picture distribution (majority and best constant of the fold's shared-picture training
                utterances). Lifts with 95% CIs over speaker groups.
  permutation   Is the probe's output related to its input at all? Speaker groups, not utterances, are the
                independent units: each group's predictions are averaged into one, and the statistic is the mean over
                groups of the group's Recall@1 (or km), so every group counts once. The null shuffles the group
                predictions among the groups of the same fold, which keeps exactly the probe's distribution of
                guesses and breaks any link to the input; a probe that spreads its guesses can lose to the majority
                on Recall@1 and still beat this null. 2,000 shuffles; one-sided p for Recall@1 (higher) and km (lower).
                Also the pincode-balanced Recall@1 (mean over pincodes of each pincode's recall), where the majority
                scores 100/K. Done for the `all` and `shared` tracks and the transfer set.
  conditional   The same group-level test, but shuffling only among groups that the channel probe
                (dense_channel_rebuild: background noise and levels, scripts/channel_features.py) places at the same
                pincode, within each fold. Under the null "the probe adds nothing about location beyond what the
                channel gives", groups with the same channel-based guess are exchangeable; a probe that still beats
                this null carries location the channel probe does not. Strata with one group are left unshuffled.
  lookalike     Residual look-alike speakers: for each test utterance, the highest ECAPA centroid cosine between its
                speaker code and any training code of the same district in another group. Codes were merged into
                groups at 0.757 (5th percentile of same-speaker split-half cosines) with average linkage, so two codes
                in different groups can still be close. Shares of test utterances above that threshold, for the
                benchmark folds and, for comparison, our speaker-disjoint split.
  holm          Holm adjustment of the paired benchmark comparisons (results/benchmark/paired.csv), per track.

Output: results/benchmark/checks_{transfer,permutation,conditional,lookalike,holm}.csv
Run:    uv run python scripts/benchmark_checks.py
"""
import numpy as np
import pandas as pd
import torch

from sal.benchmark import Scorer, find_runs, pool_folds
from sal.paths import DATA, OUTPUTS, RESULTS
from sal.probes import geo

B = RESULTS / "benchmark"
rng = np.random.default_rng(0)
N_PERM, N_BOOT, THRESH = 2000, 1000, 0.757
sp = pd.read_parquet(B / "splits.parquet")
d = sp.merge(pd.read_parquet(B / "meta.parquet")[["metadata_idx", "latitude", "longitude"]], on="metadata_idx")
K = sum(c.startswith("all_f") for c in d.columns)
d["fold_all"] = d[[f"all_f{k}" for k in range(K)]].eq("test").to_numpy().argmax(axis=1)
d["fold_shared"] = d[[f"shared_f{k}" for k in range(K)]].eq("test").to_numpy().argmax(axis=1)
pin_xyz = {p: geo.latlon_to_xyz(torch.tensor(np.radians([r.latitude])), torch.tensor(np.radians([r.longitude]))).numpy()[0]
           for p, r in d.groupby("pincode")[["latitude", "longitude"]].first().iterrows()}
runs = find_runs(RESULTS / "runs")


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def pooled(cfg, track):
    fr = runs.get((cfg, track))
    if not fr or sorted(fr) != list(range(K)):
        return None
    return pool_folds(fr)


def probes(track):
    return sorted(c for (c, t) in runs if t == track and pooled(c, t) is not None)


# ---------------------------------------------------------------- transfer: all-picture models on shared pictures
def transfer_sets():
    u = d[d["all_f0"].notna() & (d["prompt_type"] == "GENERIC")].sort_values("metadata_idx").reset_index(drop=True)
    refs = {"own majority": [], "shared majority": [], "shared constant": []}
    for k in range(K):
        tr = d[d[f"all_f{k}"] == "train"]
        trs = tr[tr["prompt_type"] == "GENERIC"]
        te = u[u["fold_all"] == k]
        const = geo.best_constant(np.radians(trs[["latitude", "longitude"]].to_numpy(np.float64)))
        for name, v in (("own majority", pin_xyz[tr["pincode"].mode().iloc[0]]),
                        ("shared majority", pin_xyz[trs["pincode"].mode().iloc[0]]), ("shared constant", const)):
            refs[name].append(pd.DataFrame({"metadata_idx": te["metadata_idx"].to_numpy(), "x": v[0], "y": v[1], "z": v[2]}))
    return u, {k: pd.concat(v) for k, v in refs.items()}


rows = []
u_tr, refs = transfer_sets()
sc = Scorer(u_tr)
ref_xyz = {k: sc.align(v) for k, v in refs.items()}
sc.reference = {k.replace(" ", "_"): sc.hits(v) for k, v in ref_xyz.items()}
for name, v in ref_xyz.items():
    rows.append({"probe": name, "kind": "reference", **sc.score(v, n=N_BOOT)})
for cfg in probes("all"):
    rows.append({"probe": cfg, "kind": "probe", **sc.score(sc.align(pooled(cfg, "all")), n=N_BOOT)})
pd.DataFrame(rows).to_csv(B / "checks_transfer.csv", index=False)
print("transfer: all-picture models on shared-picture test utterances")
print(pd.DataFrame(rows)[["probe", "recall1", "recall1_vs_own_majority", "recall1_vs_shared_majority",
                          "recall1_vs_shared_majority_lo", "recall1_vs_shared_majority_hi", "km_vs_shared_constant",
                          "km_vs_shared_constant_lo", "km_vs_shared_constant_hi"]].round(1).to_string(index=False))


# ---------------------------------------------------------------- permutation null over speaker groups
def permutation(u, pred, folds, strata=None):
    """u: utterances (group, latitude, longitude, pincode); pred: (N, 3) aligned; folds: (N,) test fold;
    strata: optional (N,) label constant within a group: groups are shuffled only within fold x stratum."""
    sc = Scorer(u)
    # candidates rounded to 6 decimals before the cosine, exactly as geo.recall_at_k does (near-ties otherwise flip)
    cxyz = unit(geo.latlon_to_xyz(*sc.cand.round(decimals=6).unbind(-1)).numpy())
    t_xyz = geo.latlon_to_xyz(sc.t[:, 0], sc.t[:, 1]).numpy()
    true_c = np.argmax(unit(geo.latlon_to_xyz(*sc.t.round(decimals=6).unbind(-1)).numpy()) @ cxyz.T, axis=1)
    # the fast scoring below must agree with the scorer that produced the headline numbers
    ref, pu = sc.hits(pred), unit(pred)
    fast_hit = 100 * (np.argmax(pu @ cxyz.T, axis=1) == true_c)
    fast_km = geo.EARTH_RADIUS_KM * np.arccos(np.clip(np.sum(pu * t_xyz, axis=1), -1, 1))
    assert np.array_equal(fast_hit, ref["recall1"]) and np.allclose(fast_km, ref["km"], atol=0.05), "scoring mismatch"
    g_codes, g_idx = np.unique(u["group"].to_numpy(), return_inverse=True)
    G = len(g_codes)
    gp = np.zeros((G, 3))
    np.add.at(gp, g_idx, unit(pred))
    gp = unit(gp)                                                # one pooled prediction per group
    g_fold = np.zeros(G, int)
    g_fold[g_idx] = folds
    g_stratum = np.zeros(G, int)
    if strata is not None:
        g_stratum[g_idx] = strata
    pin_codes, pin_idx = np.unique(u["pincode"].to_numpy(), return_inverse=True)
    n_g = np.bincount(g_idx, minlength=G)
    n_p = np.bincount(pin_idx, minlength=len(pin_codes))

    def stats(gpred):
        p = gpred[g_idx]
        hit = (np.argmax(p @ cxyz.T, axis=1) == true_c).astype(float)
        km = geo.EARTH_RADIUS_KM * np.arccos(np.clip(np.sum(p * t_xyz, axis=1), -1, 1))
        g_hit = np.bincount(g_idx, hit, G) / n_g
        g_km = np.bincount(g_idx, km, G) / n_g
        p_hit = np.bincount(pin_idx, hit, len(n_p)) / n_p
        return 100 * g_hit.mean(), g_km.mean(), 100 * p_hit.mean()

    obs = stats(gp)
    null = np.empty((N_PERM, 3))
    by_fold = [np.flatnonzero((g_fold == k) & (g_stratum == s)) for k in range(K) for s in np.unique(g_stratum)]
    by_fold = [ix for ix in by_fold if len(ix) > 1]
    for i in range(N_PERM):
        perm = np.arange(G)
        for ix in by_fold:
            perm[ix] = rng.permutation(ix)
        null[i] = stats(gp[perm])
    return {"groups": G, "pincodes": len(n_p),
            "group_recall1": obs[0], "null_group_recall1": null[:, 0].mean(),
            "p_recall1": (1 + (null[:, 0] >= obs[0]).sum()) / (N_PERM + 1),
            "group_km": obs[1], "null_group_km": null[:, 1].mean(),
            "p_km": (1 + (null[:, 1] <= obs[1]).sum()) / (N_PERM + 1),
            "pincode_recall1": obs[2], "null_pincode_recall1": null[:, 2].mean(),
            "p_pincode_recall1": (1 + (null[:, 2] >= obs[2]).sum()) / (N_PERM + 1),
            "majority_pincode_recall1": 100 / len(n_p)}


prow = []
for track in ("all", "shared"):
    u = d[d[f"{track}_f0"].notna()].sort_values("metadata_idx").reset_index(drop=True)
    sc_t = Scorer(u)
    for cfg in probes(track):
        prow.append({"set": track, "probe": cfg, **permutation(u, sc_t.align(pooled(cfg, track)), u[f"fold_{track}"].to_numpy())})
for cfg in probes("all"):
    prow.append({"set": "transfer", "probe": cfg, **permutation(u_tr, sc.align(pooled(cfg, "all")), u_tr["fold_all"].to_numpy())})
pr = pd.DataFrame(prow)
pr.to_csv(B / "checks_permutation.csv", index=False)
print("\npermutation null over speaker groups (group-mean Recall@1, km; pincode-balanced Recall@1)")
print(pr[["set", "probe", "group_recall1", "null_group_recall1", "p_recall1", "group_km", "null_group_km", "p_km",
          "pincode_recall1", "null_pincode_recall1", "p_pincode_recall1"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- conditional on the channel probe
COND = "dense_channel_rebuild"


def channel_strata(u, track_pred_cfg_track, sc_):
    """Per utterance: the candidate pincode nearest to its group's pooled channel-probe prediction."""
    cp = unit(sc_.align(pooled(*track_pred_cfg_track)))
    g_codes, g_idx = np.unique(u["group"].to_numpy(), return_inverse=True)
    gp = np.zeros((len(g_codes), 3))
    np.add.at(gp, g_idx, cp)
    cxyz = unit(geo.latlon_to_xyz(*sc_.cand.round(decimals=6).unbind(-1)).numpy())
    return np.argmax(unit(gp) @ cxyz.T, axis=1)[g_idx]


crow = []
for setname, track in (("all", "all"), ("shared", "shared"), ("transfer", "all")):
    if pooled(COND, track) is None:
        continue
    u = u_tr if setname == "transfer" else d[d[f"{track}_f0"].notna()].sort_values("metadata_idx").reset_index(drop=True)
    sc_s = sc if setname == "transfer" else Scorer(u)
    strata = channel_strata(u, (COND, track), sc_s)
    folds = u["fold_all" if setname == "transfer" else f"fold_{track}"].to_numpy()
    for cfg in probes(track):
        if cfg == COND:
            continue
        r = permutation(u, sc_s.align(pooled(cfg, track)), folds, strata)
        crow.append({"set": setname, "probe": cfg, "strata": len(np.unique(strata)), **r})
cr = pd.DataFrame(crow)
cr.to_csv(B / "checks_conditional.csv", index=False)
print("\nconditional on the channel probe: shuffles only among groups with the same channel-based guess")
print(cr[["set", "probe", "group_recall1", "null_group_recall1", "p_recall1", "group_km", "null_group_km", "p_km"]]
      .round(3).to_string(index=False))

# ---------------------------------------------------------------- residual look-alike speakers
emb = np.load(OUTPUTS / "features" / "ecapa_rebuild" / "emb.npy").astype(np.float64)
idx = pd.read_parquet(OUTPUTS / "features" / "ecapa_rebuild" / "index.parquet")
emb = pd.DataFrame(unit(emb), index=idx["metadata_idx"].to_numpy())
code_c = emb.loc[d["metadata_idx"]].set_axis(d.index).groupby(d["speaker"].to_numpy()).mean()
code_c = pd.DataFrame(unit(code_c.to_numpy()), index=code_c.index)
code_info = d.groupby("speaker").agg(group=("group", "first"), district=("district", "first"))
rebuild = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet")[["metadata_idx", "speaker_split"]]


def max_cos(test_codes, train_codes, same_group_ok):
    out = {}
    for c in test_codes:
        di, gr = code_info.loc[c, "district"], code_info.loc[c, "group"]
        cand = [t for t in train_codes if code_info.loc[t, "district"] == di and (same_group_ok or code_info.loc[t, "group"] != gr)]
        out[c] = float((code_c.loc[cand].to_numpy() @ code_c.loc[c].to_numpy()).max()) if cand else np.nan
    return pd.Series(out)


lrow = []
for k in range(K):
    te, tr = d[d[f"all_f{k}"] == "test"], d[d[f"all_f{k}"] == "train"]
    mc = max_cos(te["speaker"].unique(), tr["speaker"].unique(), same_group_ok=False)
    w = te["speaker"].map(mc)
    lrow.append({"split": f"benchmark all_f{k}", "test_utterances": len(te),
                 **{f"share_above_{t}": 100 * (w >= t).mean() for t in (0.70, THRESH, 0.80)}, "max": w.max()})
ss = d.merge(rebuild, on="metadata_idx")
te, tr = ss[ss["speaker_split"] == "test"], ss[ss["speaker_split"] == "train"]
mc = max_cos(te["speaker"].unique(), tr["speaker"].unique(), same_group_ok=True)
w = te["speaker"].map(mc)
lrow.append({"split": "rebuild speaker_split", "test_utterances": len(te),
             **{f"share_above_{t}": 100 * (w >= t).mean() for t in (0.70, THRESH, 0.80)}, "max": w.max()})
lk = pd.DataFrame(lrow)
lk.to_csv(B / "checks_lookalike.csv", index=False)
print("\nresidual look-alike speakers: % of test utterances whose code has a training code of the same district above")
print(lk.round(2).to_string(index=False))

# ---------------------------------------------------------------- Holm adjustment of the paired comparisons
pp = pd.read_csv(B / "paired.csv")
hrow = []
for _, g in pp.groupby("track"):
    g = g.sort_values("recall1_p").reset_index(drop=True)
    m = len(g)
    adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(g["recall1_p"])]).clip(max=1)
    hrow.append(g.assign(recall1_p_holm=adj))
pd.concat(hrow)[["track", "comparison", "a", "b", "recall1_diff", "recall1_lo", "recall1_hi", "recall1_p",
                 "recall1_p_holm"]].to_csv(B / "checks_holm.csv", index=False)
print("\nHolm-adjusted p of the paired comparisons")
print(pd.concat(hrow)[["track", "comparison", "a", "recall1_diff", "recall1_p", "recall1_p_holm"]].round(4).to_string(index=False))
