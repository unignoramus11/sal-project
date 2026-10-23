"""Scores on the cross-validated benchmark (scripts/make_benchmark.py), one row per probe and track.

Every probe has one run per fold (run names ..._<track>_f<k>_bench_s<seed>); test predictions are averaged over
seeds within a fold and pooled over the 5 folds, so each utterance of the track is scored once, by a model that
never saw its speaker group (sal.benchmark). Baselines are computed per fold from that fold's training part and
pooled the same way:
  majority          the fold's most frequent training pincode (reference for Recall@1 and state lifts)
  best constant     the location with the lowest mean km over the fold's training utterances (reference for km)
  1-NN transcript   location of the most similar training transcript (TF-IDF over character 1-4-grams), no
                    location learning; W2V2 and ZIPA transcripts
  picture lookup    majority pincode of the training utterances that describe the same picture, else the majority
Reported: Recall@1 over the track's pincodes, speaker-macro Recall@1 (every speaker group counts once), state
accuracy (predicted point snapped to the nearest pincode), mean km; lifts over the majority and km gain over the
best constant, paired per utterance; 95% CIs by bootstrap over speaker groups. Paired probe comparisons on the same
utterances: BiGRU vs shuffled BiGRU, bigram vs unigram, phones vs characters, ZIPA vs W2V2.

Output: results/benchmark/scores{tag}.csv, paired{tag}.csv, folds{tag}.csv (Recall@1 per fold)
Run:    uv run python scripts/benchmark_eval.py [--tag _smoke]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from sal.benchmark import Scorer, find_runs, pool_folds
from sal.paths import OUTPUTS, RESULTS
from sal.probes import geo

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="", help="select runs named ..._bench<tag>_s<seed> (e.g. _smoke)")
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()

B = RESULTS / "benchmark"
sp = pd.read_parquet(B / "splits.parquet")
meta = pd.read_parquet(B / "meta.parquet")[["metadata_idx", "latitude", "longitude"]]
d = sp.merge(meta, on="metadata_idx")
K = sum(c.startswith("all_f") for c in d.columns)
pin_xyz = {p: geo.latlon_to_xyz(torch.tensor(np.radians([r.latitude])), torch.tensor(np.radians([r.longitude]))).numpy()[0]
           for p, r in d.groupby("pincode")[["latitude", "longitude"]].first().iterrows()}
tx = {m: pd.read_parquet(OUTPUTS / "transcripts" / f"{m}_rebuild.parquet", columns=["metadata_idx", "processed_transcript"])
      .set_index("metadata_idx")["processed_transcript"] for m in ("w2v2", "zipa")}
runs = find_runs(RESULTS / "runs", args.tag)
rows, pairs, folds = [], [], []


def baselines(track):
    """Pooled out-of-fold predictions of the baselines: {name: frame metadata_idx, x, y, z, fold}."""
    out = {k: [] for k in ("majority", "best constant", "1-NN transcript (W2V2)", "1-NN transcript (ZIPA)",
                           "picture lookup")}
    for f in range(K):
        col = f"{track}_f{f}"
        tr, te = d[d[col] == "train"], d[d[col] == "test"]
        maj = tr["pincode"].mode().iloc[0]
        const = geo.best_constant(np.radians(tr[["latitude", "longitude"]].to_numpy(np.float64)))
        lk = tr.groupby("image")["pincode"].agg(lambda s: s.mode().iloc[0])
        preds = {"majority": np.tile(pin_xyz[maj], (len(te), 1)), "best constant": np.tile(const, (len(te), 1)),
                 "picture lookup": np.stack([pin_xyz[p] for p in te["image"].map(lk).fillna(maj).astype(int)])}
        for m, name in (("w2v2", "1-NN transcript (W2V2)"), ("zipa", "1-NN transcript (ZIPA)")):
            vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True, min_df=2)
            xa = vec.fit_transform(tx[m].loc[tr["metadata_idx"]])
            nn = np.asarray((vec.transform(tx[m].loc[te["metadata_idx"]]) @ xa.T).argmax(axis=1)).ravel()
            preds[name] = np.stack([pin_xyz[p] for p in tr["pincode"].to_numpy()[nn]])
        for k, v in preds.items():
            out[k].append(pd.DataFrame({"metadata_idx": te["metadata_idx"].to_numpy(), "x": v[:, 0], "y": v[:, 1],
                                        "z": v[:, 2], "fold": f}))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


for track in ("all", "shared"):
    u = d[d[f"{track}_f0"].notna()].sort_values("metadata_idx").reset_index(drop=True)
    sc = Scorer(u)
    fold_of = u.set_index("metadata_idx").apply(lambda r: next(f for f in range(K) if r[f"{track}_f{f}"] == "test"), axis=1)
    base = baselines(track)
    sc.reference = {"majority": sc.hits(sc.align(base["majority"])), "constant": sc.hits(sc.align(base["best constant"]))}
    preds = {k: (sc.align(v), "baseline", 0) for k, v in base.items()}
    for (cfg, tr_), fr in sorted(runs.items()):
        if tr_ != track:
            continue
        if sorted(fr) != list(range(K)):
            print(f"skipped {cfg} ({track}): folds {sorted(fr)} of {K}")
            continue
        pooled = pool_folds(fr)
        preds[cfg] = (sc.align(pooled), "probe", int(pooled["seeds"].min()))
    for name, (p, kind, seeds) in preds.items():
        r = sc.score(p, n=args.boot)
        rows.append({"track": track, "probe": name, "kind": kind, "seeds_per_fold": seeds, **r})
        h = sc.hits(p)["recall1"]
        per = pd.Series(h).groupby(fold_of.loc[u["metadata_idx"]].to_numpy()).mean()
        folds.append({"track": track, "probe": name, **{f"f{f}": per[f] for f in range(K)}, "sd": per.std(ddof=1)})
    for m in ("w2v2", "zipa", "rp_w2v2", "rp_zipa", "rp_whisper"):
        cand = [(f"tp_{m}_rebuild_phone_gru", f"tp_{m}_rebuild_phone_shuffle", "order (BiGRU vs shuffled)"),
                (f"tp_{m}_rebuild_phone_bigram", f"tp_{m}_rebuild_phone_unigram", "order (phone bigram vs unigram)"),
                (f"tp_{m}_rebuild_char_bigram", f"tp_{m}_rebuild_char_unigram", "order (char bigram vs unigram)"),
                (f"tp_{m}_rebuild_phone_bigram", f"tp_{m}_rebuild_char_bigram", "units (phones vs chars, bigram)")]
        for a, b, what in cand:
            if a in preds and b in preds:
                pairs.append({"track": track, "comparison": what, "a": a, "b": b, **sc.paired(preds[a][0], preds[b][0], args.boot)})
    for tok in ("phone_bigram", "phone_gru"):
        a, b = f"tp_zipa_rebuild_{tok}", f"tp_w2v2_rebuild_{tok}"
        if a in preds and b in preds:
            pairs.append({"track": track, "comparison": f"recognizer (ZIPA vs W2V2, {tok})", "a": a, "b": b,
                          **sc.paired(preds[a][0], preds[b][0], args.boot)})

out = pd.DataFrame(rows)
out.to_csv(B / f"scores{args.tag}.csv", index=False)
pd.DataFrame(pairs).to_csv(B / f"paired{args.tag}.csv", index=False)
pd.DataFrame(folds).to_csv(B / f"folds{args.tag}.csv", index=False)
cols = ["track", "probe", "seeds_per_fold", "recall1", "recall1_macro", "recall1_vs_majority", "recall1_vs_majority_lo",
        "recall1_vs_majority_hi", "state", "km", "km_vs_constant"]
print(out[cols].round(1).to_string(index=False))
if pairs:
    print(pd.DataFrame(pairs)[["track", "comparison", "a", "recall1_diff", "recall1_lo", "recall1_hi", "km_diff"]]
          .round(2).to_string(index=False))
