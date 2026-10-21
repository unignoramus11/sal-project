"""Scores on the official Vaani-Hi split, separated by whether a test clip's exact audio is also in training.

For every probe run on the official release (results/runs/*_official_prism_s*: TP-GRU, bag probes, ECAPA) the
seed-ensemble prediction is scored on all test rows, on the rows whose byte-identical audio is a train row
("duplicated"), and on the rest ("unique"). Two references are scored the same way: a nearest-neighbour transcript
matcher (TF-IDF over character 1-4-grams, no location learning) and an exact-audio lookup. CIs: bootstrap over
clusters of identical audio (pseudo-speaker clusters once available would be stricter).

Output: results/official/scores_by_duplication.csv
Run:    uv run python scripts/official_eval.py
"""
import re

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from sal.paths import DATA, OUTPUTS, RESULTS
from sal.probes import geo
from sal.stats import cluster_bootstrap

D = DATA / "vaani_hi_official"
h = pd.read_parquet(D / "audio_hash.parquet")[["metadata_idx", "h", "split", "pincode", "latitude", "longitude"]]
train_h = set(h.loc[h["split"] == "train", "h"])
te = h[h["split"] == "test"].copy()
te["dup"] = te["h"].isin(train_h)
loc = h.groupby("pincode")[["latitude", "longitude"]].first()
xyz_pin = {p: geo.latlon_to_xyz(torch.tensor([np.radians(r.latitude)]), torch.tensor([np.radians(r.longitude)])).numpy()[0]
           for p, r in loc.iterrows()}
rows = []


def score(name, pred_xyz):
    """pred_xyz: (len(te), 3) aligned with te."""
    t = torch.tensor(np.radians(te[["latitude", "longitude"]].to_numpy()))
    p = torch.tensor(pred_xyz, dtype=torch.float64)
    hit = 100 * geo.recall_at_k(p, t, 1).numpy().astype(float)
    km = geo.error_km(p, t).numpy()
    r = {"probe": name}
    for part, m in (("all", np.ones(len(te), bool)), ("duplicated", te["dup"].to_numpy()), ("unique", ~te["dup"].to_numpy())):
        ci = cluster_bootstrap(hit[m], te["h"].to_numpy()[m], n=1000)
        r.update({f"{part}_recall1": ci["estimate"], f"{part}_lo": ci["lo"], f"{part}_hi": ci["hi"],
                  f"{part}_km": float(km[m].mean()), f"{part}_n": int(m.sum())})
    rows.append(r)


# probes run on the official release
runs = RESULTS / "runs"
configs = sorted({re.sub(r"_s\d+$", "", d.name) for d in runs.glob("*_official_*prism_s*") if (d / "metrics.json").exists()})
for cfg in configs:
    parts = []
    for d in sorted(runs.glob(f"{cfg}_s*")):
        p = pd.read_parquet(d / "predictions.parquet")
        p = p[p["split"] == "test"]
        v = p[["x", "y", "z"]].to_numpy()
        p[["x", "y", "z"]] = v / np.linalg.norm(v, axis=1, keepdims=True)
        parts.append(p)
    e = pd.concat(parts).groupby("metadata_idx")[["x", "y", "z"]].mean()
    score(f"{cfg} ({len(parts)} seeds)", e.loc[te["metadata_idx"]].to_numpy())

# nearest-neighbour transcript matcher and exact-audio lookup
tr_rows = h[h["split"] == "train"]
tx = pd.read_parquet(OUTPUTS / "transcripts" / "w2v2_official.parquet", columns=["metadata_idx", "processed_transcript"]).set_index("metadata_idx")
vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True, min_df=2)
xa = vec.fit_transform(tx.loc[tr_rows["metadata_idx"], "processed_transcript"])
xb = vec.transform(tx.loc[te["metadata_idx"], "processed_transcript"])
nn = np.asarray((xb @ xa.T).argmax(axis=1)).ravel()
score("nearest-neighbour transcript (1-NN)", np.stack([xyz_pin[p] for p in tr_rows["pincode"].to_numpy()[nn]]))
maj = tr_rows["pincode"].mode().iloc[0]
lookup = tr_rows.groupby("h")["pincode"].agg(lambda s: s.mode().iloc[0])
score("exact-audio lookup, else majority", np.stack([xyz_pin[p] for p in te["h"].map(lookup).fillna(maj).astype(int)]))
score("majority", np.stack([xyz_pin[maj]] * len(te)))

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "official" / "scores_by_duplication.csv", index=False)
cols = ["probe", "all_recall1", "duplicated_recall1", "unique_recall1", "unique_lo", "unique_hi", "all_km", "unique_km"]
print(f"test rows {len(te)}: duplicated {int(te['dup'].sum())}, unique {int((~te['dup']).sum())}")
print(out[cols].round(1).to_string(index=False))


# ---- corrected version (scripts/official_prepare.py): released split on clean rows, and pseudo-speaker split
R = RESULTS / "official"
cm = pd.read_parquet(R / "meta_clean.parquet").merge(pd.read_parquet(R / "splits_clean.parquet"), on=["metadata_idx", "pincode", "state", "district"])
crow = []
for proto in ("prism", "speaker_split"):
    tr_c, te_c = cm[cm[proto] == "train"], cm[cm[proto] == "test"]
    cand = te_c.groupby("pincode")[["latitude", "longitude"]].first()
    t = torch.tensor(np.radians(te_c[["latitude", "longitude"]].to_numpy()))
    xyz_c = {p: geo.latlon_to_xyz(torch.tensor([np.radians(r.latitude)]), torch.tensor([np.radians(r.longitude)])).numpy()[0]
             for p, r in cm.groupby("pincode")[["latitude", "longitude"]].first().iterrows()}

    def cscore(name, pred, proto=proto, te_c=te_c, t=t):
        p = torch.tensor(pred, dtype=torch.float64)
        hit = 100 * geo.recall_at_k(p, t, 1).numpy().astype(float)
        ci = cluster_bootstrap(hit, te_c["speaker"].to_numpy(), n=1000)
        crow.append({"protocol": proto, "probe": name, "recall1": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"],
                     "mean_km": float(geo.error_km(p, t).numpy().mean()), "test_rows": len(te_c),
                     "test_pseudo_speakers": te_c["speaker"].nunique()})

    maj_c = tr_c["pincode"].mode().iloc[0]
    cscore("majority", np.stack([xyz_c[maj_c]] * len(te_c)))
    xa = vec.fit_transform(tx.loc[tr_c["metadata_idx"], "processed_transcript"])
    xb = vec.transform(tx.loc[te_c["metadata_idx"], "processed_transcript"])
    nn = np.asarray((xb @ xa.T).argmax(axis=1)).ravel()
    cscore("nearest-neighbour transcript (1-NN)", np.stack([xyz_c[p] for p in tr_c["pincode"].to_numpy()[nn]]))
    for cfg in sorted({re.sub(r"_s\d+$", "", d.name) for d in runs.glob(f"*_official_*{proto}_clean_s*") if (d / "metrics.json").exists()}):
        parts = []
        for d in sorted(runs.glob(f"{cfg}_s*")):
            p = pd.read_parquet(d / "predictions.parquet")
            p = p[p["split"] == "test"]
            v = p[["x", "y", "z"]].to_numpy()
            p[["x", "y", "z"]] = v / np.linalg.norm(v, axis=1, keepdims=True)
            parts.append(p)
        e = pd.concat(parts).groupby("metadata_idx")[["x", "y", "z"]].mean()
        cscore(f"{cfg} ({len(parts)} seeds)", e.loc[te_c["metadata_idx"]].to_numpy())
cr = pd.DataFrame(crow)
cr.to_csv(R / "scores_clean.csv", index=False)
print(cr.round(1).to_string(index=False))
