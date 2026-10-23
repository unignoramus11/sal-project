"""Three checks of the rebuilt Vaani-Hi evaluation, scored on the existing runs (no retraining).

  look-alike codes  speaker codes come from file names. Codes of one district whose voices are as close as two halves
                    of one speaker's recordings form a speaker group (scripts/make_benchmark.py). Test utterances of
                    the speaker-disjoint split whose group has a code in train are "look-alike"; every speaker-split
                    probe is rescored on all test utterances and on the strict rest, as lift over the majority
                    baseline of the same utterances (bootstrap over speaker groups)
  picture lookup    predict the location from the picture ID alone (majority pincode of the train utterances describing
                    the same picture, else the train majority), on all utterances and on shared-picture ones: how
                    much the prompt reveals once speakers are disjoint
  coordinates       each pincode's coordinate in the rebuild (mean of its India Post offices, as in PRiSM) against the
                    densest cluster of its offices (sal.data.rebuild.robust_pincode_coordinates), for the rebuild and
                    the official release; and how much of each probe's mean error the misplaced pincodes carry

Needs results/benchmark/{speaker_groups,coordinates}.csv (scripts/make_benchmark.py).
Output: results/rebuild_checks/{lookalike.csv, picture_lookup.csv, coordinates.csv, report.txt}
Run:    uv run python scripts/rebuild_checks.py
"""
import re

import numpy as np
import pandas as pd
import torch

from sal.data.rebuild import robust_pincode_coordinates
from sal.paths import DATA, RESULTS
from sal.probes import geo
from sal.stats import cluster_bootstrap

OUT = RESULTS / "rebuild_checks"
OUT.mkdir(parents=True, exist_ok=True)
lines = []

src = DATA / "vaani_hi_rebuild"
meta = pd.read_parquet(src / "vaani_hi_rebuild.parquet", columns=["metadata_idx", "latitude", "longitude"])
d = (meta.merge(pd.read_parquet(src / "splits.parquet"), on="metadata_idx")
         .merge(pd.read_parquet(src / "prompts.parquet")[["metadata_idx", "prompt_type", "image"]], on="metadata_idx"))
for c in ("speaker", "state", "district"):
    d[c] = d[c].astype(object)
grp = pd.read_csv(RESULTS / "benchmark" / "speaker_groups.csv", index_col=0)["group"]
d["group"] = d["speaker"].map(grp)
assert d["group"].notna().all()


def xyz(latlon_deg):
    r = np.radians(np.asarray(latlon_deg, dtype=np.float64))
    return geo.latlon_to_xyz(torch.tensor(r[:, 0]), torch.tensor(r[:, 1])).numpy()


def ensemble(cfg, idx):
    """Seed-ensemble test prediction of a run config, aligned with metadata_idx `idx`."""
    parts = []
    for f in sorted((RESULTS / "runs").glob(f"{cfg}_s*/predictions.parquet")):
        p = pd.read_parquet(f)
        p = p[p["split"] == "test"].copy()
        v = p[["x", "y", "z"]].to_numpy()
        p[["x", "y", "z"]] = v / np.linalg.norm(v, axis=1, keepdims=True)
        parts.append(p)
    e = pd.concat(parts).groupby("metadata_idx")[["x", "y", "z"]].mean()
    return e.loc[idx].to_numpy(), len(parts)


# ---- 1. look-alike speaker codes across the speaker-disjoint split
tr = d[d["speaker_split"] == "train"]
te = d[d["speaker_split"] == "test"].reset_index(drop=True)
te["lookalike"] = te["group"].isin(set(tr["group"]))
t = torch.tensor(np.radians(te[["latitude", "longitude"]].to_numpy()))
cand = torch.unique(t, dim=0)
maj = tr["pincode"].mode().iloc[0]
maj_xyz = xyz(tr.loc[tr["pincode"] == maj, ["latitude", "longitude"]].iloc[:1].to_numpy())
maj_hit = 100 * geo.recall_at_k(torch.tensor(np.repeat(maj_xyz, len(te), 0)), t, 1, cand).numpy().astype(float)
lines.append(f"look-alike codes: {int(te['lookalike'].sum())} of {len(te)} speaker-split test utterances "
             f"({te.loc[te['lookalike'], 'speaker'].nunique()} of {te['speaker'].nunique()} test speakers) have a "
             f"look-alike code in train")
cfgs = sorted({re.sub(r"_s\d+$", "", p.name) for p in (RESULTS / "runs").glob("*_rebuild_*speaker_split*_s*")
               if (p / "predictions.parquet").exists() and "control" not in p.name and "generic" not in p.name})
rows = []
for cfg in cfgs:
    pred, n = ensemble(cfg, te["metadata_idx"])
    hit = 100 * geo.recall_at_k(torch.tensor(pred), t, 1, cand).numpy().astype(float)
    km = geo.error_km(torch.tensor(pred), t).numpy()
    r = {"probe": cfg, "seeds": n}
    for part, m in (("all", np.ones(len(te), bool)), ("lookalike", te["lookalike"].to_numpy()),
                    ("strict", ~te["lookalike"].to_numpy())):
        ci = cluster_bootstrap(hit[m] - maj_hit[m], te["group"].to_numpy()[m], n=1000)
        r.update({f"{part}_recall1": hit[m].mean(), f"{part}_majority": maj_hit[m].mean(), f"{part}_lift": ci["estimate"],
                  f"{part}_lift_lo": ci["lo"], f"{part}_lift_hi": ci["hi"], f"{part}_km": km[m].mean(), f"{part}_n": int(m.sum())})
    rows.append(r)
la = pd.DataFrame(rows)
la.to_csv(OUT / "lookalike.csv", index=False)
lines.append(la[["probe", "all_lift", "strict_lift", "strict_lift_lo", "strict_lift_hi", "lookalike_lift"]]
             .round(1).to_string(index=False))

# ---- 2. picture lookup on shared-picture utterances
prow = []
for prompts, proto in [(pr, pt) for pr in ("shared", "all") for pt in ("prism", "speaker_split")]:
    g = d[d["prompt_type"] == "GENERIC"] if prompts == "shared" else d
    gtr, gte = g[g[proto] == "train"], g[g[proto] == "test"]
    m = gtr["pincode"].mode().iloc[0]
    lk = gtr.groupby("image")["pincode"].agg(lambda s: s.mode().iloc[0])
    pin = gte["image"].map(lk).fillna(m).astype(int)
    st = gtr.groupby("image")["state"].agg(lambda s: s.mode().iloc[0])
    pst = gte["image"].map(st).fillna(gtr["state"].mode().iloc[0])
    for name, hit in (("picture lookup", 100.0 * (pin.to_numpy() == gte["pincode"].to_numpy())),
                      ("majority", 100.0 * (gte["pincode"].to_numpy() == m))):
        ci = cluster_bootstrap(hit, gte["speaker"].to_numpy(), n=1000)
        state = (pst.to_numpy() == gte["state"].to_numpy()) if name == "picture lookup" else \
            (gte["state"].to_numpy() == gtr["state"].mode().iloc[0])
        prow.append({"prompts": prompts, "protocol": proto, "baseline": name, "recall1": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"],
                     "state_acc": 100 * state.mean(), "test_utts": len(gte), "test_speakers": gte["speaker"].nunique(),
                     "test_pictures_unseen": 100 * (~gte["image"].isin(lk.index)).mean()})
pl = pd.DataFrame(prow)
pl.to_csv(OUT / "picture_lookup.csv", index=False)
lines.append("picture lookup:\n" + pl.round(1).to_string(index=False))

# ---- 3. coordinates
rob = robust_pincode_coordinates(DATA / "external" / "pincode.csv")
crow = []
for name, path in (("rebuild", src / "vaani_hi_rebuild.parquet"),
                   ("official", DATA / "vaani_hi_official" / "audio_hash.parquet")):
    m = pd.read_parquet(path, columns=["pincode", "district", "latitude", "longitude"])
    c = m.groupby("pincode").agg(district=("district", "first"), rows=("district", "size"),
                                 lat=("latitude", "first"), lon=("longitude", "first"))
    r = rob.loc[c.index.astype(str)]
    a, b = np.radians(c[["lat", "lon"]].to_numpy(np.float64)), np.radians(r[["latitude", "longitude"]].to_numpy())
    c["km_from_cluster"] = 2 * 6378.1 * np.arcsin(np.sqrt(
        np.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2))
    c["cluster_lat"], c["cluster_lon"] = r["latitude"].to_numpy(), r["longitude"].to_numpy()
    c["offices"], c["cluster"] = r["offices"].to_numpy(), r["cluster"].to_numpy()
    crow.append(c.reset_index().assign(data=name))
    off = c[c["km_from_cluster"] > 50].sort_values("km_from_cluster", ascending=False)
    lines.append(f"coordinates, {name}: {len(c)} pincodes; median shift to the office cluster "
                 f"{c['km_from_cluster'].median():.0f} km; above 50 km: "
                 + (", ".join(f"{p} ({o.district}, {o.rows} rows) {o.km_from_cluster:.0f} km" for p, o in off.iterrows()) or "none"))
co = pd.concat(crow)
co.to_csv(OUT / "coordinates.csv", index=False)

# share of each probe's mean error from the misplaced rebuild pincodes, and its distance structure
bad = co[(co["data"] == "rebuild") & (co["km_from_cluster"] > 50)]["pincode"].tolist()
mis = te["pincode"].isin(bad).to_numpy()
lines.append(f"rebuild speaker-split test utterances at misplaced pincodes {bad}: {int(mis.sum())} of {len(te)}")
rc = co[co["data"] == "rebuild"].set_index("pincode")


def km(p, q, cols):
    a, b = np.radians(rc.loc[p, cols].to_numpy(np.float64)), np.radians(rc.loc[q, cols].to_numpy(np.float64))
    return 2 * 6378.1 * np.arcsin(np.sqrt(np.sin((b[0] - a[0]) / 2) ** 2 + np.cos(a[0]) * np.cos(b[0]) * np.sin((b[1] - a[1]) / 2) ** 2))


for p in bad:
    others = [q for q in rc.index if q != p]
    near_r = min(others, key=lambda q: km(p, q, ["lat", "lon"]))
    near_c = min(others, key=lambda q: km(p, q, ["cluster_lat", "cluster_lon"]))
    lines.append(f"  {p} ({rc.loc[p, 'district']}): nearest other pincode in the rebuild {near_r} "
                 f"({rc.loc[near_r, 'district']}, {km(p, near_r, ['lat', 'lon']):.0f} km); at its office cluster "
                 f"{near_c} ({rc.loc[near_c, 'district']}, {km(p, near_c, ['cluster_lat', 'cluster_lon']):.0f} km)")
(OUT / "report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
