"""A speaker-disjoint, topic-controlled benchmark from the rebuilt Vaani-Hi, with every speaker tested once.

What it changes, and why (docs/benchmark.md):
  speakers        PRiSM's random split puts the speaker of 97.9% of test utterances in train: folds are speaker-disjoint
  look-alike      speaker codes come from file names; codes of one district whose voices are as close as two halves
  codes           of one speaker's recordings (ECAPA centroid cosine above the 5th percentile of same-speaker
                  split-half cosines; average-linkage clustering within district) form one speaker group, and groups
                  never cross folds
  test size       one speaker-disjoint split tests 223 speakers (119 on shared pictures): 5-fold cross-validation
                  over speaker groups, pincode-stratified, so every group is tested once
  topic           68% of utterances describe a picture named after the speaker's own district: track "shared" trains
                  and tests on shared-picture utterances only; pincodes with fewer than 4 speaker groups on shared
                  pictures are left out of it, so a tested pincode always has training speakers
  duplicates      byte-identical audio: one copy kept
  coordinates     the rebuild (like PRiSM) places a pincode at the mean of its India Post offices, and some offices
                  are misplaced by hundreds of km: every pincode is placed at the densest cluster of its offices
Fold f of a track: test = speaker groups of fold f; val = 12.5% of the other groups (pincode-stratified); train = the
rest. Both tracks use the same fold of each group, so the same speakers are tested in fold f of both.
Per-speaker weighting and baselines are in scripts/benchmark_eval.py.

Output: results/benchmark/splits.parquet (metadata_idx, speaker, group, pincode, state, district, prompt_type, image,
        all_f0..all_f4, shared_f0..shared_f4 with train/val/test, missing = not in the track),
        results/benchmark/meta.parquet (metadata_idx, latitude, longitude, pincode, state, district),
        results/benchmark/speaker_groups.csv, coordinates.csv, report.txt
Run:    uv run python scripts/make_benchmark.py
"""
import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import AgglomerativeClustering

from sal.data.rebuild import robust_pincode_coordinates
from sal.data.splits import leakage, speaker_disjoint_splits, speaker_folds
from sal.paths import DATA, OUTPUTS, RESULTS

K, MIN_GROUPS = 5, 4
OUT = RESULTS / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
lines = []

src = DATA / "vaani_hi_rebuild"
t = pq.read_table(src / "vaani_hi_rebuild.parquet", columns=["metadata_idx", "audio", "speaker", "pincode", "state",
                                                            "district", "latitude", "longitude"]).to_pandas()
t["h"] = [hashlib.blake2b(a["bytes"] if isinstance(a, dict) else a, digest_size=12).hexdigest() for a in t.pop("audio")]
for c in ("speaker", "state", "district"):
    t[c] = t[c].astype(object)
t[["latitude", "longitude"]] = t[["latitude", "longitude"]].astype(np.float64)
d = t.merge(pd.read_parquet(src / "prompts.parquet")[["metadata_idx", "prompt_type", "image"]], on="metadata_idx")
assert len(d) == len(t)

# duplicates
dup = d[d["h"].duplicated(keep=False)].groupby("h").agg(rows=("metadata_idx", "size"), speakers=("speaker", "nunique"))
d = d.sort_values("metadata_idx").drop_duplicates("h", keep="first").reset_index(drop=True)
lines.append(f"duplicates: {len(dup)} recordings appear more than once ({int(dup['rows'].sum())} rows, "
             f"{int((dup['speakers'] > 1).sum())} across speaker codes); one copy kept -> {len(d)} rows")

# coordinates: the densest cluster of each pincode's India Post offices (sal.data.rebuild.robust_pincode_coordinates)
rob = robust_pincode_coordinates(DATA / "external" / "pincode.csv")
loc = d.groupby("pincode")[["latitude", "longitude"]].first()
new = rob.loc[loc.index.astype(str), ["latitude", "longitude"]].to_numpy()
a, b = np.radians(loc.to_numpy()), np.radians(new)
km = 2 * 6378.1 * np.arcsin(np.sqrt(np.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0])
                                    * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2))
moved = pd.DataFrame({"pincode": loc.index, "district": d.groupby("pincode")["district"].first().loc[loc.index].values,
                      "rows": d.groupby("pincode").size().loc[loc.index].values, "rebuild_lat": loc["latitude"].values,
                      "rebuild_lon": loc["longitude"].values, "lat": new[:, 0], "lon": new[:, 1], "km": km,
                      "offices": rob.loc[loc.index.astype(str), "offices"].values,
                      "cluster": rob.loc[loc.index.astype(str), "cluster"].values}).sort_values("km", ascending=False)
moved.to_csv(OUT / "coordinates.csv", index=False)
lines.append(f"coordinates: each pincode placed at the densest cluster of its offices; shift from the rebuild's mean "
             f"of offices: median {np.median(km):.0f} km, above 50 km for "
             + ", ".join(f"{r.pincode} ({r.district}, {r.rows} rows, {r.km:.0f} km; cluster {r.cluster} of {r.offices} offices)"
                         for r in moved[moved["km"] > 50].itertuples()))
for p, (la, lo) in zip(loc.index, new, strict=True):
    d.loc[d["pincode"] == p, ["latitude", "longitude"]] = [la, lo]

# look-alike speaker codes
x = np.load(OUTPUTS / "features" / "ecapa_rebuild" / "emb.npy").astype(np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True)
row = pd.read_parquet(OUTPUTS / "features" / "ecapa_rebuild" / "index.parquet").reset_index().set_index("metadata_idx")["index"]
assert d["metadata_idx"].isin(row.index).all()
d["emb_row"] = d["metadata_idx"].map(row)
codes = d.groupby("speaker").agg(district=("district", "first"), n=("metadata_idx", "size"),
                                 rows=("emb_row", list))
assert d.groupby("speaker")["district"].nunique().max() == 1
cent = np.stack([x[r].mean(0) for r in codes["rows"]])
cent /= np.linalg.norm(cent, axis=1, keepdims=True)
half = []
for r in codes["rows"]:
    if len(r) >= 4:
        p = rng.permutation(r)
        u, v = x[p[: len(p) // 2]].mean(0), x[p[len(p) // 2:]].mean(0)
        half.append(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
sim = float(np.quantile(half, 0.05))
grp = pd.Series(index=codes.index, dtype=object)
for di, g in codes.groupby("district"):
    k = np.flatnonzero(codes.index.isin(g.index))
    lab = (AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=1 - sim)
           .fit_predict(cent[k]) if len(k) > 1 else [0])
    grp.iloc[k] = [f"{di}_g{q}" for q in lab]
codes["group"] = grp
d["group"] = d["speaker"].map(grp)
per = codes.groupby("group").agg(codes=("n", "size"), utts=("n", "sum"))
lines.append(f"speaker groups: same-speaker split-half centroid cosine, 5th percentile {sim:.3f} ({len(half)} codes with "
             f">= 4 utterances); {len(codes)} codes -> {len(per)} groups, {int((per['codes'] > 1).sum())} with several "
             f"codes (max {per['codes'].max()} codes, largest group {per['utts'].max()} utterances = "
             f"{100 * per['utts'].max() / len(d):.1f}%)")
codes.drop(columns="rows").to_csv(OUT / "speaker_groups.csv")

# folds over speaker groups
d["fold"] = speaker_folds(d, k=K, speaker_col="group", group_col="pincode", seed=42)
assert d.groupby("group")["fold"].nunique().max() == 1
for track in ("all", "shared"):
    m = d if track == "all" else d[d["prompt_type"] == "GENERIC"]
    n_grp = m.groupby("pincode")["group"].nunique()
    keep = n_grp.index[n_grp >= MIN_GROUPS]
    if len(keep) < len(n_grp):
        lines.append(f"{track}: pincodes left out (fewer than {MIN_GROUPS} speaker groups): "
                     f"{ {int(p): int(n_grp[p]) for p in n_grp.index.difference(keep)} }")
    m = m[m["pincode"].isin(keep)]
    lines.append(f"{track}: {len(m)} utterances, {m['speaker'].nunique()} speaker codes, {m['group'].nunique()} groups, "
                 f"{m['pincode'].nunique()} pincodes")
    for f in range(K):
        col = f"{track}_f{f}"
        rest = m[m["fold"] != f]
        sv = speaker_disjoint_splits(rest, speaker_col="group", group_col="pincode",
                                     fractions={"train": 0.875, "val": 0.125, "test": 0.0}, seed=42 + f)
        d[col] = pd.Series(np.where(m["fold"] == f, "test", None), index=m.index)
        d.loc[rest.index, col] = sv
        lk = leakage(d.dropna(subset=[col]), col, speaker_col="group")
        part = d[col].value_counts()
        lines.append(f"  {col}: train {part.get('train', 0)}, val {part.get('val', 0)}, test {part.get('test', 0)} "
                     f"utterances; test groups {d.loc[d[col] == 'test', 'group'].nunique()}; group leakage {lk['test']:.0%}; "
                     f"test pincodes without training groups "
                     f"{len(set(d.loc[d[col] == 'test', 'pincode']) - set(d.loc[d[col] == 'train', 'pincode']))}")
    tested = pd.concat([d.loc[d[f"{track}_f{f}"] == "test", "metadata_idx"] for f in range(K)])
    assert tested.is_unique and len(tested) == len(m), track
    lines.append(f"  {track}: every utterance of the track is tested exactly once")

cols = ["metadata_idx", "speaker", "group", "pincode", "state", "district", "prompt_type", "image"]
folds = [f"{t}_f{f}" for t in ("all", "shared") for f in range(K)]
d[cols + folds].to_parquet(OUT / "splits.parquet", index=False)
d[["metadata_idx", "latitude", "longitude", "pincode", "state", "district"]].to_parquet(OUT / "meta.parquet", index=False)
(OUT / "report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
