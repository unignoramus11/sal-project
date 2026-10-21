"""Audit of the official Vaani-Hi release (changelinglab/vaani-hi-geo @ 264fb053), from the release alone.

  duplicates       identical audio (hash of the WAV bytes) appearing in several rows, and whether the copies carry
                   different pincodes, genders or splits
  leakage          share of val/test rows whose exact audio is also a train row
  lookup baseline  predict a test row's location from the train row with identical audio (majority pincode of the
                   copies), else the majority training pincode: what memorisation alone scores
  coordinates      each pincode's released coordinates against the India Post directory (data/external/pincode.csv):
                   distance to the median of the pincode's offices with valid coordinates, and the number of offices
                   with invalid ones (outside India's bounding box, e.g. latitude and longitude swapped)
Why duplicates exist (PRiSM's src/recipe/geolocation/local/vaanihindi_sample.py): every district's 4 shards write
their first 600 clips to WAVS/{state}_{district}_{i}.wav with i restarting at 0 per shard, so the shards overwrite
each other's files while the metadata keeps one row per shard and clip index; rows of the same index then share
whichever file was written last, under different labels.

Output: results/official/{duplicates,leakage,lookup_baseline,coordinates_audit,composition}.csv, audit.txt
Run:    uv run python scripts/official_audit.py
"""
import glob
import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from sal.paths import DATA, RESULTS
from sal.probes import geo

OUT = RESULTS / "official"
OUT.mkdir(parents=True, exist_ok=True)
lines = []


def haversine(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6378.1 * np.arcsin(np.sqrt(h))


parts = []
for f in sorted(glob.glob(str(DATA / "vaani_hi_official" / "data" / "*.parquet"))):
    t = pq.read_table(f).to_pandas()
    t["h"] = [hashlib.blake2b(a["bytes"] if isinstance(a, dict) else a, digest_size=12).hexdigest() for a in t["audio"]]
    parts.append(t.drop(columns=["audio"]))
d = pd.concat(parts, ignore_index=True)
d.to_parquet(DATA / "vaani_hi_official" / "audio_hash.parquet", index=False)
lines.append(f"rows {len(d)}, splits {d['split'].value_counts().to_dict()}, pincodes {d['pincode'].nunique()}, "
             f"districts {d['district'].nunique()}, states {d['state'].nunique()}")
comp = d.groupby(["state", "district", "pincode"]).agg(rows=("split", "size")).reset_index()
comp.to_csv(OUT / "composition.csv", index=False)

# duplicates
copies = d.groupby("h").size()
g = d[d["h"].isin(copies[copies > 1].index)].groupby("h").agg(
    copies=("split", "size"), pincodes=("pincode", "nunique"), genders=("gender", "nunique"),
    districts=("district", "nunique"), splits=("split", lambda s: ",".join(sorted(set(s)))))
g.to_csv(OUT / "duplicates.csv")
lines.append(f"distinct audio clips {len(copies)}; copies per clip {copies.value_counts().sort_index().to_dict()}")
lines.append(f"duplicated clips {len(g)} ({int(g['copies'].sum())} rows): copies with different pincodes {(g['pincodes'] > 1).mean():.1%}, "
             f"different genders {(g['genders'] > 1).mean():.1%}, different districts {(g['districts'] > 1).mean():.1%}")

# leakage of exact audio
train_h = set(d.loc[d["split"] == "train", "h"])
leak = [{"split": s, "rows": int((d["split"] == s).sum()), "exact_audio_in_train": float(d.loc[d["split"] == s, "h"].isin(train_h).mean())}
        for s in ("val", "test")]
pd.DataFrame(leak).to_csv(OUT / "leakage.csv", index=False)
for r in leak:
    lines.append(f"{r['split']}: {r['exact_audio_in_train']:.1%} of {r['rows']} rows have their exact audio in train")

# lookup baseline
tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
loc = d.groupby("pincode")[["latitude", "longitude"]].first()
maj = tr["pincode"].mode().iloc[0]
lookup = tr.groupby("h")["pincode"].agg(lambda s: s.mode().iloc[0])
dup = te["h"].isin(lookup.index)


def score(pins, target):
    p = np.radians(loc.loc[pins].to_numpy())
    xyz = geo.latlon_to_xyz(torch.tensor(p[:, 0]), torch.tensor(p[:, 1]))
    return geo.summarise(xyz, np.radians(target[["latitude", "longitude"]].to_numpy()))


res = {"majority": score([maj] * len(te), te),
       "exact_audio_lookup_else_majority": score(te["h"].map(lookup).fillna(maj).astype(int).tolist(), te)}
pd.DataFrame(res).T.to_csv(OUT / "lookup_baseline.csv")
lines.append(f"test rows with their exact audio in train: {dup.mean():.1%}; the train copy has the same pincode in "
             f"{(te.loc[dup, 'pincode'] == te.loc[dup, 'h'].map(lookup)).mean():.1%} of them")
for k, v in res.items():
    lines.append(f"{k}: Recall@1 {v['recall1']:.1f}, mean km {v['mean_km']:.0f} (test {v['n']})")

# coordinates
ref = pd.read_csv(DATA / "external" / "pincode.csv", dtype={"Pincode": int}, low_memory=False)
for c in ("Latitude", "Longitude"):
    ref[c] = pd.to_numeric(ref[c], errors="coerce")
rows = []
for p, o in d.groupby("pincode").agg(lat=("latitude", "first"), lon=("longitude", "first"),
                                    district=("district", "first"), n=("split", "size")).iterrows():
    r = ref[ref["Pincode"] == p]
    ok = r[r["Latitude"].between(6, 38) & r["Longitude"].between(68, 98)]
    mlat, mlon = ok["Latitude"].median(), ok["Longitude"].median()
    rows.append({"pincode": p, "district": o["district"], "rows": o["n"], "released_lat": o["lat"],
                 "released_lon": o["lon"], "offices": len(r), "offices_invalid": len(r) - len(ok),
                 "unfiltered_mean_lat": r["Latitude"].mean(), "unfiltered_mean_lon": r["Longitude"].mean(),
                 "valid_median_lat": mlat, "valid_median_lon": mlon,
                 "km_from_valid_median": haversine(o["lat"], o["lon"], mlat, mlon)})
c = pd.DataFrame(rows).sort_values("km_from_valid_median", ascending=False)
c.to_csv(OUT / "coordinates_audit.csv", index=False)
bad = c[c["km_from_valid_median"] > 100]
for r in bad.itertuples():
    lines.append(f"pincode {r.pincode} ({r.district}, {r.rows} rows): released at {r.released_lat:.2f}N {r.released_lon:.2f}E, "
                 f"{r.km_from_valid_median:.0f} km from its valid offices; {r.offices_invalid} of {r.offices} offices have "
                 f"invalid coordinates and the release equals their unfiltered mean ({r.unfiltered_mean_lat:.4f}, "
                 f"{r.unfiltered_mean_lon:.4f})")
(OUT / "audit.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
