"""Reference baselines that use no speech content.

  centroid          predict the mean training location (normalised xyz) for every utterance
  majority          predict the most frequent training location
  best_constant     predict the location with the lowest mean km over training utterances (spherical
                    geometric median): the strongest content-free reference for km errors
  speaker_lookup    predict the location where the same speaker's training utterances were recorded
                    (falls back to `majority` when the speaker is not in train). It needs only the speaker
                    ID, so its score under a split is the share of the task solvable by speaker memorisation.

With prompts = generic, every split is first restricted to utterances about shared pictures (as the probes'
--prompts generic runs are), so their scores have a matching reference.

Output: results/baselines.csv (one row per baseline, protocol, prompt subset and evaluated split)
Run:    uv run python scripts/baselines.py
"""
import numpy as np
import pandas as pd
import torch

from sal.data.prompts import prompt_type
from sal.paths import DATA, RESULTS
from sal.probes import geo

meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet",
                       columns=["metadata_idx", "latitude", "longitude"])
meta = meta.merge(pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet"), on="metadata_idx")
lat = np.radians(meta["latitude"].to_numpy())
lon = np.radians(meta["longitude"].to_numpy())
targets = np.stack([lat, lon], 1)
xyz = geo.latlon_to_xyz(torch.tensor(lat), torch.tensor(lon)).numpy()
generic = prompt_type(meta["metadata_idx"]) == "GENERIC"

rows = []
for proto, prompts in [(p, k) for k in ("all", "generic") for p in ("prism", "speaker_split")]:
    keep = generic if prompts == "generic" else np.ones(len(meta), bool)
    tr = (meta[proto].to_numpy() == "train") & keep
    majority_pin = meta.loc[tr, "pincode"].mode().iloc[0]
    maj_xyz = xyz[np.flatnonzero(meta["pincode"].to_numpy() == majority_pin)[0]]
    cen_xyz = xyz[tr].mean(0)
    const_xyz = geo.best_constant(targets[tr])
    spk_home = meta[tr].groupby("speaker")["pincode"].agg(lambda s: s.mode().iloc[0])
    pin_xyz = {p: xyz[np.flatnonzero(meta["pincode"].to_numpy() == p)[0]] for p in meta["pincode"].unique()}
    for split in ("val", "test"):
        m = (meta[proto].to_numpy() == split) & keep
        t = targets[m]
        preds = {
            "centroid": np.repeat(cen_xyz[None], m.sum(), 0),
            "majority": np.repeat(maj_xyz[None], m.sum(), 0),
            "best_constant": np.repeat(const_xyz[None], m.sum(), 0),
        }
        spk = meta.loc[m, "speaker"].to_numpy()
        known = np.array([s in spk_home.index for s in spk])
        preds["speaker_lookup"] = np.stack([pin_xyz[spk_home[s]] if k else maj_xyz for s, k in zip(spk, known)])
        for name, p in preds.items():
            s = geo.summarise(p, t)
            rows.append({"baseline": name, "protocol": proto, "prompts": prompts, "split": split, **s,
                         "speaker_coverage": float(known.mean()) if name == "speaker_lookup" else np.nan})

df = pd.DataFrame(rows)
RESULTS.mkdir(exist_ok=True)
df.to_csv(RESULTS / "baselines.csv", index=False)
print(df[df.split == "test"].round(2).to_string(index=False))
