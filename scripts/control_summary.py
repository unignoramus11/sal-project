"""Real vs control-task scores per probe (speaker-identification capacity; sal.data.control).

For each probe with control runs: test Recall@1 and mean km on real labels and on control labels (mean ± sd
over seeds, from metrics.json, which scores against the labels the probe was trained on), each with its own
majority baseline, and the lifts over those baselines. Under PRiSM's random split a control lift above zero
means the probe recognises training speakers in the test set; control_share = control lift / real lift is
the part of the real lift that speaker-constant random labels reproduce.

Output: results/tables/control.csv
Run:    uv run python scripts/control_summary.py
"""
import json
import re

import numpy as np
import pandas as pd
import torch

from sal.data.control import control_targets
from sal.paths import DATA, RESULTS
from sal.probes import geo

runs = RESULTS / "runs"
meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet",
                       columns=["metadata_idx", "latitude", "longitude"])
meta = meta.merge(pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet"), on="metadata_idx")
meta = meta.sort_values("metadata_idx").reset_index(drop=True)
real_t = np.radians(meta[["latitude", "longitude"]].to_numpy(np.float64))
ctrl_t = control_targets(meta)


def majority(t, proto):
    tr, te = meta[proto].to_numpy() == "train", meta[proto].to_numpy() == "test"
    vals, c = np.unique(t[tr], axis=0, return_counts=True)
    m = vals[c.argmax()]
    xyz = geo.latlon_to_xyz(torch.tensor([m[0]]), torch.tensor([m[1]])).expand(te.sum(), 3).double()
    return geo.summarise(xyz, t[te])


def seeds(config):
    ms = [json.loads((d / "metrics.json").read_text())["metrics"]["test"]
          for d in sorted(runs.glob(f"{config}_s*")) if (d / "metrics.json").exists()]
    return np.array([m["recall1"] for m in ms]), np.array([m["mean_km"] for m in ms])


rows = []
for d in sorted({re.sub(r"_s\d+$", "", p.name) for p in runs.glob("*_control_s*")}):
    real = d.replace("_control", "")
    proto = "speaker_split" if "_speaker_split" in d else "prism"
    (r1, km), (c1, ckm) = seeds(real), seeds(d)
    if not len(r1) or not len(c1):
        continue
    mr, mc = majority(real_t, proto), majority(ctrl_t, proto)
    lift_r, lift_c = r1.mean() - mr["recall1"], c1.mean() - mc["recall1"]
    rows.append({"probe": real, "protocol": proto, "seeds_real": len(r1), "seeds_control": len(c1),
                 "real_recall1": r1.mean(), "real_sd": r1.std(ddof=1), "real_majority": mr["recall1"],
                 "control_recall1": c1.mean(), "control_sd": c1.std(ddof=1), "control_majority": mc["recall1"],
                 "real_lift": lift_r, "control_lift": lift_c,
                 "control_share": lift_c / lift_r if lift_r > 0 else np.nan,
                 "real_km": km.mean(), "control_km": ckm.mean(), "control_majority_km": mc["mean_km"]})
t = pd.DataFrame(rows)
(RESULTS / "tables").mkdir(exist_ok=True)
t.to_csv(RESULTS / "tables" / "control.csv", index=False)
print(t.round(2).to_string(index=False))
