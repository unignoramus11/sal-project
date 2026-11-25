"""State-level accuracy: is the location signal regional rather than pincode-level?

Each seed-ensemble prediction (scripts/summarise.py: ensemble) is snapped to the nearest candidate location of
the test split, as Recall@1 does, and scored as correct if that location lies in the true state. Reported for
every config on all test utterances and on the shared-picture subset, with a 95% speaker-cluster bootstrap CI,
next to the same score for the majority baseline (the most frequent training location of the same protocol
and prompt subset) and the paired difference.

Output: results/tables/state_level.csv
Run:    uv run python scripts/state_level.py
"""

import numpy as np
import pandas as pd
import torch
from summarise import config_of, ensemble

from sal.data.prompts import prompt_type
from sal.paths import DATA, RESULTS
from sal.probes import geo
from sal.stats import paired_cluster_bootstrap

meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet",
                       columns=["metadata_idx", "latitude", "longitude"])
meta = meta.merge(pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet"), on="metadata_idx")
meta = meta.set_index("metadata_idx")
meta["generic"] = prompt_type(meta.index) == "GENERIC"


def snapped_state(pred_xyz: np.ndarray, test_idx: pd.Index) -> np.ndarray:
    cand = meta.loc[test_idx].groupby("pincode").agg(lat=("latitude", "first"), lon=("longitude", "first"),
                                                     state=("state", "first"))
    cxyz = geo.latlon_to_xyz(torch.tensor(np.radians(cand["lat"].to_numpy())),
                             torch.tensor(np.radians(cand["lon"].to_numpy()))).numpy()
    p = pred_xyz / np.linalg.norm(pred_xyz, axis=1, keepdims=True)
    return cand["state"].to_numpy()[np.argmax(p @ cxyz.T, axis=1)]


rows = []
configs = sorted({config_of(d.name)[0] for d in (RESULTS / "runs").iterdir() if (d / "metrics.json").exists()})
for cfg in configs:
    if "_control" in cfg or "_official_" in cfg or "_bench" in cfg:   # other data/splits: official_eval.py, benchmark_eval.py
        continue
    e = ensemble(cfg)
    idx = pd.Index(e["metadata_idx"])
    true = meta.loc[idx, "state"].to_numpy()
    hit = (snapped_state(e[["x", "y", "z"]].to_numpy(), idx) == true).astype(float)
    proto = "speaker_split" if "_speaker_split" in cfg else "prism"
    tr = meta[meta[proto] == "train"]
    if cfg.endswith("_generic"):
        tr = tr[tr["generic"]]
    maj_state = tr.loc[tr["pincode"] == tr["pincode"].mode().iloc[0], "state"].iloc[0]
    maj_hit = (true == maj_state).astype(float)
    spk = e["speaker"].to_numpy()
    g = meta.loc[idx, "generic"].to_numpy()
    r = {"config": cfg}
    for name, m in (("all", np.ones(len(e), bool)), ("shared", g)):
        if m.sum() == 0:
            continue
        d = paired_cluster_bootstrap(100 * hit[m], 100 * maj_hit[m], spk[m], n=1000)
        r.update({f"{name}_state_acc": 100 * hit[m].mean(), f"{name}_majority": 100 * maj_hit[m].mean(),
                  f"{name}_lift": d["diff"], f"{name}_lift_lo": d["lo"], f"{name}_lift_hi": d["hi"]})
    rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(RESULTS / "tables" / "state_level.csv", index=False)
print(t.round(1).to_string(index=False))
