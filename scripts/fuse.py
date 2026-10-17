"""Late fusion of two probes at the prediction level (RQ3: transcript probe + prosody probe).

For each seed present in both configs: unit-normalise both predicted directions, combine as
w * a + (1 - w) * b, pick w in {0, 0.05, ..., 1} by validation mean km, and score test with that w.
w = 1 means "a alone", w = 0 "b alone", so fusion can only beat a single probe if both carry information.

Output: results/tables/fusion_<a>__<b>.csv (one row per seed) and a printed summary.
Run:    uv run python scripts/fuse.py tp_zipa_rebuild_phone_gru_speaker_split prosody_gru_speaker_split
"""
import sys

import numpy as np
import pandas as pd
import torch

from sal.paths import DATA, RESULTS
from sal.probes import geo

a_cfg, b_cfg = sys.argv[1], sys.argv[2]
runs = RESULTS / "runs"
coords = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet",
                         columns=["metadata_idx", "latitude", "longitude"]).set_index("metadata_idx")


def load(cfg, seed):
    p = pd.read_parquet(runs / f"{cfg}_s{seed}" / "predictions.parquet").set_index(["split", "metadata_idx"]).sort_index()
    v = p[["x", "y", "z"]].to_numpy()
    return p.index, v / np.linalg.norm(v, axis=1, keepdims=True)


def score(idx, pred):
    t = torch.tensor(np.radians(coords.loc[idx.get_level_values(1), ["latitude", "longitude"]].to_numpy()))
    return geo.summarise(torch.tensor(pred), t)


seeds = sorted({int(d.name.rsplit("_s", 1)[1]) for d in runs.glob(f"{a_cfg}_s*")} &
               {int(d.name.rsplit("_s", 1)[1]) for d in runs.glob(f"{b_cfg}_s*")})
rows = []
for seed in seeds:
    ia, pa = load(a_cfg, seed)
    ib, pb = load(b_cfg, seed)
    assert ia.equals(ib), "the two runs must cover the same utterances"
    val = ia.get_level_values(0) == "val"
    test = ia.get_level_values(0) == "test"
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    val_km = [score(ia[val], w * pa[val] + (1 - w) * pb[val])["mean_km"] for w in grid]
    w = float(grid[int(np.argmin(val_km))])
    rows.append({"seed": seed, "w": w,
                 **{f"fused_{k}": v for k, v in score(ia[test], w * pa[test] + (1 - w) * pb[test]).items()},
                 **{f"a_{k}": v for k, v in score(ia[test], pa[test]).items()},
                 **{f"b_{k}": v for k, v in score(ia[test], pb[test]).items()}})
df = pd.DataFrame(rows)
out = RESULTS / "tables"
out.mkdir(exist_ok=True)
df.to_csv(out / f"fusion_{a_cfg}__{b_cfg}.csv", index=False)
print(df[["seed", "w", "a_recall1", "b_recall1", "fused_recall1", "a_mean_km", "b_mean_km", "fused_mean_km"]].round(2).to_string(index=False))
