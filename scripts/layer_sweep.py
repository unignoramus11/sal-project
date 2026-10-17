"""Layer sweep on time-averaged features: which layers carry location information?

For every layer of `pooled.npy` (N, L, D) a small MLP (LayerNorm, 256 hidden, angular loss) is trained on
the train split and scored on validation. Test is never used. Layers are ranked by validation mean km
under the chosen protocol; the top k are the ones whose full frames get cached for weighted-layer probes.

Output: results/layer_sweep/<features>_<protocol>.csv (one row per layer and seed) and a printed ranking.
Run:    python scripts/layer_sweep.py --features outputs/features/w2v2_rebuild --protocol speaker_split
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn

from sal.paths import DATA, RESULTS
from sal.probes.tp_data import DenseBatcher
from sal.probes.train import TrainConfig, train_probe

ap = argparse.ArgumentParser()
ap.add_argument("--features", type=Path, required=True)
ap.add_argument("--protocol", choices=["prism", "speaker_split"], default="speaker_split")
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
ap.add_argument("--dataset", type=Path, default=DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet")
ap.add_argument("--splits", type=Path, default=DATA / "vaani_hi_rebuild" / "splits.parquet")
ap.add_argument("--device", default="cpu")
ap.add_argument("--top", type=int, default=3)
args = ap.parse_args()

meta_f = json.loads((args.features / "meta.json").read_text())
index = pd.read_parquet(args.features / "index.parquet")
pooled = np.load(args.features / "pooled.npy", mmap_mode="r")
meta = pd.read_parquet(args.dataset, columns=["metadata_idx", "latitude", "longitude"]).merge(
    pd.read_parquet(args.splits), on="metadata_idx")
row = {mi: k for k, mi in enumerate(index["metadata_idx"])}
meta = meta[meta["metadata_idx"].isin(row)].sort_values("metadata_idx").reset_index(drop=True)
take = np.array([row[mi] for mi in meta["metadata_idx"]])
targets = np.radians(meta[["latitude", "longitude"]].to_numpy(np.float64))
splits = {s: np.flatnonzero(meta[args.protocol].to_numpy() == s) for s in ("train", "val")}


def mlp(d):
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 3))


rows = []
for l, name in enumerate(meta_f["layers"]):
    w = meta_f["widths"][l]
    x = np.ascontiguousarray(pooled[take, l, :w], dtype=np.float32)
    for seed in args.seeds:
        cfg = TrainConfig(lr=1e-3, batch_size=256, min_epochs=5, max_epochs=60, patience=8, val_per_epoch=1,
                          seed=seed, device=args.device)
        r = train_probe(lambda: mlp(w), DenseBatcher(x, targets), targets, splits, cfg, log=lambda s: None)
        m = r["metrics"]["val"]
        rows.append({"layer": l, "name": name, "seed": seed, "val_recall1": m["recall1"],
                     "val_mean_km": m["mean_km"], "val_median_km": m["median_km"], "epochs": r["epochs_run"]})
        print(f"layer {l:2d} {name:10s} seed {seed}: val R@1 {m['recall1']:.1f}, mean km {m['mean_km']:.0f}", flush=True)

df = pd.DataFrame(rows)
out = RESULTS / "layer_sweep"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / f"{args.features.name}_{args.protocol}.csv", index=False)
rank = df.groupby(["layer", "name"])[["val_mean_km", "val_recall1"]].mean().sort_values("val_mean_km")
print(rank.to_string())
print("top layers by validation km:", list(rank.index.get_level_values("layer")[: args.top]))
